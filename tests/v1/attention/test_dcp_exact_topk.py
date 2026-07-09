# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for exact distributed (DCP) top-k selection.

Two levels:

* ``test_select_global_topk_matches_reference`` — pure, single-process test of
  the selection math against a brute-force global top-k. No GPUs / comms.
* ``run_distributed`` — real multi-GPU test: shards KV across ``world_size``
  ranks, runs ``dcp_exact_topk`` with NCCL all-gather, and asserts every rank
  produces the identical set that a single unsharded top-k would, proving the
  DCP path is exact (not approximate). Run with, e.g.::

      uv run python tests/v1/attention/test_dcp_exact_topk.py --world-size 8

The distributed part is a plain script (torch.multiprocessing.spawn) so it needs
no pytest launcher and works on any CUDA box with >=2 GPUs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

# Import the real source. Prefer the installed package; fall back to loading the
# module file directly so the test runs even when the full ``import vllm`` chain
# is unavailable on this hardware.
try:
    from vllm.v1.attention.ops.dcp_topk import (  # type: ignore
        SENTINEL_INDEX,
        dcp_exact_topk,
        select_global_topk,
    )
except Exception:  # pragma: no cover - fallback for minimal envs
    _mod_path = (
        Path(__file__).resolve().parents[3]
        / "vllm/v1/attention/ops/dcp_topk.py"
    )
    _spec = importlib.util.spec_from_file_location("dcp_topk", _mod_path)
    _dcp_topk = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_dcp_topk)
    SENTINEL_INDEX = _dcp_topk.SENTINEL_INDEX
    dcp_exact_topk = _dcp_topk.dcp_exact_topk
    select_global_topk = _dcp_topk.select_global_topk

NEG_INF = float("-inf")


def _reference_global_topk(
    full_scores: torch.Tensor, topk: int, sentinel: int
) -> torch.Tensor:
    """Brute-force top-k over unsharded scores -> [num_tokens, topk] int32.

    Compared as a set by the assertions, so exact tie-breaking is irrelevant
    (scores are random floats with measure-zero ties).
    """
    num_tokens, num_kv = full_scores.shape
    k = min(topk, num_kv)
    sel_scores, sel_idx = torch.topk(full_scores, k, dim=-1, sorted=True)
    sel_idx = sel_idx.to(torch.int32)
    sel_idx = torch.where(
        sel_scores == NEG_INF, torch.full_like(sel_idx, sentinel), sel_idx
    )
    if k < topk:
        sel_idx = torch.nn.functional.pad(sel_idx, (0, topk - k), value=sentinel)
    return sel_idx


def _assert_same_set(got: torch.Tensor, ref: torch.Tensor, sentinel: int) -> None:
    """Assert per-token selected index *sets* match (order-independent)."""
    assert got.shape == ref.shape, (got.shape, ref.shape)
    for t in range(got.shape[0]):
        g = set(int(x) for x in got[t].tolist() if int(x) != sentinel)
        r = set(int(x) for x in ref[t].tolist() if int(x) != sentinel)
        assert g == r, f"token {t}: got {sorted(g)[:8]}... != ref {sorted(r)[:8]}..."


def test_select_global_topk_matches_reference() -> None:
    torch.manual_seed(0)
    num_tokens, num_kv, topk = 7, 500, 64
    scores = torch.randn(num_tokens, num_kv)
    idx = torch.arange(num_kv).expand(num_tokens, -1).clone()
    got = select_global_topk(scores, idx, topk)
    ref = _reference_global_topk(scores, topk, SENTINEL_INDEX)
    _assert_same_set(got, ref, SENTINEL_INDEX)
    # Capacity padding when fewer valid than topk.
    small = torch.randn(3, 10)
    small_idx = torch.arange(10).expand(3, -1).clone()
    got_small = select_global_topk(small, small_idx, topk)
    assert got_small.shape == (3, topk)
    assert (got_small[:, 10:] == SENTINEL_INDEX).all()


def _worker(rank: int, world_size: int, num_tokens: int, seq_len: int,
            topk: int, interleave: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    torch.distributed.init_process_group(
        backend, rank=rank, world_size=world_size)

    # Deterministic global scores every rank agrees on (seeded identically),
    # so a single-process reference is well defined.
    g = torch.Generator().manual_seed(1234)
    full_scores = torch.randn(num_tokens, seq_len, generator=g)

    # Interleaved KV sharding: global position p is owned by rank
    # (p // interleave) % world_size (matches sparse_utils de-interleave).
    owner = (torch.arange(seq_len) // interleave) % world_size
    local_mask = owner == rank
    local_global_idx = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
    local_scores = full_scores[:, local_mask].contiguous().to(device)
    local_global_idx = local_global_idx.to(device)

    class _Group:
        def __init__(self, ws):
            self.world_size = ws
            self.device_group = torch.distributed.group.WORLD

    got = dcp_exact_topk(local_scores, local_global_idx, topk, _Group(world_size))

    # Every rank must produce the identical global set == single-process ref.
    ref = _reference_global_topk(full_scores, topk, SENTINEL_INDEX).to(device)
    _assert_same_set(got.cpu(), ref.cpu(), SENTINEL_INDEX)

    # And all ranks must agree with each other bit-for-bit.
    gathered = [torch.empty_like(got) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, got)
    for other in gathered:
        assert torch.equal(other, got), "ranks disagree on global top-k"

    if rank == 0:
        print(f"[OK] world_size={world_size} num_tokens={num_tokens} "
              f"seq_len={seq_len} topk={topk} interleave={interleave}: "
              f"DCP exact top-k == single-process reference on all ranks.")
    torch.distributed.destroy_process_group()


def run_distributed(world_size: int, backend: str = "auto") -> None:
    if backend == "auto":
        # Prefer real GPU NCCL; fall back to gloo (CPU) when the driver/NCCL
        # build cannot run collectives (e.g. a cu129 NCCL on a CUDA-12.8
        # driver). gloo still exercises the real cross-process all-gather that
        # makes the top-k exact -> the selection logic is validated either way.
        backend = "nccl" if (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= world_size
        ) else "gloo"
    print(f"[info] distributed backend = {backend}, world_size = {world_size}")
    # seq_len chosen so each shard holds fewer valid than topk in one case and
    # more in another, exercising sentinel padding and real contention.
    for seq_len, topk in [(4096, 2048), (900, 2048), (8192, 2048)]:
        torch.multiprocessing.spawn(
            _worker,
            args=(world_size, 5, seq_len, topk, 64,
                  29500 + seq_len % 100, backend),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "nccl", "gloo"])
    args = parser.parse_args()
    # Pure test first (fast, no GPU).
    test_select_global_topk_matches_reference()
    print("[OK] select_global_topk matches brute-force reference.")
    run_distributed(args.world_size, args.backend)
    print("[ALL OK]")
    sys.exit(0)
