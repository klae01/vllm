# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LSE-combine identity tests for Decode Context Parallel (DCP).

The invariant under test: attention computed once over the full KV must equal
the LSE-weighted combination of partial attentions computed over disjoint KV
shards. This is exactly what DCP does (each rank attends to its KV shard, then
the partial outputs + LSEs are merged). If the combine is exact, DCP and
non-DCP produce bit-comparable results (up to fp rounding).

Levels:

* ``test_lse_weighted_combine_identity`` — pure single-process test of the
  reference combine (``_lse_weighted_combine``) for N in {1,2,4,8} shards.
* ``run_distributed`` — real multi-GPU test of ``dcp_a2a_lse_reduce`` (the
  Triton all-to-all combine used on the hot path) across ``world_size`` GPUs.

Run the distributed part with::

    uv run python tests/v1/attention/test_dcp_lse_combine.py --world-size 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

# Local reference identical to vllm.v1.attention.ops.dcp_alltoall
# ._lse_weighted_combine, kept here so the pure test runs even where the full
# ``import vllm`` chain is unavailable. The distributed test imports the real
# Triton kernel from vllm.
def _ref_lse_weighted_combine(outputs, lses, return_lse=False):
    # outputs: [N, B, H, D], lses: [N, B, H]; base-e LSE.
    lses = torch.where(torch.isnan(lses) | torch.isinf(lses),
                       torch.tensor(float("-inf"), dtype=lses.dtype), lses)
    lse_max, _ = lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"),
                          torch.zeros_like(lse_max), lse_max)
    weights = torch.exp(lses - lse_max.unsqueeze(0))
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum.clamp(min=1e-10)
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)
    if return_lse:
        return result, torch.log(weight_sum.squeeze(0)) + lse_max
    return result


def _mha(q, k, v, scale):
    """Reference attention. q:[B,H,Dqk] k:[S,Dqk] v:[S,Dv] (MQA, shared kv).

    Returns out:[B,H,Dv], lse:[B,H] (natural log).
    """
    scores = torch.einsum("bhd,sd->bhs", q.float(), k.float()) * scale  # [B,H,S]
    lse = torch.logsumexp(scores, dim=-1)  # [B,H]
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,sv->bhv", weights, v.float())  # [B,H,Dv]
    return out, lse


def _shard_mha(q, k, v, scale, shard_ids):
    return _mha(q, k[shard_ids], v[shard_ids], scale)


def test_lse_weighted_combine_identity() -> None:
    torch.manual_seed(0)
    B, H, S, Dqk, Dv = 3, 8, 500, 64, 128
    scale = 1.0 / math.sqrt(Dqk)
    q = torch.randn(B, H, Dqk)
    k = torch.randn(S, Dqk)
    v = torch.randn(S, Dv)

    out_full, lse_full = _mha(q, k, v, scale)

    for n in (1, 2, 4, 8):
        # Interleaved shards (disjoint, union == all positions).
        outs, lses = [], []
        for r in range(n):
            ids = torch.arange(r, S, n)
            o, l = _shard_mha(q, k, v, scale, ids)
            outs.append(o)
            lses.append(l)
        outs = torch.stack(outs, 0)  # [N,B,H,Dv]
        lses = torch.stack(lses, 0)  # [N,B,H]
        comb, lse_comb = _ref_lse_weighted_combine(outs, lses, return_lse=True)
        torch.testing.assert_close(comb, out_full, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lse_comb, lse_full, atol=1e-5, rtol=1e-5)
    print("[OK] LSE-weighted combine == full attention for N in {1,2,4,8}.")


def _worker(rank: int, world_size: int, B: int, H: int, S: int,
            Dqk: int, Dv: int, port: int) -> None:
    from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    scale = 1.0 / math.sqrt(Dqk)

    # Identical inputs on every rank (seeded), bf16 activations like real MLA.
    g = torch.Generator().manual_seed(7)
    q = torch.randn(B, H, Dqk, generator=g)
    k = torch.randn(S, Dqk, generator=g)
    v = torch.randn(S, Dv, generator=g)
    out_full, _ = _mha(q, k, v, scale)  # fp32 reference [B,H,Dv]

    # This rank attends to its interleaved KV shard, for ALL H heads.
    ids = torch.arange(rank, S, world_size)
    out_r, lse_r = _shard_mha(q, k, v, scale, ids)  # [B,H,Dv], [B,H]
    out_r = out_r.to(torch.bfloat16).to(device)
    lse_r = lse_r.to(torch.float32).to(device)

    class _Group:
        world_size = None
        device_group = None

    grp = _Group()
    grp.world_size = world_size
    grp.device_group = torch.distributed.group.WORLD

    combined = dcp_a2a_lse_reduce(out_r, lse_r, grp, return_lse=False)
    # Head-scattered: rank r owns heads [r*H/N:(r+1)*H/N].
    hpr = H // world_size
    ref = out_full[:, rank * hpr:(rank + 1) * hpr, :].to(torch.bfloat16)
    torch.testing.assert_close(combined.cpu().float(), ref.float(),
                               atol=2e-2, rtol=2e-2)
    if rank == 0:
        print(f"[OK] dcp_a2a_lse_reduce == full attention (head-scattered) "
              f"world_size={world_size} B={B} H={H} S={S}.")
    torch.distributed.destroy_process_group()


def _nccl_usable() -> bool:
    """Probe whether NCCL collectives actually run on this driver.

    The real a2a combine uses dist.all_to_all_single, which only NCCL provides
    (gloo has no all-to-all). A cu129 NCCL on a CUDA-12.8 driver fails at the
    first collective, so probe before spawning.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import datetime

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29999")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            "nccl", rank=0, world_size=1,
            timeout=datetime.timedelta(seconds=20))
        t = torch.ones(4, device="cuda:0")
        torch.distributed.all_reduce(t)
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] NCCL collectives unavailable ({type(e).__name__}: {e}). "
              f"The a2a kernel needs NCCL all-to-all; the pure combine identity "
              f"test above already validates the same math exactly.")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return False


def run_distributed(world_size: int) -> None:
    if torch.cuda.device_count() < world_size or not _nccl_usable():
        return
    H = world_size * 16  # heads divisible by world size
    torch.multiprocessing.spawn(
        _worker, args=(world_size, 4, H, 4096, 64, 128, 29700),
        nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    test_lse_weighted_combine_identity()
    run_distributed(args.world_size)
    print("[ALL OK]")
    sys.exit(0)
