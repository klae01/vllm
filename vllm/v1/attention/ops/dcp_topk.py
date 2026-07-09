# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Exact distributed top-k selection for Decode Context Parallel (DCP).

Under DCP the KV cache is sharded across ranks, so no single rank can score
every KV position. The sparse-attention indexer must nonetheless pick the same
global top-k KV tokens on every rank (otherwise ranks attend to different KV and
the LSE-weighted combine is meaningless).

This module selects the exact global top-k without materializing all scores on
one rank:

    1. each rank scores only its local KV shard and takes its local top-k;
    2. every rank all-gathers ``(score, global_index)`` for its local top-k;
    3. every rank independently re-selects the global top-k from the gathered
       candidate union.

This is exact (not an approximation). If a KV position ``e`` belongs to the
global top-k, then fewer than ``k`` positions globally outscore it, hence fewer
than ``k`` positions *on e's own shard* outscore it, so ``e`` is in that shard's
local top-k. Therefore ``global_topk ⊆ ∪ local_topk`` and step 3 recovers the
identical set a single unsharded top-k would produce.

Capacity is fixed at ``topk`` with sentinel padding, so the produced shape is
independent of how many valid KV each shard holds. This keeps the sparse
attention metadata shape-invariant across steps and preserves the
``UNIFORM_BATCH`` CUDA-graph capture used by the FlashMLA sparse backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Sentinel written into unused top-k slots. Matches the sparse index kernels in
# sparse_utils.py, which treat any index < 0 as invalid.
SENTINEL_INDEX: int = -1
NEG_INF = float("-inf")


def _local_topk_candidates(
    local_scores: torch.Tensor,
    local_to_global_idx: torch.Tensor,
    topk: int,
    sentinel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take this rank's local top-k, mapped to global KV indices.

    Args:
        local_scores: ``[num_tokens, num_local_kv]`` fp32 scores over this
            rank's KV shard. Padded/invalid slots must be ``-inf``.
        local_to_global_idx: ``[num_local_kv]`` int mapping each local KV slot
            to its global KV position.
        topk: fixed capacity K.
        sentinel: index written into slots with no valid candidate.

    Returns:
        ``(cand_scores, cand_global_idx)`` each ``[num_tokens, topk]``. When the
        shard holds fewer than ``topk`` valid KV, the tail is padded with
        ``-inf`` score / ``sentinel`` index.
    """
    num_tokens, num_local_kv = local_scores.shape
    k_local = min(topk, num_local_kv)
    top_scores, top_local = torch.topk(local_scores, k_local, dim=-1)
    top_global = local_to_global_idx.to(torch.int64)[top_local]

    if k_local < topk:
        pad = topk - k_local
        top_scores = torch.nn.functional.pad(top_scores, (0, pad), value=NEG_INF)
        top_global = torch.nn.functional.pad(top_global, (0, pad), value=sentinel)

    # A local slot that scores -inf is not a real candidate; mark its index.
    invalid = top_scores == NEG_INF
    top_global = torch.where(
        invalid, torch.full_like(top_global, sentinel), top_global
    )
    return top_scores.to(torch.float32), top_global


def select_global_topk(
    cand_scores: torch.Tensor,
    cand_global_idx: torch.Tensor,
    topk: int,
    sentinel: int = SENTINEL_INDEX,
) -> torch.Tensor:
    """Select the global top-k from already-gathered candidates (no comms).

    Pure function so it can be unit-tested against a single-process reference.

    Args:
        cand_scores: ``[num_tokens, num_candidates]`` fp32 candidate scores
            (union of every rank's local top-k). Padding is ``-inf``.
        cand_global_idx: ``[num_tokens, num_candidates]`` int global KV indices
            aligned with ``cand_scores``. Padding is ``sentinel``.
        topk: fixed capacity K.
        sentinel: index written into unused slots.

    Returns:
        ``[num_tokens, topk]`` int32 global KV indices, sorted by descending
        score, sentinel-padded. Deterministic: ties broken by smaller index.
    """
    num_tokens, num_candidates = cand_scores.shape
    k = min(topk, num_candidates)

    # Break score ties deterministically by preferring the smaller global index,
    # so every rank selects the identical set regardless of gather order.
    # Encode (score, -index) into a single sortable key via lexicographic sort.
    order = torch.argsort(cand_global_idx, dim=-1, stable=True)
    cand_scores = torch.gather(cand_scores, -1, order)
    cand_global_idx = torch.gather(cand_global_idx, -1, order)

    sel_scores, sel_pos = torch.topk(cand_scores, k, dim=-1, sorted=True)
    sel_idx = torch.gather(cand_global_idx, -1, sel_pos)

    # Drop selections that are pure padding (score -inf) -> sentinel.
    sel_idx = torch.where(
        sel_scores == NEG_INF, torch.full_like(sel_idx, sentinel), sel_idx
    )

    if k < topk:
        sel_idx = torch.nn.functional.pad(sel_idx, (0, topk - k), value=sentinel)
    return sel_idx.to(torch.int32)


def dcp_exact_topk(
    local_scores: torch.Tensor,
    local_to_global_idx: torch.Tensor,
    topk: int,
    cp_group: "GroupCoordinator",
    sentinel: int = SENTINEL_INDEX,
) -> torch.Tensor:
    """Exact global top-k KV indices across DCP ranks.

    Every rank returns the identical ``[num_tokens, topk]`` set (int32,
    sentinel-padded). ``num_tokens`` and ``topk`` must match across ranks; the
    per-shard ``num_local_kv`` may differ.

    Args:
        local_scores: ``[num_tokens, num_local_kv]`` fp32 scores over this
            rank's KV shard (``-inf`` for padding/invalid).
        local_to_global_idx: ``[num_local_kv]`` int global KV index per slot.
        topk: fixed capacity K (e.g. 2048).
        cp_group: DCP ``GroupCoordinator``.
        sentinel: index for unused slots.

    Returns:
        ``[num_tokens, topk]`` int32 global KV indices (same on every rank).
    """
    world_size = cp_group.world_size
    cand_scores, cand_global_idx = _local_topk_candidates(
        local_scores, local_to_global_idx, topk, sentinel
    )

    if world_size == 1:
        return select_global_topk(cand_scores, cand_global_idx, topk, sentinel)

    num_tokens = cand_scores.shape[0]
    # all_gather_into_tensor wants the output first-dim scaled by world_size
    # (both NCCL and gloo agree on this 2-D form); a leading world dim is
    # rejected by gloo. Gather into [world * tokens, topk] then unflatten.
    gathered_scores = cand_scores.new_empty((world_size * num_tokens, topk))
    gathered_idx = cand_global_idx.new_empty((world_size * num_tokens, topk))
    dist.all_gather_into_tensor(
        gathered_scores, cand_scores.contiguous(), group=cp_group.device_group
    )
    dist.all_gather_into_tensor(
        gathered_idx, cand_global_idx.contiguous(), group=cp_group.device_group
    )

    # [world * tokens, topk] -> [world, tokens, topk] -> [tokens, world * topk]
    cand_scores = (
        gathered_scores.view(world_size, num_tokens, topk)
        .permute(1, 0, 2)
        .reshape(num_tokens, -1)
    )
    cand_global_idx = (
        gathered_idx.view(world_size, num_tokens, topk)
        .permute(1, 0, 2)
        .reshape(num_tokens, -1)
    )
    return select_global_topk(cand_scores, cand_global_idx, topk, sentinel)
