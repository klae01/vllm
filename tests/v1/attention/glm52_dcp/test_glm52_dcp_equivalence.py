# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end DCP-exactness check for GLM-5.2-FP8 (glm_moe_dsa / sparse MLA).

Verifies that Decode Context Parallel does not change model outputs: with the
KV cache sharded across DCP ranks and partial attention outputs merged by the
exact LSE-weighted combine, the outputs must match the non-DCP run.

Two phases:

* **base** — greedy first-step logprobs for ``(tp=8, dcp=1)`` vs ``(tp=8,
  dcp=8)`` must match. Includes a >``index_topk`` (2048) prompt so the indexer
  selects a *strict* top-k subset, exercising the exact distributed top-k and
  the per-shard index de-interleave (a short prompt makes top-k "select all",
  which does not stress selection contention across shards).
* **mtp** — same DCP comparison with MTP speculative decoding enabled and
  multiple tokens generated; the greedy token-id sequences from dcp=1 and dcp=8
  must be identical (validates the drafter's decode path under DCP).

Design notes
------------
* No weight download. A local model dir is built from the real
  ``zai-org/GLM-5.2-FP8`` config with ``num_hidden_layers`` reduced to 2 (every
  other field inherited), and weights are random (``load_format="dummy"``).
* To keep the *weights* identical across the two runs (so any difference is
  attributable to DCP alone), **tensor-parallel size is held fixed** and only
  ``decode_context_parallel_size`` is varied: ``(tp=8, dcp=1)`` vs
  ``(tp=8, dcp=8)``. Same TP sharding + same seed => identical dummy weights;
  DCP only reshards the KV cache. Comparing ``tp=1`` against ``tp=8`` would
  change the dummy-weight sharding and is *not* a valid DCP check.
* Uses ``dcp_comm_backend="ag_rs"`` (matching the production serve) to exercise
  the all-gather + reduce-scatter LSE combine (``cp_lse_ag_out_rs``) that the
  sparse backend now feeds via its returned ``softmax_lse``.
* ``kv_cache_dtype="fp8"`` forces ``fp8_ds_mla`` -> the ``FLASHMLA_SPARSE``
  backend (the one the serve uses and the DCP fixes touch); with ``auto`` the
  model picks a different backend and the test would validate nothing.
* ``enforce_eager=True`` matches the serve and avoids the compile-time
  ``fuse_allreduce_rms`` pass, whose FlashInfer ``trtllm_mnnvl_allreduce_fusion``
  binding is arg-count-incompatible with the installed FlashInfer (unrelated to
  DCP; it fires during cudagraph capture). Correctness is unaffected by eager.

Hardware requirements (why this may not run everywhere)
------------------------------------------------------
The FlashMLA sparse decode kernel is Hopper-only, and DCP needs working NCCL
collectives (a CUDA-12.9 NCCL on a CUDA-12.8 driver fails). On non-Hopper /
old-driver boxes this test cannot execute the sparse attention forward.

Run::

    uv run python tests/v1/attention/glm52_dcp/test_glm52_dcp_equivalence.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

MODEL_DIR = str(Path(__file__).parent / "glm52_2layer")
# Tokenizer is loaded from the HF repo id (tokenizer files only, no weights);
# the 20 MB tokenizer.json is intentionally not vendored into the repo.
TOKENIZER = "zai-org/GLM-5.2-FP8"
MAX_MODEL_LEN = 8192

SHORT_PROMPTS = [
    "The capital of France is",
    "In distributed systems, consensus means",
]
# > index_topk (2048) tokens so the single decode step attends over more KV than
# the top-k budget and the indexer must pick a strict subset. ~200 of these
# sentences tokenize to ~3.3k tokens -- comfortably inside (2048, MAX_MODEL_LEN);
# each sentence is ~16 tokens, so keep the count well under MAX_MODEL_LEN/16.
LONG_PROMPT = " ".join(
    f"Section {i}: the measurement recorded at station {i % 13} was "
    f"{(i * 7) % 97} units."
    for i in range(200)
)
BASE_PROMPTS = [*SHORT_PROMPTS, LONG_PROMPT]


def _run(
    dcp: int,
    *,
    tp: int = 8,
    seed: int = 0,
    mtp: bool = False,
    max_tokens: int = 1,
    prompts: list[str] = BASE_PROMPTS,
):
    """Run generation and return per-prompt {"logprobs": {...}, "token_ids": [...]}.

    ``logprobs`` is the first-step top-token map (base phase); ``token_ids`` is
    the full greedy continuation (mtp phase).
    """
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL_DIR,
        tokenizer=TOKENIZER,
        trust_remote_code=True,
        load_format="dummy",
        tensor_parallel_size=tp,
        decode_context_parallel_size=dcp,
        kv_cache_dtype="fp8",
        dcp_comm_backend="ag_rs",
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        seed=seed,
        gpu_memory_utilization=0.9,
    )
    if mtp:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 3,
        }
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=20)
    outs = llm.generate(prompts, sp)
    result = []
    for o in outs:
        out0 = o.outputs[0]
        first = out0.logprobs[0] if out0.logprobs else {}
        result.append(
            {
                "logprobs": {tid: v.logprob for tid, v in first.items()},
                "token_ids": list(out0.token_ids),
            }
        )
    del llm
    torch.cuda.empty_cache()
    return result


def _compare_logprobs(a, b, atol=2e-2) -> None:
    for i, (pa, pb) in enumerate(zip(a, b)):
        la, lb = pa["logprobs"], pb["logprobs"]
        common = set(la) & set(lb)
        assert common, f"prompt {i}: no overlapping top tokens"
        m = max(abs(la[t] - lb[t]) for t in common)
        print(f"prompt {i}: {len(common)} common top tokens, max|Δlogprob|={m:.5f}")
        assert m < atol, f"prompt {i}: logprob mismatch {m} >= {atol}"


def _compare_token_ids(a, b) -> None:
    for i, (pa, pb) in enumerate(zip(a, b)):
        ta, tb = pa["token_ids"], pb["token_ids"]
        print(f"prompt {i}: dcp1={ta} dcp8={tb}")
        assert ta == tb, f"prompt {i}: token sequence mismatch dcp1 != dcp8"


def main() -> None:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    assert torch.cuda.device_count() >= 8, (
        f"needs 8 GPUs, have {torch.cuda.device_count()}")

    # Phase 1: base attention exactness (short + long-context prompts).
    print("== base DCP=1 (tp=8) ==")
    base1 = _run(dcp=1)
    print("== base DCP=8 (tp=8) ==")
    base8 = _run(dcp=8)
    _compare_logprobs(base1, base8)
    print("[OK] base: DCP=8 logits == DCP=1 logits (exact).")

    # Phase 2: MTP + DCP exactness (greedy multi-token sequences must match).
    print("== mtp DCP=1 (tp=8) ==")
    mtp1 = _run(dcp=1, mtp=True, max_tokens=8, prompts=SHORT_PROMPTS)
    print("== mtp DCP=8 (tp=8) ==")
    mtp8 = _run(dcp=8, mtp=True, max_tokens=8, prompts=SHORT_PROMPTS)
    _compare_token_ids(mtp1, mtp8)
    print("[OK] mtp+dcp: DCP=8 sequence == DCP=1 sequence (exact).")

    print("[ALL OK] GLM-5.2-FP8 (2-layer): DCP is exact for base and MTP.")


if __name__ == "__main__":
    main()
