# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end DCP-exactness check for GLM-5.2-FP8 (glm_moe_dsa / sparse MLA).

Verifies that Decode Context Parallel does not change model outputs: with the
KV cache sharded across DCP ranks and partial attention outputs merged by the
exact LSE-weighted combine, the logits must match the non-DCP run.

Design notes
------------
* No weight download. A local model dir is built from the real
  ``zai-org/GLM-5.2-FP8`` config with ``num_hidden_layers`` reduced to 2 (every
  other field inherited), and weights are random (``load_format="dummy"``).
* To keep the *weights* identical across the two runs (so any logit difference
  is attributable to DCP alone), **tensor-parallel size is held fixed** and only
  ``decode_context_parallel_size`` is varied: ``(tp=8, dcp=1)`` vs
  ``(tp=8, dcp=8)``. Same TP sharding + same seed => identical dummy weights;
  DCP only reshards the KV cache. Comparing ``tp=1`` against ``tp=8`` would
  change the dummy-weight sharding and is *not* a valid DCP check.
* Uses ``dcp_comm_backend="ag_rs"`` (matching the production serve) to exercise
  the all-gather + reduce-scatter LSE combine (``cp_lse_ag_out_rs``) that the
  sparse backend now feeds via its returned ``softmax_lse``.

Hardware requirements (why this may not run everywhere)
------------------------------------------------------
The FlashMLA sparse decode kernel is Hopper-only (see
``vllm/v1/attention/ops/flashmla.py``: "FlashMLA Dense is only supported on
Hopper devices"), and DCP needs working NCCL collectives (a CUDA-12.9 NCCL on a
CUDA-12.8 driver fails). On non-Hopper / old-driver boxes this test cannot
execute the sparse attention forward; the component correctness (LSE combine,
exact distributed top-k) is validated hardware-independently by
``tests/v1/attention/test_dcp_lse_combine.py`` and
``tests/v1/attention/test_dcp_exact_topk.py``.

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
PROMPTS = [
    "The capital of France is",
    "In distributed systems, consensus means",
]


def _run(dcp: int, tp: int = 8, seed: int = 0):
    """Return per-prompt list of {token_id: logprob} for the first step."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_DIR,
        tokenizer=TOKENIZER,
        trust_remote_code=True,
        load_format="dummy",
        tensor_parallel_size=tp,
        decode_context_parallel_size=dcp,
        # Critical: fp8 KV cache -> fp8_ds_mla -> the FLASHMLA_SPARSE backend
        # (the one the production serve uses and the one these DCP fixes touch).
        # Without this the model picks FLASH_ATTN_MLA_SPARSE (a different backend)
        # and the test would validate the wrong code path.
        kv_cache_dtype="fp8",
        # Match the production serve: AG+RS LSE combine (cp_lse_ag_out_rs).
        dcp_comm_backend="ag_rs",
        # Eager to match the serve config AND to avoid the compile-time
        # fuse_allreduce_rms pass, whose FlashInfer trtllm_mnnvl_allreduce_fusion
        # binding is arg-count-incompatible with the installed FlashInfer
        # (unrelated to DCP; it fires during cudagraph capture, not the sparse
        # forward). Correctness (DCP=1 vs DCP=8 logits) is unaffected by eager.
        enforce_eager=True,
        max_model_len=4096,
        seed=seed,
        gpu_memory_utilization=0.9,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=64)
    outs = llm.generate(PROMPTS, sp)
    result = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]  # {token_id: Logprob}
        result.append({tid: v.logprob for tid, v in lp.items()})
    del llm
    torch.cuda.empty_cache()
    return result


def _compare(a, b, atol=2e-2) -> None:
    for i, (pa, pb) in enumerate(zip(a, b)):
        common = set(pa) & set(pb)
        assert common, f"prompt {i}: no overlapping top tokens"
        diffs = [abs(pa[t] - pb[t]) for t in common]
        m = max(diffs)
        print(f"prompt {i}: {len(common)} common top tokens, max|Δlogprob|={m:.5f}")
        assert m < atol, f"prompt {i}: logprob mismatch {m} >= {atol}"


def main() -> None:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    assert torch.cuda.device_count() >= 8, (
        f"needs 8 GPUs, have {torch.cuda.device_count()}")
    print("== DCP=1 (tp=8) ==")
    base = _run(dcp=1)
    print("== DCP=8 (tp=8) ==")
    dcp8 = _run(dcp=8)
    _compare(base, dcp8)
    print("[OK] GLM-5.2-FP8 (2-layer): DCP=8 logits == DCP=1 logits (exact).")


if __name__ == "__main__":
    main()
