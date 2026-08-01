# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_dcp_gathered_ws_index,
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# For FP8 sparse attention we have two implementations:
# 1. Mixed batch mode: use the FP8 decode kernel for both prefill and decode this is
#    done by treating all tokens as single batch.
# 2. Separate prefill and decode mode: use the BF16 prefill kernel for prefill
#    (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using
#    the FP8 decode kernel for decode.
# Currently we use #1 when the number of heads per rank is low (i.e. TP) since the BF16
# prefill kernel requires padding the number of heads to 128 while the decode does not
# so when the per-rank head count is below MIN_HEADS_FOR_BF16_PREFILL we use the mixed
# batch mode (#1).
MIN_HEADS_FOR_BF16_PREFILL = 32

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

For DeepSeek V3.2, in the "FP8 with scale" format, each token's KV cache is 656
Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.

For DeepSeek V4, in the "FP8 with scale" format, each token's KV cache is 584
Bytes, structured as:
-   **First 448 bytes:** The "quantized NoPE" part, containing 448
    `float8_e4m3` values.
-   **Next 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
-   **Last 8 bytes:** Scale factors, containing 7 `ue8m0` values + 1B pad.
    The first `ue8m0` is the scale for the first 64 `float8_e4m3` values,
    the second for the next 64, and so on.
"""


class FlashMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashMLASparseMetadataBuilder"]:
        return FlashMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        return FlashMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V3.2 layout: 512 NoPE + 64 RoPE = 576.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom storage format. See module docstring.
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


@dataclass
class FlashMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    seq_lens: torch.Tensor | None = None
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    cp_kv_cache_interleave_size: int = 1

    @dataclass
    class FP8KernelMetadata:
        scheduler_metadata: FlashMLASchedMeta
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    @dataclass
    class FP8SeparatePrefillDecode:
        @dataclass
        class Decode:
            seq_lens: torch.Tensor
            kernel_metadata: "FlashMLASparseMetadata.FP8KernelMetadata"
            decode_query_len: int  # needed for reshape in spec decode

        @dataclass
        class Prefill:
            # Request ID for each token: -1 for decode tokens, request index
            # (0, 1, 2, ...) for prefill tokens.
            # Shape: [num_actual_tokens]
            request_ids: torch.Tensor

            # Workspace start offsets for all prefill requests
            # Shape: [num_prefill_reqs], adjusted in-place per chunk to be
            # 0-indexed within each chunk. Used to map prefill tokens to workspace
            # offsets in convert_logical_index_to_physical_index
            workspace_starts: torch.Tensor

            @dataclass
            class Chunk:
                """Metadata for a chunk of prefill requests.

                Prefill requests may be chunked to fit within the fixed workspace size.
                """

                tokens_slice: slice
                block_table: torch.Tensor
                req_start_idx: int
                workspace_starts: torch.Tensor
                chunk_tot_seqlen: int
                # DCP fast path: this rank's slice of the chunk's query tokens
                # (absolute token indices within the batch).
                stripe_slice: slice | None = None

            chunks: list[Chunk]
            # DCP fast path: workspace_starts/chunk_tot_seqlen are padded
            # per-rank shard row counts and each rank computes only its
            # stripe of query tokens over the DCP-allgathered workspace.
            dcp_fast: bool = False

        num_prefills: int = 0
        num_decodes: int = 0
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0

        decode: Decode | None = None
        prefill: Prefill | None = None

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None = None
    fp8_use_mixed_batch: bool = False


def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    return max_model_len * 5


class FlashMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    require_uniform_decodes: ClassVar[bool] = True
    metadata_cls = FlashMLASparseMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        num_q_heads = self.model_config.get_num_attention_heads(parallel_config)
        if current_platform.is_device_capability_family(100):
            threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
                num_q_heads, 1024
            )
        else:
            threshold = {16: 128, 32: 128, 64: 256, 128: 256}.get(num_q_heads, 256)
        # supports_dcp_with_varlen keeps the spec-decode reorder threshold
        # under DCP; without it the threshold is forced to 1 and MTP decode
        # batches would be misclassified as prefills on DCP ranks.
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )

        sm_count = num_compute_units(device.index)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        # FP8 decode kernel only supports h_q = 64 or 128, so we need to pad
        self.fp8_decode_padded_heads = (
            FlashMLASparseImpl._compute_fp8_decode_padded_heads(self.num_heads)
        )

        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"
        # DCP prefill fast path: each rank attends its stripe of prefill query
        # tokens (with DCP-gathered heads) over the full context assembled by
        # a per-layer workspace allgather; complete stripe outputs are routed
        # through the exact LSE combine via -inf LSE on non-owned rows. Only
        # wired for the ag_rs combine; PCP changes the q-gather semantics.
        self.dcp_rank = 0
        self._dcp_prefill_fast_ok = False
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
            self._dcp_prefill_fast_ok = (
                parallel_config.dcp_comm_backend == "ag_rs" and not self.use_pcp
            )
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # Shape: [max_num_seqs], all elements = topk_tokens (constant for full-CG)
        self.topk_tokens_tensor = torch.full(
            (max_num_seqs,), self.topk_tokens, device=device, dtype=torch.int32
        )
        # Shape: [max_num_seqs], all elements = max_model_len
        self.max_model_len_tensor = torch.full(
            (max_num_seqs,),
            self.model_config.max_model_len,
            device=device,
            dtype=torch.int32,
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (max_num_seqs, 1), dtype=torch.int32, device=self.device
        )

        # Equation taken from FlashMLA/csrc/api/sparse_decode.h
        # For sparse FP8 decode, the formula depends on architecture:
        # - SM90 (Hopper): num_sm_parts = num_sms / s_q / (h_q/64)
        # - SM100 (Blackwell head64/head64x2): num_sm_parts = num_sms / s_q
        # - SM100 (Blackwell head128): num_sm_parts = num_sms / s_q / 2
        # For max buffer size, use s_q = 1 (the case that produces largest output)
        # Use padded head count since that's what will be passed to the kernel
        h_q = self.fp8_decode_padded_heads
        if current_platform.is_device_capability_family(100):
            # SM100 head64 or head64x2 uses full SM count
            max_num_sm_parts = sm_count
        else:
            # SM90 uses h_q/64 divisor
            max_num_sm_parts = sm_count // max(1, h_q // 64)
        self.tile_scheduler_metadata_buffer = torch.empty(
            # TileSchedulerMetaDataSize = 8
            # see: FlashMLA/csrc/params.h
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device=device,
        )
        # Sized for per-request batching (num_decodes + 1)
        self.num_splits_buffer = torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        )

    def _build_fp8_mixed_decode_prefill(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "FlashMLASparseMetadata.FP8KernelMetadata":
        """Build FP8 metadata treating MQA tokens as one batch.

        The scheduler initializes lazily from the runtime query shape, which may
        be the full batch or only decodes when prefills use dense MHA. This avoids
        the BF16 prefill kernel's head-padding overhead at high TP.
        """
        num_tokens = common_attn_metadata.num_actual_tokens

        # Use padded head count since that's what the kernel will see
        padded_heads = self.fp8_decode_padded_heads

        # Build metadata for all tokens as a single batch
        scheduler_metadata, _ = get_mla_metadata(
            cache_seqlens=self.topk_tokens_tensor[:1],  # Single batch
            num_q_tokens_per_head_k=num_tokens * padded_heads,
            topk=self.topk_tokens,
            num_heads_q=padded_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler_metadata,
            cache_lens=self.max_model_len_tensor[:1],
            dummy_block_table=self.dummy_block_table[:1],
        )

        return fp8_metadata

    def _build_fp8_separate_prefill_decode(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        metadata: FlashMLASparseMetadata,
        dcp_fast: bool = False,
    ) -> "FlashMLASparseMetadata.FP8SeparatePrefillDecode":
        num_tokens = common_attn_metadata.num_actual_tokens

        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            metadata.num_decodes,
            metadata.num_prefills,
            metadata.num_decode_tokens,
            num_tokens - metadata.num_decode_tokens,
        )

        decode_query_len = 0
        active_num_decodes = num_decodes
        if num_decodes > 0:
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            decode_query_len = (query_start_loc_cpu[1] - query_start_loc_cpu[0]).item()
            assert decode_query_len > 0
            active_num_decodes = num_decode_tokens // decode_query_len
            assert active_num_decodes * decode_query_len == num_decode_tokens

        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=active_num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
        )

        # Extract prefill sequence lengths (context + query, not just query)
        # Decode requests come first in the batch, prefill requests follow
        prefill_request_id = None
        prefill_workspace_starts = None
        prefill_chunks = None

        # For pure decode batches, prefill_request_id will be None
        # For mixed batches, it will have -1 for decode and request_id for prefill
        if num_prefills > 0:
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below), so no D2H sync is needed.
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

            prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:]
            if dcp_fast:
                # Workspace rows are per-rank shard counts, padded so every
                # rank's compact segment uses identical per-request offsets
                # (required for the symmetric allgather layout).
                stride = self.dcp_world_size * self.cp_kv_cache_interleave_size
                prefill_ws_lens_cpu = (
                    (prefill_seq_lens_cpu + stride - 1)
                    // stride
                    * self.cp_kv_cache_interleave_size
                ).to(torch.int32)
            else:
                prefill_ws_lens_cpu = prefill_seq_lens_cpu

            # Build prefill_request_id: -1 for decode, request index for
            # prefill. This enables a single
            # convert_logical_index_to_physical_index call for all tokens
            prefill_request_id = torch.full(
                (num_tokens,), -1, dtype=torch.int32, device=self.device
            )
            # Map prefill tokens to their request IDs (0, 1, 2, ...)
            for req_idx in range(num_prefills):
                # Get query token range for this prefill request
                global_req_idx = num_decodes + req_idx
                req_query_start = query_start_loc_cpu[global_req_idx]
                req_query_end = query_start_loc_cpu[global_req_idx + 1]
                prefill_request_id[req_query_start:req_query_end] = req_idx

            # will be adjusted by chunk loop
            prefill_workspace_starts_cpu = torch.zeros(
                num_prefills, dtype=torch.int32, pin_memory=True
            )
            prefill_workspace_starts_cpu[1:] = torch.cumsum(
                prefill_ws_lens_cpu[:-1], dim=0
            )
            # populated by non-blocking copy after prefill_workspace_starts_cpu is
            # updated by each chunk
            prefill_workspace_starts = torch.empty(
                num_prefills, dtype=torch.int32, device=self.device
            )

            # Chunk prefill requests to fit within workspace size
            max_prefill_buffer_size = get_prefill_workspace_size(
                self.vllm_config.model_config.max_model_len
            )
            if dcp_fast:
                # Workspace lens are per-rank shard counts and the attention
                # runs over a dcp_world_size x larger allgathered tensor; cap
                # the padded chunk rows so the gathered tensor stays within
                # the same byte envelope as the non-DCP workspace.
                max_prefill_buffer_size //= self.dcp_world_size
            chunk_bounds = split_prefill_chunks(
                prefill_ws_lens_cpu, max_prefill_buffer_size
            )

            prefill_chunks = []
            for chunk_start, chunk_end in chunk_bounds:
                # Adjust workspace_starts in-place per chunk to be
                # 0-indexed within each chunk
                # Example: seq_lens=[10,15,20,5], chunks=[[0,2],[2,4]]
                #   Initial: workspace_starts=[0,10,25,45]
                #   After:   workspace_starts=[0,10,0,20]
                #           (chunk 0 starts at 0, chunk 1 starts at 0)
                offset = prefill_workspace_starts_cpu[chunk_start].item()
                prefill_workspace_starts_cpu[chunk_start:chunk_end] -= offset

                chunk_tot_seqlen = prefill_ws_lens_cpu[chunk_start:chunk_end].sum()
                token_start = query_start_loc_cpu[num_decodes + chunk_start].item()
                token_end = query_start_loc_cpu[num_decodes + chunk_end].item()
                tokens_slice = slice(token_start, token_end)

                stripe_slice = None
                if dcp_fast:
                    # Split the chunk's query tokens evenly across DCP ranks;
                    # this rank computes only its stripe.
                    num_chunk_tokens = token_end - token_start
                    stripe = -(-num_chunk_tokens // self.dcp_world_size)
                    s_start = min(token_start + self.dcp_rank * stripe, token_end)
                    s_end = min(s_start + stripe, token_end)
                    stripe_slice = slice(s_start, s_end)

                # Create chunk view of gpu tensor
                chunk_workspace_starts = prefill_workspace_starts[chunk_start:chunk_end]
                chunk_block_table = common_attn_metadata.block_table_tensor[
                    num_decodes + chunk_start : num_decodes + chunk_end
                ]

                prefill_chunks.append(
                    FP8Meta.Prefill.Chunk(
                        tokens_slice=tokens_slice,
                        block_table=chunk_block_table,
                        req_start_idx=chunk_start,
                        workspace_starts=chunk_workspace_starts,
                        chunk_tot_seqlen=chunk_tot_seqlen,
                        stripe_slice=stripe_slice,
                    )
                )

            prefill_workspace_starts.copy_(
                prefill_workspace_starts_cpu, non_blocking=True
            )

            fp8_metadata.prefill = FP8Meta.Prefill(
                request_ids=prefill_request_id,
                workspace_starts=prefill_workspace_starts,
                chunks=prefill_chunks,
                dcp_fast=dcp_fast,
            )

        if num_decodes > 0:
            # Use padded head count since that's what the kernel will see
            scheduler_metadata, _ = get_mla_metadata()

            kernel_meta = FlashMLASparseMetadata.FP8KernelMetadata(
                scheduler_metadata=scheduler_metadata,
                dummy_block_table=self.dummy_block_table[:active_num_decodes],
                cache_lens=self.max_model_len_tensor[:active_num_decodes],
            )
            fp8_metadata.decode = FP8Meta.Decode(
                seq_lens=common_attn_metadata.seq_lens[:active_num_decodes],
                kernel_metadata=kernel_meta,
                decode_query_len=decode_query_len,
            )

        return fp8_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        # Under DCP, batches with prefill tokens take the separate path with the
        # DCP fast prefill (workspace allgather + per-rank token stripes); the
        # mixed batch path would run every prefill token on every rank against
        # its 1/dcp KV shard with shard-filtered top-k (dcp x redundant, badly
        # shaped rows). Decode-only batches keep the mixed path so captured
        # decode CUDA graphs are unchanged.
        dcp_prefill_fast = (
            self._dcp_prefill_fast_ok
            and self.use_fp8_kv_cache
            and metadata.num_prefills > 0
        )
        fp8_use_mixed_batch = (
            self.num_heads < MIN_HEADS_FOR_BF16_PREFILL and not dcp_prefill_fast
        )
        metadata.fp8_use_mixed_batch = fp8_use_mixed_batch
        if self.use_fp8_kv_cache:
            if fp8_use_mixed_batch:
                metadata.fp8_extra_metadata = self._build_fp8_mixed_decode_prefill(
                    common_attn_metadata
                )
            else:
                metadata.fp8_extra_metadata = self._build_fp8_separate_prefill_decode(
                    common_attn_metadata, metadata, dcp_fast=dcp_prefill_fast
                )

        return metadata


class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
    # The FlashMLA decode kernel emits a natural-log softmax LSE alongside the
    # attention output (``lse_base_on_e`` stays True). Exposing it lets the MLA
    # common forward combine the per-rank partial outputs exactly under decode
    # context parallelism (mla_attention.py routes it through cp_lse_ag_out_rs
    # / dcp_a2a_lse_reduce).
    can_return_lse_for_decode: bool = True

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor, num_tokens: int, num_heads: int
    ) -> torch.Tensor:
        """Normalize a decode-kernel LSE to ``[num_tokens, num_heads]`` fp32.

        The FP8 decode kernel pads the query heads to 64/128 and returns the
        LSE for the *padded* head count, with the head and (batch, seq) axes
        ordered either way depending on the kernel build. Reconcile purely from
        the element count so no layout assumption is needed: ``heads_eff`` (the
        padded head count) is ``numel // num_tokens``; the axis whose size is
        ``heads_eff`` is the head axis, and every other axis is a token axis.
        Move the head axis last, collapse the rest into ``tokens`` (preserving
        order), then slice the padded heads back to ``num_heads`` so the DCP
        combine sees exactly ``[tokens, heads]``.
        """
        lse = lse.to(torch.float32)
        total = lse.numel()
        assert total % num_tokens == 0, (
            f"LSE numel {total} not divisible by num_tokens {num_tokens}"
        )
        heads_eff = total // num_tokens
        head_axis = next((i for i, s in enumerate(lse.shape) if s == heads_eff), None)
        assert head_axis is not None, (
            f"no LSE axis matches padded head count {heads_eff} "
            f"in shape {tuple(lse.shape)}"
        )
        lse = lse.movedim(head_axis, -1).reshape(num_tokens, heads_eff)
        if heads_eff != num_heads:
            lse = lse[:, :num_heads]
        return lse.contiguous()

    @staticmethod
    def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128
        # Compute padded head count for decode
        return 64 if num_heads <= 64 else 128

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        self.softmax_scale = scale
        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)
        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )

        if kv_cache_dtype == "fp8_ds_mla":
            # Reserve workspace during initialization
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            self.prefill_workspace_shape = (prefill_workspace_size, head_size)
            self.q_concat_buffer, self.prefill_bf16_workspace = (
                current_workspace_manager().get_simultaneous(
                    (q_concat_shape, torch.bfloat16),
                    (self.prefill_workspace_shape, torch.bfloat16),
                )
            )
        else:
            (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
                (q_concat_shape, torch.bfloat16),
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The BF16 sparse kernel does not emit an LSE, so the per-rank partial
        # outputs cannot be combined across DCP shards. DCP with this backend
        # requires the fp8_ds_mla KV cache (decode kernel returns the LSE).
        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "FlashMLA sparse with a BF16 KV cache does not support decode "
                "context parallelism; use --kv-cache-dtype fp8_ds_mla."
            )
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill). req_id_per_token covers the whole batch; slice it
        # to the MQA tokens (q may exclude prefill tokens routed to dense MHA).
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        attn_out = self._bf16_flash_mla_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length,
        )
        return attn_out, None

    def _forward_fp8_kv_separate_prefill_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)
        num_decodes = fp8_metadata.num_decodes
        num_mqa_tokens = q.shape[0]
        num_decode_tokens = fp8_metadata.num_decode_tokens
        num_prefill_tokens = num_mqa_tokens - num_decode_tokens
        assert num_prefill_tokens in (0, fp8_metadata.num_prefill_tokens), (
            "FP8 sparse MLA expects either the decode subset or the full batch"
        )

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill_workspace = False
        if num_prefill_tokens > 0:
            assert fp8_metadata.prefill is not None
            prefill_request_ids = fp8_metadata.prefill.request_ids
            prefill_workspace_starts = fp8_metadata.prefill.workspace_starts
            has_prefill_workspace = True

        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
        dcp_fast = self.dcp_world_size > 1 and has_prefill_workspace
        if dcp_fast:
            assert fp8_metadata.prefill is not None
            if not fp8_metadata.prefill.dcp_fast:
                raise NotImplementedError(
                    "FlashMLA sparse DCP prefill is only supported with the "
                    "ag_rs dcp_comm_backend and without prefill context "
                    "parallelism (the DCP fast-path metadata was not built)."
                )

        raw_topk_indices = topk_indices
        topk_length: torch.Tensor | None = None
        if self.dcp_world_size > 1:
            # Decode-context-parallel: decode rows attend this rank's KV shard
            # (globally-consistent top-k filtered to owned slots; the partial
            # outputs are merged by the LSE-weighted combine in the MLA common
            # forward). Prefill rows take the DCP fast path: their raw
            # req-local indices are converted per chunk to offsets in the
            # DCP-allgathered workspace below.
            topk_indices = raw_topk_indices[:num_decode_tokens]
            if num_decode_tokens > 0:
                topk_indices, topk_length = triton_filter_and_convert_dcp_index(
                    attn_metadata.req_id_per_token[:num_decode_tokens],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=(
                        attn_metadata.cp_kv_cache_interleave_size
                    ),
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                )
        else:
            topk_indices, topk_length = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                HAS_PREFILL_WORKSPACE=has_prefill_workspace,
                prefill_workspace_request_ids=prefill_request_ids,
                prefill_workspace_starts=prefill_workspace_starts,
                return_valid_counts=True,
            )

        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)

        def _fp8_decode(
            q: torch.Tensor,
            topk_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Reshape q: (num_decode_tokens, num_heads, head_dim)
            #         -> (num_decodes, seq_len, num_heads, head_dim)
            q = reshape_query_for_spec_decode(q, num_decodes)
            seq_len = q.shape[1]
            # Reshape topk_indices: (num_decode_tokens, topk)
            #                    -> (num_decodes, seq_len, topk)
            topk_indices = topk_indices.view(num_decodes, seq_len, -1)
            assert fp8_metadata.decode is not None
            attn_out, lse = self._fp8_flash_mla_kernel(
                q=q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                topk_indices=topk_indices,
                kernel_metadata=fp8_metadata.decode.kernel_metadata,
            )
            # Reshape output: (num_decodes, seq_len, num_heads, head_dim_v)
            #              -> (num_decode_tokens, num_heads, head_dim_v)
            attn_out = reshape_attn_output_for_spec_decode(attn_out)
            lse = self._normalize_lse(lse, attn_out.shape[0], attn_out.shape[1])
            return attn_out, lse

        # Decode rows are always the prefix [:num_decode_tokens] of the batch.
        # Under DCP the full-batch LSE is assembled below (decode rows carry
        # the real kernel LSE; prefill stripe rows are complete so they route
        # through the combine with weight one via the -inf trick).
        decode_lse: torch.Tensor | None = None

        # Pure decode: direct call without allocation
        if num_decode_tokens > 0 and num_prefill_tokens == 0:
            assert fp8_metadata.decode is not None
            attn_out, decode_lse = _fp8_decode(q, topk_indices)
        else:
            # Mixed or pure prefill: allocate output tensor. Use the runtime
            # head count: under DCP the query heads arrive all-gathered
            # (H = num_heads * dcp_world_size).
            attn_out = q.new_empty(
                (num_mqa_tokens, q.shape[1], self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            if num_decode_tokens > 0:
                decode_out, decode_lse = _fp8_decode(
                    q[:num_decode_tokens],
                    topk_indices[:num_decode_tokens],
                )
                attn_out[:num_decode_tokens] = decode_out

            assert fp8_metadata.prefill is not None
            prefill_meta = fp8_metadata.prefill
            if dcp_fast:
                from vllm.distributed.parallel_state import get_dcp_group

                logger.info_once(
                    "FlashMLA sparse: DCP prefill fast path active "
                    "(workspace allgather + per-rank query stripes)"
                )
                # Rows outside this rank's stripes must contribute exactly
                # zero through the DCP combine (their LSE stays -inf).
                attn_out[num_decode_tokens:] = 0
                for chunk in prefill_meta.chunks:
                    tot = int(chunk.chunk_tot_seqlen)
                    local_ws = self.prefill_bf16_workspace[:tot]
                    ops.cp_gather_and_upconvert_fp8_kv_cache(
                        kv_c_and_k_pe_cache,
                        local_ws,
                        chunk.block_table,
                        chunk.workspace_starts,
                        len(chunk.block_table),
                    )
                    # Collective: every DCP rank contributes the same padded
                    # row count, even when its query stripe is empty.
                    gathered_ws = get_dcp_group().all_gather(local_ws, dim=0)

                    stripe = chunk.stripe_slice
                    assert stripe is not None
                    if stripe.stop <= stripe.start:
                        continue
                    ws_indices, ws_topk_length = (
                        triton_convert_req_index_to_dcp_gathered_ws_index(
                            prefill_meta.request_ids[stripe],
                            raw_topk_indices[stripe],
                            prefill_meta.workspace_starts,
                            per_rank_rows=tot,
                            dcp_size=self.dcp_world_size,
                            cp_kv_cache_interleave_size=(
                                attn_metadata.cp_kv_cache_interleave_size
                            ),
                        )
                    )
                    attn_out[stripe] = self._bf16_flash_mla_kernel(
                        q[stripe],
                        gathered_ws,
                        ws_indices,
                        ws_topk_length,
                    )
            else:
                assert topk_length is not None
                for chunk in prefill_meta.chunks:
                    chunk_workspace = self.prefill_bf16_workspace[
                        : chunk.chunk_tot_seqlen
                    ]
                    ops.cp_gather_and_upconvert_fp8_kv_cache(
                        kv_c_and_k_pe_cache,
                        chunk_workspace,
                        chunk.block_table,
                        chunk.workspace_starts,
                        len(chunk.block_table),
                    )

                    chunk_q = q[chunk.tokens_slice]
                    chunk_topk_indices_workspace = topk_indices[chunk.tokens_slice]
                    chunk_topk_length = topk_length[chunk.tokens_slice]

                    attn_out[chunk.tokens_slice] = self._bf16_flash_mla_kernel(
                        chunk_q,
                        chunk_workspace,
                        chunk_topk_indices_workspace,
                        chunk_topk_length,
                    )

        if dcp_fast:
            # Full-batch LSE for the DCP combine: real LSE on decode rows,
            # 0 on this rank's complete prefill stripe rows, -inf elsewhere
            # (the combine kernel maps -inf to weight zero).
            lse = q.new_full(
                (num_mqa_tokens, q.shape[1]), float("-inf"), dtype=torch.float32
            )
            if decode_lse is not None:
                lse[:num_decode_tokens] = decode_lse
            assert fp8_metadata.prefill is not None
            for chunk in fp8_metadata.prefill.chunks:
                stripe = chunk.stripe_slice
                if stripe is not None and stripe.stop > stripe.start:
                    lse[stripe] = 0.0
            return attn_out, lse

        return attn_out, decode_lse

    def _forward_fp8_kv_mixed_batch(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mixed batch FP8 forward path that treats all tokens as one batch.

        This is equivalent to main branch's approach and avoids the BF16
        prefill kernel which has head padding overhead when num_heads is small.
        Used when use_mixed_batch is True.
        """
        # Convert per-request indices to global cache slots. Under DCP the KV
        # cache is sharded across cp ranks, so the globally-consistent top-k
        # (produced by the indexer's DCP top-k merge) must be de-interleaved to
        # this rank's owned slots -- otherwise every rank reads KV it does not
        # hold. The fp8 kernel skips sentinel (invalid) indices, so the valid
        # counts are only needed by the BF16 prefill kernel and are ignored
        # here.
        if self.dcp_world_size > 1:
            topk_indices, _ = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(
                    attn_metadata.cp_kv_cache_interleave_size
                ),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[: topk_indices.shape[0]],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        num_tokens = q.shape[0]
        _attn_out, _lse = self._fp8_flash_mla_kernel(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim: (T, H, D) -> (1, T, H, D)
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
            kernel_metadata=fp8_metadata,
        )

        # Output is (1, T, H, D_v), squeeze back to (T, H, D_v)
        attn_out = _attn_out.squeeze(0)
        # Match the LSE head count to attn_out's own head count, NOT
        # self.num_heads. Under DCP the query heads are all-gathered across cp
        # ranks (H = num_heads * dcp_world_size) before this call, and the DCP
        # combine needs the LSE for *all* gathered heads to reduce-scatter them
        # back to the locally-owned heads. Passing self.num_heads here would
        # drop the extra DCP heads and corrupt the combine.
        lse = self._normalize_lse(_lse, num_tokens, attn_out.shape[1])
        return attn_out, lse

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        # Derive the pad target from the runtime head count rather than the
        # per-rank count from __init__: under DCP the query heads arrive
        # all-gathered across cp ranks (H = num_heads * dcp_world_size).
        actual_num_heads = q.size(2)
        assert actual_num_heads <= 128, (
            f"FP8 sparse decode kernel supports at most 128 query heads, got "
            f"{actual_num_heads} (num_heads * dcp_world_size?)"
        )
        padded_num_heads = self._compute_fp8_decode_padded_heads(actual_num_heads)

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )

        # Slice output back to actual head count if we padded. The kernel emits
        # LSE with the padded head count too, so slice it on the head axis as
        # well; otherwise the extra padded-head LSE entries corrupt the
        # downstream normalization and the DCP LSE-weighted combine.
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            if lse.shape[-1] == padded_num_heads:
                # LSE laid out (..., H): heads on the last axis.
                lse = lse[..., :actual_num_heads]
            elif lse.dim() >= 2 and lse.shape[-2] == padded_num_heads:
                # LSE laid out (..., H, 1) or (..., H, X): heads on axis -2.
                lse = lse[..., :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        # Runtime head count: under the DCP fast path the query heads arrive
        # all-gathered (H = num_heads * dcp_world_size); otherwise per-rank.
        num_q_heads = q.shape[1]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if num_q_heads % self.prefill_padding != 0:
            assert self.prefill_padding % num_q_heads == 0
            logger.warning_once(
                f"Padding num_heads from {num_q_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, :num_q_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            topk_length=topk_length,
        )[0]

        output = output[:, :num_q_heads, :]
        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        if not use_fp8_cache:
            attn_out, lse = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        elif attn_metadata.fp8_use_mixed_batch:
            attn_out, lse = self._forward_fp8_kv_mixed_batch(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        else:
            attn_out, lse = self._forward_fp8_kv_separate_prefill_decode(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )

        # ``lse`` (natural-log softmax LSE, [num_tokens, num_heads], fp32) is
        # consumed by the MLA common forward only when dcp_world_size > 1;
        # otherwise it is ignored, so returning it never changes single-rank
        # numerics.
        return attn_out, lse
