# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft step outside the Mono draft program, with ATOM's BF16 layer weights.

``forward`` returns the complete hyper-connection state and the prepared head input of a draft step: the Mono draft
program's (``runner.DraftRunner.serve``) when it serves the step, else ``native``'s: a decode step of more than 4
requests, the startup runs before the program is attached, or a refused build. ``native`` runs the QSA attention fused
(``attention``: the target's norm / RoPE / cache publication and the index publication in two launches) and, for one
row, the experts through ATOM's CK stages on compact descriptors (``experts.NativeExperts``).
``compact_forward``: a prefill of at most 16 rows (no head input), ATOM's attention then the selected rows' MLP.
"""

import json
import logging

import torch

from atom.model_ops.qwen4_exp.ops.gated import sigmoid_mul
from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)
from atom.models.qwen4_exp_mono.draft import qsa_index
from atom.models.qwen4_exp_mono.draft.experts import NativeExperts
from atom.models.qwen4_exp_mono.draft.qsa_epilogue import qsa_norm_rope_cache
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def forward(model, positions, hidden, selected_rows):
    mono_draft = getattr(model, "_mono_draft", None)
    if mono_draft is not None:
        got = mono_draft.serve(model, hidden, selected_rows)
        if got is not None:
            return got
    return native(model, positions, hidden, selected_rows)


def native(model, positions, hidden, selected_rows):
    layer = model.layers[0]
    mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(hidden)
    if not getattr(model, "_draft_qsa_fused_logged", False):
        assert not torch.cuda.is_current_stream_capturing()
        model._draft_qsa_fused_logged = True
        sa = layer.self_attn
        logger.info(
            "Qwen mono fallback draft: fused QSA (kv %s, index %s)",
            sa.kv_format,
            sa.index_format,
        )
    block = attention(layer.self_attn, positions, mixed)
    if selected_rows is not None:
        streams, block, raw = [
            torch.index_select(x, 0, selected_rows) for x in (streams, block, raw)
        ]
    hc = layer.mlp_hyper_connection
    mixed, (streams, raw) = hc.mix_fused(streams, (block, raw))
    if mixed.shape[0] == 1:
        if not hasattr(model, "_q1_experts"):
            assert not torch.cuda.is_current_stream_capturing()
            validate_layer(layer)
            ex = layer.mlp.experts
            assert ex.top_k == 10 and ex.renormalize and ex.scoring_func == "softmax"
            assert ex.expert_map is None and not ex.apply_router_weight_on_input
            model._q1_experts = NativeExperts(ex.w13_weight, ex.w2_weight, 10)
            logger.info("Qwen mono fallback draft: one-row experts %s",
                        json.dumps({"topk": 10, "expert_dtype": str(ex.w13_weight.dtype),
                                    "block_m": model._q1_experts.bm}))  # fmt: skip
        mlp = layer.mlp
        logits = mlp.gate(mixed)
        weights, ids = mlp.experts.select_experts(
            hidden_states=mixed,
            router_logits=logits[:, :512],
            top_k=10,
            use_grouped_topk=False,
            renormalize=True,
            scoring_func="softmax",
        )
        routed = model._q1_experts(mixed, ids, weights)
        shared = mlp.shared_expert(mixed)
        out = sigmoid_mul(shared, logits[:, 512:], routed)
    else:
        out = layer.mlp(mixed)
    head_input, (state, _) = model.hyper_connection_mixer.mix_fused(streams, (out, raw))
    return state.view(-1, model.hc_count, model.hidden_size), head_input


def attention(module, positions, mixed):
    """The draft's QSA attention for q1 / q4 decode rows on the target's validated preparation arithmetic (the first
    q4 pass publishes every raw row before group compression; RoPE positions come from the draft's own position
    cache, including its +1); anything else ATOM's module."""
    fc = get_forward_context()
    meta = fc.attn_metadata.qsa_metadata
    if meta is None or fc.context.is_prefill:
        return module(positions, mixed)
    requests = meta.block_tables.shape[0]
    rows = mixed.shape[0]
    if rows not in (requests, 4 * requests):
        return module(positions, mixed)
    assert fc.context.is_draft
    assert (
        module.compressed_key_scale is None and module.k_scale_cache is None
    )  # (BF16 draft caches)
    assert module.rope_position_cache is not None
    assert module.indexer.compress_ratio == 4 and module.indexer.index_head_dim == 128
    qkv = module.qkv_proj(mixed)
    gate, q, k, v = qkv.split(
        [module.q_size, module.q_size, module.kv_size, module.kv_size], -1
    )
    qsa = meta.for_tokens(rows)
    query = qsa_norm_rope_cache(
        q.view(rows, module.num_heads, module.head_dim),
        k.view(rows, module.num_kv_heads, module.head_dim),
        v.view(rows, module.num_kv_heads, module.head_dim),
        module.qk_norm,
        module.rotary_emb,
        positions,
        module.k_cache,
        module.v_cache,
        qsa.slot_mapping,
    )
    index_query = qsa_index.prepare(
        module, mixed, positions, qsa, query_length=4 if rows == 4 * requests else 1
    )
    selected = qsa_select_paged_tokens(
        index_query,
        module.compressed_key_cache,
        qsa.block_tables,
        qsa.token_to_req,
        qsa.logical_positions,
        qsa.seq_lens,
        module.indexer.token_topk,
        module.indexer.compress_ratio,
        max_seq_len=qsa.max_seq_len,
        allow_flydsl=False,
        compressed_k_scale=module.compressed_key_scale,
    )
    attn_out = qsa_sparse_paged_gqa(
        query,
        module.k_cache,
        module.v_cache,
        selected,
        qsa.block_tables,
        qsa.token_to_req,
        softmax_scale=module.scaling,
        num_decode_requests=qsa.block_tables.shape[0],
        k_scale=module.k_scale_cache,
        v_scale=module.v_scale_cache,
    )
    return module.o_proj(sigmoid_mul(attn_out.reshape(gate.shape[0], -1), gate))


def compact_forward(layer, positions, hidden, selected_rows=None):
    """ATOM's QSA call (it publishes every row's caches), then only the selected rows' MLP: the rows the head and the
    next draft step use. No state is compacted."""
    mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(hidden)
    block = layer.self_attn(positions, mixed)
    if selected_rows is not None:
        streams = torch.index_select(streams, 0, selected_rows)
        block = torch.index_select(block, 0, selected_rows)
        raw = torch.index_select(raw, 0, selected_rows)
    hc = layer.mlp_hyper_connection
    mixed, (streams, raw) = hc.mix_fused(streams, (block, raw))
    return hc.apply_pending(streams, (layer.mlp(mixed), raw))


def validate_layer(layer):
    assert layer.tp_size == 1 and layer.ple is None and not layer.fused_hc
    assert layer.layer_type in ("full_attention", "qwen_sparse_attention")
    assert layer._attn_input_quant_dtype() is None
    assert not layer.mlp.fuse_shared and layer.mlp.shared_expert is not None
    assert layer.mlp.n_routed_experts == 512
    assert layer.mlp.experts.w13_weight.dtype == torch.bfloat16
    assert layer.mlp.experts.w2_weight.dtype == torch.bfloat16
    for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
        assert hc.hc_count == 4 and hc.hidden_size == 2560 and hc.fused_rank == 320
        assert hc.hc_norm.group_size == 2560
        assert hc.hc_norm.variance_epsilon == 1e-6
        assert hc.fused_w_cat.shape == (324, 10240)
        assert hc.fused_w_cat.dtype == torch.bfloat16
