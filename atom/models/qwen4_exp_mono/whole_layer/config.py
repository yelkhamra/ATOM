"""Qwen3.8-Flash-Next (HF / ATOM ``qwen4_exp``) geometry, checked against a checkpoint's ``config.json``.

Every per-layer size is fixed: :func:`geometry` raises unless the text config matches :data:`G`; the model-level counts
(layers, experts, layer types, PLE layers) are read from it.

Layer kinds (``Geometry.layer_kind``): ``delta`` (Gated DeltaNet + MoE under two hyper-connections), ``delta_ple`` (the
same after the PLE block; ``ple_layer_ids`` is 1-based), ``qsa`` (query-sparse attention + MoE) and ``mtp`` (the MTP
drafter's layer, index ``layers``: a QSA layer with its own weights and caches whose RoPE positions run
``ROPE_OFFSET["mtp"]`` ahead of its cache rows: ATOM's draft writes token ``t + 1``'s key at row ``t``).
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from atom.mono.program.geometry import (
    AttnGeometry,
    GDNGeometry,
    HCGeometry,
    IndexerGeometry,
    MoEGeometry,
)

QSA_TYPES = ("full_attention", "qwen_sparse_attention")
QSA_KINDS = ("qsa", "mtp")  # kinds whose mixer is QSA (one recipe, one kernel)
ROPE_OFFSET = {"qsa": 0, "mtp": 1}  # RoPE position - cache row, per QSA kind


@dataclass(frozen=True)
class Geometry:
    hidden: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    eps: float = 1e-6
    # Gated DeltaNet (linear attention)
    gdn_k_heads: int = 16
    gdn_v_heads: int = 48
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    conv_kernel: int = 4
    # QSA: gated GQA with partial RoPE and the block indexer
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 10_000_000.0
    index_heads: int = 4
    index_dim: int = 128
    index_budget: int = 2048  # tokens: the top index_budget / index_compress blocks
    index_compress: int = 4
    # MoE
    top_k: int = 10
    moe_inter: int = 640
    shared_inter: int = 640
    ple_dim: int = 2560
    # model-level counts
    layers: int = 48
    experts: int = 512
    layer_types: tuple = field(default=tuple("full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)))
    ple_layers: tuple = (1,)  # 0-based

    @property
    def hc_width(self) -> int:
        return self.hc_count * self.hidden  # 10240: the four residual streams

    @property
    def gdn_value_width(self) -> int:
        return self.gdn_v_heads * self.gdn_v_dim  # 6144

    @property
    def conv_dim(self) -> int:
        return 2 * self.gdn_k_heads * self.gdn_k_dim + self.gdn_value_width  # 10240: q | k | v through the conv

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim  # 6144

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim  # 512

    @property
    def hc(self) -> HCGeometry:
        return HCGeometry(self.hidden, self.hc_count, self.hc_lowrank, self.eps)

    @property
    def gdn(self) -> GDNGeometry:
        """The recurrent state is stored fp32 (the checkpoint's ``mamba_ssm_dtype``, ATOM's state pool)."""
        return GDNGeometry(
            self.gdn_k_heads, self.gdn_v_heads, self.gdn_k_dim, self.gdn_v_dim, self.conv_kernel, self.eps, torch.float32
        )

    @property
    def attn(self) -> AttnGeometry:
        return AttnGeometry(self.q_heads, self.kv_heads, self.head_dim, self.rotary_dim, self.rope_theta, self.eps)

    @property
    def indexer(self) -> IndexerGeometry:
        blocks = self.index_budget // self.index_compress  # 512 selected blocks at most
        return IndexerGeometry(
            self.index_heads, self.index_dim, self.index_compress, blocks, self.rotary_dim, self.rope_theta, self.eps
        )

    @property
    def moe(self) -> MoEGeometry:
        return MoEGeometry(self.hidden, self.experts, self.top_k, self.moe_inter, 1, self.shared_inter)

    def layer_kind(self, layer: int) -> str:
        """The kind of decoder layer ``layer``; index ``layers`` is the MTP drafter's (``mtp``)."""
        if layer == self.layers:
            return "mtp"
        if not 0 <= layer < self.layers:
            raise IndexError(f"layer {layer} of {self.layers}")
        if self.layer_types[layer] in QSA_TYPES:
            if layer in self.ple_layers:
                raise ValueError("PLE on a QSA layer is not supported")
            return "qsa"
        return "delta_ple" if layer in self.ple_layers else "delta"


G = Geometry()

# Per-layer sizes the kernels are written for: config.json key -> Geometry field.
_FIXED = {
    "hidden_size": "hidden",
    "hc_count": "hc_count",
    "hc_lowrank": "hc_lowrank",
    "rms_norm_eps": "eps",
    "linear_num_key_heads": "gdn_k_heads",
    "linear_num_value_heads": "gdn_v_heads",
    "linear_key_head_dim": "gdn_k_dim",
    "linear_value_head_dim": "gdn_v_dim",
    "linear_conv_kernel_dim": "conv_kernel",
    "num_attention_heads": "q_heads",
    "num_key_value_heads": "kv_heads",
    "head_dim": "head_dim",
    "indexer_n_heads": "index_heads",
    "indexer_head_dim": "index_dim",
    "indexer_budget": "index_budget",
    "indexer_compress_ratio": "index_compress",
    "num_experts_per_tok": "top_k",
    "moe_intermediate_size": "moe_inter",
    "shared_expert_intermediate_size": "shared_inter",
    "ple_embed_dim": "ple_dim",
}
# Fixed by the model's code rather than a size: checked too.
_REQUIRED = {
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "indexer_kv_heads": 1,
    "output_gate_type": "sigmoid",
    "hidden_act": "silu",
    "mamba_ssm_dtype": "float32",
}


def geometry(config: dict) -> Geometry:
    """The ``Geometry`` of an HF ``config.json`` dict (or its ``text_config``); ``ValueError`` on any per-layer size
    other than :data:`G`'s."""
    text = config.get("text_config", config)
    bad = {key: (text[key], getattr(G, name)) for key, name in _FIXED.items() if text.get(key) != getattr(G, name)}
    bad.update({key: (text.get(key), want) for key, want in _REQUIRED.items() if text.get(key) != want})
    rope = text.get("rope_parameters", {})
    rotary = int(text["head_dim"] * rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0)))
    if rotary != G.rotary_dim:
        bad["rotary_dim"] = (rotary, G.rotary_dim)
    if float(rope.get("rope_theta", 0)) != G.rope_theta:
        bad["rope_theta"] = (rope.get("rope_theta"), G.rope_theta)
    if bad:
        raise ValueError(f"unsupported Qwen3.8-Flash-Next geometry (got, supported): {bad}")
    geo = replace(
        G,
        layers=int(text["num_hidden_layers"]),
        experts=int(text["num_experts"]),
        layer_types=tuple(text["layer_types"]),
        ple_layers=tuple(i - 1 for i in text.get("ple_layer_ids", ())),
    )
    if len(geo.layer_types) != geo.layers:
        raise ValueError(f"{len(geo.layer_types)} layer_types for {geo.layers} layers")
    for i in range(geo.layers):
        geo.layer_kind(i)  # raises on PLE-on-QSA
    return geo


def load_geometry(checkpoint) -> Geometry:
    return geometry(json.loads((Path(checkpoint) / "config.json").read_text()))
