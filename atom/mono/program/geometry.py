"""Geometry of the shared layer pieces: what blocks take as arguments.

The Qwen model package builds these from its own geometry (``whole_layer.config.Geometry``); blocks
(``atom.mono.program.blocks``) never read a model's constants. They are frozen (hashable): blocks put them in their
build keys.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNGeometry:
    """A gated delta net mixer (Qwen3.5 / Qwen3-Next style).

    q, k: ``k_heads`` heads of ``k_dim``; v (and z): ``v_heads`` heads of ``v_dim``; each
    k-head serves ``v_heads // k_heads`` consecutive v-heads. The causal conv has
    ``conv_kernel`` taps over the concatenated pre-conv qkv rows (``conv_dim``). The
    gated output norm uses ``eps``. The recurrent state [v_heads, k_dim, v_dim] (V
    contiguous) is *stored* in ``state_dtype`` and computed in fp32.
    """

    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_kernel: int
    eps: float
    state_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.v_heads % self.k_heads:
            raise ValueError(f"{self}: v_heads must be a multiple of k_heads")

    @property
    def rep(self) -> int:
        """v-heads per k-head."""
        return self.v_heads // self.k_heads

    @property
    def key_width(self) -> int:
        return self.k_heads * self.k_dim

    @property
    def value_width(self) -> int:
        return self.v_heads * self.v_dim

    @property
    def conv_dim(self) -> int:
        """Pre-conv qkv channels: q | k | v."""
        return 2 * self.key_width + self.value_width


@dataclass(frozen=True)
class AttnGeometry:
    """Gated multi-head attention with GQA and partial RoPE (Qwen3.5 full attention).

    ``q_heads`` query heads over ``kv_heads`` KV heads of ``head_dim``; RoPE rotates the
    first ``rotary_dim`` dims (base ``rope_theta``, HF's half-split rotation); q and k
    are RMS-normed per head (zero-centred ``1 + w``, ``eps``) before RoPE. The fused
    projection is q_proj (per head [q, gate]) | k_proj | v_proj.
    """

    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    eps: float

    def __post_init__(self):
        if self.q_heads % self.kv_heads or self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError(f"unsupported attention geometry {self}")

    @property
    def rep(self) -> int:
        """Query heads per KV head."""
        return self.q_heads // self.kv_heads

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class HCGeometry:
    """Hyper-connections: ``streams`` residual streams of ``hidden`` (the flat row is
    stream-major, ``streams * hidden``); each sub-layer's HC mixes them through a low-rank
    path of ``lowrank`` columns and normalizes each stream with ``eps``."""

    hidden: int
    streams: int
    lowrank: int
    eps: float

    @property
    def width(self) -> int:
        """The flat stream row: streams * hidden."""
        return self.streams * self.hidden


@dataclass(frozen=True)
class MoEGeometry:
    """A routed MoE over rows of ``hidden``: ``experts`` routed SwiGLU experts of
    ``inter`` (softmax router, the top ``top_k`` renormalized) plus ``shared`` always-on
    experts of ``shared_inter`` with a sigmoid gate."""

    hidden: int
    experts: int
    top_k: int
    inter: int
    shared: int = 1
    shared_inter: int = 0

    def __post_init__(self):
        if not 0 < self.top_k <= self.experts:
            raise ValueError(f"unsupported MoE geometry {self}")


@dataclass(frozen=True)
class IndexerGeometry:
    """A block indexer for sparse attention: ``heads`` index query heads of ``head_dim``
    score one key per block of ``compress`` tokens; a query attends the tokens of its ``top_blocks`` best blocks plus the incomplete tail. Queries and
    block keys are RMS-normed per head (zero-centred, ``eps``) and rotated on their first
    ``rotary_dim`` dims (base ``rope_theta``, half-split rotation)."""

    heads: int
    head_dim: int
    compress: int
    top_blocks: int
    rotary_dim: int
    rope_theta: float
    eps: float

    def __post_init__(self):
        if self.rotary_dim % 2 or self.rotary_dim > self.head_dim or self.top_blocks < 1:
            raise ValueError(f"unsupported indexer geometry {self}")
