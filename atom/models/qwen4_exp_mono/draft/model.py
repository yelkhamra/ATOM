# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen Mono's MTP drafter: the draft step returns the complete hyper-connection state and the head input, the first
step's MLP runs on each request's selected row only, and the draft ids come from the certificate heads (``head``).

The drafter (``spec_decode.drafter``) builds ``MonoMTP`` in place of the Native ``Qwen4ExpMTP`` when Mono serves the
deployment (``config.config_refusal``). A draft step of at most 16 rows runs ``draft.native.forward`` (the Mono draft
program when it serves the step, ``runner.DraftRunner``, else the fallback draft), a prefill of at most 16 rows
``draft.native.compact_forward``, anything larger ATOM's own layer.
"""

import torch
from torch import nn

from atom.config import Config
from atom.model_ops.embed_head import ParallelLMHead
from atom.models.qwen4_exp_mtp import Qwen4ExpMTP as NativeMTP
from atom.models.qwen4_exp_mtp import Qwen4ExpMultiTokenPredictor as NativePredictor


class MonoPredictor(NativePredictor):
    def _fc_sum(self, input_ids, hidden_states, inputs_embeds, n, inbuf=None):
        """``fc_hidden(norm(state)) + fc_embedding(norm(embed))`` per stream, [n, streams, H] (into ``inbuf``, the
        Mono draft program's registered input, when given and large enough)."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        hidden = self.pre_fc_norm_hidden(hidden_states.reshape(n, -1))
        hidden = self.fc_hidden(hidden.reshape(-1, self.hidden_size))
        if inbuf is not None and n <= inbuf.shape[0]:
            # (the Mono draft program's registered input: its kernel binds it, no staging copy)
            return torch.add(hidden.view(n, self.hc_count, self.hidden_size), embedding[:, None, :],
                             out=inbuf[:n].view(n, self.hc_count, self.hidden_size))  # fmt: skip
        return hidden.view(n, self.hc_count, self.hidden_size) + embedding[:, None, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        selected_rows: torch.Tensor | None = None,
        with_head_input: bool = False,
    ):
        n = hidden_states.shape[0]
        inbuf = getattr(getattr(self, "_mono_draft", None), "inbuf", None)
        hidden = self._fc_sum(input_ids, hidden_states, inputs_embeds, n, inbuf)
        if n <= 16:
            from atom.models.qwen4_exp_mono.draft import native

            if with_head_input:
                return native.forward(self, positions, hidden.flatten(1), selected_rows)
            hidden = native.compact_forward(
                self.layers[0], positions, hidden.flatten(1), selected_rows
            )
        else:
            assert selected_rows is None
            hidden = self.layers[0](positions, hidden.flatten(1), None)
        hidden = hidden.view(-1, self.hc_count, self.hidden_size)
        if with_head_input:
            return hidden, self.hyper_connection_mixer.mix(hidden.flatten(1))[0]
        return hidden


class MonoMTP(NativeMTP):
    draft_graph_supports_mrope = True
    reuse_draft_graph_step_buffers = True
    # the step returns (state, head input) and the first step takes ``selected_rows`` (``EagleProposer``)
    draft_returns_head_input = True

    @staticmethod
    def draft_graph_hidden_state_shape(draft_hf) -> tuple[int, ...]:
        """The draft carries the whole hyper-connection bundle between steps."""
        return (draft_hf.hc_count, draft_hf.hidden_size)

    def __init__(self, atom_config: Config):
        nn.Module.__init__(self)
        self.config = atom_config.hf_config
        self.model = MonoPredictor(atom_config)
        assert self.model.layers[0].tp_size == 1
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size, prefix="lm_head"
        )

    def forward_with_head(self, input_ids, positions, hidden_states):
        return self.model(input_ids, positions, hidden_states, with_head_input=True)

    def forward_selected_with_head(
        self, input_ids, positions, hidden_states, selected_rows
    ):
        return self.model(
            input_ids,
            positions,
            hidden_states,
            selected_rows=selected_rows,
            with_head_input=True,
        )

    def prepare_draft_head(self):
        """After the draft weights load, before warm-up and graph capture: the draft heads (``head.heads.prepare``)."""
        from atom.models.qwen4_exp_mono.draft import native
        from atom.models.qwen4_exp_mono.head.heads import prepare

        native.validate_layer(self.model.layers[0])
        assert self.lm_head.tp_size == 1 and self.lm_head.bias is None
        assert self.lm_head.weight.shape == (248320, 2560)
        self._screen_head, self._batch_heads = prepare(self.lm_head.weight)

    def _draft_argmax(self, head_input, *, out):
        """1 row: the global-certificate head; 2 / 4 rows: the batched W4 head; others: ATOM's argmax head. The ids
        are the BF16 head's argmax (ties low) on every route."""
        if (
            head_input.ndim == 2
            and head_input.shape[0] in self._batch_heads
            and head_input.shape[1] == 2560
            and head_input.dtype == torch.bfloat16
            and head_input.is_contiguous()
            and out.shape == (head_input.shape[0],)
            and out.dtype == torch.int32
            and out.is_contiguous()
        ):
            return self._batch_heads[head_input.shape[0]].run(head_input, out=out)
        if head_input.shape == (1, 2560):
            return self._screen_head.run(head_input, out=out)
        return self.lm_head.compute_argmax_token(head_input, out=out)

    def compute_draft_ids_with_head(self, head_input, *, out):
        return self._draft_argmax(head_input, out=out)

    def compute_draft_ids(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0, *, out: torch.Tensor
    ):
        return self._draft_argmax(self._head_input(hidden_states), out=out)
