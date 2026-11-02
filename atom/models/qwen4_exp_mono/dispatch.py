# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Qwen3.8-Flash-Next forwards take the Mono path.

The installer (``models.registry.mono_decode_installers``) wraps the loaded model after load: a served MTP verify step
(``plan.verify_step``) runs every decoder layer as one Mono program, reading and writing ATOM's GDN pools and QSA caches
in place; every other step runs ATOM's own model. The drafter's layer is attached to the MTP predictor
(``runner.DraftRunner``, asked first by every draft step). Both are built eagerly here: the kernels of every served
shape, the drafter's weights bound; the target's on the first eager step of each shape (before graph capture).
"""

import logging

import torch
from torch import nn

from atom.models.qwen4_exp_mono.config import config_refusal, geometry

logger = logging.getLogger("atom")


class MonoTargetModel(nn.Module):
    """``Qwen4ExpForConditionalGeneration`` with its served verify steps on Mono layer programs."""

    def __init__(self, model: nn.Module, runner) -> None:
        super().__init__()
        self.model = model
        self.runner = runner

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        meta = self.runner.step_meta(input_ids, intermediate_tensors, inputs_embeds)
        if meta is None:
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return self.runner.forward(input_ids, meta)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    """``model`` wrapped for Qwen Mono decode, or ``model`` itself when Mono does not serve this deployment."""
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("Qwen mono off: %s", why)
        return model
    from atom.models.qwen4_exp_mono.draft.model import MonoMTP
    from atom.models.qwen4_exp_mono.runner import DraftRunner, TargetRunner

    geo = geometry(atom_config)
    if drafter is not None and isinstance(drafter.model, MonoMTP):
        inner = drafter.model.model
        inner._mono_draft = DraftRunner(inner, atom_config, geo)
    return MonoTargetModel(model, TargetRunner(model, atom_config, geo))
