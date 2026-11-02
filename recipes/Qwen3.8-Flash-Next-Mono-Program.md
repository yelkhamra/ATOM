# Qwen3.8-Flash-Next Mono decode

Mono runs every target decoder layer of an MTP3 verify step, and the MTP drafter's layer, as one whole-layer
program per layer (FlyDSL, gfx950). The service entry is `atom/models/qwen4_exp_mono/dispatch.py`
(`models.registry.mono_decode_installers`); the layer recipes and schedules are in
`atom/models/qwen4_exp_mono/whole_layer/`, the shared program framework in `atom/mono/program/`.

## Deployment contract

`atom/models/qwen4_exp_mono/config.py:config_refusal` checks it once; any other deployment runs Native for the
whole model (target, drafter and the PLE table placement), decided before any state is written.

| Item | Served configuration |
| --- | --- |
| Model | Full 48-layer Qwen3.8-Flash-Next Quark MXFP4 checkpoint (geometry read from its `config.json`) |
| GPU / parallelism | MI355X / gfx950, TP1, no EP / DP / PP / CP / TBO, not a plugin |
| Speculative decode | MTP, 3 draft tokens; classic verify (`ATOM_ENABLE_REPLAYSSM=0`) |
| Caches | `--kv_cache_dtype fp8 --index-cache-dtype bf16 --block-size 64`; FP32 GDN state; BF16 draft caches |

| Step | Route |
| --- | --- |
| Target verify, 1-4 requests x 4 rows | Mono: 36 GDN + 12 QSA layer programs; the PLE layer runs Native PLE first |
| Draft first step (4 rows a request) / later steps (1 row), 1-4 requests | Mono BF16 draft program |
| Draft head | 1 row: W4 global-certificate screen; 2 / 4 rows: batched W4 screen; 3 rows and others: Native argmax (all return the BF16 head's argmax) |
| More than 4 requests, prefill, mixed steps | Native target; fallback draft (fused QSA attention, Native BF16 MLP) |

Mono keeps its own padded, interleaved A8W4 copy of every target layer's experts (~1.5 GB a layer) next to
ATOM's A4W4 tensors, and the BF16 PLE n-gram table in page-locked host memory.

## Enable

`ATOM_MONO_ENABLE` (default on) is the only switch; the engine switches below are required.

```bash
export ATOM_MONO_ENABLE=1 ATOM_ENABLE_REPLAYSSM=0 ATOM_GDN_SSM_DTYPE=fp32 AITER_LOG_LEVEL=WARNING
python -m atom.entrypoints.openai_server \
  --model /models/Qwen3.8-Flash-Next-Quark-MXFP4 --trust-remote-code -tp 1 \
  --gpu-memory-utilization 0.95 --max-num-seqs 160 \
  --max-num-batched-tokens 16384 --max-model-len 262144 \
  --block-size 64 --kv_cache_dtype fp8 --index-cache-dtype bf16 \
  --enable-prefix-caching --method mtp --num-speculative-tokens 3 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","*mlp.gate","*shared_expert_gate","*mlp.experts*","*mlp.shared_expert*","*hyper_connection*","*indexer*","*.ple.*","model.visual.*","mtp.*"]}'
```

The server log names the outcome: `Qwen mono off: <reason>` for a refused deployment; otherwise
`Qwen mono draft: layer 48 (bf16) hooked ... served {...}` and `Qwen mono: 48 layers ... served {...}`, where a
shape whose build failed is listed as not served (and warned about) and runs Native. A Native control run sets
`ATOM_MONO_ENABLE=0`.

## Validate

- Unit tests: `tests/test_qwen_mono_plan.py` (CPU, runs in CI: deployment refusals, step plan, metadata puts,
  pool and expert layout checks), `tests/test_qwen_mono_ple.py` and `tests/test_qwen_mono_program_prep.py` (GPU:
  PLE table lookup, verify prep and draft metadata), `tests/test_qwen4_exp_mtp.py`, `tests/test_draft_graph.py`.
- Kernels: compile the 16 programs (target `delta` / `qsa` and draft `mtp` S4 / S1, batches 1-4) on MI355X SPX and
  compare normalized ISA, resources and stage lists with the accepted build.
- Service: same-card Native -> Mono -> Native with fixed prompts; the GPU trace must contain the `flash_delta` and
  `flash_qsa` program symbols (the draft program is a `flash_qsa` build). Accuracy: the full 1319-question GSM8K
  under real MTP (no forced acceptance). Compare scores, not text: under real MTP the greedy text of a build
  differs run to run, and under a forced acceptance length the output does not depend on the prompt.
