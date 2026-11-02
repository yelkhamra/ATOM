# ATOM maintenance

The Qwen3.8-Flash-Next Mono decode lives in `atom/models/qwen4_exp_mono/` (entry `dispatch.py`, deployment
contract `config.py`), its layer recipes in `whole_layer/` and the shared program framework in `atom/mono/program/`.
See `recipes/Qwen3.8-Flash-Next-Mono-Program.md` for the deployment and its validation. Keep experiment patches,
measurements and independent numerical references outside the repository.
