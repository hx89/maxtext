# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo identity

This is the NVIDIA fork of [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext).

- `origin` → `github.com/nvjax-svc-0/maxtext` — main branch is `te/main` (NOT `main`). Open PRs against `te/main`.
- The fork carries NVIDIA-specific work on top of upstream: TransformerEngine (TE) fused MoE router/permutation, TE Grouped GEMM, and DeepEP HybridEP dispatch/combine. These are the practical reasons most changes land here instead of upstream.

## Package layout quirk

Source lives under `src/MaxText/` but the installed package is lowercase `maxtext`. **Always import as `from maxtext.X import ...`** — the codebase does this consistently. The mixed-case directory is a legacy-compat shim handled by `build_hooks.py`:

- On case-insensitive filesystems (macOS default, Windows), only `maxtext` is exposed.
- On case-sensitive filesystems (Linux CI/containers), both `MaxText/` and `maxtext/` import paths work.

If you grep and find `from MaxText...`, that's almost certainly a bug — fix to lowercase.

## Common commands

Install in editable mode (required after pulling, since the package is source-built):
```
pip install -e .
```

Run pre-training (the canonical entry point — note module path, not script path):
```
python3 -m maxtext.trainers.pre_train.train src/MaxText/configs/base.yml run_name=<name> [key=value ...]
```

Other trainers follow the same pattern:
```
python3 -m maxtext.trainers.post_train.sft.train_sft ...
python3 -m maxtext.trainers.post_train.rl.train_rl ...
python3 -m maxtext.trainers.post_train.distillation.train_distill ...
python3 -m maxtext.inference.decode ...
```

Tests (pytest, configured via `pytest.ini`):
```
pytest tests/unit/<file>_test.py             # one file
pytest tests/unit/<file>_test.py::TestClass::test_name  # one test
pytest -m cpu_only                            # run only CPU-marked tests
DECOUPLE_GCLOUD=TRUE pytest tests/unit        # offline mode (no GCS/Vertex deps)
```

Hardware markers auto-skip when hardware is missing (see `tests/conftest.py`): `tpu_only`, `gpu_only`, `cpu_only`, `tpu_backend`. `external_serving` / `external_training` are deselected entirely under `DECOUPLE_GCLOUD=TRUE`. Several flaky/heavy unit tests are unconditionally `--ignore`d in `pytest.ini`; check there before assuming a missing test was deleted.

Lint / format (mirrors CI in `.github/workflows/CodeQuality.yml`):
```
pre-commit run --all-files                    # codespell + pylint + pyink + mdformat
pre-commit run --from-ref <base> --to-ref HEAD  # what CI actually runs
```
`pyink` uses `--pyink-indentation=2 --line-length=122` — match this when writing code by hand.

## Config system

Configs are YAML + Pydantic, not argparse:

- `src/MaxText/configs/base.yml` — full default config for pre-training (also reused by `decode`, `inference_microbenchmark`, `train_compile`).
- `src/MaxText/configs/models/<name>.yml` — per-model overrides selected by `model_name=<name>` on the CLI.
- `src/MaxText/configs/types.py` — `MaxTextConfig` Pydantic schema. **This is the source of truth for valid keys, types, defaults, and cross-field validation.** When adding a config flag, add it here AND in `base.yml`.
- `src/MaxText/configs/pyconfig.py` — loads YAML via omegaconf, applies CLI overrides (`key=value`), env overrides (prefix `M_`, e.g. `M_run_name`), and validates against `MaxTextConfig`. Each entry-point module is mapped to its default YAML in `_CONFIG_FILE_MAPPING` near the top of this file.
- `src/MaxText/configs/decoupled_base_test.yml` — auto-substituted for `base.yml` in tests when `DECOUPLE_GCLOUD=TRUE` (see `tests/utils/test_helpers.py`).

## Architecture (big picture)

JAX/Flax with an opinionated training stack: Flax (model), Optax (optimizer), Orbax (checkpointing), Grain/HF datasets (data), Tunix (post-training).

- `src/MaxText/trainers/` — entry points. `pre_train/train.py` holds the canonical training loop and is the largest single file to understand.
- `src/MaxText/models/` — per-architecture model assembly (Llama, Gemma, DeepSeek, Mixtral, Qwen, GPT-OSS, Kimi, ...). These are thin assemblies built on `layers/`.
- `src/MaxText/layers/` — reusable building blocks: attention variants (`attentions.py`, `attention_mla.py`, `attention_op.py`), MoE (`moe.py` is the central file), MTP, pipeline parallelism, normalizations, embeddings, quantization. NNX variants live alongside Linen versions (`nnx_decoders.py`, `train_state_nnx.py`).
- `src/MaxText/kernels/` — custom Pallas / Mosaic / Megablox kernels (TPU and GPU).
- `src/MaxText/inference/` — decode loops, JetStream/Pathways serving glue, KV cache, microbenchmarks.
- `src/MaxText/integration/` — Tunix and vLLM bridges (post-training + RL sampling).
- `src/MaxText/common/` — checkpointing, profiling, metric logging, GCloud stub (`gcloud_stub.py` gates everything that touches GCP so decoupled mode works).
- `src/MaxText/utils/` — sharding, gradient accumulation, vocabulary tiling, GCS helpers, model creation.
- `src/dependencies/` — Dockerfiles, requirements (generated per-platform: `tpu`, `tpu-post-train`, `cuda12`, `runner`, `docs`), and shell helpers (`preflight.sh`, `rto_setup.sh`, `setup.sh`). These ship inside the wheel.
- `tests/` — `unit/` (fast, often CPU), `integration/` (heavier, often hardware-marked), `end_to_end/` (shell scripts per model under `tpu/` and `gpu/`), `post_training/`, `inference/`.
- `benchmarks/` — perf harness, separate from `tests/`.

## NVIDIA-fork-specific concepts

These flags only exist on this fork — searching upstream won't find them. All live in `MaxTextConfig` (`src/MaxText/configs/types.py`) and are consumed in `src/MaxText/layers/moe.py`:

- `te_router_and_permutation_impl` — fused TE router + permutation. Requires `sparse_matmul=True`. Implementations in `layers/te_router.py`, `layers/te_permutation.py`.
- `te_use_gmm` + `te_gmm_quantization` — TE Grouped GEMM for MoE matmuls. Requires `sparse_matmul=True` and an explicit quantization mode (`TEGroupedGemmQuantizationType`, e.g. `te_mxfp8`). Empty quantization is rejected at validation time.
- `use_hybrid_ep` + `hybrid_ep_pad_multiple` — DeepEP HybridEP for MoE dispatch/combine over the NVLink domain (GPU only). `hybrid_ep_pad_multiple` defaults to 128 because `te_mxfp8` requires that alignment; do not lower it without checking the quant mode. Imports `jax_deep_ep` lazily inside `moe.py`.
- `moe_permutation_group_align_size` — shared MT/TE permutation padding alignment.

When debugging MoE perf or correctness on this fork, start at `src/MaxText/layers/moe.py` (the dispatch/combine code paths branch on the flags above) rather than the model files.

## Things to know before editing

- Don't add `Co-Authored-By` trailers to commits (per user-level instruction).
- The `pyconfig` system silently accepts unknown CLI keys only when `override_model_config: True` — if your `key=value` override is being ignored, that's why.
- `gcloud_stub.is_decoupled()` (controlled by `DECOUPLE_GCLOUD=TRUE`) is the gate for everything GCP-touching. New code that calls Vertex / GCS / Goodput must go through this stub or be guarded, otherwise decoupled CI breaks.
- `pre_train/train.py` imports `pathwaysutils` for side effects — don't strip it as "unused".
- Tests under `tests/unit/` that compare against PyTorch / external references are listed in `pytest.ini`'s `--ignore` block; they are run separately, not skipped at the marker level.
