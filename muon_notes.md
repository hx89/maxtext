# Muon optimizer in MaxText — notes

Reference for using Muon in this fork. Source files: `src/MaxText/optimizers/optimizers.py`, `src/MaxText/utils/muon_utils.py`, `src/MaxText/configs/base.yml`, `src/MaxText/configs/types.py`. Test fixtures with concrete dim numbers per model: `tests/unit/optimizers_test.py`.

## How to enable

Set `opt_type=muon` (default `adamw`):

```
python3 -m maxtext.trainers.pre_train.train src/MaxText/configs/base.yml \
  run_name=<name> model_name=<deepseek*|qwen3*|gemma3*|llama2*> \
  opt_type=muon
```

Knobs (all in `base.yml`, schema at `configs/types.py:1345` `class Muon`):

- `muon_beta` — momentum decay (default `0.95`).
- `muon_weight_decay` — multiplied with LR (default `0`).
- `muon_consistent_rms` — `None` ⇒ width scaling on updates; set to `0.2` for consistent-RMS scaling (recommended in the in-file comment).
- AdamW-side params reused for non-Muon-eligible weights (embeddings, norms, biases): `adam_b1`, `adam_b2`, `adam_eps`, `adam_eps_root`, `adam_weight_decay`, `mu_dtype`. Muon-style updates apply only to matmul weights; everything else falls through to AdamW under the hood (`optax.contrib._muon.muon`).

## Supported models

`types.py:2764` allowlist: `decoder_block ∈ {DEEPSEEK, QWEN3, GEMMA3, LLAMA2}`. Anything else hard-fails at config validation with:

```
python3 -m maxtext.utils.muon_utils <model_name> True
```

That command prints the auto-generated `MuonDimensionNumbers` so you can verify them, then add the new `decoder_block` to the allowlist if they look right. The rules are in `transform_logic` (`utils/muon_utils.py:49`).

## Trainer support

- Pre-train: yes (`trainers/pre_train/train.py`).
- SFT: yes (`trainers/post_train/sft/train_sft.py`).
- Distillation: **explicitly rejected** (`trainers/post_train/distillation/train_distill.py:91`).
- RL/DPO: inherit the same optimizer factory; not specifically tested.

The optimizer factory needs `model` passed in so `get_muon_weight_dimension_numbers` can walk the param tree — that's what the `# pass in model for muon` comments at the call sites are about.

## MuonClip is a separate switch

```
use_qk_clip=True  qk_clip_threshold=100.0
```

Despite the name, this is **not** auto-enabled by `opt_type=muon`. Validation (`types.py:2777-2787`):

- requires `attention_type=mla`
- incompatible with `attn_logits_soft_cap`

Implementation: `utils/qk_clip_utils.py`.

## Which params get Muon updates

Rules in `transform_logic`. Test fixtures in `tests/unit/optimizers_test.py` confirm what falls out for real models.

### Excluded — falls through to AdamW

A param is **not** Muon-updated if its path contains any of these substrings: `scale`, `bias`, `embedding`, `logits_dense`. Concretely:

- Token embedder (`token_embedder.embedding`)
- LM head (`decoder.logits_dense.kernel`) — explicit by-name exclusion even though it's a 2D matrix
- All RMSNorm / LayerNorm scales: `decoder_norm`, `pre/post_self_attention_layer_norm`, `pre/post_ffw_norm`, `kv_norm`, `q_norm`, `key_norm`, `query_norm`
- All biases: MLP biases, gate bias (DS3), attention biases
- Gemma 4 `per_expert_scale` (matches `scale`)

### Muon-updated

Three rules, applied in order:

**1. MoE expert matmuls** (path contains `MoeBlock_0` AND leaf is `wi_0/wi_1/wo`): `mdn((-2,), (-1,))`.
Shape `(num_experts, in, out)` — axis 0 is treated as a batch dim, NS runs independently per expert.

**2. Self-attention projections** (path contains `self_attention`):
- `out.kernel` → `mdn((0, -2), (-1,))`
- `query.kernel`, `key.kernel`, `value.kernel`, `wq_b.kernel`, `wkv_b.kernel` → `mdn((0,), (-2, -1))`
- MLA "down" projections `wq_a.kernel`, `wkv_a.kernel` fall through to rule 3

**3. Default 2D fallback**: `mdn((0,), (-1,))`. Catches:
- Dense MLP `wi_0/wi_1/wo`
- Shared experts (`<MoeBlock>.shared_experts.{wi_0, wi_1, wo}`) — NOT inside `MoeBlock_0`, so they get standard 2D, not per-expert
- MoE router `MoeBlock_0.gate.kernel` (the comment says "exclude gate" but the conditional only matches `wi_*/wo`; gate falls through and **is** Muon-updated)
- MLA down-projections `wq_a/wkv_a`

### Per-architecture summary

| Component | Llama / Qwen3 dense | Gemma 3 | DeepSeek 2/3, Kimi K2 |
| --- | --- | --- | --- |
| Token embedding | AdamW | AdamW | AdamW |
| LM head (`logits_dense`) | AdamW | AdamW | AdamW |
| Pre/post norms | AdamW | AdamW | AdamW |
| Q/K/V proj | Muon | Muon | Muon (incl. `wq_b/wkv_b`; `wq_a/wkv_a` use standard 2D rule) |
| Attn output proj | Muon | Muon | Muon |
| Q/K norms | — | AdamW | AdamW (DS3 only) |
| Dense MLP `wi_0/wi_1/wo` | Muon | Muon | Muon (in `dense_layers`) |
| MoE expert MLP `wi_0/wi_1/wo` | — | — | **Muon, per-expert** |
| MoE router (`gate`) | — | — | Muon |
| MoE shared experts | — | — | Muon (treated as dense MLP) |
| Biases | AdamW | AdamW | AdamW |

## Interaction with FSDP and EP

MaxText doesn't add Muon-aware sharding logic — it calls `optax.contrib._muon.muon` and lets GSPMD route the collectives based on existing param sharding. The whole story is (a) what `MuonDimensionNumbers` declare and (b) how those axes map to mesh axes via `logical_axis_rules`.

`MuonDimensionNumbers((row_axes,), (col_axes,))` reshapes each weight to a 2D matrix `M` for the Newton–Schulz iteration. **Any axis NOT listed as row or col is a batch axis** — NS runs independently per slice along it. That's the key.

### EP (`expert` mesh axis)

Default MoE param sharding (`layers/moe.py:417`):
```
wi_kernel_axes = ("exp", "embed_moe", "mlp_moe")
wo_kernel_axes = ("exp", "mlp_moe", "embed_moe")
```
and `['exp', 'expert']` in `base.yml:515` maps the `exp` logical axis to the `expert` mesh axis.

Because Muon's MoE rule treats axis 0 as a batch dim, **NS runs independently for each expert on its EP shard, with zero cross-shard communication for orthogonalization**. EP is essentially free for Muon — each rank does NS over its locally-held experts. Optimizer state (momentum) is sharded along `expert` exactly like the param.

This is why `transform_logic`'s MoE branch gates on the path string `"MoeBlock_0"` (`models.py:2717` keeps the literal name for back-compat). A custom MoE module with a different attribute name fails the path check, falls through to `((0,), (-1,))`, and NS would try to flatten across experts. Don't rename without updating `transform_logic`.

### FSDP (`fsdp` / `fsdp_transpose` mesh axes)

The axes Muon uses as **rows/cols** are the ones FSDP shards:

- `embed_moe` → `[fsdp, fsdp_transpose, sequence, tensor_transpose, context]` (`base.yml:517–520`)
- `mlp_moe` → `[fsdp_transpose, tensor, tensor_sequence, autoregressive]` (`base.yml:516`)
- Dense `embed`/`mlp` → similar.

Newton–Schulz needs `M Mᵀ M` and `(M Mᵀ)² M`, so when row OR col is FSDP-sharded, GSPMD inserts an all-gather/reduce-scatter inside the optimizer step. Practically:

- **FSDP-on-`embed` (1D FSDP)**: gather along one axis only. XLA usually schedules it cheaply alongside the existing forward/backward all-gathers. This is the tested path.
- **`use_2d_fsdp_sharding=True`** (`base.yml:253`): both `fsdp` and `fsdp_transpose` shard the MoE weights. NS now needs collectives along **two** axes. Allowed but more expensive.
- **`shard_exp_on_fsdp=True`** (`base.yml:251`): `exp` axis maps to `fsdp` (DSv3 layout, `wi_kernel_axes = ("embed_moe", None, "mlp_moe")`). Now FSDP shards what Muon treats as either a batch axis or a row — the optimizer's gather pattern changes, but the validator doesn't reject it. Worth profiling.
- `moe_fsdp_use_two_stage_all_gather` (`base.yml:248`) is a forward-pass knob; doesn't change the optimizer.

### Memory accounting

Muon stores **one momentum buffer per param** (no second moment for Muon-eligible params; AdamW fallback for embeddings/norms still has `mu` and `nu`). Sharding inheritance is automatic — momentum gets the same `PartitionSpec` as the param. Roughly half the optimizer-state memory of pure AdamW for matmul weights.

### Failure modes

- **Allowlist hard-fail**: only the four `decoder_block`s above are whitelisted. Other models raise at config validation. Run `python3 -m maxtext.utils.muon_utils <model_name> True`, eyeball the dim numbers, then add to the allowlist.
- **Custom MoE attribute name**: if your block isn't named `MoeBlock_0` somewhere on the path, Muon will treat the expert axis as row/col and flatten across experts — almost certainly wrong under EP. Same caveat for renaming `wi_0/wi_1/wo`.
- **Pipeline parallelism (`stage` axis) + scan**: `transform_logic` doesn't special-case the optional `L` (scan) layer dim that gets prepended when `scan_layers=True`. The standard rule `((0,), (-1,))` is technically wrong for a `(L, embed, mlp)` tensor under scan — in practice the test fixtures show this works because `L` gets treated as an extra batch axis by optax's reshape. If you hit weird shapes under PP+scan, run `muon_utils` with `scan_layers=True/False` and compare.
- **MTP / Engram / mHC / multimodal heads**: not in the test fixtures and not specifically handled. Params fall through to defaults — exclusion if leaf name contains `scale/bias/embedding`, otherwise standard 2D `((0,), (-1,))`. If you enable these, run `muon_utils` and verify the printed dim numbers.
