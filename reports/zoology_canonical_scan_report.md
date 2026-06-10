# zoology_canonical_scan — Report

**Status**: **PARTIAL — stopped at 50 / 860 runs** for compute reasons (multi-day local job; user is considering external GPUs).
**Results file**: `/home/blake/zoology/zoology_canonical_scan_results.jsonl`
**Experiment script**: `/home/blake/zoology/zoology/experiments/zoology_canonical_scan.py`
**Runner**: `/home/blake/zoology/run_zoology_canonical_scan.py`

## What this sweep tested

The **actual Zoology paper canonical setup** with our CLA additions on top:
- Zoology's exact published task config (multi-task MQAR curriculum, multi-test eval at 7 difficulty levels).
- Zoology's exact baseline definitions (`add_attention`, `add_gla`, `add_gated_delta_net` from `zoology/experiments/models_repo.py`).
- Their `Hybrid([BaseConv, kernel])` wrapping, slice_keys eval, 4-LR sweep with max-over-LRs reporting convention.
- A **state-scan** axis added on top: each architecture evaluated at 7 state-size points from 2k to 524k floats, with CLA cells in recipe / no-recipe pairs.

The goal: produce numbers that are directly comparable to Zoology Figure 2, then show how CLA + recipe sits on the state-efficiency curve relative to those canonical baselines and a MHA ceiling.

## Task configuration

| Parameter | Value |
|---|---|
| vocab_size | 8,192 |
| MLP / state_mixer | `torch.nn.Identity` |
| max_position_embeddings | 0 |
| d_model | 128 |
| n_layers | 2 |
| block_type | TransformerBlock |
| **Hybrid `BaseConv` prefix** | `BaseConv(l_max=1024, kernel_size=3, implicit_long_conv=True)` applied to EVERY kernel (per Zoology canonical) |
| **Multi-task train** | 5 train configs: kv=4/seq=64 with 100k ex, kv=8/128/20k, kv=16/256/20k, kv=32/256/20k, kv=64/256/20k. ~180k examples combined. |
| **Multi-test eval** | 7 test configs: kv=4/seq=64, kv=8/64, kv=16/64, kv=32/128, kv=64/256, kv=128/512, kv=256/1024. 1,000 examples each. |
| slice_keys | `["num_kv_pairs"]` — per-difficulty accuracy reported |
| batch (train, test) | (256, 32) |
| max_epochs | 32 |
| **4-LR sweep** | `np.logspace(-3, -1.5, 4)` = {1.00e-3, 3.16e-3, 1.00e-2, 3.16e-2}, max-over-LRs reported per Zoology convention |
| weight_decay | 0 |
| early-stopping | valid/accuracy ≥ 0.99 |
| seeds | {1337, 42, 7, 0, 1} (5 per cell × 4 LRs = 20 runs per cell) |

## Cells

**43 cells × 4 LRs × 5 seeds = 860 total runs.**

### Baselines (match Zoology canonical model definitions: num_heads=2, use_gate=False for GDN, dropout=0.1 for MHA)

| Cell | Implementation | Kwargs | State |
|---|---|---|---:|
| `mha` | `zoology.mixers.attention.MHA` | `num_heads=2, dropout=0.1` | (quadratic — no recurrent state; serves as ceiling) |
| `baseline-gla-s2k` | `fla.layers.gla.GatedLinearAttention` | `num_heads=2, expand_k=0.5, expand_v=0.5, use_short_conv=False` | 2,048 |
| `baseline-gla-s8k` | same | `expand_k=1.0, expand_v=1.0` | 8,192 |
| `baseline-gla-s33k` | same | `expand_k=2.0, expand_v=2.0` | 32,768 |
| `baseline-gla-s64k` | same | `expand_k=2.0, expand_v=4.0` | 65,536 |
| `baseline-gla-s128k` | same | `expand_k=4.0, expand_v=4.0` | 131,072 |
| `baseline-gla-s256k` | same | `expand_k=4.0, expand_v=8.0` | 262,144 |
| `baseline-gla-s524k` | same | `expand_k=8.0, expand_v=8.0` | 524,288 |
| `baseline-gdn-s2k` | `fla.layers.gated_deltanet.GatedDeltaNet` | `num_heads=2, head_dim=32, expand_v=1, use_gate=False, use_short_conv=True, conv_size=4` | 2,048 |
| `baseline-gdn-s8k` | same | `head_dim=64, expand_v=1` | 8,192 |
| `baseline-gdn-s33k` | same | `head_dim=128, expand_v=1` | 32,768 |
| `baseline-gdn-s64k` | same | `head_dim=128, expand_v=2` | 65,536 |
| `baseline-gdn-s128k` | same | `head_dim=256, expand_v=1` | 131,072 |
| `baseline-gdn-s256k` | same | `head_dim=256, expand_v=2` | 262,144 (FLA default at num_heads=2) |
| `baseline-gdn-s524k` | same | `head_dim=256, expand_v=4` | 524,288 |

### CLA-GLA state-scan (`zoology.mixers.cla.ChunkedLinearAttention`, `writer=softmax_gla`, `reader=softmax_linear`, `n_heads=4`, `num_chunks=8`, `use_short_conv=True`, `route_on='kq'`, `tie_routers=False`) — each state point × {recipe, no recipe}

| Cell | d_qk × d_v | State (V+1) |
|---|---|---:|
| `cla-gla-s2k-*` | 7×8 | 2,016 |
| `cla-gla-s9k-*` | 16×16 | 8,704 |
| `cla-gla-s34k-*` | 32×32 | 33,792 |
| `cla-gla-s66k-*` | 32×64 | 66,560 |
| `cla-gla-s133k-*` | 64×64 | 133,120 |
| `cla-gla-s264k-*` | 64×128 | 264,192 |
| `cla-gla-s528k-*` | 128×128 | 528,384 |

### CLA-GDN state-scan (same module, `writer=softmax_gdn`) — each state point × {recipe, no recipe}

| Cell | d_qk × d_v | State |
|---|---|---:|
| `cla-gdn-s2k-*` | 8×8 | 2,048 |
| `cla-gdn-s8k-*` | 16×16 | 8,192 |
| `cla-gdn-s33k-*` | 32×32 | 32,768 |
| `cla-gdn-s65k-*` | 32×64 | 65,536 |
| `cla-gdn-s131k-*` | 64×64 | 131,072 |
| `cla-gdn-s262k-*` | 64×128 | 262,144 |
| `cla-gdn-s524k-*` | 128×128 | 524,288 |

### Recipe spec

Identical to canonical_state_scan: `MQAR_ROUTER_STD_WRITE=1.0`, `MQAR_ROUTER_STD_READ=0.05`, `MQAR_CURR_MODE=linear`, writer LR mult 3.0→0.3, reader LR mult 0.3→3.0 over 32 epochs.

## Results — partial (49 of 860 runs)

The runner completed only the first 3 cells worth of work (sweep is cell-major, then LR-major, then seed-major):

### MHA (5 seeds × 4 LRs = 20 runs, complete)

| LR | n | grok | mean | per-seed |
|---|---:|---:|---:|---|
| 1.00e-3 | 5 | **5/5** | **1.000** | 1.000 1.000 1.000 1.000 1.000 |
| 3.16e-3 | 5 | **5/5** | **1.000** | 1.000 1.000 1.000 1.000 1.000 |
| 1.00e-2 | 5 | **5/5** | **1.000** | 1.000 1.000 1.000 1.000 1.000 |
| 3.16e-2 | 5 | 0/5 | 0.034 | 0.052 0.005 0.013 0.067 0.032 |

**Max-over-LR: 5/5 grok, mean 1.000.** MHA is a clean ceiling at canonical task once `BaseConv` prefix supplies positional information. LR=3.16e-2 is too aggressive and collapses training.

### baseline-gla-s2k (5 seeds × 4 LRs = 20 runs, complete)

| LR | n | grok | mean | per-seed |
|---|---:|---:|---:|---|
| 1.00e-3 | 5 | 0/5 | 0.695 | 0.582 0.720 0.721 0.732 0.718 |
| 3.16e-3 | 5 | 0/5 | 0.682 | 0.617 0.734 0.733 0.712 0.612 |
| 1.00e-2 | 5 | 0/5 | **0.726** (best) | 0.735 0.730 0.724 0.718 0.723 |
| 3.16e-2 | 5 | 0/5 | 0.557 | 0.509 0.544 0.611 0.595 0.524 |

**Max-over-LR: 0/5 grok, mean 0.726** at LR=1.00e-2. FLA-GLA at small (2k) state is genuinely state-limited — no LR rescues it at this task.

### baseline-gla-s8k (partial — 2 of 4 LRs done, 9 runs)

| LR | n | grok | mean | per-seed |
|---|---:|---:|---:|---|
| 1.00e-3 | 5 | 0/5 | 0.820 | 0.821 0.843 0.833 0.795 0.809 |
| 3.16e-3 | 4 | 0/4 | 0.827 (in progress) | 0.829 0.831 0.818 0.829 |

**Max-over-LR (so far): mean 0.827 at LR=3.16e-3.**

## Key findings (preliminary)

1. **MHA solves canonical task** at 3 of 4 LRs with strict 5/5 grok. Confirms the BaseConv prefix supplies enough positional information for attention to nail MQAR. Top of the curve.
2. **baseline-gla at 2k state plateaus around 0.7** — even the best LR can't push it past 0.73 mean. State-limited at canonical kv=16 with seq up to 256.
3. **LR=3.16e-2 is consistently too high** in observed cells. Likely will be true for most cells; the 4-LR sweep's upper bound was speculative.

Nothing yet for any CLA cell or for baseline-GDN at canonical — those require resuming the sweep.

## What's needed to finish

810 runs remaining (cells: baseline-gla-s33k onwards, baseline-gdn × 7 states, CLA-GLA × 7 states × 2 recipes, CLA-GDN × 7 states × 2 recipes). Per-run timings observed:

- MHA (quadratic): ~1 min/run at LR=1e-3 (grok at epoch 0), ~13 min/run at LR=3.16e-2 (never converges, runs full 32 epochs)
- baseline-gla-s2k: ~13 min/run (no grok, runs full epochs)
- baseline-gla-s8k: ~13 min/run

**Estimated remaining wall time on the local RTX 3080 Ti**: ~7-15 days, with the d_qk=128 CLA cells likely OOM-ing without batch reduction. Resumption on larger-memory GPUs (A6000/A100) recommended.

## Known caveats

- **VRAM pressure**: even small-state baseline cells consume ~12 GB on the 3080 Ti due to the BaseConv `l_max=1024` long-conv allocations. The d_qk=128 CLA cells (s264k, s528k for GLA; s262k, s524k for GDN) are projected to OOM without batch reduction or gradient checkpointing.
- **Some cell d_qk values not present here that were in canonical_state_scan**: `cla-gla-s640` (d_qk=4) was dropped because Triton's `chunk_gla` kernel requires K ≥ 16.
- **All baseline kwargs now match Zoology canonical** (num_heads=2, use_gate=False for GDN, dropout=0.1 for MHA) — different from canonical_state_scan where baselines used non-canonical num_heads=4 / use_gate=True / dropout=0.

## Outputs

- `zoology_canonical_scan_results.jsonl` — one JSON per run; fields include `max_acc`, `grok_ep`, `epochs_run`, `slice_accs` (per-difficulty kv accuracy), `peakiness` (CLA cells — router weight stds and softmax entropy), `valid_acc_curve` (per-epoch trajectory), `env`, `returncode`, `stderr_tail`.

## Next steps

1. Resume on larger-memory GPU (or in parallel across multiple GPUs by sharding `idx % N`).
2. Pre-emptively reduce batch to 128 for d_qk=128 CLA cells to avoid OOM.
3. Once complete, the headline plot is **accuracy vs state size** with one curve per family (FLA-GLA, FLA-GDN, CLA-GLA+recipe, CLA-GDN+recipe), MHA as a flat horizontal ceiling, and recipe-no-recipe pairs as dashed lines.
