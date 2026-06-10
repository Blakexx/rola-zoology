# Working in the zoology repo

This repo holds the **experiment runners, cloud infrastructure, and results** for
the CLA/RoLA project. It is conceptually one project with the CLA repo at
`/mnt/c/Users/Blake/Documents/VSCode/CLA/`, which holds the model implementation
(`cla_bench.py`), planning docs, and papers. **Read that repo's `CLAUDE.md` and
`resources/INDEX.md` first** for project state and design docs.

> **KEEP THIS DOC ALIVE.** Whenever you hit something non-obvious — a constraint,
> a failure mode, a config rule, a "wait, that's not how it works" — **append it to
> the Gotchas section immediately**, in the same turn you discover it. Assume your
> context will be compressed and a future you will have only this file. Document
> the catch, the symptom, and the fix. Err toward over-documenting operational
> traps; they are cheap to write and expensive to rediscover.

## The two-repo relationship (IMPORTANT)

- `cla_bench.py` is **authored** in the CLA repo. A copy lives here at the repo
  root (`/home/blake/zoology/cla_bench.py`) because the Docker image and runners
  `from cla_bench import ...`.
- **After editing `cla_bench.py` in the CLA repo you MUST copy it here**
  (`cp /mnt/c/Users/Blake/Documents/VSCode/CLA/cla_bench.py /home/blake/zoology/cla_bench.py`)
  and rebuild the image. The build copies whatever is in *this* repo.

## Python environment

Use the CLA repo's venv — there is no interpreter in this repo:
`/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python` (Python 3.10; PyTorch,
FLA, Triton, zoology deps, google-cloud-aiplatform all installed).

## Sensitive values

GCP project ID, bucket names, billing live ONLY in `cloud/.env` (gitignored).
Never commit or paste them into code, comments, PRs, or files outside that env.
Source it before any cloud command: `set -a; source cloud/.env; set +a`.

---

## How models are built (zoology/experiments/rla_sweep.py helpers)

All architectures share a backbone and differ only in the **sequence mixer**.
Build every model — baselines included — with these shared helpers so they are
comparable:

- `wrap_hybrid(kernel_kwargs)` → `Hybrid([base_conv_mixer, kernel])`. **Every**
  MQAR model gets a `BaseConv` (kernel_size=3 short conv) in front of the mixer;
  the MQAR input format depends on it. Do not skip it.
- `base_conv_mixer` — the canonical short-conv dict.
- `mha` — the canonical softmax-attention dict (num_heads=2, dropout=0.1). The MHA
  ceiling is `wrap_hybrid(mha)` with `max_position_embeddings=0`, identical to the
  linear baselines except the kernel. (MQAR is content-based recall → attention
  needs no position embeddings; abs pos-emb fails to extrapolate past trained
  lengths.) **Do not hand-roll a bare `sequence_mixer=mha()` — that is a strawman,
  not a ceiling.** See `memory/feedback_mha_baseline_must_use_wrap_hybrid.md`.
- `cla_bench.rola_instance(name, d_qk, d_v, num_chunks, n_heads=4, route_topk=, row_topk=)`
  → kwargs for a named RoLA instance (`rola-rla-asym`, `rola-rla-sym`, `rola-sse`),
  wrapped with `wrap_hybrid(...)`.

Key `cla_bench` concepts: `ChunkedLinearAttention` is the CLA/RoLA kernel. Axes:
`writer`/`reader` (softmax_linear | softmax_gla), `route_on` (`'x'` residual
stream — the paper scheme, scores higher; `'kq'` per-head — older sweeps),
`tie_routers` (symmetric routing), `route_topk`/`route_always_select` (sparse
partition routing), `phi` (`elu` | `softmax`), `normalized` (V+1), `row_topk`
(SSE Eq.7 row-sparse classification). Recurrent **state size** =
`nc * n_heads * d_qk * (d_v + 1)` (n_heads=4).

## Defining an experiment

Add `zoology/experiments/<name>.py` exposing `load_configs_and_envs()` →
`(configs, configs_envs)` where `configs` is a list of `TrainConfig`. Select it at
runtime via the `RLA_CONFIG` env var (the runner imports
`zoology/experiments/{RLA_CONFIG}.py`). Reuse `data_ext` /
`TRAIN_CONFIGS`/`TEST_CONFIGS` from `cla_router_width_v2.py` for the standard
extended-kv test set (kv 4…1024). Experiment files are baked into the image at
build time, **but you do NOT need a rebuild to change a config** — see the
`CONFIG_GCS_DIR` refresh below. Only `cla_bench.py` changes require a rebuild.

---

## Cloud workflow: build → submit → monitor → sync

### 1. Build the image — ONLY for `cla_bench.py` changes

```sh
set -a; source cloud/.env; set +a
gcloud builds submit --config=cloud/cloudbuild.yaml --project="$GCP_PROJECT_ID" .
```

Tags `…/cla-sweep/sweep:latest` (~7–10 min). **Submit only AFTER the build
finishes** — submitting against a stale `:latest` runs old code.

**Rebuild is needed ONLY when `cla_bench.py` changes** (it's baked at the image
root). For **experiment config changes** (`zoology/experiments/*.py`) do NOT
rebuild — instead upload the config to GCS and pass `CONFIG_GCS_DIR`:

```sh
# upload the changed config(s) — include any modules they import
gsutil cp zoology/experiments/<name>.py [imported_configs.py] "gs://$GCS_BUCKET/configs/"
# then submit with the refresh env var (seconds, no build):
$PY cloud/submit.py --runner run_rla_sweep.py \
  --gcs-results "gs://$GCS_BUCKET/<name>/results.jsonl" \
  --extra-env RLA_CONFIG=<name> \
  --extra-env CONFIG_GCS_DIR="gs://$GCS_BUCKET/configs" \
  --spot --regions us-central1 --display-name-prefix <tag>
```

`shard_runner` pulls every `.py` from `CONFIG_GCS_DIR` into `zoology/experiments/`
*before* importing (shard_runner.py:101). Upload **all** config files the target
imports (e.g. a smoke that imports its parent sweep needs both uploaded).

### 2. Submit

```sh
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
$PY cloud/submit.py \
  --runner run_rla_sweep.py \
  --gcs-results "gs://$GCS_BUCKET/<name>/results.jsonl" \
  --extra-env RLA_CONFIG=<name> \
  --spot \
  --regions us-central1[,europe-west4,asia-southeast1] \
  --num-shards N \
  --display-name-prefix <tag>
```

Required: `--runner`, `--gcs-results`. Gotchas that bit before:
- **`--spot` is mandatory in practice** — without it you request on-demand A100s
  (≈0 quota) and get `429 ResourceExhausted`. Spot quota: 8 per region in
  us-central1 / europe-west4 / asia-southeast1 (24 total).
- The experiment is chosen by `--extra-env RLA_CONFIG=<name>`, NOT a positional arg.
- A transient `429` right after jobs finish is Vertex quota lag — retry, or use a
  different `--regions`.

### 3. Sharding (parallelism)

`--num-shards M` splits the config list by `i % M == shard_n`; one GPU per shard,
all run concurrently. `--regions a,b,c` round-robins shards across regions (each
region's staging bucket must match — handled by `REGION_STAGING_BUCKETS`). The
runner **resume-skips** runs already present (`ok=True`) in the gcs-results file,
so re-submitting after preemption only reruns what's missing. **A slow sweep is
almost always running sequentially in one shard — reshard it.** (E.g. the SSE GLA
writer expands to `nc` virtual heads → very slow at high nc; parallelize.)

### 4. Monitor (cross-region)

```sh
$PY cloud/jobs.py status                          # state counts, all regions
$PY cloud/jobs.py list --name-contains <tag>      # list matching jobs
$PY cloud/jobs.py cancel --name-contains <tag> -y # bulk cancel (requires a filter)
```

For a failure cause, read the job's logs:
`gcloud logging read 'resource.type="ml_job" resource.labels.job_id="<id>" (textPayload:Traceback OR severity>=ERROR)' --project=$GCP_PROJECT_ID --limit=20 --order=asc`

### 5. Sync results

```sh
$PY sync_cloud_results.py
```

Parallel-fetches every `results.jsonl` in the bucket → writes
`cloud_results_cache.jsonl` (all records) + `cloud_results_summary.md` (per-state
table). **Read these local files instead of re-querying GCS.** Dedup by `run_id`
keeps `ok=True` over `ok=False`, then higher `max_acc` (so a fixed rerun
supersedes a broken record only if its run_id matches — a renamed config coexists).

---

## Results schema (per run, in results.jsonl)

The runner (`run_rla_sweep.py:parse_stdout`) captures the full per-epoch ×
per-slice matrix and derives: `slice_accs_best` (best-overall checkpoint —
**rigorous default**), `slice_accs_final` (last epoch), `slice_accs` (per-slice
max over epochs — envelope, back-compat), `best_overall`, `final_overall`,
`best_ep_idx`, `epochs_run`, `ok`, `peakiness`, `state_size`, `n_params`.
The paper-facing flat dataset is built separately in the rola_paper repo
(`results/build_results.py` → tidy-long `results.jsonl` + `manifest.json`).

## Gotchas / hard-won lessons (see also `memory/`)

- **Rebuild ONLY for `cla_bench.py` changes.** Experiment-config changes do NOT
  need a rebuild — upload to `gs://$BUCKET/configs/` and pass `CONFIG_GCS_DIR`
  (§1). Don't burn 8-min rebuilds on config edits.
- **Copy cla_bench.py from the CLA repo to here** before building.
- **Always `--spot`**; staging bucket must match region (auto via
  `REGION_STAGING_BUCKETS`).
- **Build before submit** — stale `:latest` runs old code / missing configs.
- **Reshard slow sweeps** rather than waiting on one sequential shard.
- **Build every baseline with `wrap_hybrid` + the canonical kernel dicts**; a
  config differing in backbone/conv/pos-emb is a strawman, not a ceiling.
- **Never run `backfill_flops.py` during a live sweep** — its atomic rewrite
  orphans the runner's append fd and corrupts state.
- `route_on='x'` (residual-stream routing) is the paper scheme and scores higher
  than the older `route_on='kq'`; filter on it when comparing.
- **MQAR data needs `vocab_size > input_seq_len`** (NOT `> num_kv`). The generator
  asserts it. seq ≈ 4×kv, so kv=2048→seq8192 needs vocab>8192, kv=4096→seq16384
  needs vocab>16384. Symptom: `AssertionError` at epoch 0, before training. And
  bump vocab **uniformly across train+eval** or eval tokens fall outside the
  trained embedding (untrained → fails for non-rank reasons).
- **CUDA grid limit on FLA chunk kernels: `B·n_heads·nc ≤ 65535`.** They grid over
  batch×virtual-heads; exceeding it crashes with `Triton Error [CUDA]: invalid
  argument` (e.g. nc≥128 at batch 128). Fixed in `cla_bench` via `_headchunked` —
  splits the virtual-head dim into grid-safe sub-launches (mathematically exact;
  heads are independent). This caps achievable nc at batch 128 to ~127 *without*
  the helper.
- **`_headchunked` fixes the grid, NOT peak memory.** The full virtual-head
  expansion `[B,L,H·nc,d]` is materialized before the (chunked) kernel call, so
  nc=512 @ batch 128 still OOMs (~29 GB: v_virt/g_virt/q_flat/k_flat ≈13 GB +
  kernel + autograd). nc≤256 @ batch 128 fits ~40 GB. To go higher at constant
  batch you must chunk the *expansion* too (loop state-groups), not just the call.
- **Per-kernel FLA head-dim limits (monoliths, big d_qk):** RLA ran to ≈770, GLA
  to ≈320, and **GDN hard-asserts head dim ≤ 256** (`fla/ops/common/chunk_delta_h.py`:
  `assert K <= 256`). So a GDN *wide* monolith is impossible above d_qk=256 (~12k
  state) — it can't reach 20k/40k as a wide monolith; use a square monolith
  (d_qk=d_v≤256) instead. Also **GDN routed at high nc OOMs** (the delta-rule
  `chunk_scaled_dot_kkt` intermediates are memory-heavy + fragmentation) — pass
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Vertex `JOB_STATE_SUCCEEDED` ≠ the run succeeded.** Check the `ok` field in
  results.jsonl. And the record's `stderr_tail` is usually truncated to wandb
  noise — pull the real traceback with `gcloud logging read … resource.labels.job_id=<id>`.
- **SSE is NOT a RoLA instance** — `rola-sse` was removed from `cla_bench`
  deliberately; do not reintroduce it. Cite the SSE paper (arXiv 2507.16577) for
  SSE; see `memory/project_sse_loses_at_matched_state`.
- **Eval is hardwired per-epoch and can dominate run time.** Cost knobs:
  (1) **`EVAL_EVERY_N`** env var (train.py) — eval every N epochs + always epoch 0
  and the last (default 1). (2) **Test batch is eval-only** (no accuracy effect) —
  size it as large as fits; it was the cause of a 6× slowdown once (batch 4 vs 64
  → 16× more eval iterations; ~121 → ~1970 valid steps/epoch). Eval memory is
  bounded by the longest-seq logits (`B·seq·vocab·4`) and, at high nc, the
  virtual-head expansion — so size test batch per-nc.
- **Default job timeout is 90 min (`DEFAULT_TIMEOUT_SECONDS=5400`) — too short for
  high-nc runs.** A nc=256 run is ~3 hr (training at batch 128 forces head-chunking
  → slower step); it gets `JOB_STATE_CANCELLED` at ~epoch 16 if you don't raise
  `--timeout-seconds` (use ~12600 for nc≤256). Symptom: CANCELLED with elapsed≈5400s.
- **`train.py` / package changes need a REBUILD** — `CONFIG_GCS_DIR` only refreshes
  `experiments/*.py`, NOT `train.py` or `cla_bench.py`. Those are baked.
- **Large vocab blows up CE/logits memory → OOM in `cross_entropy_loss`, not the
  mixer.** Logits are `[B, L, vocab]` fp32 (train.py has NO amp/bf16 — fp32 only).
  At vocab=32768, batch 128, seq 256 that's ~4.3 GB just for logits (+softmax/grad
  ~2-3×); the seq-16384 eval slice is ~8.6 GB even at batch 4. The kv=4096 →
  vocab>16384 requirement is what triggers it. Levers (no code change): pass
  `--extra-env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reclaims
  fragmentation — we saw 8 GB reserved-but-unallocated), or cap kv lower so vocab
  can shrink (kv=2048 needs only vocab>8192). Real fix would be bf16/chunked-CE in
  train.py (code + rebuild). Symptom: OOM traceback ends in `cross_entropy`.

- **`.gcloudignore` unanchored `_*.py` silently stripped every `__init__.py` from the build context**
  (gitignore semantics: no leading `/` = match any depth; `__init__.py` starts with `_`). Symptom:
  Cloud Build fails with `FileNotFoundError: .../fla_rola/__init__.py` in a setup.py, or packages
  import as namespace packages with missing module code. Fix (2026-06-09): anchored to `/_*.py` and
  `/_*.pkl`. When adding ignore patterns for "top-level temp files", ALWAYS anchor with a leading slash.
