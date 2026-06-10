"""Runner for RLA sweeps (full sweep and pilot share this runner).

Usage:
  python run_rla_sweep.py --config rla_sweep   # full 400-run sweep
  python run_rla_sweep.py --config rla_pilot   # 6-run G2 pilot
  python run_rla_sweep.py --config rla_sweep --shard 0/4   # one shard

Per-run JSONL fields:
  run_id, idx, ok, returncode, elapsed,
  max_acc, grok_ep, epochs_run,
  state_floats (read from STATE_FLOATS_JSON — actual instantiated state, NOT formula),
  n_params (total trainable, parsed from Zoology stdout if present),
  slice_accs (per-difficulty kv accuracy from slice_keys=["num_kv_pairs"]),
  valid_acc_curve, env, stderr_tail.
"""
import argparse, sys, os, json, time, pickle, subprocess, re, signal
from pathlib import Path
import importlib.util

ZOO = Path(__file__).resolve().parent
PYTHON = sys.executable


def load_configs(name):
    spec = importlib.util.spec_from_file_location(
        name, ZOO / f"zoology/experiments/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.configs, mod.configs_envs


def load_configs_and_envs():
    """Shard-runner entry point. Uses RLA_CONFIG env var (default rla_sweep)."""
    return load_configs(os.environ.get('RLA_CONFIG', 'rla_sweep'))


def parse_stdout(stdout):
    # Full per-epoch curve. Each COMPLETED validation epoch prints one line
    # ("Valid Epoch E/T: 100%|...") that contains the overall accuracy AND every
    # per-kv-slice accuracy together — i.e. all from the SAME checkpoint. We
    # capture the whole [epoch x slice] matrix so any aggregation (final-epoch,
    # best-checkpoint-by-overall, max-over-epochs) is derivable downstream and
    # per-slice numbers stay coherent (same checkpoint), rather than taking an
    # independent per-slice max from possibly-different epochs.
    epoch_curve = []  # list of {epoch, overall, slices:{kv:acc}}, one per completed epoch
    for m in re.finditer(r'Valid Epoch (\d+)/\d+: 100%\|([^\n\r]*)', stdout):
        epoch = int(m.group(1))
        blob = m.group(2)
        mo = re.search(r'valid/accuracy=([\d.]+)', blob)
        if not mo:
            continue
        overall = float(mo.group(1))
        if not (0 <= overall <= 1):
            continue
        slices = {}
        for ms in re.finditer(r'valid/num_kv_pairs/accuracy-(\d+)[^\d.]+([\d.]+)', blob):
            kv, acc = int(ms.group(1)), float(ms.group(2))
            if 0 <= acc <= 1:
                slices[kv] = acc
        # Keep the last reading for each epoch index (Lightning reprints the
        # final 100% line; dedupe by overwriting).
        epoch_curve.append({"epoch": epoch, "overall": overall, "slices": slices})
    # Dedupe by epoch (keep last occurrence — the completed line).
    _by_ep = {}
    for e in epoch_curve:
        _by_ep[e["epoch"]] = e
    epoch_curve = [_by_ep[k] for k in sorted(_by_ep)]

    pairs = [(e["epoch"], e["overall"]) for e in epoch_curve]
    max_acc = max((a for _, a in pairs), default=0.0)
    grok_ep = next((e for e, a in pairs if a >= 0.99), None)
    epochs_run = max((e for e, _ in pairs), default=0)
    valid_acc_curve = pairs

    # Back-compat: independent per-slice max over epochs (the old field).
    slice_accs = {}
    for e in epoch_curve:
        for kv, acc in e["slices"].items():
            slice_accs[kv] = max(slice_accs.get(kv, 0), acc)
    # Coherent single-checkpoint slice breakdowns (the rigorous views):
    #   final_epoch      : last completed epoch's slices
    #   best_checkpoint  : slices from the epoch with the highest overall acc
    slice_accs_final = epoch_curve[-1]["slices"] if epoch_curve else {}
    best_ep = max(epoch_curve, key=lambda e: e["overall"], default=None) if epoch_curve else None
    slice_accs_best = best_ep["slices"] if best_ep else {}
    best_overall = best_ep["overall"] if best_ep else 0.0
    final_overall = epoch_curve[-1]["overall"] if epoch_curve else 0.0
    best_ep_idx = best_ep["epoch"] if best_ep else None

    # CLA STATE_FLOATS_JSON — one per kernel layer. Capture all & sum n_heads ×
    # num_chunks × per_entry from each. For state-axis we use the per-layer
    # state (all kernel layers are identical), so just take the first.
    state_records = []
    for m in re.finditer(r'STATE_FLOATS_JSON (\{.*?\})', stdout):
        try:
            state_records.append(json.loads(m.group(1)))
        except Exception:
            pass
    state_floats = state_records[0]['state_floats'] if state_records else None
    state_per_layer = [r['state_floats'] for r in state_records] if state_records else []

    # MODEL_STATS_JSON line emitted by zoology/logger.py.
    n_params = None
    model_state_total = None
    m = re.search(r'MODEL_STATS_JSON (\{.*?\})', stdout)
    if m:
        try:
            ms = json.loads(m.group(1))
            n_params = ms.get('num_parameters')
            model_state_total = ms.get('state_size')
        except Exception:
            pass

    # FLOPS_JSON line (also from logger.py) — forward FLOPs at training batch
    # × seq_len, with disk-backed caching per unique cell config.
    # The flops_by_op dict is a nested dict so the older non-greedy regex won't
    # match cleanly; we use a more permissive single-line greedy match.
    fwd_flops = None
    flops_batch = None
    flops_seq = None
    flops_cache_hit = None
    flops_by_op = None
    flops_error = None
    m = re.search(r'^FLOPS_JSON (\{.*\})\s*$', stdout, re.MULTILINE)
    if m:
        try:
            fs = json.loads(m.group(1))
            fwd_flops = fs.get('forward_flops')
            flops_batch = fs.get('batch')
            flops_seq = fs.get('seq_len')
            flops_cache_hit = fs.get('cache_hit')
            flops_by_op = fs.get('flops_by_op')
            flops_error = fs.get('error')
        except Exception as e:
            flops_error = f"runner json parse failed: {e}; raw: {m.group(1)[:300]}"
    else:
        flops_error = "FLOPS_JSON not found in stdout"

    # PEAKINESS_JSON: end-of-training router-weight stds + softmax entropy on
    # a real test batch. Captures whether CLA chunks are getting discriminative
    # routing or collapsing to uniform/single-chunk.
    peakiness = None
    m = re.search(r'PEAKINESS_JSON (\{.*?\})\s*$', stdout, re.MULTILINE)
    if m:
        try:
            peakiness = json.loads(m.group(1))
        except Exception:
            pass

    # RANK_JSON: realized effective-attention rank (num_rank, eff_rank) emitted per
    # eval forward per kernel layer (CLA_MEASURE_RANK=1). Group by epoch, mean over
    # eval batches + layers → rank_curve; rank_final = the last (best-trained) epoch.
    rank_recs = []
    for mm in re.finditer(r'RANK_JSON (\{.*?\})\s*$', stdout, re.MULTILINE):
        try:
            rank_recs.append(json.loads(mm.group(1)))
        except Exception:
            pass
    rank_curve, rank_final = [], None
    if rank_recs:
        # The diagnostic now emits multi-tolerance numerical rank (rank_1e-01..rank_1e-04)
        # + eff_rank + sv_ratio_128/256, measured PER (epoch, seq-len) — each MQAR slice has
        # its own L (kv=1024 → longest). Group by (epoch, seqlen) so we don't average across
        # slices; carry every rank_* / sv_ratio_* / eff_rank key present (tolerant of missing).
        RANK_KEYS = sorted({k for r in rank_recs for k in r
                            if k.startswith('rank_') or k.startswith('mrank_') or k.startswith('sv_ratio_')
                            or k in ('eff_rank', 'meff_rank', 'stable_rank', 'mstable_rank')})
        by = {}
        for r in rank_recs:
            by.setdefault((r.get('epoch', -1), r.get('seqlen', r.get('seq_len', -1))), []).append(r)
        def mean(recs, key):
            vals = [x[key] for x in recs if key in x]
            return round(sum(vals) / len(vals), 3) if vals else None
        for (ep, sl) in sorted(by):
            recs = by[(ep, sl)]
            row = {'epoch': ep, 'seqlen': sl, 'n': len(recs)}
            row.update({k: mean(recs, k) for k in RANK_KEYS})
            rank_curve.append(row)
        # rank_final: the hardest slice (longest seq-len) at the last epoch — that's where
        # the task demands the most rank and where rank should exceed d_model under routing.
        last_ep = max(r['epoch'] for r in rank_curve)
        cand = [r for r in rank_curve if r['epoch'] == last_ep]
        rank_final = max(cand, key=lambda r: r.get('seqlen') or -1)
        rank_final = {**rank_final, 'nc_dqk': rank_recs[0].get('nc_dqk'),
                      'd_model': rank_recs[0].get('d_model'), 'nc': rank_recs[0].get('nc')}

    return dict(
        max_acc=max_acc, grok_ep=grok_ep, epochs_run=epochs_run,
        valid_acc_curve=valid_acc_curve, slice_accs=slice_accs,
        # Rigorous + flexible reporting: full per-epoch x per-slice matrix plus
        # the coherent single-checkpoint views derived from it.
        epoch_curve=epoch_curve,
        slice_accs_final=slice_accs_final,
        slice_accs_best=slice_accs_best,
        final_overall=final_overall,
        best_overall=best_overall,
        best_ep_idx=best_ep_idx,
        state_floats=state_floats, state_per_layer=state_per_layer,
        model_state_total=model_state_total,
        n_params=n_params,
        fwd_flops=fwd_flops,
        flops_by_op=flops_by_op,
        flops_batch=flops_batch,
        flops_seq=flops_seq,
        flops_cache_hit=flops_cache_hit,
        flops_error=flops_error,
        peakiness=peakiness,
        rank_curve=rank_curve,
        rank_final=rank_final,
    )


def run_one(idx, run_id, cfg, env_overrides, results_file=None):
    tmp_path = Path(f"/tmp/_rla_single_{idx}.py")
    tmp_path.write_text(
        "import pickle\n"
        f"with open('/tmp/_rla_cfg_{idx}.pkl', 'rb') as f:\n"
        "    configs = [pickle.load(f)]\n"
    )
    with open(f"/tmp/_rla_cfg_{idx}.pkl", "wb") as f:
        pickle.dump(cfg, f)

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    # Disable wandb's stdout interception so our FLOPS_JSON / MODEL_STATS_JSON
    # / STATE_FLOATS_JSON lines reliably reach the subprocess's captured stdout.
    # Without this, wandb's console-capture eats some of our prints (notably
    # FLOPS_JSON), leaving the runner unable to parse FLOPs.
    env["WANDB_CONSOLE"] = "off"
    # Clear any stale MQAR_* env from previous runs (recipe/router knobs).
    for k in list(env):
        if k.startswith("MQAR_ROUTER_STD") or k.startswith("MQAR_CURR"):
            del env[k]
    env.update(env_overrides)
    env["CLA_PRINT_STATE_JSON"] = "1"

    t0 = time.time()
    proc = None
    # Forward SIGTERM (Vertex spot preemption hits PID 1 = shard_runner, which
    # imports + calls this synchronously, so the handler installed here runs
    # in that same process). Without forwarding, the training subprocess never
    # receives SIGTERM and its checkpoint handler never fires.
    def _forward_sigterm(signum, frame):
        if proc is not None and proc.poll() is None:
            print(f"\n[run_one] forwarding SIGTERM → subprocess pid={proc.pid}", flush=True)
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception as e:
                print(f"[run_one] SIGTERM forward failed: {e}", flush=True)
    old_handler = signal.signal(signal.SIGTERM, _forward_sigterm)
    try:
        # Popen + tee: stream subprocess stdout to parent's stdout in real time
        # (so Cloud Logging / local terminal see Lightning progress, epoch
        # metrics, etc. as they happen) AND buffer for post-hoc parsing of
        # FLOPS_JSON, STATE_FLOATS_JSON, MODEL_STATS_JSON, valid/accuracy lines.
        proc = subprocess.Popen(
            [PYTHON, "-m", "zoology.launch", str(tmp_path)],
            cwd=str(ZOO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,  # line-buffered
        )
        buffer = []
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)  # tee → parent stdout
                buffer.append(line)
                # Cheap safety net for the 7200s timeout: check elapsed every line
                if time.time() - t0 > 7200:
                    proc.kill()
                    raise subprocess.TimeoutExpired(proc.args, 7200)
        finally:
            proc.wait()
        full_stdout = "".join(buffer)
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        parsed = parse_stdout(full_stdout)
        return {
            "run_id": run_id, "idx": idx, "ok": ok, "elapsed": elapsed,
            "returncode": proc.returncode, "env": env_overrides,
            **parsed,
            "stderr_tail": full_stdout[-2000:] if not ok else "",
        }
    except subprocess.TimeoutExpired:
        return {"run_id": run_id, "idx": idx, "ok": False, "elapsed": 7200, "error": "timeout"}
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        try:
            Path(f"/tmp/_rla_cfg_{idx}.pkl").unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rla_sweep",
                    help="experiment module under zoology/experiments/ (rla_sweep | rla_pilot)")
    ap.add_argument("--results", default=None,
                    help="output jsonl path (default: <config>_results.jsonl in cwd)")
    ap.add_argument("--shard", default=None,
                    help="N/M to process only idx where idx %% M == N")
    args = ap.parse_args()

    configs, envs = load_configs(args.config)
    results_file = Path(args.results) if args.results else ZOO / f"{args.config}_results.jsonl"
    print(f"Loaded {len(configs)} {args.config} configs", flush=True)
    print(f"Results → {results_file}", flush=True)

    shard_n, shard_m = None, None
    if args.shard:
        n, m = args.shard.split("/")
        shard_n, shard_m = int(n), int(m)
        print(f"Shard {shard_n}/{shard_m} — processing idx where idx %% {shard_m} == {shard_n}", flush=True)

    completed = set()
    if results_file.exists():
        for line in open(results_file):
            try:
                r = json.loads(line)
                if r.get("ok"):
                    completed.add(r["idx"])
            except Exception:
                pass
    if completed:
        print(f"Resuming: skipping {len(completed)} done", flush=True)

    with open(results_file, "a") as fout:
        for i, (cfg, env_o) in enumerate(zip(configs, envs)):
            if shard_m is not None and i % shard_m != shard_n:
                continue
            if i in completed:
                continue
            run_id = cfg.run_id
            print(f"\n[{i+1}/{len(configs)}] {run_id} | env: {env_o}", flush=True)
            result = run_one(i, run_id, cfg, env_o, results_file)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            status = "OK" if result.get("ok") else f"FAIL ({result.get('error', result.get('returncode'))})"
            elapsed = result.get('elapsed', 0)
            epochs_run = result.get('epochs_run', 0) or 0
            n_ep = epochs_run + 1
            s_per_ep = elapsed / max(n_ep, 1)
            fwd = result.get('fwd_flops')
            fwd_str = (f"{fwd/1e9:.1f}G" if isinstance(fwd, (int, float)) and fwd else "n/a")
            cache_str = "cached" if result.get('flops_cache_hit') else "fresh"
            print(f"  → {status}, max_acc={result.get('max_acc', 0):.3f}, "
                  f"grok_ep={result.get('grok_ep')}, "
                  f"state={result.get('state_floats')}, "
                  f"params={result.get('n_params')}, "
                  f"fwd_flops={fwd_str} ({cache_str}), "
                  f"t={elapsed:.0f}s ({s_per_ep:.0f}s/ep × {n_ep}ep)", flush=True)


if __name__ == "__main__":
    main()
