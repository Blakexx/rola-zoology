"""Runner for TRUE Zoology canonical state-scan sweep.

20 cells × 4 LRs × 5 seeds = 400 runs. Estimate: ~10-15 min per run at seq_len up
to 256 with multi-task training → ~80-100 hours total. Multi-day job.
"""
import sys, os, json, time, pickle, subprocess
from pathlib import Path
import importlib.util

ZOO = Path("/home/blake/zoology")
PYTHON = "/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python"
RESULTS_FILE = ZOO / "zoology_canonical_scan_results.jsonl"


def load_configs_and_envs():
    spec = importlib.util.spec_from_file_location("zcs",
        ZOO / "zoology/experiments/zoology_canonical_scan.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.configs, mod.configs_envs


def run_one(idx, run_id, cfg, env_overrides):
    tmp_path = Path(f"/tmp/_zcs_single_{idx}.py")
    tmp_path.write_text(
        "import pickle\n"
        f"with open('/tmp/_zcs_cfg_{idx}.pkl', 'rb') as f:\n"
        "    configs = [pickle.load(f)]\n"
    )
    with open(f"/tmp/_zcs_cfg_{idx}.pkl", "wb") as f:
        pickle.dump(cfg, f)

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    for k in list(env):
        if k.startswith("MQAR_ROUTER_STD") or k.startswith("MQAR_CURR"):
            del env[k]
    env.update(env_overrides)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "zoology.launch", str(tmp_path)],
            cwd=str(ZOO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=7200,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        import re
        pairs = re.findall(
            r'Valid Epoch (\d+)/\d+: 100%\|[^\n\r]*valid/accuracy=([\d.]+)', proc.stdout)
        pairs = [(int(e), float(a)) for e, a in pairs if 0 <= float(a) <= 1]
        max_acc = max((a for _, a in pairs), default=0.0)
        grok_ep = next((e for e, a in pairs if a >= 0.99), None)
        epochs_run = max((e for e, _ in pairs), default=0)
        # Full per-epoch valid_acc curve (for plotting / time-to-grok analysis).
        valid_acc_curve = [(e, a) for e, a in pairs]
        # Per-slice accuracy from Zoology's slice_keys=["num_kv_pairs"]
        slice_accs = {}
        for m in re.finditer(r'valid/accuracy/num_kv_pairs=(\d+)[^\d.]+([\d.]+)', proc.stdout):
            kv = int(m.group(1)); acc = float(m.group(2))
            if 0 <= acc <= 1:
                slice_accs[kv] = max(slice_accs.get(kv, 0), acc)
        # CLA router peakiness (printed by train.py _report_peakiness as PEAKINESS_JSON {...}).
        peak_match = re.search(r'PEAKINESS_JSON (\{.*\})', proc.stdout)
        peakiness = None
        if peak_match:
            try:
                peakiness = json.loads(peak_match.group(1))
            except Exception:
                peakiness = {"parse_error": peak_match.group(1)[:200]}
        return {"run_id": run_id, "idx": idx, "ok": ok, "elapsed": elapsed,
                "max_acc": max_acc, "grok_ep": grok_ep, "epochs_run": epochs_run,
                "returncode": proc.returncode, "env": env_overrides,
                "slice_accs": slice_accs, "peakiness": peakiness,
                "valid_acc_curve": valid_acc_curve,
                "stderr_tail": proc.stdout[-500:] if not ok else ""}
    except subprocess.TimeoutExpired:
        return {"run_id": run_id, "idx": idx, "ok": False, "elapsed": 7200, "error": "timeout"}


def main():
    configs, envs = load_configs_and_envs()
    print(f"Loaded {len(configs)} zoology-canonical-scan configs", flush=True)
    completed = set()
    if RESULTS_FILE.exists():
        for line in open(RESULTS_FILE):
            try:
                r = json.loads(line)
                if r.get("ok"): completed.add(r["idx"])
            except: pass
    if completed:
        print(f"Resuming: skipping {len(completed)} done", flush=True)
    with open(RESULTS_FILE, "a") as fout:
        for i, (cfg, env_o) in enumerate(zip(configs, envs)):
            if i in completed: continue
            run_id = cfg.run_id
            print(f"\n[{i+1}/{len(configs)}] {run_id} | env: {env_o}", flush=True)
            result = run_one(i, run_id, cfg, env_o)
            fout.write(json.dumps(result) + "\n"); fout.flush()
            status = "OK" if result.get("ok") else f"FAIL ({result.get('error', result.get('returncode'))})"
            print(f"  → {status}, max_acc={result.get('max_acc', 0):.3f}, "
                  f"grok_ep={result.get('grok_ep')}, t={result.get('elapsed', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
