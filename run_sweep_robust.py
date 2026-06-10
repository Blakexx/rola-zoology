"""Subprocess-based sweep driver: runs each Zoology config in its own Python process.

Why: Zoology's default `python -m zoology.launch sweep.py` runs all configs in one
process. After ~6 runs, CUDA state accumulates and WSL2 TDR fires, killing the whole
sweep. Wrapping each config in a subprocess isolates crashes.

Usage: python run_sweep_robust.py <sweep_config.py>
"""
import sys, os, json, time, subprocess, importlib.util
from pathlib import Path

ZOOLOGY_DIR = Path("/home/blake/zoology")
PYTHON = "/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python"

def load_configs(config_path: str):
    spec = importlib.util.spec_from_file_location("swept", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.configs

def run_one(config_idx: int, total: int, run_id: str, config):
    """Spawn a single-config launcher subprocess."""
    # Write a tiny "just this one config" module
    tmp_path = Path(f"/tmp/_zoology_single_{config_idx}.py")
    tmp_path.write_text(
        "import pickle, sys\n"
        f"with open('/tmp/_zoology_cfg_{config_idx}.pkl', 'rb') as f:\n"
        "    configs = [pickle.load(f)]\n"
    )
    import pickle
    with open(f"/tmp/_zoology_cfg_{config_idx}.pkl", "wb") as f:
        pickle.dump(config, f)

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    # Router init_std defaults (set per-config via run_id pattern below if present).
    env.setdefault("MQAR_ROUTER_STD_WRITE", "0.3")
    env.setdefault("MQAR_ROUTER_STD_READ", "0.3")
    # Per-config override: parse "w<X>_r<Y>" from run_id → set env vars.
    # e.g. "stdsweep_w0.3_r0.1_lr1e-3_s7" → MQAR_ROUTER_STD_WRITE=0.3, MQAR_ROUTER_STD_READ=0.1
    import re as _re
    m_std = _re.search(r"w(\d+\.?\d*)_r(\d+\.?\d*)", run_id)
    if m_std:
        env["MQAR_ROUTER_STD_WRITE"] = m_std.group(1)
        env["MQAR_ROUTER_STD_READ"] = m_std.group(2)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "zoology.launch", str(tmp_path)],
            cwd=str(ZOOLOGY_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # merge streams
            text=True, timeout=1800,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        # Extract final accuracy from merged output. tqdm writes progress to stderr,
        # which is now merged into proc.stdout.
        import re
        combined = proc.stdout
        # Parse epoch + accuracy pairs from "Valid Epoch N/M: 100%|...valid/accuracy=X"
        epoch_acc_pairs = re.findall(
            r'Valid Epoch (\d+)/\d+: 100%\|[^\n\r]*valid/accuracy=([\d.]+)', combined)
        pairs = [(int(e), float(a)) for e, a in epoch_acc_pairs if 0 <= float(a) <= 1]
        max_acc = max((a for _, a in pairs), default=0.0)
        grok_ep = next((e for e, a in pairs if a >= 0.99), None)
        return {
            "run_id": run_id, "idx": config_idx, "ok": ok,
            "elapsed": elapsed, "max_acc": max_acc, "grok_ep": grok_ep,
            "returncode": proc.returncode,
            "stderr_tail": combined[-500:] if not ok else "",
        }
    except subprocess.TimeoutExpired:
        return {"run_id": run_id, "idx": config_idx, "ok": False, "elapsed": 1800,
                "error": "timeout"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_sweep_robust.py <sweep_config.py>"); sys.exit(1)
    config_path = sys.argv[1]
    configs = load_configs(config_path)
    total = len(configs)
    print(f"Loaded {total} configs from {config_path}", flush=True)

    results_file = Path("/home/blake/zoology/sweep_results.jsonl")
    # Resume: skip already-completed indices
    completed = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok"): completed.add(r["idx"])
                except: pass
        print(f"Resuming: skipping {len(completed)} completed configs", flush=True)

    with open(results_file, "a") as fout:
        for i, cfg in enumerate(configs):
            if i in completed:
                continue
            run_id = getattr(cfg, "run_id", f"config_{i}")
            print(f"\n[{i+1}/{total}] {run_id} ...", flush=True)
            result = run_one(i, total, run_id, cfg)
            fout.write(json.dumps(result) + "\n"); fout.flush()
            status = "OK" if result.get("ok") else f"FAIL ({result.get('error', result.get('returncode'))})"
            print(f"  → {status}, max_acc={result.get('max_acc', 0):.3f}, "
                  f"grok_ep={result.get('grok_ep')}, t={result.get('elapsed', 0):.0f}s", flush=True)

if __name__ == "__main__":
    main()
