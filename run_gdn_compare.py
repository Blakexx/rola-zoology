"""GDN comparison runner: baseline GDN vs CLA-GDN with asym init + curriculum.

Per-config env vars based on run_id substring:
  - "cla-gdn-asym" → MQAR_ROUTER_STD_WRITE=1.0, MQAR_ROUTER_STD_READ=0.05,
                     MQAR_CURR_MODE=linear, writer 3.0→0.3, reader 0.3→3.0
  - "baseline-gdn" → no env (default training)
"""
import sys, os, json, time, pickle, subprocess
from pathlib import Path
import importlib.util

ZOO = Path("/home/blake/zoology")
PYTHON = "/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python"
RESULTS_FILE = ZOO / "gdn_results.jsonl"

def load_configs():
    spec = importlib.util.spec_from_file_location("gdn_compare",
        ZOO / "zoology/experiments/gdn_compare.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.configs

ASYM_CURR_ENV = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0",
    "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3",
    "MQAR_CURR_R_LR_PHASE2": "3.0",
}

def env_for_run(run_id):
    if "cla-gdn-asym" in run_id:
        return dict(ASYM_CURR_ENV)
    return {}

def run_one(idx, run_id, cfg):
    tmp_path = Path(f"/tmp/_gdn_single_{idx}.py")
    tmp_path.write_text(
        "import pickle\n"
        f"with open('/tmp/_gdn_cfg_{idx}.pkl', 'rb') as f:\n"
        "    configs = [pickle.load(f)]\n"
    )
    with open(f"/tmp/_gdn_cfg_{idx}.pkl", "wb") as f:
        pickle.dump(cfg, f)

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    # Clear and apply
    for k in list(env):
        if k.startswith("MQAR_ROUTER_STD") or k.startswith("MQAR_CURR"):
            del env[k]
    env.update(env_for_run(run_id))

    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "zoology.launch", str(tmp_path)],
            cwd=str(ZOO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=1800,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        import re
        pairs = re.findall(
            r'Valid Epoch (\d+)/\d+: 100%\|[^\n\r]*valid/accuracy=([\d.]+)', proc.stdout)
        pairs = [(int(e), float(a)) for e, a in pairs if 0 <= float(a) <= 1]
        max_acc = max((a for _, a in pairs), default=0.0)
        grok_ep = next((e for e, a in pairs if a >= 0.99), None)
        return {"run_id": run_id, "idx": idx, "ok": ok, "elapsed": elapsed,
                "max_acc": max_acc, "grok_ep": grok_ep, "returncode": proc.returncode,
                "stderr_tail": proc.stdout[-500:] if not ok else ""}
    except subprocess.TimeoutExpired:
        return {"run_id": run_id, "idx": idx, "ok": False, "elapsed": 1800, "error": "timeout"}


def main():
    configs = load_configs()
    print(f"Loaded {len(configs)} GDN configs", flush=True)
    completed = set()
    if RESULTS_FILE.exists():
        for line in open(RESULTS_FILE):
            try:
                r = json.loads(line)
                if r.get("ok"): completed.add(r["idx"])
            except: pass
    with open(RESULTS_FILE, "a") as fout:
        for i, cfg in enumerate(configs):
            if i in completed: continue
            run_id = cfg.run_id
            env = env_for_run(run_id)
            print(f"\n[{i+1}/{len(configs)}] {run_id} | env: {env}", flush=True)
            result = run_one(i, run_id, cfg)
            fout.write(json.dumps(result) + "\n"); fout.flush()
            status = "OK" if result.get("ok") else f"FAIL ({result.get('error', result.get('returncode'))})"
            print(f"  → {status}, max_acc={result.get('max_acc', 0):.3f}, "
                  f"grok_ep={result.get('grok_ep')}, t={result.get('elapsed', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
