"""Curriculum LR sweep: test if per-router LR schedules recover the winner cell.

Same task as std_sweep_confirm. Compares conditions with DEFAULT router init
(std=0.02), per-router LR schedules, vs peaky init for reference.

Each subprocess gets specific MQAR_CURR_* / MQAR_ROUTER_STD_* env vars based on
the condition name.
"""
import os, sys, json, time, pickle, subprocess
from pathlib import Path

import numpy as np
sys.path.insert(0, "/home/blake/zoology")
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

# Same task spec as the std_sweep_confirm.
VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1, 2]  # 6 seeds per condition

# 2x2 ablation: similarity kind × init kind, both with writer-first curriculum LR.
#
#   sim_kind: softmax (dot-product) vs cosine_softmax (scale-invariant)
#   init_kind: uniform (Zoology default std=0.02) vs asym (peaky write w=1.0, flat read r=0.05)
#
# Hypothesis: cosine is std-agnostic, so cos_uniform ≈ cos_asym.
#             dot needs the asym init, so dot_asym > dot_uniform.
#             Curriculum LR may help both rows recover from uniform init.

# Gradual curriculum: writer LR mult linearly 3.0 → 0.3 across 24 epochs;
# reader LR mult linearly 0.3 → 3.0. They cross at epoch 12 at value 1.65.
# Both routers always receive gradient (no freezing), with shifting emphasis.
_CURR_WRITER_FIRST = {
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0",  # writer start
    "MQAR_CURR_W_LR_PHASE2": "0.3",  # writer end
    "MQAR_CURR_R_LR_PHASE1": "0.3",  # reader start
    "MQAR_CURR_R_LR_PHASE2": "3.0",  # reader end
}
_ASYM_INIT = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
}
_SYMPEAK_INIT = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "1.0",
}

# Cosine sim normalizes vectors → init std cancels out; only need 1 cosine row.
# Init is only an axis to ablate for the dot-product (softmax) sim.
#
# (condition_name, env_overrides, mixer_overrides)
# Dropped cos_curr after 2/6 runs both at 0.13 (failed clearly). Indices must
# stay stable for resume logic — slot 12-17 used to be cos_curr; we cleaned
# those out of curriculum_results.jsonl, so dot_sympeak_curr inherits idx 12-17.
CONDITIONS = [
    ("dot_uniform_curr",     {**_CURR_WRITER_FIRST},                  {}),
    ("dot_asym_curr",        {**_CURR_WRITER_FIRST, **_ASYM_INIT},    {}),
    ("dot_sympeak_curr",     {**_CURR_WRITER_FIRST, **_SYMPEAK_INIT}, {}),
    # Hard-uniform reader phase 1 + asym init + curriculum. Tests if removing
    # phase-1 reader noise improves over the gradual coupling.
    ("hard_uniform_asym",    {**_CURR_WRITER_FIRST, **_ASYM_INIT,
                              "MQAR_HARD_UNIFORM_READER_UNTIL": "12"},
                             {}),
]


def make_config(seed: int, run_id: str, mixer_overrides: dict) -> TrainConfig:
    data = DataConfig(
        train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                                  num_examples=TRAIN_EXAMPLES, num_kv_pairs=NUM_KV,
                                  random_non_queries=False)],
        test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                                 num_examples=2_000, num_kv_pairs=NUM_KV,
                                 random_non_queries=False)],
        batch_size=(BATCH, BATCH // 4),
        cache_dir="/tmp/zoology_cache_stdsweep",
    )
    mixer_kwargs = dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                        writer="softmax_gla", reader="softmax_linear",
                        tie_routers=False)
    mixer_kwargs.update(mixer_overrides)
    return TrainConfig(
        data=data,
        model=ModelConfig(
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.cla.ChunkedLinearAttention",
                kwargs=mixer_kwargs,
            ),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL, n_layers=2,
            max_position_embeddings=0, vocab_size=VOCAB,
        ),
        logger=LoggerConfig(project_name="cla-gla-curriculum", entity=""),
        max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
        run_id=run_id,
        early_stopping_threshold=0.99,
        early_stopping_metric="valid/accuracy",
    )


PYTHON = "/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python"
ZOOLOGY_DIR = Path("/home/blake/zoology")
RESULTS_FILE = Path("/home/blake/zoology/curriculum_results.jsonl")


def run_one(idx: int, total: int, run_id: str, condition_env: dict, cfg: TrainConfig):
    tmp_path = Path(f"/tmp/_curr_single_{idx}.py")
    tmp_path.write_text(
        "import pickle\n"
        f"with open('/tmp/_curr_cfg_{idx}.pkl', 'rb') as f:\n"
        "    configs = [pickle.load(f)]\n"
    )
    with open(f"/tmp/_curr_cfg_{idx}.pkl", "wb") as f:
        pickle.dump(cfg, f)

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    # Clear any prior router/curriculum env (clean slate per run).
    for k in list(env.keys()):
        if k.startswith("MQAR_ROUTER_STD") or k.startswith("MQAR_CURR"):
            del env[k]
    env.update(condition_env)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "zoology.launch", str(tmp_path)],
            cwd=str(ZOOLOGY_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=1800,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        import re
        combined = proc.stdout
        pairs = re.findall(
            r'Valid Epoch (\d+)/\d+: 100%\|[^\n\r]*valid/accuracy=([\d.]+)', combined)
        pairs = [(int(e), float(a)) for e, a in pairs if 0 <= float(a) <= 1]
        max_acc = max((a for _, a in pairs), default=0.0)
        grok_ep = next((e for e, a in pairs if a >= 0.99), None)
        return {"run_id": run_id, "idx": idx, "ok": ok, "elapsed": elapsed,
                "max_acc": max_acc, "grok_ep": grok_ep, "returncode": proc.returncode,
                "condition_env": condition_env,
                "stderr_tail": combined[-500:] if not ok else ""}
    except subprocess.TimeoutExpired:
        return {"run_id": run_id, "idx": idx, "ok": False, "elapsed": 1800,
                "error": "timeout", "condition_env": condition_env}


def main():
    configs = []
    for cond_name, env, mixer_over in CONDITIONS:
        for seed in SEEDS:
            run_id = f"curr_{cond_name}_lr{LR:.1e}_s{seed}"
            configs.append((cond_name, env, mixer_over, seed, run_id))
    print(f"Loaded {len(configs)} configs across {len(CONDITIONS)} conditions × {len(SEEDS)} seeds",
          flush=True)

    completed = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        completed.add(r["idx"])
                except: pass
        print(f"Resuming: skipping {len(completed)} completed configs", flush=True)

    with open(RESULTS_FILE, "a") as fout:
        for i, (cond_name, env, mixer_over, seed, run_id) in enumerate(configs):
            if i in completed:
                continue
            cfg = make_config(seed, run_id, mixer_over)
            print(f"\n[{i+1}/{len(configs)}] {run_id} | env: {env} | mixer: {mixer_over}", flush=True)
            result = run_one(i, len(configs), run_id, env, cfg)
            fout.write(json.dumps(result) + "\n"); fout.flush()
            status = "OK" if result.get("ok") else f"FAIL ({result.get('error', result.get('returncode'))})"
            print(f"  → {status}, max_acc={result.get('max_acc', 0):.3f}, "
                  f"grok_ep={result.get('grok_ep')}, t={result.get('elapsed', 0):.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
