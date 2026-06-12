"""RLA Sweep — CLOUD-ONLY subset (d ∈ {48, 64}).

These cells were hitting VRAM spillover on the local 12 GB 3080 Ti
(FLA pads d=48 → BK=BV=64 internally → ~4× larger intermediate tensors →
allocator thrashing into system RAM, observed as 4× training slowdown).
On A100 40 GB there's no VRAM pressure, so all cells run at batch=256 —
no OOM_CELLS asymmetry vs the local d ∈ {16,24,32} run.

6 cells × 4 LRs × 5 seeds = 120 runs.

Imports from rla_sweep.py to keep the cell definitions (rla_baseline, cla_rla,
wrap_hybrid) and hyperparameters (LRS, SEEDS, D_MODEL, VOCAB, data) in sync.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, wrap_hybrid, rla_baseline, cla_rla,
    LRS, SEEDS, D_MODEL, VOCAB,
)


# Cloud-only cells: d ∈ {48, 64}. Run d=48 baseline first so the smoke-test
# (--max-runs=1) hits the cell whose utilization we actually care about.
CELLS = [
    ("rla-baseline-d48",    rla_baseline(48),   {}),
    ("rla-baseline-d64",    rla_baseline(64),   {}),
    ("cla-rla-d48-nc2",     cla_rla(48,  2),    {}),
    ("cla-rla-d48-nc4",     cla_rla(48,  4),    {}),
    ("cla-rla-d64-nc2",     cla_rla(64,  2),    {}),
    ("cla-rla-d64-nc4",     cla_rla(64,  4),    {}),
]

assert len(CELLS) == 6, f"expected 6 cells, got {len(CELLS)}"

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    for lr in LRS:
        for seed in SEEDS:
            configs.append(
                TrainConfig(
                    data=data,  # all batch=256 on cloud A100 — no OOM concern
                    model=ModelConfig(
                        block_type="TransformerBlock",
                        sequence_mixer=mixer,
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=0,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="rla-sweep-cloud", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 120, f"expected 120 configs, got {len(configs)}"


def load_configs_and_envs():
    return configs, configs_envs
