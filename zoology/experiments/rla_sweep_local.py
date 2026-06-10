"""RLA Sweep — LOCAL-ONLY subset (d ∈ {16, 24, 32} CLA-RLA cells).

The d ∈ {16, 24, 32} baselines + MHA-ceiling are already done locally (84
rows in rla_sweep_results.jsonl). This config covers only the remaining
CLA-RLA cells at small d. The d ∈ {48, 64} cells go to cloud (separate
config rla_sweep_cloud.py).

10 cells × 4 LRs × 5 seeds = 200 runs.

Cells:
  cla-rla-d16-nc{2,4,8,16,32,64}  — d=16 column + interpolation
  cla-rla-d24-nc{2,4}
  cla-rla-d32-nc{2,4}

OOM_CELLS at d=16 (nc=16, 32, 64) keep batch=128 — those are the actual
small-VRAM cells, unrelated to the d=48/64 sysmem-spillover issue. Other
cells stay batch=256.

Imports from rla_sweep.py to keep cell definitions in sync.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, data_small_batch, wrap_hybrid, cla_rla, OOM_CELLS,
    LRS, SEEDS, D_MODEL, VOCAB,
)


CELLS = [
    # ---- 3a Decoupling grid at small d ----
    ("cla-rla-d16-nc2",     cla_rla(16,  2),    {}),
    ("cla-rla-d16-nc4",     cla_rla(16,  4),    {}),
    ("cla-rla-d24-nc2",     cla_rla(24,  2),    {}),
    ("cla-rla-d24-nc4",     cla_rla(24,  4),    {}),
    ("cla-rla-d32-nc2",     cla_rla(32,  2),    {}),
    ("cla-rla-d32-nc4",     cla_rla(32,  4),    {}),
    # ---- 3b Interpolation column at d=16 ----
    ("cla-rla-d16-nc8",     cla_rla(16,  8),    {}),
    ("cla-rla-d16-nc16",    cla_rla(16, 16),    {}),
    ("cla-rla-d16-nc32",    cla_rla(16, 32),    {}),
    ("cla-rla-d16-nc64",    cla_rla(16, 64),    {}),
]

assert len(CELLS) == 10, f"expected 10 cells, got {len(CELLS)}"

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    cell_data = data_small_batch if name in OOM_CELLS else data
    for lr in LRS:
        for seed in SEEDS:
            configs.append(
                TrainConfig(
                    data=cell_data,
                    model=ModelConfig(
                        block_type="TransformerBlock",
                        sequence_mixer=mixer,
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=0,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="rla-sweep-local", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 200, f"expected 200 configs, got {len(configs)}"


def load_configs_and_envs():
    return configs, configs_envs
