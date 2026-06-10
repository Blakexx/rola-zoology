"""RLA Pilot — G2 from resources/rla_sweep.md.

Anchors the d-axis of the main sweep. Pilot finds smallest d that solves at
chunk-rich (n_chunks=8) setting.

Cells: n_chunks=8 × d ∈ {16, 24, 32}  =  3 cells
Seeds: {1337, 42}                     =  2 seeds
LR:    1e-2                           =  1 LR
Total: 6 runs.

Reuses rla_sweep.py task config (multi-task MQAR canonical) verbatim.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, data_small_batch, wrap_hybrid, cla_rla, D_MODEL, VOCAB,
)

PILOT_SEEDS = [1337, 42]
PILOT_LR = 1e-2
PILOT_DS = [16, 24, 32]
PILOT_NC = 8

# Recipe-off, matching the rla_sweep config decision (recipe ablation confirmed
# recipe-on hurts plain LA across all measured (init × curriculum × scale × LR)
# combinations at d=16 nc=8).
# d=32 nc=8 OOMs at batch=256 on local 3080 Ti (cudaErrorUnknown during backward
# at 108s) → use batch=128 for that cell. Same threshold motivates the main
# sweep's OOM_CELLS list; running d=32 nc=8 at batch=128 also validates that
# fix works before the 6-day sweep launches.
CELLS = [(f"cla-rla-d{d}-nc{PILOT_NC}", cla_rla(d, PILOT_NC), {}) for d in PILOT_DS]
SMALL_BATCH_DS = {32}  # any d with activations > ~250 MB per virt tensor

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    # Extract d from cell name "cla-rla-d{d}-nc{nc}".
    d = int(name.split("-d")[1].split("-")[0])
    cell_data = data_small_batch if d in SMALL_BATCH_DS else data
    for seed in PILOT_SEEDS:
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
                logger=LoggerConfig(project_name="rla-pilot", entity=""),
                max_epochs=32, learning_rate=PILOT_LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{PILOT_LR:.2e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"],
            )
        )
        configs_envs.append(env)

assert len(configs) == 6, f"expected 6 pilot configs, got {len(configs)}"
