"""Recipe-LR-scale ablation. Isolates whether the recipe's curriculum LR scale
is the determining factor for whether asymmetric router init helps or hurts.

Discovery context: pilot at d=16 nc=8 base_lr=1e-2 found recipe-on (0.892)
worse than recipe-off (0.940). Suspect: writer curriculum's `3.0 × base_lr =
3e-2` peak LR at epoch 0 thrashes the peaky-initialized W_route.

Ablation grid: d=16 nc=8 × 4 base_lrs × 3 writer-curriculum scales × 1 seed
= 12 runs (~5 hr local).

Writer curriculum w_p1 → w_p2 maintains the default 10× anneal ratio:
  scale=1.0: w 1.0→0.1
  scale=2.0: w 2.0→0.2
  scale=3.0: w 3.0→0.3   (default recipe)
Reader curriculum stays at default 0.3→3.0 so writer is the lone varying knob.
Init asymmetry (writer std=1.0, reader std=0.05) stays on in all variants.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, wrap_hybrid, cla_rla, D_MODEL, VOCAB,
)

ABL_SEEDS = [1337]
ABL_BASE_LRS = [1e-4, 4.64e-4, 2.15e-3, 1e-2]
WRITER_SCALES = [1.0, 2.0, 3.0]
D = 16
NC = 8


def recipe_env(w_scale):
    """Recipe with variable writer curriculum scale. Reader stays at default."""
    w_p1 = w_scale
    w_p2 = w_scale / 10.0  # preserve 10x anneal ratio
    return {
        "MQAR_ROUTER_STD_WRITE": "1.0",
        "MQAR_ROUTER_STD_READ":  "0.05",
        "MQAR_CURR_MODE": "linear",
        "MQAR_CURR_W_LR_PHASE1": f"{w_p1}", "MQAR_CURR_W_LR_PHASE2": f"{w_p2}",
        "MQAR_CURR_R_LR_PHASE1": "0.3",     "MQAR_CURR_R_LR_PHASE2": "3.0",
    }


CELLS = []
for w_scale in WRITER_SCALES:
    name = f"cla-rla-d{D}-nc{NC}-wscale{w_scale:g}"
    CELLS.append((name, cla_rla(D, NC), recipe_env(w_scale)))

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    for lr in ABL_BASE_LRS:
        for seed in ABL_SEEDS:
            configs.append(
                TrainConfig(
                    data=data,
                    model=ModelConfig(
                        block_type="TransformerBlock",
                        sequence_mixer=mixer,
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=0,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="rla-recipe-ablation", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 12, f"expected 12 ablation configs, got {len(configs)}"
