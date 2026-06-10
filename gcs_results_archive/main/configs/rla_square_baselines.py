"""RLA-square (d_qk = d_v) baselines at the unified-sweep states.

Complements cla_router_width_v2 (which has only wide-asym RLA refs).
Extended test set (kv up to 1024).

State targets (matching the d=10, d_v=12 CLA scan):
  state 2080 → d=23 (4*23*24 = 2208)
  state 4160 → d=32 (4*32*33 = 4224)
  state 6240 → d=39 (4*39*40 = 6240)
  state 9360 → d=48 (4*48*49 = 9408)
  state 12480 → d=55 (4*55*56 = 12320)
  state 16640 → d=64 (4*64*65 = 16640)

6 runs total.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import rla_asymmetric
from zoology.experiments.cla_router_width_v2 import data_ext


SHAPES = [23, 32, 39, 48, 55, 64]  # d_qk = d_v
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for d in SHAPES:
    cell = f"rla-d{d}-dv{d}"
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(rla_asymmetric(d, d)),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="rla-square-baselines", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_linear_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
