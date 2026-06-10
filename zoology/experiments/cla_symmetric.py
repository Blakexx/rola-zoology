"""Symmetric CLA variant sweep: d_qk = d_v at the 6 unified-sweep states.

Mirrors the wide-asym CLA scan (d_qk=10, d_v=12) but with d_qk = d_v = 11
so the routing key space and value space have the same width. State targets
land within ~1.5% of the asym versions; nc varies from 4 to 32.

Routing: mlp-relu with hidden = d_qk = 11 (narrow-router optimum from
cla_router_width_v2).

6 runs (seed 1337, lr 1e-2, extended kv test set).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp
from zoology.experiments.cla_router_width_v2 import data_ext


NCS = [4, 8, 12, 18, 24, 32]
D = 11  # d_qk = d_v
LR = 1e-2
SEED = 1337
ROUTER_HIDDEN = D  # h = d_qk = 11

configs = []
configs_envs = []
for nc in NCS:
    cell = f"cla-rla-nc{nc}-d{D}-dv{D}"
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(
                    cla_rla_mlp(D, D, nc, route_hidden_dim=ROUTER_HIDDEN, route_act='relu')
                ),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-symmetric", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_mlp-relu-h{ROUTER_HIDDEN}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
