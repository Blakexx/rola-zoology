"""Router width sweep at fixed shape (d_qk=10, d_v=12).

Existing data:
  - hidden=32 for nc ∈ {4, 8, 12, 18, 24, 32} (cla_state_scan_dv12)
  - hidden=10 for nc=18 (from cla_dv_nc_scan)

This sweep fills in:
  - hidden=10 (base = d_qk) for nc ∈ {4, 8, 12, 24, 32} = 5 runs
  - hidden=128 (very wide) for nc ∈ {4, 8, 12, 18, 24, 32}    = 6 runs

11 runs total, all mlp-relu, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp


NCS = [4, 8, 12, 18, 24, 32]
D_QK, D_V = 10, 12
LR = 1e-2
SEED = 1337

# (nc, hidden_dim) pairs to actually run (skip existing combos)
TO_RUN = []
for nc in NCS:
    # h=10 — skip nc=18 (already in cache)
    if nc != 18:
        TO_RUN.append((nc, 10))
    # h=128 — always
    TO_RUN.append((nc, 128))

configs = []
configs_envs = []
for nc, hidden in TO_RUN:
    cell = f"cla-rla-nc{nc}-d{D_QK}-dv{D_V}"
    kernel = cla_rla_mlp(D_QK, D_V, nc, route_hidden_dim=hidden, route_act='relu')
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-router-width", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_mlp-relu-h{hidden}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 11


def load_configs_and_envs():
    return configs, configs_envs
