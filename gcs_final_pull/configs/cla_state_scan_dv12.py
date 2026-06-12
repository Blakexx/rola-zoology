"""State-axis scan at the best-performing CLA shape (d_qk=10, d_v=12),
with wide-router mlp-relu (hidden_dim=32 ≈ 3.2× d_qk).

The shape (d_qk=10, d_v=12) won at state 9,360 with hidden_dim=10 (0.974).
This sweep tests whether the same shape transfers to other state regimes,
and whether widening the router (10 → 32) lifts performance further.

State = nc · 4 · 10 · 13 = 520 · nc.

  nc |  state
   4 |  2080
   8 |  4160
  12 |  6240
  18 |  9360   ← direct comparison with prior hidden=10 result
  24 | 12480
  32 | 16640

6 runs, mlp-relu hidden=32, seed=1337, lr=1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp


NCS = [4, 8, 12, 18, 24, 32]
D_QK = 10
D_V = 12
HIDDEN = 32
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for nc in NCS:
    cell = f"cla-rla-nc{nc}-d{D_QK}-dv{D_V}"
    kernel = cla_rla_mlp(D_QK, D_V, nc, route_hidden_dim=HIDDEN, route_act='relu')
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
            logger=LoggerConfig(project_name="cla-state-scan-dv12", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_mlp-relu-h{HIDDEN}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
