"""Apply state-9.2k learnings to state ~4600.

At state ~9.2k we discovered:
  - mlp-relu routing beats linear by +0.01 to +0.07
  - d_v ∈ {10, 12} beats d_v=8 at matched state
  - mid-nc (d_qk ≥ 10) beats extreme nc

This sweep applies those learnings to the previously CLA-weak state ~4600 regime,
where prior runs had CLA stuck at 0.83-0.87 vs RLA 0.92.

Shape matrix (d_qk ≥ 10 throughout):

  d_v | nc | d_qk | state
   8  | 16 |  8   | 4608   (mlp-relu only — linear baseline already at 0.863)
  10  |  5 | 21   | 4620
  10  |  7 | 15   | 4620
  12  |  4 | 22   | 4576
  12  |  8 | 11   | 4576
  16  |  4 | 17   | 4624
  16  |  5 | 14   | 4760

RLA wide-asym refs at matched state:
  rla-d105-dv10 → state 4620
  rla-d89-dv12  → state 4628
  rla-d68-dv16  → state 4624

6 CLA × {linear, mlp-relu} + 1 dv8 mlp-relu + 3 RLA = 16 runs (seed 1337, lr 1e-2).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp, rla_asymmetric


# (nc, d_qk, d_v)
CLA_SHAPES_BOTH_ROUTING = [
    ( 5, 21, 10), (7, 15, 10),
    ( 4, 22, 12), (8, 11, 12),
    ( 4, 17, 16), (5, 14, 16),
]
# Cells that get only mlp-relu (linear baseline already in cache)
CLA_MLP_ONLY = [(16, 8, 8)]

LR = 1e-2
SEED = 1337

configs = []
configs_envs = []


def _add(cell, kernel, run_tag):
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
            logger=LoggerConfig(project_name="cla-4600-scan", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for nc, d_qk, d_v in CLA_SHAPES_BOTH_ROUTING:
    cell = f"cla-rla-nc{nc}-d{d_qk}-dv{d_v}"
    _add(cell, cla_rla_mlp(d_qk, d_v, nc, route_hidden_dim=None), "linear")
    _add(cell, cla_rla_mlp(d_qk, d_v, nc, route_hidden_dim=d_qk, route_act='relu'), "mlp-relu")

for nc, d_qk, d_v in CLA_MLP_ONLY:
    cell = f"cla-rla-nc{nc}-d{d_qk}-dv{d_v}"
    _add(cell, cla_rla_mlp(d_qk, d_v, nc, route_hidden_dim=d_qk, route_act='relu'), "mlp-relu")

_add("rla-d105-dv10", rla_asymmetric(105, 10), "linear")
_add("rla-d89-dv12",  rla_asymmetric( 89, 12), "linear")
_add("rla-d68-dv16",  rla_asymmetric( 68, 16), "linear")

assert len(configs) == 16


def load_configs_and_envs():
    return configs, configs_envs
