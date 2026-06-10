"""d_v = 16, nc-scan at state ≈ 9216, with linear + mlp-relu routing + RLA ref.

CLA shapes (d_qk ≥ 10 throughout):
  nc |  d_qk |  state
   5 |   27  |  9180
   8 |   17  |  9248
  11 |   12  |  8976
  14 |   10  |  9520

RLA wide-asym ref: rla-d135-dv16 → state 9180.

4 CLA × {linear, mlp-relu} + 1 RLA = 9 runs (seed 1337, lr 1e-2).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp, rla_asymmetric


SHAPES = [(5, 27, 16), (8, 17, 16), (11, 12, 16), (14, 10, 16)]
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
            logger=LoggerConfig(project_name="cla-dv16-scan", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for nc, d_qk, d_v in SHAPES:
    cell = f"cla-rla-nc{nc}-d{d_qk}-dv{d_v}"
    _add(cell, cla_rla_mlp(d_qk, d_v, nc, route_hidden_dim=None), "linear")
    _add(cell, cla_rla_mlp(d_qk, d_v, nc, route_hidden_dim=d_qk, route_act='relu'), "mlp-relu")

# Wide-asym RLA reference at d_v=16, state ≈ 9216
_add("rla-d135-dv16", rla_asymmetric(135, 16), "linear")

assert len(configs) == 9


def load_configs_and_envs():
    return configs, configs_envs
