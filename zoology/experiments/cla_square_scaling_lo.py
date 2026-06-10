"""Bottom of the square-scaling curve: nc=2 and nc=3.

Maps the smallest real RoLA points (nc=1 is the monolith identity). Same design
as cla_square_scaling: base square x=12, route_on='x', tied + untied routing,
with matched square AND wide-asym (d_qk-scaled) monoliths.

State = 624 * nc:
  nc=2 -> 1248   nc=3 -> 1872

Matched monoliths:
  square (d_qk=d_v=D, 4*D*(D+1)≈state):
    1248 -> d17-dv17 (4*17*18=1224)
    1872 -> d21-dv21 (4*21*22=1848)
  wide-asym (d_qk-scaled, d_v=12, 4*D*13=state -> D=12*nc):
    1248 -> d24-dv12   1872 -> d36-dv12

Runs: 2 nc * {RoLA-tied, RoLA-untied, square-mono, wide-mono} = 8.
Rich-logging image, extended kv test, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import rla_asymmetric
from zoology.experiments.cla_square_scaling import rola_square
from zoology.experiments.cla_router_width_v2 import data_ext


X = 12
# (nc, square-monolith D)
SCALE = [(2, 17), (3, 21)]
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []


def _add(cell, mixer_kwargs, run_tag):
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(mixer_kwargs),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-square-scaling-lo", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for nc, D in SCALE:
    cell = f"cla-rla-nc{nc}-d{X}-dv{X}"
    _add(cell, rola_square(X, nc, tie=False), "x-untied")
    _add(cell, rola_square(X, nc, tie=True),  "x-tied")
    _add(f"rla-d{D}-dv{D}", rla_asymmetric(D, D), "linear")          # square monolith
    _add(f"rla-d{12*nc}-dv{X}", rla_asymmetric(12 * nc, X), "linear")  # wide-asym monolith

assert len(configs) == 8


def load_configs_and_envs():
    return configs, configs_envs
