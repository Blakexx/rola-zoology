"""Companion to cla_square_scaling: the d_qk-scaled (wide) monolith baseline.

The square-scaling sweep compares RoLA (tied/untied) against the SQUARE monolith
(d_qk=d_v). The experimental design also scales the monolith the other way:
grow d_qk while holding d_v at the base value (12). That gives a wide-asymmetric
monolith, the classic "spend the state budget on more key dimensions" baseline.

State of rla-d{D}-dv12 = 4*D*13 = 52*D, which equals the RoLA state 624*nc
exactly when D = 12*nc. So these match the sqscale states with no rounding:

  state  | RoLA nc | wide-asym monolith
   2496  |   4     | rla-d48-dv12
   4992  |   8     | rla-d96-dv12
   9984  |  16     | rla-d192-dv12
  19968  |  32     | rla-d384-dv12
  39936  |  64     | rla-d768-dv12

5 runs, new (rich-logging) image, extended kv test, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import rla_asymmetric
from zoology.experiments.cla_router_width_v2 import data_ext


D_V = 12
NCS = [4, 8, 16, 32, 64]   # wide monolith d_qk = 12 * nc
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for nc in NCS:
    d_qk = 12 * nc
    cell = f"rla-d{d_qk}-dv{D_V}"
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(rla_asymmetric(d_qk, D_V)),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-square-scaling-widemono", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_linear_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 5


def load_configs_and_envs():
    return configs, configs_envs
