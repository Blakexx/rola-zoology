"""route_on='x' linear RoLA at low/mid states, asym + sym, to complete the
crossover curve.

We already have route_on='x' at high state (nc 48/63/64/76/77 -> 25k-40k, all
wins). This fills the low/mid grid matching the kq sweeps so the x-routing
crossover (lose at small state, win at large) can be drawn.

asym: d_qk=10, d_v=12  -> state = nc*4*10*13 = 520*nc
sym:  d_qk=d_v=11      -> state = nc*4*11*12 = 528*nc

nc in {4, 8, 12, 18, 24, 32}:
  asym states: 2080, 4160, 6240, 9360, 12480, 16640
  sym  states: 2112, 4224, 6336, 9504, 12672, 16896

12 runs, route_on='x' linear, extended kv test set, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_route_on_x import cla_rla_x_linear
from zoology.experiments.cla_router_width_v2 import data_ext


NCS = [4, 8, 12, 18, 24, 32]
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
            logger=LoggerConfig(project_name="cla-route-on-x-lowstate", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for nc in NCS:
    # asymmetric: d_qk=10, d_v=12
    _add(f"cla-rla-nc{nc}-d10-dv12",
         cla_rla_x_linear(10, 12, nc), "x-linear-asym")
    # symmetric: d_qk=d_v=11
    _add(f"cla-rla-nc{nc}-d11-dv11",
         cla_rla_x_linear(11, 11, nc), "x-linear-sym")

assert len(configs) == 12


def load_configs_and_envs():
    return configs, configs_envs
