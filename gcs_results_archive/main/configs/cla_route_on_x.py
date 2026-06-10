"""CLA with route_on='x' (residual-stream routing), linear router.

Compares the original CLA routing scheme (residual stream) against our default
route_on='kq' (key/query routing) at the high-state regime where CLA's
architectural advantage is clearest.

Shape variants per state:
  - asymmetric: d_qk=10, d_v=12 (same as the winning kq config)
  - symmetric:  d_qk=d_v=11 (square)

State targets (matching cla_high_state):
  25k: asym nc=48 → 24,960; sym nc=47 → 24,816
  33k: asym nc=64 → 33,280; sym nc=63 → 33,264
  40k: asym nc=77 → 40,040; sym nc=76 → 40,128

6 runs (3 states × {asym, sym}). Extended kv test set.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext


def cla_rla_x_linear(d_qk, d_v, num_chunks):
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": d_qk, "d_v": d_v, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_linear", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": False,
            "route_on": "x",   # ← switches from default 'kq' to residual-stream
        },
    )


# (state_label, asym_nc, sym_nc)
TARGETS = [
    ("25k", 48, 47),   # 24,960  / 24,816
    ("33k", 64, 63),   # 33,280  / 33,264
    ("40k", 77, 76),   # 40,040  / 40,128
]

LR = 1e-2
SEED = 1337
D_QK_ASYM, D_V_ASYM = 10, 12
D_SYM = 11

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
            logger=LoggerConfig(project_name="cla-route-on-x", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for _label, asym_nc, sym_nc in TARGETS:
    asym_cell = f"cla-rla-nc{asym_nc}-d{D_QK_ASYM}-dv{D_V_ASYM}"
    _add(asym_cell,
         cla_rla_x_linear(D_QK_ASYM, D_V_ASYM, asym_nc),
         "x-linear-asym")
    sym_cell = f"cla-rla-nc{sym_nc}-d{D_SYM}-dv{D_SYM}"
    _add(sym_cell,
         cla_rla_x_linear(D_SYM, D_SYM, sym_nc),
         "x-linear-sym")

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
