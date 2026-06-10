"""High-state extension: CLA + RLA at ~25k, ~33k, ~40k on the hard-kv test set.

Now that we have the kv∈{...,512,1024} extrapolation test, push the state axis
beyond the 16k saturation point to see whether (a) RLA's wide-asym scales to
huge d_qk gracefully, (b) CLA scales gracefully via nc, (c) which param-scaling
strategy on CLA helps most.

Shapes (CLA stays d_qk=10, d_v=12 with mlp-relu routing):
  state ~25k: nc=48 → 24,960
  state ~33k: nc=64 → 33,280
  state ~40k: nc=77 → 40,040

RLA matched (d_v=12 wide-asym AND d_qk=d_v square):
  state 25k: rla-d481-dv12  /  rla-d79-dv79  (state 25,280)
  state 33k: rla-d640-dv12  /  rla-d91-dv91  (state 33,488)
  state 40k: rla-d770-dv12  /  rla-d100-dv100 (state 40,400)

CLA parameter-scaling variants at state ~40k:
  - h=10 (narrow router, baseline)
  - h=32 (wider router)
  - h=10 + dm=160 (d_model bump, proven at 16k)
  - h=10 + dm=192 (more aggressive d_model)
  - h=10 + n_layers=3 (depth instead of width)

Per-state runs (5):
  CLA h=10 baseline + RLA wide + RLA square + (only at 40k: 4 more scaling variants)
  = 3 states × 3 base + 4 extra at 40k = 13 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp, rla_asymmetric
from zoology.experiments.cla_router_width_v2 import data_ext


LR = 1e-2
SEED = 1337
ROUTER_HIDDEN = 10
D_QK, D_V = 10, 12

# (state_label, nc, rla_wide_d_qk, rla_sq_d)
TARGETS = [
    ("25k", 48, 481,  79),   # 24,960 / 25,012 / 25,280
    ("33k", 64, 635,  91),   # 33,280 / 33,020 / 33,488
    ("40k", 77, 770, 100),   # 40,040 / 40,040 / 40,400
]

configs = []
configs_envs = []


def _add(cell, mixer_kwargs, run_tag, d_model=D_MODEL, n_layers=2):
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(mixer_kwargs),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=d_model, n_layers=n_layers,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-high-state", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


# Base lineup: CLA baseline + RLA wide + RLA square at each state
for _label, nc, rla_wide_d, rla_sq_d in TARGETS:
    cla_cell = f"cla-rla-nc{nc}-d{D_QK}-dv{D_V}"
    _add(cla_cell,
         cla_rla_mlp(D_QK, D_V, nc, route_hidden_dim=ROUTER_HIDDEN, route_act='relu'),
         f"mlp-relu-h{ROUTER_HIDDEN}")
    _add(f"rla-d{rla_wide_d}-dv12",
         rla_asymmetric(rla_wide_d, 12),
         "linear")
    _add(f"rla-d{rla_sq_d}-dv{rla_sq_d}",
         rla_asymmetric(rla_sq_d, rla_sq_d),
         "linear")

# Extra CLA param-scaling variants only at state ~40k
NC_40K = 77
cell_40k = f"cla-rla-nc{NC_40K}-d{D_QK}-dv{D_V}"
# h=32 router
_add(cell_40k,
     cla_rla_mlp(D_QK, D_V, NC_40K, route_hidden_dim=32, route_act='relu'),
     "mlp-relu-h32")
# d_model = 160
_add(cell_40k,
     cla_rla_mlp(D_QK, D_V, NC_40K, route_hidden_dim=ROUTER_HIDDEN, route_act='relu'),
     f"mlp-relu-h{ROUTER_HIDDEN}-dm160",
     d_model=160)
# d_model = 192
_add(cell_40k,
     cla_rla_mlp(D_QK, D_V, NC_40K, route_hidden_dim=ROUTER_HIDDEN, route_act='relu'),
     f"mlp-relu-h{ROUTER_HIDDEN}-dm192",
     d_model=192)
# n_layers = 3
_add(cell_40k,
     cla_rla_mlp(D_QK, D_V, NC_40K, route_hidden_dim=ROUTER_HIDDEN, route_act='relu'),
     f"mlp-relu-h{ROUTER_HIDDEN}-nl3",
     n_layers=3)

assert len(configs) == 13


def load_configs_and_envs():
    return configs, configs_envs
