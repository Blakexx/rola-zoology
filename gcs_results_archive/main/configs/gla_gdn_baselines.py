"""GLA and GDN baselines at the 6 state targets of the router-width sweep.

Same 6 states as cla_router_width_v2: nc · 4 · 10 · 13 for nc ∈ {4,8,12,18,24,32}.
Extended test set (kv up to 1024) to match.

Architectures tested:
  - GLA wide-asym (d_v=12, like RLA wide): 6 cells   — cla_bench.RecurrentGLA
  - GLA square (d_qk = d_v):                6 cells
  - GDN (zoology's GatedDeltaNet, head_dim=32 fixed by d_model/num_heads;
         expand_v tuned to match state):    6 cells

NOTE: GDN's zoology wrapper hardcodes head_dim = hidden_size // num_heads, so
"wide vs square" isn't really tunable for GDN — only expand_v (i.e. d_v) varies.

18 runs total, seed 1337, lr 1e-2, max_epochs 32, extended test kv∈{...,512,1024}.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB

# Reuse the same extended data config (kv up to 1024).
from zoology.experiments.cla_router_width_v2 import data_ext


def gla_recurrent(d_qk, d_v):
    return dict(
        name="zoology.mixers.cla.RecurrentGLA",
        kwargs={"d_qk": d_qk, "d_v": d_v, "n_heads": 4},
    )


def gdn(expand_v):
    # head_dim is hardcoded by the wrapper to d_model//num_heads = 32.
    # expand_v sets head_v_dim = head_dim * expand_v = 32 * expand_v.
    return dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"num_heads": 4, "expand_v": expand_v, "use_short_conv": False},
    )


# Target states (nc·4·d_qk·(d_v+1) at d_qk=10, d_v=12 from the unified sweep).
TARGET_STATES = [2080, 4160, 6240, 9360, 12480, 16640]


def gla_wide_asym(target):
    """GLA at d_v=12 with d_qk = state/(4*13). State matches exactly."""
    d_qk = target // (4 * 13)
    return f"gla-d{d_qk}-dv12", gla_recurrent(d_qk, 12)


def gla_square(target):
    """GLA with d_qk = d_v = round(sqrt(state/4)). State approx but close."""
    import math
    d = round(math.sqrt(target / 4))
    return f"gla-d{d}-dv{d}", gla_recurrent(d, d)


def gdn_matched(target):
    """GDN with head_dim=32 fixed; head_v_dim picked to hit target state.
    State = num_heads · head_k_dim · head_v_dim = 4 · 32 · head_v_dim = 128 · head_v_dim.
    """
    head_v_dim = round(target / 128)
    expand_v = head_v_dim / 32  # zoology's GatedDeltaNet at d_model=128, num_heads=4
    state = 4 * 32 * head_v_dim
    return f"gdn-head32-hv{head_v_dim}", gdn(expand_v), state


LR = 1e-2
SEED = 1337

configs = []
configs_envs = []


def _add(cell, mixer_kwargs, project="gla-gdn-baselines"):
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
            logger=LoggerConfig(project_name=project, entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_linear_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for s in TARGET_STATES:
    cell_a, mix_a = gla_wide_asym(s)
    _add(cell_a, mix_a)
    cell_b, mix_b = gla_square(s)
    _add(cell_b, mix_b)
    cell_c, mix_c, _ = gdn_matched(s)
    _add(cell_c, mix_c)

assert len(configs) == 18


def load_configs_and_envs():
    return configs, configs_envs
