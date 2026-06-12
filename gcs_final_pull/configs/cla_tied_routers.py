"""CLA with TIED routers (symmetric routing): writer and reader share router weights.

Requires route_on='x' (residual-stream routing) since tied routers can't have
different inputs per role. MLP routing isn't implemented for x-routing yet,
so this sweep uses plain linear routing.

This is the canonical "symmetric routing" CLA variant — the writer's routing
decision and the reader's routing decision come from the same learned function.

Comparison vs our best CLA (untied + kq + mlp-relu) at matched state.

Shape: d_qk = 10, d_v = 12 (matching wide-asym CLA scan).
Routing: tie_routers=True, route_on='x', linear (nn.Linear router).

nc ∈ {4, 8, 12, 18, 24, 32} → states {2080, 4160, 6240, 9360, 12480, 16640}.
6 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext


def cla_rla_tied_linear(d_qk, d_v, num_chunks):
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": d_qk, "d_v": d_v, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_linear", "reader": "softmax_linear",
            "tie_routers": True,        # ← key change
            "use_short_conv": False,
            "route_on": "x",            # ← required for tied
        },
    )


NCS = [4, 8, 12, 18, 24, 32]
D_QK, D_V = 10, 12
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for nc in NCS:
    cell = f"cla-rla-nc{nc}-d{D_QK}-dv{D_V}"
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(cla_rla_tied_linear(D_QK, D_V, nc)),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-tied-routers", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_tied-x-linear_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
