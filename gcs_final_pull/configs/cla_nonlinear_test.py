"""Nonlinear (MLP) routing test on the best non-grokking CLA cell.

Same cell as the recipe test: cla-rla-nc16-d16-dv8 (state 9216, baseline 0.950).
Variant: writer + reader routers become small MLPs (d_qk -> hidden -> num_chunks
with GELU between), instead of a single linear projection.

Hidden dim = d_qk = 16 (modest; doubles router params from 2048 → 4096 per
layer, total model params +8%, state unchanged).

3 variants × 1 seed = 3 runs:
  - linear baseline (sanity check vs prior 0.950 result)
  - MLP-GELU
  - MLP-ReLU
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB


def cla_rla_mlp(d_qk, d_v, num_chunks, route_hidden_dim=None, route_act='gelu'):
    """CLA-RLA with optional MLP routing."""
    kwargs = {
        "d_qk": d_qk, "d_v": d_v, "num_chunks": num_chunks, "n_heads": 4,
        "writer": "softmax_linear", "reader": "softmax_linear",
        "tie_routers": False, "use_short_conv": False, "route_on": "kq",
    }
    if route_hidden_dim:
        kwargs["route_hidden_dim"] = route_hidden_dim
        kwargs["route_act"] = route_act
    return dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kwargs)


CELL_NAME = "cla-rla-nc16-d16-dv8"
D_QK, D_V, NC = 16, 8, 16
LR = 1e-2
SEED = 1337

VARIANTS = [
    ("linear",   cla_rla_mlp(D_QK, D_V, NC, route_hidden_dim=None)),
    ("mlp-gelu", cla_rla_mlp(D_QK, D_V, NC, route_hidden_dim=D_QK, route_act='gelu')),
    ("mlp-relu", cla_rla_mlp(D_QK, D_V, NC, route_hidden_dim=D_QK, route_act='relu')),
]

configs = []
configs_envs = []
for tag, kernel in VARIANTS:
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
            logger=LoggerConfig(project_name="cla-nonlinear-test", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{CELL_NAME}_{tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


def load_configs_and_envs():
    return configs, configs_envs
