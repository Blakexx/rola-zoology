"""Third point on the state-9216 chunked-shape sweep.

Existing data at state 9216:
  cla-rla-nc16-d16-dv8 (linear)   : 0.950
  cla-rla-nc16-d16-dv8 (mlp-gelu) : 0.954
  cla-rla-nc32-d8-dv8  (linear)   : 0.946

New shape: nc=8, d_qk=32, d_v=8 → state 8·4·32·9 = 9216 (matched).
Wider d_qk per chunk, fewer chunks. Tests both linear and mlp-gelu routing.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB


def cla_rla_mlp(d_qk, d_v, num_chunks, route_hidden_dim=None, route_act='gelu'):
    kwargs = {
        "d_qk": d_qk, "d_v": d_v, "num_chunks": num_chunks, "n_heads": 4,
        "writer": "softmax_linear", "reader": "softmax_linear",
        "tie_routers": False, "use_short_conv": False, "route_on": "kq",
    }
    if route_hidden_dim:
        kwargs["route_hidden_dim"] = route_hidden_dim
        kwargs["route_act"] = route_act
    return dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kwargs)


CELL_NAME = "cla-rla-nc8-d32-dv8"
D_QK, D_V, NC = 32, 8, 8
LR = 1e-2
SEED = 1337

VARIANTS = [
    ("linear",   cla_rla_mlp(D_QK, D_V, NC, route_hidden_dim=None)),
    ("mlp-gelu", cla_rla_mlp(D_QK, D_V, NC, route_hidden_dim=D_QK, route_act='gelu')),
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
            logger=LoggerConfig(project_name="cla-shape-nc8", entity=""),
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
