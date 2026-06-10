"""d_v ∈ {10, 12} × nc-scan at state ≈ 9216, with linear + mlp-relu routing.

Plus 2 wide-asymmetric RLA references (no chunks, big d_qk) at matched state.

CLA shape matrix (8 cells; d_qk ≥ 10 throughout):

  d_v | nc | d_qk | state
  10  |  8 | 26   | 9152
  10  | 13 | 16   | 9152
  10  | 16 | 13   | 9152
  10  | 21 | 10   | 9240
  12  |  8 | 22   | 9152
  12  | 11 | 16   | 9152
  12  | 16 | 11   | 9152
  12  | 18 | 10   | 9360

RLA references (wide-asym):
  rla-d210-dv10 → state 9240
  rla-d180-dv12 → state 9360

8 CLA × {linear, mlp-relu} + 2 RLA = 18 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB


def cla_rla_mlp(d_qk, d_v, num_chunks, route_hidden_dim=None, route_act='relu'):
    kwargs = {
        "d_qk": d_qk, "d_v": d_v, "num_chunks": num_chunks, "n_heads": 4,
        "writer": "softmax_linear", "reader": "softmax_linear",
        "tie_routers": False, "use_short_conv": False, "route_on": "kq",
    }
    if route_hidden_dim:
        kwargs["route_hidden_dim"] = route_hidden_dim
        kwargs["route_act"] = route_act
    return dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kwargs)


def rla_asymmetric(d_qk, d_v):
    return dict(
        name="zoology.mixers.cla.RecurrentLinearAttention",
        kwargs={"d_qk": d_qk, "d_v": d_v, "n_heads": 4},
    )


# (nc, d_qk, d_v)
SHAPES = [
    ( 8, 26, 10), (13, 16, 10), (16, 13, 10), (21, 10, 10),
    ( 8, 22, 12), (11, 16, 12), (16, 11, 12), (18, 10, 12),
]

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
            logger=LoggerConfig(project_name="cla-dv-nc-scan", entity=""),
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

# RLA wide-asym references (state ≈ 9216, no chunks)
_add("rla-d210-dv10", rla_asymmetric(210, 10), "linear")
_add("rla-d180-dv12", rla_asymmetric(180, 12), "linear")

assert len(configs) == 18


def load_configs_and_envs():
    return configs, configs_envs
