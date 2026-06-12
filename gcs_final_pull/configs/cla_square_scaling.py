"""Clean square-scaling design.

Base: a SQUARE RLA monolith, d_qk = d_v = x, 1 state. Scale the state budget
two ways and compare at matched state:

  (a) Monolith: grow d_qk = d_v (stays square). The baseline.
  (b) RoLA: grow the number of states nc, per-state square d_qk = d_v = x,
      in two routing modes:
        - symmetric routing  = tied read/write router  (tie_routers=True)
        - asymmetric routing = untied read/write router (tie_routers=False)
      Both use route_on='x' so the ONLY difference is the tie. (Tied routing
      requires route_on='x'; reader and writer cannot share weights under kq
      routing because the writer routes on k and the reader on q.)

At nc=1 both RoLA modes reduce to the base monolith (routing cancels), so we
run the base once and start RoLA scaling at nc=4.

Base x = 12 (square, at the d_v >= ~12 MQAR threshold).
State = 4 * nc * 12 * 13 = 624 * nc.

  nc | RoLA state | matched square monolith
   1 |    624     | d12-dv12 (base)
   4 |   2496     | d24-dv24 (2400)
   8 |   4992     | d35-dv35 (5040)
  16 |   9984     | d49-dv49 (9800)
  32 |  19968     | d70-dv70 (19880)
  64 |  39936     | d99-dv99 (39600)

Runs: base (1) + 5 nc * {RoLA-tied, RoLA-untied, square-monolith} = 16.
Extended kv test set, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import rla_asymmetric  # generic RLA(d_qk,d_v)
from zoology.experiments.cla_router_width_v2 import data_ext


def rola_square(x, num_chunks, tie):
    """RoLA with square per-state dims (d_qk=d_v=x), route_on='x', linear router.
    tie=True  -> symmetric routing (shared read/write router).
    tie=False -> asymmetric routing (separate read/write routers)."""
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": x, "d_v": x, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_linear", "reader": "softmax_linear",
            "tie_routers": tie, "use_short_conv": False, "route_on": "x",
        },
    )


X = 12
# (nc, matched square-monolith D)
SCALE = [(4, 24), (8, 35), (16, 49), (32, 70), (64, 99)]
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
            logger=LoggerConfig(project_name="cla-square-scaling", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


# Base square monolith (nc=1).
_add(f"rla-d{X}-dv{X}", rla_asymmetric(X, X), "linear")

for nc, D in SCALE:
    cell = f"cla-rla-nc{nc}-d{X}-dv{X}"
    _add(cell, rola_square(X, nc, tie=False), "x-untied")   # asymmetric routing
    _add(cell, rola_square(X, nc, tie=True),  "x-tied")     # symmetric routing
    _add(f"rla-d{D}-dv{D}", rla_asymmetric(D, D), "linear")  # matched square monolith

assert len(configs) == 16


def load_configs_and_envs():
    return configs, configs_envs
