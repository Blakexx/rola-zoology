"""Faithful SSE re-test: the row-sparse update (paper Eq.7).

The earlier rola_sse runs were missing SSE's headline mechanism: the key was a
DENSE softmax over all d_qk rows (every token smears into every state row ->
interference -> recall collapses as kv grows). Eq.7 is k = softmax(top-k(x W_k)):
each token writes to only `row_topk` of the c=d_qk rows (hard classification).
This sweep turns that on and varies row_topk to find SSE's real ceiling on MQAR.

Everything else identical to the prior faithful-SSE corner: tied write-read gate,
GLA diagonal decay, un-normalized, sparse partition top-k=1 + 1 always-on
partition, load-balance aux loss. State is unchanged by row_topk (still
nc*4*d_qk*(d_v+1)), so these stay matched to the prior SSE / RoLA points.

  d_qk=12 (c=12 rows): nc in {16,32}, row_topk in {1,2,4}   -> 6 runs
  d_qk=48 native:      nc=4,          row_topk in {2,4,8}   -> 3 runs
Extended kv, seed 1337, lr 1e-2. 9 runs total.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

D_V = 12
LR = 1e-2
SEED = 1337

# (d_qk, nc, [row_topk values])
GRID = [
    (12, 16, [1, 2, 4]),
    (12, 32, [1, 2, 4]),
    (48,  4, [2, 4, 8]),
]

configs = []
configs_envs = []
for d_qk, nc, rtks in GRID:
    for rtk in rtks:
        kw = rola_instance("rola-sse", d_qk=d_qk, d_v=D_V, num_chunks=nc,
                           n_heads=4, route_topk=1, row_topk=rtk)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        cell = f"rola-nc{nc}-d{d_qk}-dv{D_V}"
        configs.append(
            TrainConfig(
                data=data_ext,
                model=ModelConfig(
                    block_type="TransformerBlock",
                    sequence_mixer=wrap_hybrid(kernel),
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=0,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="rola-sse-rowtopk", entity=""),
                max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
                run_id=f"{cell}_sse-rtk{rtk}_lr{LR:.2e}_s{SEED}",
                early_stopping_threshold=2.0,
                early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"],
            )
        )
        configs_envs.append({})

assert len(configs) == 9


def load_configs_and_envs():
    return configs, configs_envs
