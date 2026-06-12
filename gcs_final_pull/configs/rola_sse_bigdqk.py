"""SSE in its native many-rows regime: large d_qk (many classification rows),
canonical n=4 partitions / top-k=1 (n4k1), at matched state.

The square-scaling SSE runs used d_qk=12 -> only 12 rows to classify into, far
below SSE's design point (c ~ 128). That handicapped SSE specifically. Here we
keep SSE's canonical 4 partitions and pour the matched-state budget into ROWS
(d_qk), so SSE gets its intended fine-grained classification. d_v stays at the
MQAR floor (12); buying rows costs partitions, which is the honest matched-state
trade-off.

state = nc * 4 * d_qk * (d_v+1) = 4 * 4 * d_qk * 13 = 208 * d_qk  (nc=4)
  d_qk=48  -> 9984   (rows=48, vs RoLA nc16-d12-dv12)
  d_qk=96  -> 19968  (rows=96, vs RoLA nc32-d12-dv12)
  d_qk=192 -> 39936  (rows=192 — SSE's native regime, vs RoLA nc64-d12-dv12)

3 runs, faithful SSE (softmax row classification, top-k=1 + always-on partition,
tied write-read gate, GLA decay, un-normalized, load-balance loss). Extended kv,
seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

NC = 4
D_V = 12
DQKS = [48, 96, 192]   # rows; states 9984 / 19968 / 39936 at nc=4
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for d_qk in DQKS:
    kw = rola_instance("rola-sse", d_qk=d_qk, d_v=D_V, num_chunks=NC, n_heads=4, route_topk=1)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    cell = f"rola-nc{NC}-d{d_qk}-dv{D_V}"
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
            logger=LoggerConfig(project_name="rola-sse-bigdqk", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_sse_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 3


def load_configs_and_envs():
    return configs, configs_envs
