"""RoLA-SSE across the square-scaling states, alongside the two RoLA-RLA
instances for a clean 3-way framework comparison on one image.

Instances (from cla_bench.rola_instance):
  rola-rla-asym : dense, untied, normalized, linear kernel
  rola-rla-sym  : dense, tied,   normalized, linear kernel
  rola-sse      : sparse top-k, tied (symmetric gate), un-normalized,
                  GLA-decay inner kernel  -- the SSE corner of the framework

Square base x=12, route_on='x'. nc in {4, 8, 16, 32, 64} -> state 624*nc.
SSE top-k = 2 (sparse; routes each token to its 2 highest states).

3 instances * 5 nc = 15 runs. Extended kv test, seed 1337, lr 1e-2.
The matched monoliths already exist from cla_square_scaling{,_widemono}.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

X = 12
NCS = [4, 8, 16, 32, 64]
SSE_TOPK = 2
LR = 1e-2
SEED = 1337
INSTANCES = ["rola-rla-asym", "rola-rla-sym", "rola-sse"]
# short run-id tags per instance
TAG = {"rola-rla-asym": "asym", "rola-rla-sym": "sym", "rola-sse": "sse"}

configs = []
configs_envs = []
for nc in NCS:
    for inst in INSTANCES:
        kw = rola_instance(inst, d_qk=X, d_v=X, num_chunks=nc, n_heads=4, route_topk=SSE_TOPK)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        cell = f"rola-nc{nc}-d{X}-dv{X}"
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
                logger=LoggerConfig(project_name="rola-sse", entity=""),
                max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
                run_id=f"{cell}_{TAG[inst]}_lr{LR:.2e}_s{SEED}",
                early_stopping_threshold=2.0,
                early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"],
            )
        )
        configs_envs.append({})

assert len(configs) == 15


def load_configs_and_envs():
    return configs, configs_envs
