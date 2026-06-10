"""Square Hedgehog monolith (nc=1, d_qk=d_v) at ~20k/~40k — the fair high-state
Hedgehog monolith we missed earlier (only ran WIDE d384/d768, which washed out).
At d_qk=68/96 the softmax feature map is over 68/96 dims (vs 384/768), so it should
actually train. Pairs with RoLA-Hedgehog routed (nc32=0.805, nc64=0.892) for the
fair routed-vs-monolith Hedgehog comparison.

Hedgehog is normalized (V+1): state = 4*1*d*(d+1):
  d=68 -> 18768 (~20k);  d=96 -> 37248 (~40k)
Easy regime (data_ext: kv<=1024, vocab 8192, batch 64). 2 runs, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from cla_bench import rola_instance

LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for d in (68, 96):
    kw = rola_instance("rola-hedgehog-sym", d_qk=d, d_v=d, num_chunks=1, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="rola-hh-square", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"mono-hh-square-d{d}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 2


def load_configs_and_envs():
    return configs, configs_envs
