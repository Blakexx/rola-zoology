"""GLA analogue of the RLA/GDN high-state comparison at ~20k/~40k, easy regime
(old data_ext: kv<=1024, vocab 8192, batch 64). GLA is un-normalized (just the
forget gate), so state = 4*nc*d_qk*d_v — matched to the GDN runs.

  RoLA-GLA routed (gla-sym, route_on='x', d_qk=12, d_v=12):
    nc=32 -> 18432 (~20k),  nc=64 -> 36864 (~40k)
  GLA square monolith (nc=1, d_qk=d_v): d=68 (~20k), d=96 (~40k)  [<=320, should run]
  GLA wide monolith (nc=1, d_v=12):     d_qk=384 (~20k), 768 (~40k)
     [probe: chunk_gla ran to d=320 before; 384/768 may exceed its head-dim limit]

6 runs, seed 1337, lr 1e-2. expandable_segments passed at submit for the heavy ones.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from cla_bench import rola_instance

LR = 1e-2
SEED = 1337

# (tag, nc, d_qk, d_v)
SPECS = [
    ("rola-gla-sym-nc32-d12", 32, 12, 12),   # ~20k routed
    ("rola-gla-sym-nc64-d12", 64, 12, 12),   # ~40k routed
    ("mono-gla-square-d68",    1, 68, 68),   # ~20k square monolith
    ("mono-gla-square-d96",    1, 96, 96),   # ~40k square monolith
    ("mono-gla-wide-d384",     1, 384, 12),  # ~20k wide monolith (probe)
    ("mono-gla-wide-d768",     1, 768, 12),  # ~40k wide monolith (probe)
]

configs = []
configs_envs = []
for tag, nc, d_qk, d_v in SPECS:
    kw = rola_instance("rola-gla-sym", d_qk=d_qk, d_v=d_v, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
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
            logger=LoggerConfig(project_name="rola-gla-highstate", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
