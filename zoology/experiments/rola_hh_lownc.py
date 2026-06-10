"""Low-nc Hedgehog: nc=1 and nc=2 at the high-state per-head dims (d_qk=12, d_v=12).
nc=1 is a Hedgehog monolith with d_qk=12 — its softmax feature map is over only 12
dims, so it is NOT degenerate like the wide monoliths (d_qk=384/768, which washed
out to ~0.03). This gives the fair monolith baseline + the first routing step, to
see whether routing (nc=2) already beats the monolith (nc=1) and how it compares to
the broken wide monolith.

State (Hedgehog normalized, V+1): nc=1 -> 4*12*13 = 624; nc=2 -> 1248.
Easy regime (data_ext: kv<=1024, vocab 8192, batch 64). 2 runs, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for nc in (1, 2):
    kw = rola_instance("rola-hedgehog-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
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
            logger=LoggerConfig(project_name="rola-hh-lownc", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"rola-hh-sym-nc{nc}-d12_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 2


def load_configs_and_envs():
    return configs, configs_envs
