"""Memory smoke test for the rank-scaling sweep: the single worst-case run.

rola-gla-sym at nc=512 (2048 virtual heads) is the memory-tightest config — the
GLA writer materializes q/k/g (each [B,L,H*nc,d_qk]) + v, and the extended test
set evaluates to seq=16384 (kv=4096). If this fits and trains, every other run in
rola_rank_scaling (lower nc, or GDN's scalar gate, or the RLA writer with no g) is
lighter. Run this ALONE first; if it OOMs, drop test batch to 2 or cap nc at 256.

1 run, full extended-kv eval, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL
from zoology.experiments.rola_rank_scaling import make_data, D_V, D_QK, LR, SEED, VOCAB_RANK
from rola import rola_instance

kw = rola_instance("rola-gla-sym", d_qk=D_QK, d_v=D_V, num_chunks=256, n_heads=4)
kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)

configs = [
    TrainConfig(
        data=make_data(256),
        model=ModelConfig(
            block_type="TransformerBlock",
            sequence_mixer=wrap_hybrid(kernel),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL, n_layers=2,
            max_position_embeddings=0,
            vocab_size=VOCAB_RANK,
        ),
        logger=LoggerConfig(project_name="rola-rank-smoke", entity=""),
        max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=f"smoke-rola-gla-sym-nc256-d{D_QK}_lr{LR:.2e}_s{SEED}",
        early_stopping_threshold=2.0,
        early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"],
    )
]
configs_envs = [{}]
assert len(configs) == 1


def load_configs_and_envs():
    return configs, configs_envs
