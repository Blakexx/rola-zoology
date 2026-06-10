"""RoLA-RLA at high nc (128, 256) — extends the existing curve (nc 8..64) to test
whether RoLA-RLA keeps scaling far past the d_model rank ceiling.
  nc=128 -> rank nc*d_qk = 1536, state 4*128*12*13 = 79872 (~80k)
  nc=256 -> rank 3072,            state 159744 (~160k)
(d_model=128, so these are 12x / 24x past the per-head rank ceiling.)

Same easy regime as the nc32/64 runs (data_ext: kv<=1024, vocab 8192, route_on='x',
d_qk=12, d_v=12, tied/symmetric), so directly comparable. Train batch 128 (the
baked head-chunking handles the CUDA grid limit at nc>=128); test batch 8 for the
seq-4096 eval at high nc. 2 runs, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEED = 1337
data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                  batch_size=(128, 8), cache_dir="/tmp/zoology_cache_rwext")

configs = []
configs_envs = []
for nc in (128, 256):
    kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="rola-rla-highnc", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"rola-nc{nc}-d12-dv12_sym_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 2


def load_configs_and_envs():
    return configs, configs_envs
