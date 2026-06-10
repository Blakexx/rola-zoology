"""nc=256 rank cell, split out to its own region (the 4-shard/3-region round-robin doubled
up us-central1, leaving nc=256 queued behind nc=32). Same run_id as rola_t4_rank_hardslice's
nc=256 so it writes to the same results file. Per-slice multi-tolerance rank diagnostic."""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2; SEED = 1337; nc = 256; TEST_BS = 8
data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                  batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext")
kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
configs = [TrainConfig(
    data=data,
    model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                      state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                      d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
    logger=LoggerConfig(project_name="rola-t4-rank", entity=""),
    max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED,
    run_id=f"rankHS-nc{nc}-d12dv12_sym_lr{LR:.0e}_s{SEED}",
    early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
    slice_keys=["num_kv_pairs"])]
configs_envs = [{"EVAL_EVERY_N": "10", "CLA_MEASURE_RANK": "1"}]


def load_configs_and_envs():
    return configs, configs_envs
