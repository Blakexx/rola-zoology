"""Rank re-measurement with the per-slice, multi-tolerance diagnostic.

Tests whether the apparent rank saturation at d_model is real or an artifact of (a) the
1e-2 SVD threshold and (b) measuring on easy MQAR slices. The patched _effective_attention_rank
reports num_rank at {1e-1,1e-2,1e-3,1e-4} + eff_rank + sv decay, once PER (epoch, seq-len) — so
each MQAR slice (kv=4..1024, different L) gets its own rank, including the hard kv=1024 slice
where the task demands high rank. If rank exceeds d_model on the hard slice / finer tol, routing
genuinely breaks the barrier; if it caps at d_model everywhere, the ceiling is real.

nc in {32,64,128,256} (above the crossover), sym, 1 seed, CLA_MEASURE_RANK=1, 40 epochs. T4.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEED = 1337
NCS = [32, 64, 128, 256]
TEST_BS = 8

configs, configs_envs = [], []
for nc in NCS:
    data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext")
    kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(TrainConfig(
        data=data,
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="rola-t4-rank", entity=""),
        max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=f"rankHS-nc{nc}-d12dv12_sym_lr{LR:.0e}_s{SEED}",
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    configs_envs.append({"EVAL_EVERY_N": "10", "CLA_MEASURE_RANK": "1"})

assert len(configs) == 4, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
