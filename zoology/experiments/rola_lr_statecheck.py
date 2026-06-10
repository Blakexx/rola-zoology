"""High-vs-low STATE LR verification (the cheap alternative to per-state sweeps).
Tests whether RLA's calibrated LR (1e-2) stays optimal at the EXTREMES of the state
ladder — nc=3 (~2k) and nc=256 (~160k) — at LR in {3e-3, 1e-2, 3e-2} (winner +/- a
notch). If 1e-2 wins (or ties) at BOTH extremes, optimal LR is state-invariant along
the nc axis (as theory predicts: RoLA shares projections, d_model fixed) and one LR
per kernel reuses across the whole ladder. Per-nc test batch (nc=256 eval is heavy).
data_ext, 40ep, EVAL_EVERY_N=5. 6 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

NC_TB = [(3, 64), (256, 8)]          # (nc, test_batch): low ~2k / high ~160k
LRS = [3e-3, 1e-2, 3e-2]
SEED = 1337
configs, configs_envs = [], []
for nc, tb in NC_TB:
    data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, tb), cache_dir="/tmp/zoology_cache_rwext")
    for lr in LRS:
        kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-lr-statecheck", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"lrstate-rla-nc{nc}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
