"""Norm A/B at high nc, LR-CONTROLLED, on A100 (per-state OOMs at nc64 on a 12GB local card).

Answers two things at once:
  1. Does PER-STATE norm collapse the global sym/asym gap (which grows with nc:
     +0.018 @nc16 -> +0.080 @nc64 local / +0.15 @nc64 cloud-3seed)?
  2. Is that gap an LR artifact? -> sweep LR per arm and read off best-LR-per-arm, so the
     global-vs-per-state and sym-vs-asym comparisons are not LR-confounded.

nc=64 x {global,per-state} x {sym,asym} x LR{3e-3,1e-2,3e-2} x 1 seed = 12 runs.
Everything else matched to the crossover: d_qk=d_v=12, n_heads=4, 40 epochs, batch(128,8).
A100 (40GB) holds per-state at nc64 batch128. EVAL_EVERY_N=5.
"""
import sys
sys.path.insert(0, '/workspace')                       # cloud image path
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')  # local fallback
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

SEED = 1337
NC = 64
LRS = [3e-3, 1e-2, 3e-2]
TEST_BS = 8
ARMS = [("g_sym", "rola-rla-sym"), ("g_asym", "rola-rla-asym"),
        ("ps_sym", "rola-rla-sym-ps"), ("ps_asym", "rola-rla-asym-ps")]

configs, configs_envs = [], []
for lr in LRS:
    for tag, inst in ARMS:
        data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                          batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_normab")
        kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=NC, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-norm-ab-a100", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"normABa100-nc{NC}-{tag}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})

assert len(configs) == 12, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
