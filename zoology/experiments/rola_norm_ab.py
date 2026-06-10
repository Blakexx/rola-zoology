"""Normalization A/B: GLOBAL vs PER-STATE norm, crossed with sym/asym, one code version.

Tests the hypothesis that the normalization "fix" (single global partition fn) is what
regressed RoLA and opened the sym/asym gap — vs per-state self-normalization (each routed
state normalized before the read-combine). All four arms run in ONE sweep on the current
rola.py, so the only differences are state_norm and tie_routers.

Decisive cell: nc=64 (where the global-norm sym/asym gap was largest, ~0.15). Add nc=16 for
the spread. 2 norms x 2 sym x 2 nc x 1 seed = 8 runs. Everything else copied from the
crossover: d_qk=d_v=12, n_heads=4, lr=1e-2, 40 epochs, TEST_BS=8. EVAL_EVERY_N=10 for speed.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEEDS = [1337]
NCS = [16, 64]
TEST_BS = 8
ARMS = [("g_sym", "rola-rla-sym"), ("g_asym", "rola-rla-asym"),
        ("ps_sym", "rola-rla-sym-ps"), ("ps_asym", "rola-rla-asym-ps")]

configs, configs_envs = [], []
for seed in SEEDS:
    for nc in NCS:
        for tag, inst in ARMS:
            data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                              batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext")
            kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
            kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
            configs.append(TrainConfig(
                data=data,
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-norm-ab", entity=""),
                max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"normAB-nc{nc}-d12dv12_{tag}_lr{LR:.0e}_s{seed}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 8, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
