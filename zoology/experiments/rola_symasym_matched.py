"""Sym-vs-asym matched controlled comparison — BOTH arms in ONE sweep, one code version.

Motivation: the prior overlay compared SYM (from rola_t4_crossover2, run earlier) against
ASYM (from rola_t4_symasym, run later). rola.py is untracked, so code drift between those two
runs could not be ruled out, and the gap looked much wider than the prior "sym ~= asym" result.
This config runs sym AND asym side by side, same local rola.py, same process, same seeds, so
the ONLY difference is tie_routers. If the gap vanishes -> the cloud gap was drift. If it holds
-> the asymmetry penalty is real on reciprocal MQAR.

Everything except tie_routers is copied verbatim from rola_t4_crossover / rola_t4_symasym:
d_qk=d_v=12, n_heads=4, route_on='x', lr=1e-2, 40 epochs, TEST_BS=8, EVAL_EVERY_N=5.
nc in {16,32,64} (the informative crossover region) x {sym,asym} x 2 seeds = 12 runs.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEEDS = [1337, 2024]
NCS = [16, 32, 64]
TEST_BS = 8

configs, configs_envs = [], []
for seed in SEEDS:
    for nc in NCS:
        for tag, inst in [("sym", "rola-rla-sym"), ("asym", "rola-rla-asym")]:
            data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                              batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext")
            kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
            kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
            configs.append(TrainConfig(
                data=data,
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-symasym-matched", entity=""),
                max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"symasymM-nc{nc}-d12dv12_{tag}_lr{LR:.0e}_s{seed}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "5"})

assert len(configs) == 12, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
