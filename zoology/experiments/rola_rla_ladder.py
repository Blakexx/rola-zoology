"""Stage 1 of the staged sweep plan: the RoLA-RLA ladder.

The headline state-scaling curve for the paper: RoLA-RLA at the default norm config
(global + sym, the paper default) across the full nc ladder at fixed d_qk=d_v=12.
State per layer-head = nc * (d_qk*d_v + d_qk) ; nc 2 -> 256 spans ~26x state.

Single seed / single LR (1e-2, calibrated for RLA) by design: Stage 1 gets the RoLA
numbers fast while the MQAR task definition could still change. Stage 2 adds the best
competitor (Based/Hedgehog); Stage 3 freezes the task and runs the massive multi-seed
baseline grid. Config otherwise identical to rola_kappa_norm (same data, 40 epochs).
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEEDS = [1337]
NCS = [2, 4, 8, 16, 32, 64, 128, 256]
TEST_BS = 8

configs, configs_envs = [], []
for seed in SEEDS:
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
            logger=LoggerConfig(project_name="rola-rla-ladder", entity=""),
            max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
            run_id=f"ladder-nc{nc}-d12dv12_g_sym_lr{LR:.0e}_s{seed}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 8, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
