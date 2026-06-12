"""Completes the norm x symmetry grid across the full ladder: the 22 cells not covered by
the banked g_sym / k_asym / ps_sym ladders and the nc=256 five-way (kappa-norm sweep).
Missing: g_asym + k_sym at nc 2..128, ps_asym at nc 2..256. Same protocol as the ladders
(d12/dv12, lr 1e-2, seed 1337, 40 epochs); one paper table = one protocol."""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR, SEED, TEST_BS = 1e-2, 1337, 8
ARMS = ([("g_asym", "rola-rla-asym", nc) for nc in (2, 4, 8, 16, 32, 64, 128)]
        + [("k_sym", "rola-rla-kappa-sym", nc) for nc in (2, 4, 8, 16, 32, 64, 128)]
        + [("ps_asym", "rola-rla-asym-ps", nc) for nc in (2, 4, 8, 16, 32, 64, 128, 256)])

configs, configs_envs = [], []
for tag, inst, nc in ARMS:
    kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(TrainConfig(
        data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                        batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="rola-normsym-grid", entity=""),
        max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=f"ladder-nc{nc}-d12dv12_{tag}_lr{LR:.0e}_s{SEED}",
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 22, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
