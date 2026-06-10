"""Does tied-init fix asym under GLOBAL norm? (cheap fix vs per-state)

asym's deficit under global norm is an optimization failure (can't reach r=w through the
coupled landscape). Tied-init starts read==write so asym begins in the sym basin, then is
free to specialize. If it closes the gap to sym, global+asym becomes viable -> keep the
fused kernel, skip the 2.5-4x per-state cost.

global norm, lr=1e-2 (calibrated). {asym-tieinit, asym-random} x {nc16, nc64} x 1 seed = 4.
Compare to this session's global sym (nc16 0.203, nc64 0.781) and asym-random (0.185, 0.701).
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEED = 1337
TEST_BS = 8
ARMS = [("tieinit", "rola-rla-asym-tieinit"), ("random", "rola-rla-asym")]

configs, configs_envs = [], []
for nc in (16, 64):
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
            logger=LoggerConfig(project_name="rola-tieinit-test", entity=""),
            max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"tieinit-nc{nc}-asym_{tag}_lr{LR:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 4, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
