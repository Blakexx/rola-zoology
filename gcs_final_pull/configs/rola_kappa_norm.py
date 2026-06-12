"""KAPPA normalization MQAR test: does the learned mass-compensation exponent close the
global-norm sym/asym gap?

Mechanism under test: global norm mixes states by r·D (read gate x accumulated mass), so under
SYM routing heavily-written states are automatically heavily-read — mass bias compounds, and
asymmetry is needed to decouple. kappa rescales r by (D+eps)^{-kappa(x)} (exact global at kappa=0,
exact per-state at kappa=1, learned per token/head, init ~global). Prediction: kappa-sym recovers
most of the g_asym - g_sym gap; ps_sym shouldn't show the gap at all.

Arms: g_sym, g_asym (the gap), ps_sym (no-mass control), k_sym (the test), k_asym (interaction).
All on the CURRENT verified kernel stack (fla_rola Triton; kappa rides the same kernel on
modified gates). 5 arms x nc {16,64} x 1 seed = 10 runs. Config otherwise = rola_norm_ab.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEEDS = [1337]
NCS = [256]   # biggest observed divergence is at the highest n_s; widen only if inconclusive
TEST_BS = 8
ARMS = [("g_sym", "rola-rla-sym"), ("g_asym", "rola-rla-asym"),
        ("ps_sym", "rola-rla-sym-ps"),
        ("k_sym", "rola-rla-kappa-sym"), ("k_asym", "rola-rla-kappa-asym")]

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
                logger=LoggerConfig(project_name="rola-kappa-norm", entity=""),
                max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"kappa-nc{nc}-d12dv12_{tag}_lr{LR:.0e}_s{seed}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 5, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
