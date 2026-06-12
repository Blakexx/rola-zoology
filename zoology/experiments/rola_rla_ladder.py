"""Stage 1 of the staged sweep plan: the RoLA-RLA ladder — CANONICAL CELL (kappa-asym).

The headline state-scaling curve for the paper: RoLA-RLA with asymmetric routing
(independent read/write routers — the canonical RoLA form, what reviewers care about and
what LM runs use) + kappa normalization (which closes the global-norm asym tail gap:
kv1024 g_asym 0.885 -> k_asym 0.946 = k_sym), across the full nc ladder at d_qk=d_v=12.
The earlier g_sym ladder (same file, ladder-*_g_sym rows) is kept as the asym~sym
equivalence section. CLA_MEASURE_RANK=1: every run also emits the DISTRIBUTIONAL
realized-rank diagnostic (per-slice dists + spectrum quantiles, unmasked object).
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
        kw = rola_instance("rola-rla-kappa-asym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-rla-ladder", entity=""),
            max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
            run_id=f"ladder-nc{nc}-d12dv12_k_asym_lr{LR:.0e}_s{seed}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "10", "CLA_MEASURE_RANK": "1"})

assert len(configs) == 8, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
