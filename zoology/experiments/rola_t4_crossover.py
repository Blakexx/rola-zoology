"""RoLA-RLA crossover on T4 — the KNOWN-GOOD cells (pure-torch fused kernel, smoke-
confirmed on sm75 incl nc=256 @ seq4096). The clean re-run on the current impl, now
MULTI-SEED for the variance story. Monolith baselines (wide d768 / GDN — untested T4
memory) are deferred to a later launch.

8-state nc ladder x 3 seeds = 24 runs. d_qk=d_v=12, sym routing, route_on='x', lr=1e-2
(RoLA LR is nc-invariant, confirmed). T4 tuning: EVAL_EVERY_N=5 (eval is the wall-clock
cost on T4); rank diagnostic on seed 1337 only. Per-nc test batch shrinks at high nc so
the seq-4096 (kv=1024) eval fits 16GB.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEEDS = [1337, 2024, 7]
# T4 16GB: the seq-4096 eval logits are batch*4096*vocab(8192)*4B, independent of nc.
# batch 64 -> 8.6GB -> OOM (the A100-sized batches did this). batch 8 -> ~1GB, smoke-
# confirmed at nc=256. Cap test batch at 8 for ALL cells (binding slice is seq4096).
NC_LADDER = [3, 4, 8, 16, 32, 64, 128, 256]
TEST_BS = 8

configs, configs_envs = [], []
for seed in SEEDS:
    for nc, test_bs in [(nc, TEST_BS) for nc in NC_LADDER]:
        data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                          batch_size=(128, test_bs), cache_dir="/tmp/zoology_cache_rwext")
        kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-t4-crossover", entity=""),
            max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
            run_id=f"rolaT4-nc{nc}-d12dv12_sym_lr{LR:.0e}_s{seed}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        env = {"EVAL_EVERY_N": "5", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        if seed == 1337:
            env["CLA_MEASURE_RANK"] = "1"   # rank diagnostic once, not every seed
        configs_envs.append(env)

assert len(configs) == 24, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
