"""State-extremes LR check for the OTHER crossover kernels (RLA done in
rola_lr_statecheck). Tests whether each kernel's routed LR holds across the nc
ladder by checking nc=3 (~2k) and nc=256 (~high) at its routed winner +/- 1 notch.
Measures the LR x state interaction PER KERNEL (it's kernel-dependent — GDN shifted
2 notches monolith->routed, RLA only 1). Per-nc test batch. data_ext, 40ep, EVAL_EVERY_N=5.
4 kernels x 2 nc x 3 LR = 24 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

# (kernel, d_qk, d_v, [LRs around routed winner])
KERNELS = [("rola-gla-sym",      12, 12, [3e-3, 1e-2, 3e-2]),
           ("rola-hedgehog-sym", 12, 12, [3e-4, 1e-3, 3e-3]),
           ("rola-based-sym",     4, 12, [3e-3, 1e-2, 3e-2]),
           ("rola-rebased-sym",   5, 12, [1e-3, 3e-3, 1e-2])]
NC_TB = [(3, 64), (256, 8)]
SEED = 1337
configs, configs_envs = [], []
for k, dqk, dv, LRS in KERNELS:
    for nc, tb in NC_TB:
        data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                          batch_size=(128, tb), cache_dir="/tmp/zoology_cache_rwext")
        for lr in LRS:
            kw = rola_instance(k, d_qk=dqk, d_v=dv, num_chunks=nc, n_heads=4)
            kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
            configs.append(TrainConfig(
                data=data,
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-lr-statecheck2", entity=""),
                max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
                run_id=f"lrs2-{k}-nc{nc}_lr{lr:.0e}_s{SEED}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 24


def load_configs_and_envs():
    return configs, configs_envs
