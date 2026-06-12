"""LR calibration STAGE A (coarse): 4 RoLA inner kernels x 5 LRs, 1 seed, nc=16
(~10k state, the learning regime). Full data_ext test set (hard kv slices are what
discriminate LR). 40 epochs, EVAL_EVERY_N=5. Winner per cell feeds Stage B (fine,
3 LR x 3 seeds around it). No rank measurement (not needed for LR ranking).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

KERNELS = ["rola-rla-sym", "rola-gla-sym", "rola-hedgehog-sym", "rola-gdn-sym"]
LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SEED = 1337
configs, configs_envs = [], []
for k in KERNELS:
    for lr in LRS:
        kw = rola_instance(k, d_qk=12, d_v=12, num_chunks=16, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data_ext,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-lr-probe-a", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"lrprobe-{k}-nc16_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 20


def load_configs_and_envs():
    return configs, configs_envs
