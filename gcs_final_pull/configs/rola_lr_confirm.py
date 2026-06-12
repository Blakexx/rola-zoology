"""LR Stage B (confirmation, seed-robust, multi-region): narrow grid around each
cell's Stage-A winner, 3 seeds, nc=16. RLA & GLA peaked at 1e-2; Hedgehog at 3e-3
(so sweep its lower neighborhood 1e-3->1e-2). Pick final LR by mean over seeds.
data_ext, 40 epochs, EVAL_EVERY_N=5. 21 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

CELLS = [("rola-rla-sym", [3e-3, 1e-2]),
         ("rola-gla-sym", [3e-3, 1e-2]),
         ("rola-hedgehog-sym", [1e-3, 3e-3, 1e-2])]
SEEDS = [1337, 1, 2]
configs, configs_envs = [], []
for k, lrs in CELLS:
    for lr in lrs:
        for s in SEEDS:
            kw = rola_instance(k, d_qk=12, d_v=12, num_chunks=16, n_heads=4)
            kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
            configs.append(TrainConfig(
                data=data_ext,
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-lr-confirm", entity=""),
                max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=s,
                run_id=f"lrconf-{k}-nc16_lr{lr:.0e}_s{s}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 21


def load_configs_and_envs():
    return configs, configs_envs
