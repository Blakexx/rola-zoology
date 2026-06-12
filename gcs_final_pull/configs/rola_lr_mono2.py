"""Monolith (nc=1) LR calibration for the feature-map kernels — the gap rola_lr_mono
missed (it did RLA/GLA/GDN only). Focused 3-LR grids around each kernel's routed
winner (monolith optima cluster near 3e-3, so reusing routed LR would mis-tune).
Representative ~matched-state monoliths (~9-10k); based/rebased can't go square
(feature map expands quadratically) so they scale d_v instead. EVAL_EVERY_N=5. 9 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

# (kernel, d_qk, d_v, [LRs])  -- nc=1 monolith; state ~ 4*feat*(d_v+1)
CELLS = [("rola-hedgehog-sym", 48, 48, [1e-3, 3e-3, 1e-2]),   # feat=48, state 9408 (square)
         ("rola-based-sym",     8, 51, [3e-3, 1e-2, 3e-2]),   # feat=45, state 9408 (wide-value)
         ("rola-rebased-sym",   9, 51, [1e-3, 3e-3, 1e-2])]   # feat=45, state 9408
SEED = 1337
configs, configs_envs = [], []
for k, dqk, dv, LRS in CELLS:
    for lr in LRS:
        kw = rola_instance(k, d_qk=dqk, d_v=dv, num_chunks=1, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data_ext,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-lr-mono2", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"lrm2-{k}-mono-d{dqk}v{dv}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 9


def load_configs_and_envs():
    return configs, configs_envs
