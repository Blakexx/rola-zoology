"""LR calibration (coarse) for the NEW/fixed inner kernels: faithful Hedgehog
(canonical FLA), Based (Taylor), ReBased — 5 LRs x 1 seed at nc=16, ~matched state
(~10-12k). The old Hedgehog LR (3e-3) was on the broken impl; Based/ReBased never
calibrated. d_qk chosen so feat_dim (state) is comparable: hedgehog d12->feat12,
based d4->feat15, rebased d5->feat15. data_ext, 40 epochs, EVAL_EVERY_N=5. 15 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

# Per-cell LR grids. Hedgehog/Based/ReBased: 5-pt around 1e-2/3e-3.
# GDN: its Stage-A optimum (1e-3) sat at the grid BOUNDARY and was monotone-decreasing
# in LR, so extend DOWN (1e-4, 3e-4) to actually bracket it before judging GDN's ceiling.
CELLS = [("rola-hedgehog-sym", 12, [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
         ("rola-based-sym", 4, [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
         ("rola-rebased-sym", 5, [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
         ("rola-gdn-sym", 12, [1e-4, 3e-4, 1e-3, 3e-3])]
SEED = 1337
configs, configs_envs = [], []
for k, dqk, LRS in CELLS:
    for lr in LRS:
        kw = rola_instance(k, d_qk=dqk, d_v=12, num_chunks=16, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data_ext,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-lr-kernels", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"lrk-{k}-nc16d{dqk}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 19   # 3x5 (hh/based/rebased) + 4 (gdn low-LR)


def load_configs_and_envs():
    return configs, configs_envs
