"""RoLA-GLA: optimized SCALAR-gate (shared-gram) vs per-channel (virtual-head), and
NORMALIZED vs UN-NORMALIZED — LR calibration + nc-state invariance + accuracy probe.

Scalar gate enables the shared-gram fused path (ScalarGLAKernel/_rola_gla_chunked):
~4x faster / ~9x less mem than the per-channel virtual-head path, and it FINISHES
nc=256 (per-channel timed out @ep20). Normalization is a swept axis (GLA convention is
un-normalized, like FLA's GLA; '-norm-' adds the global V+1 partition fn).

Grid (d_qk=12, d_v=12, n_heads=4; data_ext, 40ep, EVAL_EVERY_N=5):
  scalar {unnorm, norm} x nc16 x {3e-3,1e-2,3e-2}     -> LR cal + norm comparison
  scalar {unnorm, norm} x {nc3,nc256} x {3e-3,1e-2}   -> nc-state invariance
  per-channel {unnorm, norm} x nc16 @1e-2             -> scalar-vs-virtualhead accuracy ref
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

SEED = 1337
# (instance, nc, test_batch, [LRs])
CELLS = []
for scalar in ("rola-gla-scalar-sym", "rola-gla-scalar-norm-sym"):
    CELLS += [(scalar, 16, 64, [3e-3, 1e-2, 3e-2]),
              (scalar,  3, 64, [3e-3, 1e-2]),
              (scalar, 256, 8, [3e-3, 1e-2])]
# per-channel virtual-head references (both norm settings) at nc16
CELLS += [("rola-gla-sym", 16, 64, [1e-2]),
          ("rola-gla-norm-sym", 16, 64, [1e-2])]

configs, configs_envs = [], []
for name, nc, tb, LRS in CELLS:
    data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, tb), cache_dir="/tmp/zoology_cache_rwext")
    for lr in LRS:
        kw = rola_instance(name, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-gla-scalar", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"glascalar-{name}-nc{nc}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 16, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
