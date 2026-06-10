"""High-state Hedgehog, CORRECT LR (3e-3 from calibration; the old 1e-2 mis-tuned
Hedgehog by ~12pts). Matched-state comparison of FOUR ways to spend the state
budget, all phi=hedgehog, at ~20k and ~40k:
  rola   : routing (scale nc), 4 heads, shared projection   -> nc 32/64, d12 v12
  square : monolith d_qk=d_v                                -> d 68/96
  num_kq : monolith, scale d_qk only (wide)                 -> d_qk 384/768, d_v 12
  heads  : scale n_heads (separate projections), nc=1       -> H 128/256, d12 v12
State (Hedgehog normalized V+1) = n_heads*nc*d_qk*(d_v+1):
  20k pairs: rola 19968 | square 18768 | wide 19968 | heads 19968
  40k pairs: rola 39936 | square 37248 | wide 39936 | heads 39936
data_ext, 40 epochs, lr 3e-3, seed 1337, EVAL_EVERY_N=5. Per-arch test batch
(heads folds H into batch -> small test batch to bound memory). 8 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR, SEED = 3e-3, 1337
# (arch, n_heads, nc, d_qk, d_v, test_batch)
SPECS = [
    ("rola",   4,  32, 12, 12, 32), ("rola",   4,  64, 12, 12, 32),
    ("square", 4,   1, 68, 68, 32), ("square", 4,   1, 96, 96, 32),
    ("numkq",  4,   1, 384, 12, 32), ("numkq",  4,   1, 768, 12, 32),
    ("heads",  128, 1, 12, 12, 8),  ("heads",  256, 1, 12, 12, 8),
]
configs, configs_envs = [], []
for arch, H, nc, dqk, dv, tb in SPECS:
    data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, tb), cache_dir="/tmp/zoology_cache_rwext")
    kw = rola_instance("rola-hedgehog-sym", d_qk=dqk, d_v=dv, num_chunks=nc, n_heads=H)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    state = H * nc * dqk * (dv + 1)
    configs.append(TrainConfig(
        data=data,
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="rola-hh-scaling", entity=""),
        max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=f"hhscale-{arch}-h{H}nc{nc}d{dqk}v{dv}_st{state}_lr3e-03_s{SEED}",
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 8


def load_configs_and_envs():
    return configs, configs_envs
