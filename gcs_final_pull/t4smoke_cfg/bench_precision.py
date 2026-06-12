"""torch.compile vs fp16-AMP vs eager-fp32 — independent A/B on T4.

Three IDENTICAL nc=16 crossover cells, differing only by env, run to convergence so the
same run gives BOTH (a) the it/s speedup (from train logs; use STEADY-STATE it/s, not
epoch 0 — torch.compile pays a one-time compile cost on the first steps) and (b) the
fp16 science-validation (does AMP-fp16 match eager-fp32 on final hard-slice acc + grok
epoch + the per-epoch curve). If fp16 diverges from fp32 here, fp16 is unsafe for the sweep.

  fp32     : eager fp32 (current default)
  compile  : torch.compile, fp32       (COMPILE=1)
  fp16     : autocast(fp16) + GradScaler (AMP=fp16)

Needs the rebuilt image (train.py COMPILE/AMP hooks are baked, not GCS-refreshed).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR, SEED, NC, EP, TEST_BS = 1e-2, 1337, 16, 35, 8


def _cell(tag):
    data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext")
    kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=NC, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    return TrainConfig(
        data=data,
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="bench-precision", entity=""),
        max_epochs=EP, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=f"bench-nc{NC}-{tag}", early_stopping_threshold=2.0,
        early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"])


configs = [_cell("fp32"), _cell("compile"), _cell("fp16")]
configs_envs = [
    {"EVAL_EVERY_N": "5"},
    {"EVAL_EVERY_N": "5", "COMPILE": "1"},
    {"EVAL_EVERY_N": "5", "AMP": "fp16"},
]
assert len(configs) == 3


def load_configs_and_envs():
    return configs, configs_envs
