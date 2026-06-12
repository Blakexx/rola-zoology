"""T4 (sm75) viability smoke test — does the sweep run on a $0.24/hr T4 (16 GB)?

Three cells, 1 epoch each (we are testing EXECUTION + MEMORY on Turing sm75, not
grokking), ordered safe -> risky so the log tells us exactly where the boundary is:

  1. rola-rla nc=256 @ seq4096  -> PURE-TORCH kernel (_rola_chunked_parallel) +
     highest-state / longest-eval cell. Tests 16 GB headroom. Expected: PASS.
  2. gla-mono (FLA chunk_gla)   -> the real unknown: do FLA's Triton kernels run on
     sm75? If this dies with a Triton/CUDA-capability error, ALL FLA baselines
     (RLA/GLA/GDN monoliths) must go to A100.
  3. mha @ seq4096 (batch 16)   -> O(L^2) attention, the OOM-boundary cell on 16 GB.

Run on T4 via: submit.py --gpu t4 --runner t4_smoke --num-shards 1 ...
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, mha, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from zoology.experiments.gla_sweep import gla_baseline
from rola import rola_instance

LR = 1e-2
SEED = 1337
EP = 1  # smoke: one epoch, just exercise fwd/bwd + the seq4096 eval slice


def _data(test_bs):
    return DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, test_bs), cache_dir="/tmp/zoology_cache_rwext")


def _cfg(run_id, mixer, test_bs, n_layers=2):
    return TrainConfig(
        data=_data(test_bs),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(mixer),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=n_layers, max_position_embeddings=0,
                          vocab_size=VOCAB),
        logger=LoggerConfig(project_name="t4-smoke", entity=""),
        max_epochs=EP, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=run_id, early_stopping_threshold=2.0,
        early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"])


# 1. pure-torch RoLA at the heaviest state/seq (small eval batch like the crossover)
rola_nc256 = dict(name="zoology.mixers.cla.ChunkedLinearAttention",
                  kwargs=rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=256, n_heads=4))

configs = [
    _cfg("t4smoke-rola-rla-nc256", rola_nc256, test_bs=8),
    _cfg("t4smoke-gla-mono-d48", gla_baseline(48), test_bs=16),
    _cfg("t4smoke-mha-nl2", mha, test_bs=16),
]
configs_envs = [{"EVAL_EVERY_N": "1"} for _ in configs]
assert len(configs) == 3


def load_configs_and_envs():
    return configs, configs_envs
