"""RoLA-RLA crossover sweep, 2k -> 160k state, under the GLOBAL-norm fused path,
WITH realized effective-attention rank measurement (CLA_MEASURE_RANK=1).

This re-runs the canonical crossover (paper Fig./Table, was populated to nc=64)
across the full nc ladder, now with:
  - global normalization (paper's Eq. rola_output; the old per-state divide was a bug)
  - the fused chunk-parallel path (~2x faster + ~2x less mem than virtual-head)
  - per-eval-epoch realized rank logged as RANK_JSON (num_rank, eff_rank), so we can
    plot realized rank vs nc against the nc*d_qk law and the d_model cap.

State = nc * n_heads(4) * d_qk(12) * (d_v(12)+1) = nc * 624:
  nc:    3      4      8      16     32     64     128    256
  state: 1872   2496   4992   9984   19968  39936  79872  159744
  (~2k   ~2.5k  ~5k    ~10k   ~20k   ~40k   ~80k   ~160k)

Easy regime (data_ext: kv<=1024, vocab 8192, route_on='x', dense symmetric, dv=12).
Single seed 1337, lr 1e-2, 40 epochs. Test batch scaled down at high nc so the
seq-4096 (kv=1024) eval fits. Train batch 128 (fused path has no CUDA grid limit).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEED = 1337
# nc ladder spanning ~2k -> ~160k state; per-nc test batch (smaller at high nc).
NC_TESTBATCH = [(3, 64), (4, 64), (8, 64), (16, 64), (32, 48), (64, 32), (128, 16), (256, 8)]

configs = []
configs_envs = []
for nc, test_bs in NC_TESTBATCH:
    data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, test_bs), cache_dir="/tmp/zoology_cache_rwext")
    kw = rola_instance("rola-rla-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="rola-rla-crossover", entity=""),
            max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"rola-nc{nc}-d12-dv12_sym_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({"CLA_MEASURE_RANK": "1"})

assert len(configs) == 8


def load_configs_and_envs():
    return configs, configs_envs
