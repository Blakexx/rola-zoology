"""T4 16GB MEMORY smoke for the heavy baseline cells NOT covered by t4_smoke.
seq4096 eval at conservative test batch. 1 epoch (testing it FITS, not accuracy).
Wide RLA monolith uses RecurrentLinearAttention (FLA, memory-efficient) per the
#55 OOM fix; the fused chunked path materializes [bh,L,d_qk,d_v] and would OOM at
d_qk=768. If any cell OOMs here, drop its test batch or send it to A100."""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from zoology.experiments.gdn_sweep import gdn_baseline

SEED = 1337


def _data(tb):
    return DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                      batch_size=(128, tb), cache_dir="/tmp/zoology_cache_rwext")


def _cfg(rid, mixer, tb):
    return TrainConfig(
        data=_data(tb),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(mixer),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="t4-mem-smoke", entity=""),
        max_epochs=1, learning_rate=1e-2, weight_decay=0.0, seed=SEED, run_id=rid,
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"])


def rla_mono(d_qk, d_v):  # memory-efficient FLA recurrent path (the #55 fix)
    return dict(name="rola.RecurrentLinearAttention",
                kwargs={"d_qk": d_qk, "d_v": d_v, "n_heads": 4})


configs = [
    _cfg("t4mem-rla-wide-d768",  rla_mono(768, 12), tb=8),   # ~40k state, wide — the OOM risk
    _cfg("t4mem-rla-square-d99", rla_mono(99, 99),  tb=8),   # ~40k state, square
    _cfg("t4mem-gdn-d48",        gdn_baseline(48),  tb=8),   # FLA delta-rule @ seq4096
]
configs_envs = [{"EVAL_EVERY_N": "1"} for _ in configs]
assert len(configs) == 3


def load_configs_and_envs():
    return configs, configs_envs
