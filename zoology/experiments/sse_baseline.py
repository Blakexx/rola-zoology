"""Faithful SSE baseline at matched state vs the kappa-norm nc=256 cells.

Matched per-head state: RoLA nc=256 d12/dv12 -> nc*dqk*(dv+1) = 39936 floats.
SSE: N*c*dh = 39936 at dh=12 -> N*c = 3328. Two shapes (few-big vs many-small partitions),
top-1 and top-2 (+always-on partition), 2 LRs (per-arch tuning protocol). Lambda=I.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS

SEED = 1337
# Paper-faithful shapes: expand N, keep c near their base (~128). N*c*dh = 39936 @ dh=12.
ARMS = [("N26c128_t1", dict(num_partitions=26, num_rows=128, topk=1)),
        ("N26c128_t2", dict(num_partitions=26, num_rows=128, topk=2)),
        ("N16c208_t1", dict(num_partitions=16, num_rows=208, topk=1))]
LRS = [1e-2, 3e-3]

configs, configs_envs = [], []
for lr in LRS:
    for tag, kw in ARMS:
        data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                          batch_size=(128, 8), cache_dir="/tmp/zoology_cache_rwext")
        kernel = dict(name="zoology.mixers.sse.SSE",
                      kwargs=dict(n_heads=4, d_head=12, always_on=True, **kw))
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="sse-baseline", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"sse-{tag}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 6, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
