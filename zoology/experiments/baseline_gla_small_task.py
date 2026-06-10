"""Baseline GLA at the same task as std_sweep_confirm to set the bar.

Same task: vocab=512, kv=8, d_model=128, lr=2e-3, 24 epochs. 5 seeds.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]

data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                              num_examples=TRAIN_EXAMPLES, num_kv_pairs=NUM_KV,
                              random_non_queries=False)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                             num_examples=2_000, num_kv_pairs=NUM_KV,
                             random_non_queries=False)],
    batch_size=(BATCH, BATCH // 4),
    cache_dir="/tmp/zoology_cache_stdsweep",
)

# Baseline Zoology GLA (no V+1)
def gla_mixer():
    return ModuleConfig(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs=dict(num_heads=4),
    )

# Our V+1-normalized in-house GLA wrapper (apples-to-apples vs CLA-GLA)
def gla_norm():
    return ModuleConfig(
        name="zoology.mixers.cla.RecurrentGLA",
        kwargs=dict(d_qk=16, d_v=32, n_heads=4),
    )

MIXERS = [
    ("baseline-gla", gla_mixer()),
    ("inhouse-gla-norm", gla_norm()),
]

configs = []
for name, mixer in MIXERS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=0, vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="baseline-gla-smalltask", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
