"""Confirmation sweep: more seeds around the (w=1.0, r=0.05) winner.

Goal: validate that the post-fix std sweep finding (peaky-writer + flat-reader is
the unique grok corner) is reproducible across more seeds, and probe immediate
neighbors to confirm the cliff structure.

Cells:
  - (w=1.0, r=0.05) — the WINNER: 9 new seeds (total 12 with prior 3)
  - (w=1.0, r=0.1)  — cliff just past winner on read axis: 6 new seeds
  - (w=0.5, r=0.05) — cliff below winner on write axis: 6 new seeds
  - (w=1.0, r=0.01) — even flatter reader: 6 seeds (new cell)
  - (w=1.5, r=0.05) — even peakier writer: 6 seeds (new cell)

Same task/model as std_sweep_post_fix.py:
  vocab=512, seq_len=128, kv=8, d_model=128, lr=2e-3, 24 epochs, batch=128.
"""
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 512
SEQ_LEN = 128
NUM_KV = 8
TRAIN_EXAMPLES = 30_000
D_MODEL = 128
BATCH = 128
LR = 2e-3

# Original sweep used {1337, 42, 7}. Use disjoint seeds here so we can pool.
NEW_SEEDS = [0, 1, 2, 3, 11, 99, 123, 456, 789]

CELLS = [
    # (w_std, r_std, n_seeds_to_run)
    (1.0,  0.05, 9),  # winner — full 9 new seeds
    (1.0,  0.1,  6),  # cliff (read axis)
    (0.5,  0.05, 6),  # cliff (write axis)
    (1.0,  0.01, 6),  # NEW: even flatter reader
    (1.5,  0.05, 6),  # NEW: even peakier writer
]


def cla_gla_asym():
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(
            d_qk=7, d_v=8, num_chunks=8, n_heads=4,
            writer="softmax_gla", reader="softmax_linear",
            tie_routers=False,
        ),
    )


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

configs = []
for w_std, r_std, n_seeds in CELLS:
    seeds = NEW_SEEDS[:n_seeds]
    for seed in seeds:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=cla_gla_asym(),
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL,
                    n_layers=2,
                    max_position_embeddings=0,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="cla-gla-stdsweep-confirm", entity=""),
                max_epochs=24,
                learning_rate=LR,
                weight_decay=0.0,
                seed=seed,
                run_id=f"confirm_w{w_std}_r{r_std}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
