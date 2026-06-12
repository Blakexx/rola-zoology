"""Follow-up: does router init_std rescue the stuck-seed lottery problem?

We saw 2/9 cla-gla-asym seeds at kv=16 hit 90%+, and 7/9 stuck at exactly 3.3%
(degenerate fixed point). Hypothesis: default nn.Linear init (~0.02 std) leaves
the writer router too close to uniform softmax → symmetry between chunks isn't
broken early enough → gradient flow stalls.

Test: peakier router init_std should escape the degenerate point more often.
Use the same setup as cla_gla_headtohead (kv=16, 50k examples) so results
are directly comparable.

Variants:
  - cla-gla-asym-w0.1-r0.1 (std=0.1 both, slight bump from default ~0.02)
  - cla-gla-asym-w0.3-r0.3 (std=0.3 matched, in-house Block A optimum)
  - cla-gla-asym-w0.3-r0.1 (asymmetric, in-house std_split winner)
  - cla-gla-asym-w0.1-r0.3 (asymmetric mirror)

3 lrs × 4 std variants × 3 seeds = 36 runs at the slightly-faster cell.
"""
import os
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8192
SEQ_LEN = 256
D_MODEL = 128
BATCH = 256
NUM_KV = 16
TRAIN_EXAMPLES = 50_000

SEEDS = [1337, 42, 7]
LRS = [4.64e-4, 2.15e-3, 1.0e-2]

STD_VARIANTS = [  # (label, write_std, read_std)
    ("w0.1_r0.1", 0.1, 0.1),
    ("w0.3_r0.3", 0.3, 0.3),
    ("w0.3_r0.1", 0.3, 0.1),
    ("w0.1_r0.3", 0.1, 0.3),
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


configs = []
for w_std, r_std in [(v[1], v[2]) for v in STD_VARIANTS]:
    label = next(v[0] for v in STD_VARIANTS if v[1] == w_std and v[2] == r_std)
    data = DataConfig(
        train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                                  num_examples=TRAIN_EXAMPLES, num_kv_pairs=NUM_KV,
                                  random_non_queries=False)],
        test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                                 num_examples=3_000, num_kv_pairs=NUM_KV,
                                 random_non_queries=False)],
        batch_size=(BATCH, BATCH // 8),
        cache_dir="/tmp/zoology_cache",
    )
    for lr in LRS:
        for seed in SEEDS:
            # We need the std env vars set when the worker spawns. They're read inside
            # _maybe_init_router(), called at module construction (i.e., inside the
            # subprocess's TrainConfig instantiation). Setting them via TrainConfig
            # isn't supported — we need the runner to set them in env per config.
            # Solution: patch the run_id and have the wrapper script extract std from it.
            cfg = TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=cla_gla_asym(),
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL,
                    n_layers=2,
                    max_position_embeddings=0,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="cla-gla-router-std", entity=""),
                max_epochs=32,
                learning_rate=lr,
                weight_decay=0.0,
                seed=seed,
                run_id=f"cla-gla-asym_{label}_lr{lr:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
            configs.append(cfg)
