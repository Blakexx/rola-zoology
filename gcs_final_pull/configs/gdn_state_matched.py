"""State-matched FLA-GDN baseline: bring baseline-gdn down from 524,288 state to
4,096 (matched to cla-gdn-bigstate at d_qk=8, d_v=16, c=8, h=4).

FLA's GatedDeltaNet defaults to head_dim=256, expand_v=2 → 524,288 state per layer.
Setting head_dim=32, expand_v=1, num_heads=4 → 4 × 32 × 32 = 4,096 state.

Four cells: {conv_on, conv_off} × {pos_emb=0, pos_emb=128}, 5 seeds each = 20 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


def gdn_stmatched(use_short_conv: bool):
    # head_dim=32, expand_v=1, num_heads=4 → 4*32*32 = 4096 state (matches cla-gdn-bigstate).
    return ModuleConfig(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs=dict(num_heads=4, head_dim=32, expand_v=1, use_short_conv=use_short_conv),
    )


# (name, mixer, pos_emb)
RUNS = [
    ("gdn-stmatched-noconv-pos0",   gdn_stmatched(False), 0),
    ("gdn-stmatched-conv-pos0",     gdn_stmatched(True),  0),
    ("gdn-stmatched-noconv-pos128", gdn_stmatched(False), SEQ_LEN),
    ("gdn-stmatched-conv-pos128",   gdn_stmatched(True),  SEQ_LEN),
]

data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                              num_examples=TRAIN_EXAMPLES, num_kv_pairs=NUM_KV,
                              random_non_queries=False)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                             num_examples=2_000, num_kv_pairs=NUM_KV,
                             random_non_queries=False)],
    batch_size=(BATCH, BATCH // 4),
    cache_dir="/tmp/zoology_cache_h2h",
)

configs = []
for name, mixer, pos in RUNS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=pos,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="gdn-state-matched", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
