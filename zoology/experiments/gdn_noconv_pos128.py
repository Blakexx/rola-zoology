"""Re-run baseline-gdn (no short_conv) at pos_emb=128 to test conv-was-cheat hypothesis.

Prior result at pos_emb=128: baseline-gdn WITH conv → 5/5 grok, mean 0.994.
Question: does disabling short_conv kill that win?
  - If still 5/5 → conv wasn't the cheat; GDN really is robust to pos_emb
  - If fails → conv was doing the position-invariance work
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


def baseline_gdn_noconv():
    return ModuleConfig(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs=dict(num_heads=4, use_short_conv=False),
    )

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
for seed in SEEDS:
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                sequence_mixer=baseline_gdn_noconv(),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=SEQ_LEN,    # 128 (matches prior h2h)
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="gdn-noconv-pos128", entity=""),
            max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
            run_id=f"baseline-gdn-noconv_lr{LR:.1e}_s{seed}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
        )
    )
