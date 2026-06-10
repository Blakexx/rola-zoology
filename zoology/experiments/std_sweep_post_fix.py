"""(write_std, read_std) sweep for CLA-GLA-asym on a faster task.

Goal: identify optimal router init_std post-gating-fix. Cheaper task than full
Zoology setup so we can sweep more cells in less wall-clock.

Task scale (between in-house and full Zoology):
  - vocab=512, seq_len=128, num_kv=8, 30k examples, d_model=128, batch=128
  - max_epochs=24, lr=2e-3  (~3-4 min per run on 3080 Ti)

Grid: 5 × 5 × 3 = 75 runs at ~3.5 min ≈ 4-5 hours total.

NOTES on prior failures:
  - vocab=1024, d_model=64, lr=1e-3 → all seeds stuck at 9-12% (model too small)
  - vocab=2048, d_model=128, lr=1e-3 → all seeds stuck at 12% (lr too low / vocab too high)
  - vocab=512, d_model=128, lr=2e-3 → solves to 96% by epoch 18 (probe confirmed). Use this.

The runner (run_sweep_robust.py) parses "w<X>_r<Y>" from run_id and sets
MQAR_ROUTER_STD_WRITE / MQAR_ROUTER_STD_READ env vars per-config.
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

WRITE_STDS = [0.05, 0.1, 0.3, 0.5, 1.0]
READ_STDS  = [0.05, 0.1, 0.3, 0.5, 1.0]
SEEDS = [1337, 42, 7]
LR = 2e-3  # learnable on this task per probe (96% by epoch 18)


def cla_gla_asym():
    # Match the headtohead config so optimal std transfers directly.
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
for w_std in WRITE_STDS:
    for r_std in READ_STDS:
        for seed in SEEDS:
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
                    logger=LoggerConfig(project_name="cla-gla-stdsweep-postfix", entity=""),
                    max_epochs=24,
                    learning_rate=LR,
                    weight_decay=0.0,
                    seed=seed,
                    # run_id MUST contain w<X>_r<Y> for the runner to plumb env vars
                    run_id=f"stdsweep_w{w_std}_r{r_std}_lr{LR:.1e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                )
            )
