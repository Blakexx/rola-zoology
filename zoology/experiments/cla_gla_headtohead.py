"""CLA-GLA head-to-head vs baseline GLA on standard Zoology MQAR.

Figure 2 recipe:
  - random_non_queries=False (easier distractors, matches Zoology Figure 2)
  - num_examples=100k (Figure 2 recipe)
  - d_model=128, batch=256, max_epochs=32
  - lr sweep over logspace(-4, -2, 4) ≈ {1e-4, 4.6e-4, 2.2e-3, 1e-2} — pick best per model

Models (3):
  - baseline pure GLA
  - CLA-GLA asym
  - CLA-GLA sym

num_kv scaling (3):
  - 8, 16, 32 (all at seq_len=256)

Seeds (3):
  - 1337, 42, 7

Total: 3 × 3 × 4 × 3 = 108 runs.
"""
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8192
SEQ_LEN = 256
D_MODEL = 128
BATCH = 256

NUM_KV_VALUES = [16]  # kv=16 only for speed; can add {8, 32} later if needed
SEEDS = [1337, 42, 7]
LRS = list(np.logspace(-4, -2, 4))[1:]  # drop lr=1e-4 (underfits): [4.64e-4, 2.15e-3, 1e-2]
TRAIN_EXAMPLES = 50_000  # down from 100k to cut per-run time ~2×

# ---------- Model mixers ----------
def baseline_gla():
    return ModuleConfig(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs=dict(num_heads=4),
    )

def cla_gla(tie_routers: bool):
    # State: 4 heads × 8 chunks × 7 (d_qk) × (8+1) (d_v + V+1 bit) = 2016 floats.
    # 1.5% UNDER baseline GLA's 2048 state — CLA never has a state advantage.
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(
            d_qk=7, d_v=8, num_chunks=8, n_heads=4,
            writer="softmax_gla", reader="softmax_linear",
            tie_routers=tie_routers,
        ),
    )


def inhouse_gla_norm():
    # Our RecurrentGLA wrapper — same architecture as Zoology's GLA but applies V+1
    # normalization (mathematically more faithful to attention). Same h=4, d_qk=16, d_v=32
    # as Zoology baseline; with V+1: state = 4 × 16 × 33 = 2112 (3% over baseline).
    return ModuleConfig(
        name="zoology.mixers.cla.RecurrentGLA",
        kwargs=dict(d_qk=16, d_v=32, n_heads=4),
    )

MIXERS = [
    # ("baseline-gla",   baseline_gla()),       # Zoology default GLA (no V+1); skip — already failed
    ("inhouse-gla-norm", inhouse_gla_norm()),   # in-house GLA WITH V+1 — apples-to-apples vs CLA-GLA
    ("cla-gla-asym",     cla_gla(False)),
    # ("cla-gla-sym",   cla_gla(True)),         # add back later
]

# ---------- Build config list ----------
configs = []
for num_kv in NUM_KV_VALUES:
    data = DataConfig(
        train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                                  num_examples=TRAIN_EXAMPLES, num_kv_pairs=num_kv,
                                  random_non_queries=False)],
        test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                                 num_examples=3_000, num_kv_pairs=num_kv,
                                 random_non_queries=False)],
        batch_size=(BATCH, BATCH // 8),
        cache_dir="/tmp/zoology_cache",
    )
    for name, mixer in MIXERS:
        for lr in LRS:
            for seed in SEEDS:
                configs.append(
                    TrainConfig(
                        data=data,
                        model=ModelConfig(
                            sequence_mixer=mixer,
                            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                            d_model=D_MODEL,
                            n_layers=2,
                            max_position_embeddings=0,
                            vocab_size=VOCAB,
                        ),
                        logger=LoggerConfig(project_name="cla-gla-headtohead", entity=""),
                        max_epochs=32,
                        learning_rate=lr,
                        weight_decay=0.0,
                        seed=seed,
                        run_id=f"{name}_kv{num_kv}_lr{lr:.1e}_s{seed}",
                        early_stopping_threshold=0.99,
                        early_stopping_metric="valid/accuracy",
                    )
                )
