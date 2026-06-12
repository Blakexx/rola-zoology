"""Head-to-head: CLA recipe vs baselines, kernel-agnostic, on standard Zoology setup.

Models compared (5 × 5 seeds = 25 runs):
  1. cla-gla-asym     : CLA-GLA with asym init + curriculum LR (THE RECIPE)
  2. baseline-gla     : Zoology's stock GatedLinearAttention
  3. inhouse-gla-norm : Our V+1-normalized RecurrentGLA (V+1 control)
  4. cla-gdn-asym     : CLA-GDN with asym init + curriculum LR (recipe + GDN kernel)
  5. baseline-gdn     : Zoology's stock GatedDeltaNet

All under identical training conditions matching Zoology's canonical MQAR setup:
  - max_position_embeddings = 128 (matching seq_len, NOT 0 as before)
  - state_mixer = Identity (matching original_mqar_configs.py)
  - vocab=512, seq_len=128, kv=8, d_model=128, n_layers=2, n_heads=4
  - AdamW, lr=2e-3, wd=0, batch=128, 24 epochs, early-stop at acc≥0.99

The runner (run_head_to_head.py) sets per-model env vars:
  - cla-*-asym  → MQAR_ROUTER_STD_{WRITE=1.0, READ=0.05} + curriculum LR vars
  - others      → no env (default training)
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


def cla_gla_recipe():
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                    writer="softmax_gla", reader="softmax_linear",
                    tie_routers=False),
    )

def baseline_gla():
    return ModuleConfig(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs=dict(num_heads=4),
    )

def inhouse_gla_norm():
    return ModuleConfig(
        name="zoology.mixers.cla.RecurrentGLA",
        kwargs=dict(d_qk=16, d_v=32, n_heads=4),
    )

def cla_gdn_recipe():
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                    writer="softmax_gdn", reader="softmax_linear",
                    tie_routers=False),
    )

def baseline_gdn():
    return ModuleConfig(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs=dict(num_heads=4),
    )

MIXERS = [
    ("cla-gla-asym",     cla_gla_recipe()),
    ("baseline-gla",     baseline_gla()),
    ("inhouse-gla-norm", inhouse_gla_norm()),
    ("cla-gdn-asym",     cla_gdn_recipe()),
    ("baseline-gdn",     baseline_gdn()),
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
for name, mixer in MIXERS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=SEQ_LEN,   # NOTE: now 128, was 0
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="cla-h2h", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
