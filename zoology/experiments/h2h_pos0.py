"""Same as head_to_head.py but with max_position_embeddings=0 (clean comparison).

Skips cla-gla-asym (already have 6 seeds from curriculum_results.jsonl).
Adds the missing pieces:
  - baseline-gla: just seed 1 (we have 1337, 42, 7, 0 from prior runs)
  - inhouse-gla-norm: all 5 seeds
  - cla-gdn-asym (recipe + GDN): all 5 seeds
  - baseline-gdn: all 5 seeds
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS_ALL = [1337, 42, 7, 0, 1]


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
    # use_short_conv=False to disable the 1D conv (FLA default is True).
    # The conv is an architectural confound: it captures local position info
    # via a kernel-4 conv that no other model in this comparison has, so
    # baseline GDN was solving MQAR partly via the conv rather than the kernel.
    return ModuleConfig(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs=dict(num_heads=4, use_short_conv=False),
    )

def cla_gla_recipe():
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                    writer="softmax_gla", reader="softmax_linear",
                    tie_routers=False),
    )

# (name, mixer, seeds_to_run)
# Skip cla-gla-asym (already have 6/6 from curriculum sweep) and baseline-gla
# (already have 4+ seeds from earlier pipeline + this sweep).
# Run the 3 missing models at pos_emb=0:
RUNS = [
    ("baseline-gdn",     baseline_gdn(),      SEEDS_ALL),   # no conv — key test
    ("cla-gdn-asym",     cla_gdn_recipe(),    SEEDS_ALL),   # recipe + GDN
    ("inhouse-gla-norm", inhouse_gla_norm(),  SEEDS_ALL),   # V+1 control
    # cla-gla-asym at pos_emb=0 with peakiness diagnostic enabled (the prior 6/6
    # grok came from curriculum_results.jsonl before the diagnostic was added).
    ("cla-gla-asym",     cla_gla_recipe(),    SEEDS_ALL),
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
for name, mixer, seeds in RUNS:
    for seed in seeds:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=0,         # KEY: back to 0
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="cla-h2h-pos0", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
