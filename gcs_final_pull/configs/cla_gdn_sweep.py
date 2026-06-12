"""CLA-GDN config sweep — explore what GDN needs to win.

Hypotheses for why cla-gdn-asym hasn't beaten baseline-gdn:
  1. State undersized: baseline-gdn has 4×16×64 = 4096 floats. Our cla-gdn-asym
     has 4×7×8×9 = 2016 — under half. Maybe scale up.
  2. Recipe too aggressive: GDN's delta+L2-norm kernel doesn't need the strong
     symmetry-breaking that GLA's lottery requires. Maybe gentler curriculum
     or no curriculum.

Variants (5 seeds each, pos_emb=0):
  A. cla-gdn-bigstate-recipe   : d_qk=8, d_v=16, chunks=8 → 4096 state; w=1.0/r=0.05; linear curriculum 3→0.3 / 0.3→3
  B. cla-gdn-bigstate-norecipe : same arch as A, default init, no curriculum
  C. cla-gdn-bigstate-gentle   : same arch as A, w=0.5/r=0.1; linear 1.5→0.5 / 0.5→1.5
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


def cla_gdn_bigstate():
    # d_qk=8, d_v=16, num_chunks=8, n_heads=4 → 4×8×16×8 = 4096 floats/layer
    # (matched to baseline-gdn at d_qk=16, d_v=64, n_heads=4 → 4096).
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=8, d_v=16, num_chunks=8, n_heads=4,
                    writer="softmax_gdn", reader="softmax_linear",
                    tie_routers=False),
    )


def cla_gdn_smallstate():
    # Same as cla-gdn-asym (matched to baseline-gla state ≈ 2016 floats).
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                    writer="softmax_gdn", reader="softmax_linear",
                    tie_routers=False),
    )


# (name, mixer, env_overrides)
RUNS = [
    # State-matched-to-baseline-GDN (4096), varying recipe strength
    ("cla-gdn-bigstate-recipe",     cla_gdn_bigstate(),
        {"MQAR_ROUTER_STD_WRITE": "1.0",
         "MQAR_ROUTER_STD_READ":  "0.05",
         "MQAR_CURR_MODE": "linear",
         "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
         "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0"}),
    ("cla-gdn-bigstate-norecipe",   cla_gdn_bigstate(), {}),  # default init, no curr
    ("cla-gdn-bigstate-gentle",     cla_gdn_bigstate(),
        {"MQAR_ROUTER_STD_WRITE": "0.5",
         "MQAR_ROUTER_STD_READ":  "0.1",
         "MQAR_CURR_MODE": "linear",
         "MQAR_CURR_W_LR_PHASE1": "1.5", "MQAR_CURR_W_LR_PHASE2": "0.5",
         "MQAR_CURR_R_LR_PHASE1": "0.5", "MQAR_CURR_R_LR_PHASE2": "1.5"}),
    # Same arch as cla-gdn-asym but WITHOUT the recipe (isolation test: is the
    # recipe hurting, or is chunking itself the issue for GDN?).
    ("cla-gdn-smallstate-norecipe", cla_gdn_smallstate(), {}),
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
configs_envs = []
for name, mixer, env in RUNS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=0,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="cla-gdn-sweep", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
        configs_envs.append(env)
