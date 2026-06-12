"""Conv-on sweep: re-run all "models that matter" with short conv enabled, at
small/matched state, both pos_emb=0 and pos_emb=128. 5 seeds each = 40 runs.

The short conv is FLA's depthwise causal Conv1d(kernel=4) + SiLU on q,k,v.
At pos_emb=0 it provides local pattern detection without positional info.
At pos_emb=128 it stacks with the positional embedding.

Models (each cell uses 5 seeds):
  baseline-gla-conv         : FLA-GLA, default state (2048), conv ON
  inhouse-gla-norm-conv     : RecurrentGLA (V+1), state 2112, conv ON
  cla-gla-asym-conv         : CLA-GLA, state 2016, asym+curr recipe, conv ON
  cla-gdn-bigstate-recipe-conv : CLA-GDN, state 4096, asym+curr recipe, conv ON

Note: state-matched FLA-GDN with conv ON at pos=0 and pos=128 is already in
gdn_state_matched_results.jsonl, so it's not repeated here.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


def baseline_gla_conv():
    return ModuleConfig(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs=dict(num_heads=4, use_short_conv=True),
    )


def inhouse_gla_conv():
    return ModuleConfig(
        name="zoology.mixers.cla.RecurrentGLA",
        kwargs=dict(d_qk=16, d_v=32, n_heads=4, use_short_conv=True),
    )


def cla_gla_conv():
    # Same arch as cla-gla-asym; recipe applied via env overrides.
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                    writer="softmax_gla", reader="softmax_linear",
                    tie_routers=False, use_short_conv=True),
    )


def cla_gdn_bigstate_conv():
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=8, d_v=16, num_chunks=8, n_heads=4,
                    writer="softmax_gdn", reader="softmax_linear",
                    tie_routers=False, use_short_conv=True),
    )


ASYM_RECIPE_ENV = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
}


# (name, mixer, pos_emb, env_overrides)
RUNS = [
    ("baseline-gla-conv-pos0",            baseline_gla_conv(),       0,       {}),
    ("baseline-gla-conv-pos128",          baseline_gla_conv(),       SEQ_LEN, {}),
    ("inhouse-gla-norm-conv-pos0",        inhouse_gla_conv(),        0,       {}),
    ("inhouse-gla-norm-conv-pos128",      inhouse_gla_conv(),        SEQ_LEN, {}),
    ("cla-gla-asym-conv-pos0",            cla_gla_conv(),            0,       ASYM_RECIPE_ENV),
    ("cla-gla-asym-conv-pos128",          cla_gla_conv(),            SEQ_LEN, ASYM_RECIPE_ENV),
    ("cla-gdn-bigstate-recipe-conv-pos0", cla_gdn_bigstate_conv(),   0,       ASYM_RECIPE_ENV),
    ("cla-gdn-bigstate-recipe-conv-pos128", cla_gdn_bigstate_conv(), SEQ_LEN, ASYM_RECIPE_ENV),
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
for name, mixer, pos, env in RUNS:
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
                logger=LoggerConfig(project_name="conv-sweep", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
        configs_envs.append(env)
