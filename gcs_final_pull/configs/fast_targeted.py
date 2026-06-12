"""Targeted fast-benchmark sweep to resolve two open questions before the
canonical state-scan sweep:

  Q1: Does conv (with kq routing) work for CLA at pos_emb=128?
  Q2: What does small-state FLA-GDN baseline look like at this benchmark?

All CLA cells use route_on='kq' + use_short_conv=True. Same fixed ~2,048-state
config we validated on cla-gla-conv-recipe-pos0 = 0.970 mean (vs 0.687 with
route_on='x' + conv).

8 cells × 5 seeds = 40 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


# CLA-GLA: n_heads=4, c=8, d_qk=7, d_v=8 + V+1 → 2,016 state. kq routing.
cla_gla_conv = dict(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs={"d_qk": 7, "d_v": 8, "num_chunks": 8, "n_heads": 4,
            "writer": "softmax_gla", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
)

# CLA-GDN: n_heads=4, c=8, d_qk=8, d_v=8 → 2,048 state. kq routing.
cla_gdn_conv = dict(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs={"d_qk": 8, "d_v": 8, "num_chunks": 8, "n_heads": 4,
            "writer": "softmax_gdn", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
)

# baseline-gdn at small (2,048) state: num_heads=2, head_dim=32, expand_v=1.
fla_gdn_conv_small = dict(
    name="zoology.mixers.gated_delta_net.GatedDeltaNet",
    kwargs={"num_heads": 2, "head_dim": 32, "expand_v": 1,
            "use_gate": False, "use_short_conv": True, "conv_size": 4},
)


ASYM_RECIPE_ENV = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
}


# (name, kernel, pos, env)
RUNS = [
    # Q1: conv at pos=128 (the 4 CLA cells that aren't in the prior 9-result file)
    ("cla-gla-conv-recipe-pos128",     cla_gla_conv,        SEQ_LEN, ASYM_RECIPE_ENV),
    ("cla-gla-conv-norecipe-pos128",   cla_gla_conv,        SEQ_LEN, {}),
    ("cla-gdn-conv-recipe-pos128",     cla_gdn_conv,        SEQ_LEN, ASYM_RECIPE_ENV),
    ("cla-gdn-conv-norecipe-pos128",   cla_gdn_conv,        SEQ_LEN, {}),
    # Q2: small-state baseline-gdn at pos=0 and pos=128
    ("baseline-gdn-conv-pos0",         fla_gdn_conv_small,  0,       {}),
    ("baseline-gdn-conv-pos128",       fla_gdn_conv_small,  SEQ_LEN, {}),
    # Companions: CLA-GDN at pos=0 (so we can pair with baseline-gdn-conv-pos0)
    ("cla-gdn-conv-recipe-pos0",       cla_gdn_conv,        0,       ASYM_RECIPE_ENV),
    ("cla-gdn-conv-norecipe-pos0",     cla_gdn_conv,        0,       {}),
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
for name, kernel, pos, env in RUNS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=ModuleConfig(name=kernel["name"], kwargs=kernel["kwargs"]),
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=pos,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="fast-targeted", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
        configs_envs.append(env)
