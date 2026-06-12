"""GDN comparison: baseline GDN vs CLA-GDN with asym init + gradual curriculum.

Same task as the GLA experiments (vocab=512, kv=8, d_model=128, lr=2e-3, 24 epochs).
5 seeds each.

The runner sets MQAR_* env vars based on run_id pattern (see run_sweep_robust.py).
For curriculum/asym init, the run_id has 'asym_curr' which the runner needs to
recognize and apply the env vars. The standard runner only parses w/r std from
run_id like 'w1.0_r0.05'. We embed those values in run_id directly:
'cla-gdn-asym_w1.0_r0.05_lr2e-03_s1337'.

NOTE: curriculum env vars are NOT parseable from run_id by the standard runner.
We launch this via a dedicated runner that knows to apply curriculum to cla-gdn-asym
runs. See run_gdn_compare.py.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]


def baseline_gdn():
    return ModuleConfig(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs=dict(num_heads=4),
    )


def cla_gdn():
    # Match CLA-GLA topology: 8 chunks, head dims 7/8 (state ~2016 floats).
    return ModuleConfig(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs=dict(d_qk=7, d_v=8, num_chunks=8, n_heads=4,
                    writer="softmax_gdn", reader="softmax_linear",
                    tie_routers=False),
    )


MIXERS = [
    ("cla-gdn-asym",   cla_gdn()),  # combined with curriculum + asym env vars
    ("baseline-gdn",   baseline_gdn()),
]

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
for name, mixer in MIXERS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=0, vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="gdn-compare", entity=""),
                max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
