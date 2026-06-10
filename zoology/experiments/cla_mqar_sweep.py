"""CLA vs baselines on standard Zoology MQAR.

6 models × 3 num_kv × 2 seeds = 36 runs.
- Baselines: GatedLinearAttention, GatedDeltaNet (Zoology's implementations)
- Ours: CLA-GLA asym/sym, CLA-GDN asym/sym
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8192
SEQ_LEN = 256
D_MODEL = 128

# ---------- Data: sweep num_kv ----------
NUM_KVS = [16, 32, 64]
data_cfg = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                              num_examples=20_000, num_kv_pairs=kv) for kv in NUM_KVS],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                             num_examples=1_000, num_kv_pairs=kv) for kv in NUM_KVS],
    batch_size=(32, 32),
)

# ---------- Model mixers ----------
# Zoology's TransformerBlock auto-passes d_model and layer_idx; do NOT include them in kwargs.
def baseline_gla():
    return ModuleConfig(name="zoology.mixers.gla.GatedLinearAttention", kwargs=dict(
        num_heads=4))

def baseline_gdn():
    return ModuleConfig(name="zoology.mixers.gated_delta_net.GatedDeltaNet", kwargs=dict(
        num_heads=4))

def cla(writer: str, tie_routers: bool):
    return ModuleConfig(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=dict(
        d_qk=16, d_v=16, num_chunks=8, n_heads=4,
        writer=writer, reader="softmax_linear", tie_routers=tie_routers))

MODEL_CONFIGS = [
    ("baseline-gla",     baseline_gla()),
    ("baseline-gdn",     baseline_gdn()),
    ("cla-gla-asym",     cla("softmax_gla", False)),
    ("cla-gla-sym",      cla("softmax_gla", True)),
    ("cla-gdn-asym",     cla("softmax_gdn", False)),
    ("cla-gdn-sym",      cla("softmax_gdn", True)),
]

SEEDS = [1337, 42, 7]

# ---------- Build configs ----------
configs = []
for name, mixer in MODEL_CONFIGS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data_cfg,
                model=ModelConfig(
                    sequence_mixer=mixer,
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL,
                    n_layers=2,
                    max_position_embeddings=0,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="cla-zoology", entity=""),
                max_epochs=32,
                learning_rate=1e-3,
                weight_decay=1e-2,
                seed=seed,
                run_id=f"{name}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
