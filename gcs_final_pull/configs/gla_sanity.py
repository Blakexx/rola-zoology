"""Single-config sanity test: can Zoology's own GLA solve MQAR at published defaults?

num_kv=16, seq_len=256, batch=256, lr=3e-4. If this doesn't hit high accuracy,
my head_first patch is probably broken.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8192
SEQ_LEN = 256
NUM_KV = 16
D_MODEL = 128

data_cfg = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                              num_examples=20_000, num_kv_pairs=NUM_KV)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                             num_examples=1_000, num_kv_pairs=NUM_KV)],
    batch_size=(256, 32),
)

configs = [
    TrainConfig(
        data=data_cfg,
        model=ModelConfig(
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.gla.GatedLinearAttention",
                kwargs=dict(num_heads=4),
            ),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL,
            n_layers=2,
            max_position_embeddings=0,
            vocab_size=VOCAB,
        ),
        logger=LoggerConfig(project_name="gla-sanity", entity=""),
        max_epochs=32,
        learning_rate=3e-4,
        weight_decay=0.0,
        seed=1337,
        run_id="gla_kv16_bs256_lr3e4",
        early_stopping_threshold=0.99,
        early_stopping_metric="valid/accuracy",
    )
]
