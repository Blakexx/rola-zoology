"""Sanity test: does OUR CLA mixer learn normally in Zoology at num_kv=16?
If yes, the test harness is fine and only Zoology's FLA-based baselines are broken.
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
    batch_size=(64, 32),
)

configs = [
    TrainConfig(
        data=data_cfg,
        model=ModelConfig(
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.cla.ChunkedLinearAttention",
                kwargs=dict(
                    d_qk=16, d_v=16, num_chunks=8, n_heads=4,
                    writer="softmax_gla", reader="softmax_linear",
                    tie_routers=False,
                ),
            ),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL,
            n_layers=2,
            max_position_embeddings=0,
            vocab_size=VOCAB,
        ),
        logger=LoggerConfig(project_name="cla-sanity", entity=""),
        max_epochs=32,
        learning_rate=3e-4,
        weight_decay=0.0,
        seed=1337,
        run_id="cla_gla_asym_kv16",
        early_stopping_threshold=0.99,
        early_stopping_metric="valid/accuracy",
    )
]
