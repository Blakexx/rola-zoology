"""Minimal CLA test config — one small run to verify the integration works."""
import uuid
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8192
SEQ_LEN = 256
NUM_KV = 16

data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN, num_examples=20_000, num_kv_pairs=NUM_KV)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN, num_examples=1_000, num_kv_pairs=NUM_KV)],
    batch_size=(32, 32),
)

cla = ModuleConfig(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs=dict(
        d_qk=16, d_v=16, num_chunks=8, n_heads=4,
        writer="softmax_gdn", reader="softmax_linear", tie_routers=False,
    ),
)

model = ModelConfig(
    sequence_mixer=cla,
    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
    d_model=128,
    n_layers=2,
    max_position_embeddings=0,
    vocab_size=VOCAB,
)

configs = [
    TrainConfig(
        data=data,
        model=model,
        logger=LoggerConfig(project_name="cla-zoology-test", entity=""),
        max_epochs=2,  # short for smoke test
        learning_rate=1e-3,
        weight_decay=1e-2,
        seed=1337,
    )
]
