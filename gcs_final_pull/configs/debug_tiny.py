"""Minimal 2-epoch config for fast parser validation."""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

configs = [
    TrainConfig(
        data=DataConfig(
            train_configs=[MQARConfig(vocab_size=256, input_seq_len=64,
                                      num_examples=2_000, num_kv_pairs=4)],
            test_configs=[MQARConfig(vocab_size=256, input_seq_len=64,
                                     num_examples=500, num_kv_pairs=4)],
            batch_size=(64, 32),
        ),
        model=ModelConfig(
            sequence_mixer=ModuleConfig(name="zoology.mixers.attention.MHA",
                                         kwargs=dict(num_heads=1, dropout=0.0)),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=64, n_layers=2, max_position_embeddings=64, vocab_size=256,
        ),
        logger=LoggerConfig(project_name="debug", entity=""),
        max_epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=1337,
        run_id="debug_tiny",
    )
]
