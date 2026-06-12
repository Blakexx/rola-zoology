"""Tiny smoke config — runs in ~30s on any GPU. For validating cloud-image
changes via `docker run --gpus all` locally BEFORE submitting Vertex jobs.

Exercises the same code path (shard_runner → run_rla_sweep → train.py → Trainer
→ FLA kernels) but at a scale where one full training is seconds, not minutes.

If this passes locally in docker, the same image should run on Vertex
(modulo Vertex-specific env quirks like driver-version edge cases).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, cla_rla, rla_baseline

VOCAB = 1024  # small vocab — tiny LM head
D_MODEL = 64
BATCH = 8
SEQ = 32

# Single tiny MQAR task, tiny dataset (256 train examples)
data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ,
                              num_examples=256, num_kv_pairs=4)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ,
                             num_examples=64, num_kv_pairs=4)],
    batch_size=(BATCH, BATCH),
    cache_dir="/tmp/zoology_cache_smoke",
)

# Just one tiny cell, one tiny config — but exercises CLA, FLA, Triton, etc.
configs = [
    TrainConfig(
        data=data,
        model=ModelConfig(
            block_type="TransformerBlock",
            sequence_mixer=wrap_hybrid(cla_rla(16, 2)),  # d=16, nc=2
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL, n_layers=1,  # 1 layer is enough to validate
            max_position_embeddings=0,
            vocab_size=VOCAB,
        ),
        logger=LoggerConfig(project_name="rla-smoke", entity=""),
        max_epochs=2,  # 2 epochs = ~10s on any modern GPU
        learning_rate=1e-3, weight_decay=0.0, seed=0,
        run_id="smoke-tiny-d16-nc2",
        early_stopping_threshold=2.0,  # never trigger
        early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"],
    ),
]
configs_envs = [{}]


def load_configs_and_envs():
    return configs, configs_envs
