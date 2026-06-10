"""Test CLA + RLA at harder MQAR difficulties (num_kv ∈ {512, 1024}).

Existing experiments tested only up to num_kv=256. This adds longer tasks
(2x and 4x harder) at the high-state regime where the winning shape transfers
well: state ∈ {12k, 16k, 33k}.

States from (d_qk=10, d_v=12) at varying nc:
  nc=24 → state 12480
  nc=32 → state 16640
  nc=64 → state 33280

Matched RLA wide-asym refs:
  rla-d240-dv12 → state 12480
  rla-d320-dv12 → state 16640
  rla-d640-dv12 → state 33280

Train data unchanged (kv ∈ {4..64}); test data extended with kv ∈ {128, 256, 512, 1024}.
Models trained with max_position_embeddings=0 generalize to longer eval sequences.

3 states × 2 architectures (CLA, RLA) = 6 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp, rla_asymmetric


# Train data — same as the standard rla_sweep (kv up to 64)
TRAIN_CONFIGS = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
# Test data — extended with kv=512 (seq=2048) and kv=1024 (seq=4096).
# Each test config produces a slice we can read from slice_accs.
TEST_CONFIGS = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
    MQARConfig(vocab_size=VOCAB, input_seq_len=2048, num_examples=500,   num_kv_pairs=512),
    MQARConfig(vocab_size=VOCAB, input_seq_len=4096, num_examples=250,   num_kv_pairs=1024),
]

# Smaller batch for the longer-sequence test runs — avoids eval OOM at seq=4096.
data_hardkv = DataConfig(
    train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
    batch_size=(128, 64),  # (train, test) — test smaller since seq is much longer
    cache_dir="/tmp/zoology_cache_hardkv",
)


# (nc, d_qk, d_v) for the CLA shape; expected state and matched RLA d_qk
TARGETS = [
    # (nc, d_qk, d_v, rla_d_qk)
    (24, 10, 12, 240),  # state 12480
    (32, 10, 12, 320),  # state 16640
    (64, 10, 12, 640),  # state 33280
]

LR = 1e-2
SEED = 1337
HIDDEN = 32

configs = []
configs_envs = []


def _add(cell, kernel, run_tag):
    configs.append(
        TrainConfig(
            data=data_hardkv,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-hard-kv", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,  # never trigger — want full 32 epochs
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for nc, d_qk, d_v, rla_d_qk in TARGETS:
    cla_cell = f"cla-rla-nc{nc}-d{d_qk}-dv{d_v}"
    _add(cla_cell,
         cla_rla_mlp(d_qk, d_v, nc, route_hidden_dim=HIDDEN, route_act='relu'),
         f"mlp-relu-h{HIDDEN}")
    rla_cell = f"rla-d{rla_d_qk}-dv{d_v}"
    _add(rla_cell, rla_asymmetric(rla_d_qk, d_v), "linear")

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
