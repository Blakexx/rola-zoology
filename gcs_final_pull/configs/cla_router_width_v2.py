"""Router-width sweep at (d_qk=10, dv=12) WITH extended test set (kv up to 1024).

This is the merged version of the previous cla_router_width + cla_hard_kv
intent: a single sweep that varies router width AND tests at much harder
MQAR difficulties (num_kv = 512 and 1024) along with the standard kv ≤ 256.

Setup:
  - Shape fixed: d_qk=10, d_v=12.
  - nc ∈ {4, 8, 12, 18, 24, 32} (states 2080, 4160, 6240, 9360, 12480, 16640).
  - Routing widths: hidden ∈ {10 (base = d_qk), 32, 128 (wide)}.
  - Plus matched-state RLA wide-asym (d_v=12) at each nc/state.
  - Test slices: kv ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024}.

Counts:
  6 nc × 3 widths = 18 CLA runs
  6 RLA baselines  = 6 runs
  Total = 24 runs (seed 1337, lr 1e-2).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp, rla_asymmetric


TRAIN_CONFIGS = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
# Extended test: adds kv=512 (seq=2048) and kv=1024 (seq=4096).
# Each MQAR example uses 2*kv + 1 tokens, so seq_len must be ≥ 2*kv + 1.
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
data_ext = DataConfig(
    train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
    batch_size=(128, 64),  # test smaller — long sequences
    cache_dir="/tmp/zoology_cache_rwext",
)


NCS = [4, 8, 12, 18, 24, 32]
WIDTHS = [10, 32, 128]
D_QK, D_V = 10, 12
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []


def _add(cell, kernel, run_tag):
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-rw-v2", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_{run_tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,  # never trigger — want full 32 epochs
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


for nc in NCS:
    cla_cell = f"cla-rla-nc{nc}-d{D_QK}-dv{D_V}"
    for h in WIDTHS:
        _add(cla_cell,
             cla_rla_mlp(D_QK, D_V, nc, route_hidden_dim=h, route_act='relu'),
             f"mlp-relu-h{h}")
    # Matched-state RLA wide-asym ref
    rla_state = nc * 4 * D_QK * (D_V + 1)
    rla_d_qk = rla_state // (4 * (D_V + 1))
    _add(f"rla-d{rla_d_qk}-dv{D_V}", rla_asymmetric(rla_d_qk, D_V), "linear")

assert len(configs) == 24


def load_configs_and_envs():
    return configs, configs_envs
