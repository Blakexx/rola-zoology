"""MHA (softmax attention) ceiling on the exact MQAR task used everywhere else.

Identical to the RLA/RoLA baselines except the sequence-mixer kernel is full
softmax attention instead of the chunked-linear kernel: same wrap_hybrid (BaseConv
-> mixer) block, same d_model / n_layers / vocab, same max_position_embeddings=0
(MQAR is content-based recall, so attention needs no position embeddings), same
extended-kv test set, same lr=1e-2.

Uses the repo's canonical `mha` dict (num_heads=2, dropout=0.1) from rla_sweep.

2 runs (2-layer + 4-layer), seed 1337, lr 1e-2. Test batch 16: full O(L^2)
attention at seq 4096 OOMs at batch 64 (eval-memory only, not a model change).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, mha, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS

LR = 1e-2
SEED = 1337

# Same extended-kv test set as everywhere else, but smaller test batch: full
# O(L^2) attention at seq 4096 OOMs at batch 64.
data_mha = DataConfig(
    train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
    batch_size=(128, 16),
    cache_dir="/tmp/zoology_cache_rwext",
)

configs = []
configs_envs = []
for n_layers in (2, 4):
    configs.append(
        TrainConfig(
            data=data_mha,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(mha),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=n_layers,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="mha-baseline", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"mha-hybrid-nl{n_layers}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 2


def load_configs_and_envs():
    return configs, configs_envs
