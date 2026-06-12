"""TINY local diagnostic for the RoLA-GDN failure (run locally, not cloud).
Tests the prediction: if routing the (nonlinear-write) delta rule is the problem,
RoLA-GDN should degrade with nc even at small nc, and lr shouldn't rescue it.

Small + fast: d_model=128, 2 layers, train kv<=16, test kv up to 64, ~6k examples,
16 epochs. nc in {1,2,4,8} (nc=1 ~ tiny GDN monolith, should learn) + nc=4 at a
higher lr (undertraining check).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from rola import rola_instance

TRAIN = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=6_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=4_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=4_000, num_kv_pairs=16),
]
TEST = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=500, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=500, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=500, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=500, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=500, num_kv_pairs=64),
]
data = DataConfig(train_configs=TRAIN, test_configs=TEST, batch_size=(64, 64),
                  cache_dir="/tmp/zoology_cache_gdn_tiny")

SPECS = [(f"gdn-tiny-nc{nc}", nc, 1e-2) for nc in (1, 2, 4, 8)] + [("gdn-tiny-nc4-lr3e-3", 4, 3e-3)]

configs = []
configs_envs = []
for tag, nc, lr in SPECS:
    kw = rola_instance("rola-gdn-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="gdn-tiny", entity=""),
            max_epochs=16, learning_rate=lr, weight_decay=0.0, seed=1337,
            run_id=f"{tag}_lr{lr:.2e}", early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


def load_configs_and_envs():
    return configs, configs_envs
