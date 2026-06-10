"""Minimal local control: can GDN learn TRIVIAL recall (kv=4) at nc=1 at all?
RLA-nc1 vs GDN-nc1, single easy slice, small + fast (so it's dataloader-light).
If RLA-nc1 learns kv4 and GDN-nc1 doesn't, GDN training is broken/hard even for
the simplest case at nc=1 -> the routed failure is downstream of a basic GDN
optimization issue, not routing.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from rola import rola_instance

TRAIN = [MQARConfig(vocab_size=VOCAB, input_seq_len=64, num_examples=4_000, num_kv_pairs=4)]
TEST  = [MQARConfig(vocab_size=VOCAB, input_seq_len=64, num_examples=500,  num_kv_pairs=4)]
data = DataConfig(train_configs=TRAIN, test_configs=TEST, batch_size=(64, 64),
                  cache_dir="/tmp/zoology_cache_ctrl_tiny")

SPECS = [("ctrl-rla-nc1", "rola-rla-sym"), ("ctrl-gdn-nc1", "rola-gdn-sym")]

configs = []
configs_envs = []
for tag, inst in SPECS:
    kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=1, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="ctrl-tiny", entity=""),
            max_epochs=24, learning_rate=1e-2, weight_decay=0.0, seed=1337,
            run_id=f"{tag}", early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy", slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})


def load_configs_and_envs():
    return configs, configs_envs
