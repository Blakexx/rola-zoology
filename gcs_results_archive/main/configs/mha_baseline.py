"""MHA (softmax attention) ceiling on the exact MQAR task used everywhere else.

We cite MHA as the O(N)-state ceiling but never actually ran it on this task /
test set. This does. Pure multi-head softmax attention (no BaseConv prefix),
same d_model/heads/vocab and the same extended-kv test set (kv up to 1024).

Unlike the linear models (max_position_embeddings=0, implicit recency), softmax
attention needs explicit position embeddings, set to 4096 to cover the longest
eval sequence (kv=1024 -> seq 4096).

2 runs (2-layer + a 4-layer for headroom), seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext

D_MODEL = 128
MAXPOS = 4096   # longest eval sequence (kv=1024)
LR = 1e-2
SEED = 1337

def mha():
    return ModuleConfig(name="zoology.mixers.attention.MHA",
                        kwargs={"num_heads": 4})

configs = []
configs_envs = []
for n_layers in (2, 4):
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=mha(),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=n_layers,
                max_position_embeddings=MAXPOS,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="mha-baseline", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"mha-nl{n_layers}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 2


def load_configs_and_envs():
    return configs, configs_envs
