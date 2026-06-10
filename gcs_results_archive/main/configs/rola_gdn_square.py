"""GDN SQUARE monolith (nc=1, d_qk = d_v) at the high-state points (~20k, ~40k),
the square-shape counterpart to rola_gdn_highstate's wide monolith. Easy regime
(old data_ext: kv<=1024, vocab 8192, test batch 64).

State (GDN, no V+1) = 4 * 1 * d * d:
  d_qk=d_v=68 -> 18496  (~20k; routed RoLA-GDN nc32 is 18432)
  d_qk=d_v=96 -> 36864  (~40k; exact match to routed RoLA-GDN nc64)

d=68/96 are well under the wide-monolith d384/768, so more likely to clear FLA's
gated-delta-rule head-dim limit and give a real monolith datapoint.
2 runs, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from cla_bench import rola_instance

LR = 1e-2
SEED = 1337

# (tag, d) — square: d_qk = d_v = d, nc=1
SPECS = [("mono-gdn-square-d68", 68), ("mono-gdn-square-d96", 96)]

configs = []
configs_envs = []
for tag, d in SPECS:
    kw = rola_instance("rola-gdn-sym", d_qk=d, d_v=d, num_chunks=1, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
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
            logger=LoggerConfig(project_name="rola-gdn-square", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 2


def load_configs_and_envs():
    return configs, configs_envs
