"""Diagnostic: WHY does RoLA-GDN underperform its monolith?
RoLA-GDN nc32 got 0.525 while the GDN monolith got 0.78 and RoLA-GLA (also gated)
got 0.95 — so it's delta-rule-specific. This localizes it:

  nc sweep {1,2,4,8,16,32} at d_qk=12,d_v=12 (state 576*nc) — is the failure
    nc-dependent (routing the delta rule across more states corrupts it) or present
    even at low nc? nc=1 is ~a tiny GDN monolith (should learn the easy slice).
  lr probes at nc=8 and nc=32 (lr 3e-3, 1e-3) — is it just undertrained?

Easy regime (data_ext: kv<=1024, vocab 8192, batch 64). seed 1337.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

SEED = 1337
# (tag, nc, lr)
SPECS = [(f"gdn-diag-nc{nc}", nc, 1e-2) for nc in (1, 2, 4, 8, 16, 32)] + [
    ("gdn-diag-nc8-lr3e-3", 8, 3e-3),
    ("gdn-diag-nc32-lr1e-3", 32, 1e-3),
]

configs = []
configs_envs = []
for tag, nc, lr in SPECS:
    kw = rola_instance("rola-gdn-sym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
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
            logger=LoggerConfig(project_name="rola-gdn-diag", entity=""),
            max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"{tag}_lr{lr:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 8


def load_configs_and_envs():
    return configs, configs_envs
