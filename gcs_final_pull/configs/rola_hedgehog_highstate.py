"""Hedgehog analogue of the RLA/GLA/GDN high-state comparison at ~20k/~40k, easy
regime (old data_ext: kv<=1024, vocab 8192, batch 64).

Hedgehog = the SoftmaxLinearWriter with phi='hedgehog' (learnable softmax-mimic
feature map phi(x)=softmax([xW;-xW]), feature dim = d_qk). It uses the LINEAR
kernel (fused_chunk_linear_attn — no gated-kernel OOM, no K<=256 limit) and is
NORMALIZED (V+1, like RLA), so state = 4*nc*d_qk*(d_v+1) — matched to the RLA runs.

  RoLA-Hedgehog routed (route_on='x', d_qk=12, d_v=12):
    nc=32 -> 19968 (~20k),  nc=64 -> 39936 (~40k)
  Hedgehog wide monolith (nc=1, d_v=12): d_qk=384 (~20k), 768 (~40k)
    [linear kernel ran to d=770 for RLA, so these run; Hedgehog needs even d_qk]

Compares against RoLA-RLA / RLA-monolith at the same states: does the learnable
softmax-mimic feature map (Hedgehog) change the routed-vs-monolith story?
4 runs, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

LR = 1e-2
SEED = 1337

# (tag, nc, d_qk, d_v)
SPECS = [
    ("rola-hh-sym-nc32-d12", 32, 12, 12),    # ~20k routed
    ("rola-hh-sym-nc64-d12", 64, 12, 12),    # ~40k routed
    ("mono-hh-wide-d384",     1, 384, 12),   # ~20k wide monolith
    ("mono-hh-wide-d768",     1, 768, 12),   # ~40k wide monolith
]

configs = []
configs_envs = []
for tag, nc, d_qk, d_v in SPECS:
    kw = rola_instance("rola-hedgehog-sym", d_qk=d_qk, d_v=d_v, num_chunks=nc, n_heads=4)
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
            logger=LoggerConfig(project_name="rola-hedgehog-highstate", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 4


def load_configs_and_envs():
    return configs, configs_envs
