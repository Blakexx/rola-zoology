"""GDN analogue of the RLA-vs-RoLA-RLA matched-state comparison, at the high-state
points only (~20k, ~40k), back in the easy regime (old data_ext: kv<=1024, vocab
8192, seq<=4096, test batch 64). No vocab/CE-OOM, and nc<=64 stays under the CUDA
grid limit so no head-chunking — fast, clean runs like the earlier sweeps.

  RoLA-GDN (routed, gdn-sym, route_on='x', d_qk=12, d_v=12):
    nc=32 -> state 4*32*12*12 = 18432  (~20k)
    nc=64 -> state 4*64*12*12 = 36864  (~40k)
  Matched GDN monolith (nc=1, big d_qk) — the "vs monolith" half:
    d_qk=384 (~20k), d_qk=768 (~40k). These probe FLA's gated-delta-rule head-dim
    limit (GDN baselines only ever ran at d_qk=32); expected to fail, which is the
    point — GDN can't be a monolith at high state, only routed.

Compare against the existing RoLA-RLA / RLA-monolith results at 20k/40k.
4 runs, seed 1337, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from cla_bench import rola_instance

D_V = 12
LR = 1e-2
SEED = 1337

# (tag, instance, nc, d_qk)
# Wide monoliths (d_qk=384/768) DROPPED: GDN's chunk kernel hard-asserts head dim
# <= 256 (fla chunk_delta_h.py), so GDN cannot be a wide monolith above d_qk=256
# (~12k state) — it can't reach 20k/40k as a wide monolith at all. The SQUARE
# monolith (d<=96, in rola_gdn_square.py) is the runnable GDN monolith here.
SPECS = [
    ("rola-gdn-sym-nc32-d12", "rola-gdn-sym", 32, 12),   # ~20k, routed
    ("rola-gdn-sym-nc64-d12", "rola-gdn-sym", 64, 12),   # ~40k, routed
]

configs = []
configs_envs = []
for tag, inst, nc, d_qk in SPECS:
    kw = rola_instance(inst, d_qk=d_qk, d_v=D_V, num_chunks=nc, n_heads=4)
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
            logger=LoggerConfig(project_name="rola-gdn-highstate", entity=""),
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
