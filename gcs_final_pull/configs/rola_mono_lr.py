"""Canonical-monolith LR verification across the STATE range (the crossover's monolith
baselines). Monoliths scale state via WIDTH (nc=1), and muP says optimal LR tracks width
-> the d48-square LR may not transfer to 2k or 160k state. Check LR at both extremes.

Baselines (canonical): RLA = RoLA-nc=1 (verified == canonical LA, rel 3e-4); GLA = FLA
GatedLinearAttention (gla_baseline); GDN = FLA GatedDeltaNet (gdn_baseline). Square shape
(d_qk=d_v=d). State ~4d^2: d=22 -> ~2k, d=200 -> ~160k (matches RoLA crossover extremes).
3 kernels x {d22,d200} x {3e-3,1e-2,3e-2} = 18 runs. data_ext, 40ep, EVAL_EVERY_N=5.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from zoology.experiments.gla_sweep import gla_baseline
from zoology.experiments.gdn_sweep import gdn_baseline
from rola import rola_instance

SEED = 1337
DS = [22, 200]                 # ~2k and ~160k state (4*d^2)
LRS = [3e-3, 1e-2, 3e-2]


def rla_mono(d):
    # Canonical FLA linear attention (== RoLA-nc=1, verified rel 3e-4) but MEMORY-EFFICIENT:
    # RoLA-nc=1's chunked path materializes [bh,L,d_qk,d_v] and OOMs at d_qk=200 x long seq.
    return dict(name="rola.RecurrentLinearAttention",
                kwargs={"d_qk": d, "d_v": d, "n_heads": 4})


KERNELS = [("rla", rla_mono), ("gla", gla_baseline), ("gdn", gdn_baseline)]
configs, configs_envs = [], []
for kname, mk in KERNELS:
    for d in DS:
        tb = 8 if d >= 200 else 64
        data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                          batch_size=(128, tb), cache_dir="/tmp/zoology_cache_rwext")
        for lr in LRS:
            kernel = mk(d)
            configs.append(TrainConfig(
                data=data,
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-mono-lr", entity=""),
                max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
                run_id=f"monolr-{kname}-d{d}_lr{lr:.0e}_s{SEED}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 18, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
