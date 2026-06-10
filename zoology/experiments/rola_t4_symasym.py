"""Sym-vs-asym routing ablation (the ASYM half) — targeted + multi-seed, current impl.

crossover2 already has SYM at 8 nc x 3 seeds. The paper claims sym ~= asym on reciprocal
MQAR (sym slightly better), so we only need clean multi-seed ASYM at the informative
region to overlay. Below the barrier (nc<=8) both sit at ~0 on the hard slice, so we run
asym only where routing does work: nc in {16,32,64} x 3 seeds = 9 runs (~$3). Compare
kv1024 (and overall) against crossover2's sym at the same nc; overlapping 3-seed error
bars => asymmetry does not matter on reciprocal MQAR (asymmetry is tested on language).

Identical to rola_t4_crossover except rola-rla-ASYM and the reduced nc set. Same OOM-safe
test batch (8), eval cuts, lr 1e-2.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

LR = 1e-2
SEEDS = [1337, 2024, 7]
NCS = [16, 32, 64]          # the crossover region; below it sym==asym==~0 on the hard slice
TEST_BS = 8

configs, configs_envs = [], []
for seed in SEEDS:
    for nc in NCS:
        data = DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                          batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext")
        kw = rola_instance("rola-rla-asym", d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-t4-symasym", entity=""),
            max_epochs=40, learning_rate=LR, weight_decay=0.0, seed=seed,
            run_id=f"rolaT4asym-nc{nc}-d12dv12_asym_lr{LR:.0e}_s{seed}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})

assert len(configs) == 9, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
