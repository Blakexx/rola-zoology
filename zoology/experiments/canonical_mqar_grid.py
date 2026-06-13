"""Canonical MQAR matched-state grid (the fleet sweep): routed RoLA cells vs clean-room
canonical baselines, one hardware family, tier-ordered (strongest competition first) so a
budget cut leaves the crossover-critical cells done.

Stage 1 = per-(arch, shape, state) learning-rate sweep, 1 seed. Stage 2 (+2 seeds at the
best LR on claim-bearing rungs) is a separate config launched after stage 1 picks winners.

Tiers (emit order = run order):
  T1  RoLA-RLA(kappa-asym) + RoLA-GLA routed ladders;  wide RLA monolith (same-kernel control)
  T2  Based;  RLA square + heads
  T3  Hedgehog (3 shapes);  GLA (3 shapes)
  T4  GDN (3 shapes);  MHA oracle
Rank diagnostic (CLA_MEASURE_RANK) rides only the instrumented RoLA code: the routed cells
and a RoLA-RLA wide monolith (nc=1), at nc in {16,64,128,256} -> fig:rank. Canonical FLA/
zoology baselines drive accuracy (tab:main) only; they have no rank probe.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB, mha
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from zoology.experiments.canonical_baselines import baseline_cell, NCS, REF
from rola import rola_instance

SEED, TEST_BS = 1337, 8
RANK_NCS = {16, 64, 128, 256}
LRS = {  # per-method stage-1 brackets around the calibrated optimum
    "rla": [3e-3, 1e-2, 3e-2], "gla": [3e-3, 1e-2, 3e-2], "based": [3e-3, 1e-2, 3e-2],
    "hedgehog": [1e-3, 3e-3, 1e-2], "gdn": [3e-4, 1e-3, 3e-3],
    "routed": [3e-3, 1e-2, 3e-2], "mha": [1e-3, 3e-3, 1e-2],
}
configs, configs_envs = [], []


def add(kernel, run_id, lr, rank=False):
    configs.append(TrainConfig(
        data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                        batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
        model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                          state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                          d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
        logger=LoggerConfig(project_name="canonical-mqar-grid", entity=""),
        max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED, run_id=run_id,
        early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"]))
    env = {"EVAL_EVERY_N": "10"}
    if rank:
        env["CLA_MEASURE_RANK"] = "1"
    configs_envs.append(env)


def routed(inst, tag):
    for nc in NCS:
        kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        for lr in LRS["routed"]:
            add(kernel, f"grid-routed-{tag}-nc{nc}_st{REF(nc)}_lr{lr:.0e}_s{SEED}", lr,
                rank=(nc in RANK_NCS))


def monolith_rank_RLA():
    # RoLA-RLA wide monolith (nc=1, d_qk widened) via OUR code -> rank diagnostic for fig:rank.
    for nc in sorted(RANK_NCS):
        dqk = max(1, round(REF(nc) / (4 * 12)))            # H=4, d_v=12, wide keys
        kw = rola_instance("rola-rla-sym", d_qk=dqk, d_v=12, num_chunks=1, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        add(kernel, f"grid-rlamono-wide-nc{nc}_st{REF(nc)}_lr1e-02_s{SEED}", 1e-2, rank=True)


def baselines(methods, shapes):
    for method in methods:
        for shape in shapes:
            for nc in NCS:
                cell = baseline_cell(method, shape, nc)
                if cell is None:
                    continue
                kernel, st = cell
                for lr in LRS[method]:
                    add(kernel, f"grid-{method}-{shape}-nc{nc}_st{st}_lr{lr:.0e}_s{SEED}", lr)


# ---- TIER 1: routed cells + same-kernel wide-RLA control + rank monolith ----
routed("rola-rla-kappa-asym", "rla")
routed("rola-gla-scalar-sym", "gla")
monolith_rank_RLA()
baselines(["rla"], ["wide"])
# ---- TIER 2: Based + RLA square/heads ----
baselines(["based"], ["wide"])
baselines(["rla"], ["square", "heads"])
# ---- TIER 3: Hedgehog + GLA monoliths ----
baselines(["hedgehog", "gla"], ["wide", "square", "heads"])
# ---- TIER 4: GDN + MHA oracle ----
baselines(["gdn"], ["wide", "square", "heads"])
for lr in LRS["mha"]:
    add(mha, f"grid-mha_lr{lr:.0e}_s{SEED}", lr)


def load_configs_and_envs():
    return configs, configs_envs


if __name__ == "__main__":
    print(f"{len(configs)} runs; rank-instrumented: {sum('CLA_MEASURE_RANK' in e for e in configs_envs)}")
