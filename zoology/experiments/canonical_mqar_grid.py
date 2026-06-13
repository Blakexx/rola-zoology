"""MQAR matched-state grid (the fleet sweep): every method on one accuracy-vs-state plot,
one hardware family, each scaling along its OWN natural axis -- the proposed cells
(RoLA-RLA kappa-asym, RoLA-GLA; scale via nc), the canonical monolith baselines (RLA,
Hedgehog, Based, GLA, GDN; scale via wide/square/heads or feature-dim), SSE (the nearest
prior multi-state method; scales via partitions N and classifier dim c), and the MHA
oracle. Tier-ordered (strongest competition first) so a budget cut leaves the
crossover-critical cells done. Not a "monolith grid": RoLA and SSE are full participants.

Stage 1 = per-(arch, shape, state) learning-rate sweep, 1 seed. Stage 2 (+2 seeds at the
best LR on claim-bearing rungs) is a separate config launched after stage 1 picks winners.

Tiers (emit order = run order):
  T1  RoLA-RLA(kappa-asym) + RoLA-GLA routed ladders;  wide RLA monolith (same-kernel control)
  T2  Based;  RLA square + heads
  T3  Hedgehog (3 shapes);  GLA (3 shapes)
  T4  GDN (3 shapes);  MHA oracle
Rank is measured POST-HOC on saved best checkpoints (a separate full-dataset pass, all
methods), not inline -- so this sweep is train + save-best-checkpoint only. Set
SAVE_BEST_CKPT_DIR to capture checkpoints.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB, mha
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from zoology.experiments.canonical_baselines import baseline_cell, NCS, REF
from rola import rola_instance

SEED, TEST_BS = 1337, 8
# CANONICAL 5-LR grid spanning 3e-4..3e-2 (≈ Zoology logspace). The earlier 3-LR brackets were
# centered on PRIOR calibration probes (not citable in this paper's canonical data) and the
# results showed half the cells -- incl flagship routed-RLA in 6/8 -- winning at the LOW edge
# (3e-3), i.e. optimum NOT bracketed. This uniform grid makes every optimum interior and the
# methodology self-justifying. Existing {3e-3,1e-2,3e-2} runs are a SUBSET (reused); only
# {3e-4,1e-3} are added per cell. GDN/MHA already low; given the same uniform grid for cleanliness.
_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
LRS = {k: list(_GRID) for k in ("rla", "gla", "based", "hedgehog", "gdn", "routed", "mha")}
configs, configs_envs = [], []


def add(kernel, run_id, lr):
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
    configs_envs.append({"EVAL_EVERY_N": "10"})


def routed(inst, tag):
    for nc in NCS:
        kw = rola_instance(inst, d_qk=12, d_v=12, num_chunks=nc, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        for lr in LRS["routed"]:
            add(kernel, f"grid-routed-{tag}-nc{nc}_st{REF(nc)}_lr{lr:.0e}_s{SEED}", lr)




def sse_ladder():
    # SSE, the nearest prior multi-state method, scaled the way it is strongest. Per-head
    # state (N+1)*c*dh matches RoLA's nc*dqk*(dv+1)=156*nc at dh=12 -> (N+1)*c = 13*nc.
    # Two regimes per rung so SSE picks its best: NATIVE (few big partitions, its strong
    # regime) and PARALLEL (N~nc small partitions, mirroring RoLA's states). top-k=2 (the
    # matched-state winner) + top-1 on native to show the sparsity effect; LR-swept.
    for nc in (8, 16, 32, 64, 128, 256):
        budget = 13 * nc                                   # (N+1)*c target, dh=12, H=4
        native = (4, max(2, round(budget / 5)))            # few-big: N=4, c=budget/(N+1)
        par = (max(1, nc), max(2, round(budget / (nc + 1))))  # many-small: N~nc, c~13
        for (N, c), tag, topks in ((native, "native", (1, 2)), (par, "parallel", (2,))):
            for tk in topks:
                kernel = dict(name="zoology.mixers.sse.SSE",
                              kwargs=dict(n_heads=4, num_partitions=N, num_rows=c, topk=tk))
                st = 4 * (N + 1) * c * 12                   # H*(N+1)*c*dh realized total
                for lr in LRS["routed"]:
                    add(kernel, f"grid-sse-{tag}t{tk}-nc{nc}_N{N}c{c}_st{st}_lr{lr:.0e}_s{SEED}", lr)


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


# Tier selection via GRID_TIERS: "12" = tiers 1+2 (high-competition LR sweep),
# "34" = tiers 3+4 only (run separately on spot), "all"/unset = full 4-tier grid.
# Each group is a contiguous index block so sharding within a group is clean.
import os as _os
_TIERS = _os.environ.get("GRID_TIERS", "all")
if _TIERS in ("all", "12"):
    # ---- TIER 1: routed cells + same-kernel wide-RLA control ----
    routed("rola-rla-kappa-asym", "rla")
    routed("rola-gla-scalar-sym", "gla")
    baselines(["rla"], ["wide"])
    # ---- TIER 2: Based + SSE (nearest prior method) + RLA square/heads ----
    baselines(["based"], ["wide"])
    sse_ladder()
    baselines(["rla"], ["square", "heads"])
if _TIERS in ("all", "34"):
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
