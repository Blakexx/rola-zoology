"""Stage 2 of the staged sweep plan: best-competitor ladder (Based + Hedgehog).

Monolith (nc=1) Based and Hedgehog at the SAME states as the Stage-1 RoLA-RLA ladder
(S = 624*nc for nc in 2..256; state = n_heads * feat_dim * (d_v+1)). Goal: identify the
strongest non-routed competitor curve before committing to the Stage-3 massive grid.

Shapes per kernel (feat_dim must hit 12*nc with H=4, d_v=12 => feat_dim*(13)*4 = 624*nc):
  hedgehog-wide   : feat_dim = d_qk            -> d_qk = 12*nc
  hedgehog-square : d_qk = d_v = d, state 4*d*(d+1) = 624*nc -> d = sqrt(156*nc)-ish
  based-wide      : feat_dim = 1+2d+d(d-1)/2 = 12*nc (taylor expansion dim), d_v=12

Monolith optimal LR is state-dependent and monoliths are NOT yet LR-calibrated at these
states -> 2 LRs per kernel around its inner-kernel calibration (hedgehog 3e-3, based 1e-2).
3 arms x 8 states x 2 LRs x 1 seed = 48 runs. T4-able (no routed-Triton dependency).
LAUNCH GATE: hold until the Stage-1 ladder is green and the MQAR task is frozen.
"""
import math
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

SEED = 1337
NCS = [2, 4, 8, 16, 32, 64, 128, 256]
TEST_BS = 8
LRS = {"hedgehog": [1e-3, 3e-3], "based": [3e-3, 1e-2]}


def based_d_for_feat(target):
    # 1 + 2d + d(d-1)/2 = target -> pick the d whose feat_dim lands closest
    d0 = max(1, int((-3 + math.sqrt(9 + 8 * (target - 1))) / 2))
    return min((d0, d0 + 1), key=lambda d: abs(based_feat(d) - target))


def based_feat(d):
    return 1 + 2 * d + d * (d - 1) // 2


configs, configs_envs = [], []
for nc in NCS:
    target_feat = 12 * nc          # per-head feat_dim matching RoLA at this nc
    d_sq = max(1, round((-1 + math.sqrt(1 + 624 * nc)) / 2))   # d*(d+1) = 156*nc
    d_based = based_d_for_feat(target_feat)
    arms = [
        ("hh-wide",   "rola-hedgehog-sym", dict(d_qk=target_feat, d_v=12,   n_heads=4), 4 * target_feat * 13),
        ("hh-square", "rola-hedgehog-sym", dict(d_qk=d_sq,        d_v=d_sq, n_heads=4), 4 * d_sq * (d_sq + 1)),
        ("based",     "rola-based-sym",    dict(d_qk=d_based,     d_v=12,   n_heads=4), 4 * based_feat(d_based) * 13),
    ]
    for tag, inst, dims, state in arms:
        assert abs(state - 624 * nc) / (624 * nc) < 0.18, (tag, nc, state)  # based taylor dim is coarse at small nc
        kernel_name = inst.split("-")[1]
        for lr in LRS[kernel_name]:
            kw = rola_instance(inst, num_chunks=1, **dims)
            kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
            configs.append(TrainConfig(
                data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                                batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-competitor-ladder", entity=""),
                max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
                run_id=f"comp-{tag}-nc{nc}_st{state}_lr{lr:.0e}_s{SEED}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            configs_envs.append({"EVAL_EVERY_N": "10"})

assert len(configs) == 48, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
