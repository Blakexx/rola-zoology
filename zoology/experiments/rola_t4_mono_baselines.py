"""RLA MONOLITH baselines (nc=1) at the crossover's matched states — wide / square / heads.

The crossover (rola_t4_crossover) sweeps routed RoLA at d_qk=d_v=12, n_heads=4, so its state at
routed count nc is S = 156(=d_qk*d_v+d_qk) * 4(heads) * nc = 624*nc. These baselines are nc=1
(pure RLA monolith, no routing) scaled THREE ways to hit the SAME state S, so the crossover plot
can show routed-RoLA(nc) vs monolith-at-matched-state:
  - square: n_heads=4, d_qk=d_v=d        -> (d^2+d)*4 = S
  - wide:   n_heads=4, d_v=12, d_qk=w     -> (13w)*4 = 52w = S
  - heads:  d_qk=d_v=12, n_heads=H        -> 156*H = S

Monolith optimal LR is STATE-dependent (muP effect; wide head wants lower LR than the routed
1e-2), so we sweep LR per cell and take the best — required for a fair baseline.

states match nc in {8,16,32,64,128}; 3 shapes x 3 LR {3e-3,1e-2,3e-2} x 1 seed = 45 runs. T4.
"""
import math
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import TRAIN_CONFIGS, TEST_CONFIGS
from rola import rola_instance

SEED = 1337
LRS = [3e-3, 1e-2, 3e-2]
NC_TARGETS = [8, 16, 32, 64, 128]
TEST_BS = 8

configs, configs_envs = [], []
for nc in NC_TARGETS:
    S = 624 * nc
    d_sq = max(1, round((-1 + math.sqrt(1 + S)) / 2))      # square: (d^2+d)*4 = S
    w = max(1, round(S / 52))                              # wide:   52w = S
    H = max(1, round(S / 156))                             # heads:  156H = S
    shapes = [("square", dict(d_qk=d_sq, d_v=d_sq, n_heads=4)),
              ("wide",   dict(d_qk=w,    d_v=12,   n_heads=4)),
              ("heads",  dict(d_qk=12,   d_v=12,   n_heads=H))]
    for lr in LRS:
        for tag, dims in shapes:
            kw = rola_instance("rola-rla-sym", num_chunks=1, **dims)   # nc=1 = pure RLA monolith
            kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
            configs.append(TrainConfig(
                data=DataConfig(train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
                                batch_size=(128, TEST_BS), cache_dir="/tmp/zoology_cache_rwext"),
                model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                                  state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                                  d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
                logger=LoggerConfig(project_name="rola-t4-mono", entity=""),
                max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
                run_id=f"monoT4-{tag}-nc{nc}-lr{lr:.0e}_s{SEED}",
                early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"]))
            env = {"EVAL_EVERY_N": "5"}
            if nc in (16, 64, 128):
                env["CLA_MEASURE_RANK"] = "1"   # monolith side of fig:rank (wide-RLA is the same-kernel pair)
            configs_envs.append(env)

assert len(configs) == 45, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
