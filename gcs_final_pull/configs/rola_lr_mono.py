"""Monolith (nc=1) LR calibration — the GAP: all prior LR cal was routed (nc=16).
Square monoliths (d_qk=d_v=48, nc=1) at ~matched state (~9-10k) for RLA/GLA/GDN.
Answers: (a) does nc=1 prefer a different LR than nc=16 (transfer check, before any
RoLA-vs-monolith crossover), (b) is GDN weak as a MONOLITH too (isolates mechanism
mismatch from routing). GDN gets the low-LR grid. data_ext, 40ep, EVAL_EVERY_N=5. 14 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext
from rola import rola_instance

CELLS = [("rola-rla-sym", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
         ("rola-gla-sym", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
         ("rola-gdn-sym", [1e-4, 3e-4, 1e-3, 3e-3])]
D, SEED = 48, 1337  # square monolith: nc=1, d_qk=d_v=48 -> state ~9.2-9.4k (~matched nc=16)
configs, configs_envs = [], []
for k, LRS in CELLS:
    for lr in LRS:
        kw = rola_instance(k, d_qk=D, d_v=D, num_chunks=1, n_heads=4)
        kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
        configs.append(TrainConfig(
            data=data_ext,
            model=ModelConfig(block_type="TransformerBlock", sequence_mixer=wrap_hybrid(kernel),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=D_MODEL, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB),
            logger=LoggerConfig(project_name="rola-lr-mono", entity=""),
            max_epochs=40, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"lrm-{k}-mono-d{D}_lr{lr:.0e}_s{SEED}",
            early_stopping_threshold=2.0, early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"]))
        configs_envs.append({"EVAL_EVERY_N": "5"})
assert len(configs) == 14


def load_configs_and_envs():
    return configs, configs_envs
