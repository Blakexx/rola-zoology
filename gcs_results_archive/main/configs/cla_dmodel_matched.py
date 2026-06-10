"""CLA with WIDER d_model backbone (not FFN) to match RLA's parameter budget.

RLA's extra params at high state come from wider Q/K projections (d_qk=240..320).
Those params are PART OF routing — they help the model decide what to retrieve.
Adding an FFN on top of CLA doesn't help retrieval at all (the earlier FFN
runs lost ~0.02 at low state).

This sweep scales d_model from 128 → 160 (+25%). The token embedding,
Q/K/V/output projections, BaseConv, and output head all grow proportionally.
This matches RLA's total params at state ~12480 (+260K vs no-bump CLA).

CLA shape stays the same: d_qk=10, d_v=12, mlp-relu hidden=10.
nc ∈ {4, 8, 12, 18, 24, 32} → states {2080..16640}.

6 runs (seed 1337, lr 1e-2, extended kv test set).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp
from zoology.experiments.cla_router_width_v2 import data_ext


D_MODEL_BUMP = 160  # was 128; +25% → ~+260K params via embedding + projections
NCS = [4, 8, 12, 18, 24, 32]
D_QK, D_V = 10, 12
LR = 1e-2
SEED = 1337
ROUTER_HIDDEN = D_QK

configs = []
configs_envs = []
for nc in NCS:
    cell = f"cla-rla-nc{nc}-d{D_QK}-dv{D_V}"
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(
                    cla_rla_mlp(D_QK, D_V, nc, route_hidden_dim=ROUTER_HIDDEN, route_act='relu')
                ),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL_BUMP, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-dmodel-matched", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_mlp-relu-h{ROUTER_HIDDEN}-dm{D_MODEL_BUMP}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
