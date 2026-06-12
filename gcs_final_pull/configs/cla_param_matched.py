"""CLA with FFN backbone to match RLA's parameter budget at matched state.

At state 12480, RLA wide-asym has 1.32M params (d_qk=240 → big Q/K projections),
vs CLA's 1.09M (d_qk=10). The gap is ~230K. Adding a standard FFN state_mixer
(hidden_mult=4 → d_ff=512) adds ~262K params over 2 layers, bringing CLA up to
~1.35M — matched-or-slightly-above RLA's param count.

This isolates: does CLA's small accuracy deficit at high state disappear when
you give it the same param budget?

Cells (using the winning shape d_qk=10, d_v=12, mlp-relu-h10):
  state  9360, nc=18: ffn vs no-ffn
  state 12480, nc=24: ffn vs no-ffn
  state 16640, nc=32: ffn vs no-ffn

(The no-ffn versions are already in cache; just adding the ffn variants.)
3 runs at hidden_mult=4, seed 1337, extended test set.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_dv_nc_scan import cla_rla_mlp
from zoology.experiments.cla_router_width_v2 import data_ext


NCS = [4, 8, 12, 18, 24, 32]  # full state range, matching cla_router_width_v2
D_QK, D_V = 10, 12
LR = 1e-2
SEED = 1337
ROUTER_HIDDEN = 10  # narrow router wins per the v2 sweep
FFN_HIDDEN_MULT = 4  # d_ff = 4*d_model = 512 → +262K total params

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
                state_mixer=ModuleConfig(
                    name="zoology.mixers.mlp.MLP",
                    kwargs={"hidden_mult": FFN_HIDDEN_MULT},
                ),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="cla-param-matched", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_mlp-relu-h{ROUTER_HIDDEN}-ffn{FFN_HIDDEN_MULT}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 6


def load_configs_and_envs():
    return configs, configs_envs
