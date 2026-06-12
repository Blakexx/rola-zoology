"""GDN baselines at integer expand_v values (workaround for zoology
GatedDeltaNet wrapper's float-value_dim bug at line 118).

At d_model=128, num_heads=4, head_dim=32:
  expand_v=1 → head_v_dim=32, state = 4*32*32 = 4,096
  expand_v=2 → head_v_dim=64, state = 4*32*64 = 8,192
  expand_v=3 → head_v_dim=96, state = 4*32*96 = 12,288
  expand_v=4 → head_v_dim=128, state = 4*32*128 = 16,384

State grid is coarser than the CLA/RLA sweep but provides comparable points
at 4k, 8k, 12k, 16k. Skips the lower states (2080) that need expand_v<1.

4 runs, extended kv test set.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.cla_router_width_v2 import data_ext


def gdn(expand_v_int):
    """expand_v_int MUST be Python int (not float) — zoology's wrapper does
    self.value_dim = self.key_dim * self.expand_v without int() cast."""
    assert isinstance(expand_v_int, int), "expand_v must be int (zoology wrapper bug)"
    return dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"num_heads": 4, "expand_v": expand_v_int, "use_short_conv": False},
    )


EXPAND_VS = [1, 2, 3, 4]
LR = 1e-2
SEED = 1337

configs = []
configs_envs = []
for ev in EXPAND_VS:
    head_v = 32 * ev
    state = 128 * head_v  # 4 * 32 * head_v
    cell = f"gdn-head32-hv{head_v}"
    configs.append(
        TrainConfig(
            data=data_ext,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(gdn(ev)),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="gdn-baselines", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{cell}_linear_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 4


def load_configs_and_envs():
    return configs, configs_envs
