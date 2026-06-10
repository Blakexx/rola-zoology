"""GDN Sweep — kernel-swap follow-up to rla_sweep / gla_sweep.

Identical to rla_sweep.py in every dimension EXCEPT the inner kernel:
  RLA writer ('softmax_linear', V+1 ON)  →  GDN writer ('softmax_gdn', no V+1)
  RecurrentLinearAttention baseline       →  fla.layers.gated_delta_net.GatedDeltaNet

GDN already lacks V+1 by design (internal L2 norm + delta-rule update bound
state magnitude), so no V+1 flag is needed.

Reader stays 'softmax_linear'. n_heads=4, route_on='kq', tie_routers=False.
use_short_conv=False on BOTH CLA cells AND the FLA-GDN baseline — this
deviates from canonical-Zoology FLA-GDN (which uses use_short_conv=True with
conv_size=4) but holds every confound constant across the three sweeps so the
kernel is the only variable. Trade-off: comparison to Zoology's published
numbers is less direct, but the kernel-swap question is answered cleanly.

State formula WITHOUT V+1:
  state = n_heads · num_chunks · d_qk · d_v = 4 · n_chunks · d²

For the FLA-GDN baseline:
  state = n_heads · head_dim · (head_dim · expand_v) = 4 · d · d · 1 = 4d²
  matching CLA-GDN at nc=1.

OPEN QUESTION: `chunk_gated_delta_rule` (FLA's Triton kernel) is empirically
verified at d_qk ∈ {32, 64, 128} from canonical_state_scan. The grid here
includes d_qk ∈ {24, 48} which are non-power-of-2 and untested. If those
cells fail, drop them and re-grid as d ∈ {16, 32, 64} (3×3 + interp column =
14 cells × 4 LR × 5 seed = 280 runs).

20 cells × 4 LRs × 5 seeds = 400 runs (if all d values work).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, data_small_batch, wrap_hybrid, mha, OOM_CELLS,
    LRS, SEEDS, D_MODEL, VOCAB,
)


def cla_gdn(d, num_chunks):
    """CLA-GDN: softmax_gdn writer (no V+1, kernel does its own L2 norm),
    softmax_linear reader, route_on='kq', n_heads=4."""
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": d, "d_v": d, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_gdn", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": False, "route_on": "kq",
        },
    )


def gdn_baseline(d):
    """Canonical single-state GDN — FLA's GatedDeltaNet.

    Notes on the kwargs:
      num_heads=4 matches CLA-GDN's n_heads (Zoology canonical uses 2; we
      diverge here to keep the kernel-swap apples-to-apples with CLA).
      head_dim=d, expand_v=1 → state = 4·d² matching CLA-GDN at nc=1.
      use_gate=False matches Zoology canonical.
      use_short_conv=False matches the RLA/GLA sweep configs.
    """
    return dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"num_heads": 4, "head_dim": d, "expand_v": 1,
                "use_gate": False, "use_short_conv": False},
    )


# Same OOM rule as RLA/GLA. Cell names use "gdn" instead of "rla".
GDN_OOM_CELLS = {n.replace("rla", "gdn") for n in OOM_CELLS}

CELLS = [
    ("mha-ceiling",         mha,                {}),
    # Decoupling grid: nc=1 column = FLA GDN baseline
    ("gdn-baseline-d16",    gdn_baseline(16),   {}),
    ("gdn-baseline-d24",    gdn_baseline(24),   {}),   # untested kernel: d_qk=24
    ("gdn-baseline-d32",    gdn_baseline(32),   {}),
    ("gdn-baseline-d48",    gdn_baseline(48),   {}),   # untested kernel: d_qk=48
    ("gdn-baseline-d64",    gdn_baseline(64),   {}),
    # nc≥2: CLA-GDN (no V+1)
    ("cla-gdn-d16-nc2",     cla_gdn(16,  2),    {}),
    ("cla-gdn-d16-nc4",     cla_gdn(16,  4),    {}),
    ("cla-gdn-d24-nc2",     cla_gdn(24,  2),    {}),   # untested kernel: d_qk=24
    ("cla-gdn-d24-nc4",     cla_gdn(24,  4),    {}),   # untested kernel: d_qk=24
    ("cla-gdn-d32-nc2",     cla_gdn(32,  2),    {}),
    ("cla-gdn-d32-nc4",     cla_gdn(32,  4),    {}),
    ("cla-gdn-d48-nc2",     cla_gdn(48,  2),    {}),   # untested kernel: d_qk=48
    ("cla-gdn-d48-nc4",     cla_gdn(48,  4),    {}),   # untested kernel: d_qk=48
    ("cla-gdn-d64-nc2",     cla_gdn(64,  2),    {}),
    ("cla-gdn-d64-nc4",     cla_gdn(64,  4),    {}),
    # Interpolation column at d=16
    ("cla-gdn-d16-nc8",     cla_gdn(16,  8),    {}),
    ("cla-gdn-d16-nc16",    cla_gdn(16, 16),    {}),
    ("cla-gdn-d16-nc32",    cla_gdn(16, 32),    {}),
    ("cla-gdn-d16-nc64",    cla_gdn(16, 64),    {}),
]

assert len(CELLS) == 20, f"expected 20 cells, got {len(CELLS)}"

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    cell_data = data_small_batch if name in GDN_OOM_CELLS else data
    for lr in LRS:
        for seed in SEEDS:
            configs.append(
                TrainConfig(
                    data=cell_data,
                    model=ModelConfig(
                        block_type="TransformerBlock",
                        sequence_mixer=mixer,
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=0,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="gdn-sweep", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 400, f"expected 400 configs, got {len(configs)}"


def load_configs_and_envs():
    return configs, configs_envs
