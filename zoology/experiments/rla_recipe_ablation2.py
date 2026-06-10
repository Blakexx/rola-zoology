"""Recipe ablation phase 2 — fill the missing 4-corner cells.

Phase 1 (rla_recipe_ablation.py) covered "asym init + variable curriculum" at
4 base_lrs × 3 writer scales. Combined with pilots, we have 3 of 4 corners
of the (init × curriculum) ablation at d=16 nc=8:

  Kaiming init, no curriculum    →  0.940 at lr=1e-2 only (pilot v1)
  Asym init,    no curriculum    →  0.892 at lr=1e-2     (pilot v2, curriculum dead)
  Asym init,    with curriculum  →  0.916 max-over-LRs   (phase 1)
  Kaiming init, with curriculum  →  UNMEASURED

This phase fills the 4th corner AND completes the recipe-off LR scan.

Cells:
  A. recipe-off at lr ∈ {1e-4, 4.64e-4, 2.15e-3}                    →  3 runs
  B. curriculum-only (Kaiming init + default curriculum) at 4 lrs   →  4 runs
  Total: 7 runs.

Ordering: recipe-off cells first so we have the complete recipe-off LR scan
after ~75 min, then curriculum-only cells.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, wrap_hybrid, cla_rla, D_MODEL, VOCAB,
)

D = 16
NC = 8
SEED = 1337

CURRICULUM_ONLY_ENV = {
    # No MQAR_ROUTER_STD_* — falls through to Kaiming default for W_route.
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
}

CELLS = [
    # Phase A: recipe-off at the 3 missing LRs (we already have lr=1e-2 = 0.940)
    ("recipe-off-d16-nc8",          cla_rla(D, NC), {},                    1e-4),
    ("recipe-off-d16-nc8",          cla_rla(D, NC), {},                    4.64e-4),
    ("recipe-off-d16-nc8",          cla_rla(D, NC), {},                    2.15e-3),
    # Phase B: curriculum-only (no asym init) at all 4 LRs
    ("curriculum-only-d16-nc8",     cla_rla(D, NC), CURRICULUM_ONLY_ENV,   1e-4),
    ("curriculum-only-d16-nc8",     cla_rla(D, NC), CURRICULUM_ONLY_ENV,   4.64e-4),
    ("curriculum-only-d16-nc8",     cla_rla(D, NC), CURRICULUM_ONLY_ENV,   2.15e-3),
    ("curriculum-only-d16-nc8",     cla_rla(D, NC), CURRICULUM_ONLY_ENV,   1e-2),
]

configs = []
configs_envs = []
for name, kernel, env, lr in CELLS:
    mixer = wrap_hybrid(kernel)
    configs.append(
        TrainConfig(
            data=data,
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=mixer,
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB,
            ),
            logger=LoggerConfig(project_name="rla-recipe-ablation2", entity=""),
            max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=SEED,
            run_id=f"{name}_lr{lr:.2e}_s{SEED}",
            early_stopping_threshold=0.99,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append(env)

assert len(configs) == 7, f"expected 7 ablation2 configs, got {len(configs)}"
