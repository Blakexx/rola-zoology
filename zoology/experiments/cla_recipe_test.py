"""Single recipe-on rerun of cla-rla-nc16-d16-dv8 at seed=1337, lr=1e-2.

Baseline cloud run: max_acc=0.950, kv=256=0.72, no grok.
Matched-state RLA-wide-sym (rla-d48-dv48): max_acc=0.975, kv=256=0.84, no grok.

Apply the canonical training recipe (LR curriculum + asymmetric router init)
to see whether CLA closes the kv=256 gap or even induces grokking.

Recipe:
  MQAR_CURR_MODE=linear
  MQAR_CURR_W_LR_PHASE1=3.0 → PHASE2=0.3  (writers learn fast then settle)
  MQAR_CURR_R_LR_PHASE1=0.3 → PHASE2=3.0  (readers slow then fast)
  MQAR_ROUTER_STD_WRITE=1.0  (peaky writer init)
  MQAR_ROUTER_STD_READ=0.05  (gentle reader init)
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import data, wrap_hybrid, D_MODEL, VOCAB
from zoology.experiments.rla_extreme import cla_rla_asym

CELL_NAME = "cla-rla-nc16-d16-dv8"
KERNEL = cla_rla_asym(d_qk=16, d_v=8, num_chunks=16)
LR = 1e-2
SEED = 1337

RECIPE_ENV = {
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
}

configs = [
    TrainConfig(
        data=data,
        model=ModelConfig(
            block_type="TransformerBlock",
            sequence_mixer=wrap_hybrid(KERNEL),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL, n_layers=2,
            max_position_embeddings=0,
            vocab_size=VOCAB,
        ),
        logger=LoggerConfig(project_name="cla-recipe-test", entity=""),
        max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
        run_id=f"{CELL_NAME}_recipe_lr{LR:.2e}_s{SEED}",
        early_stopping_threshold=0.99,
        early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"],
    )
]
configs_envs = [RECIPE_ENV]


def load_configs_and_envs():
    return configs, configs_envs
