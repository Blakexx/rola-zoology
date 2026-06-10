"""Rank-scaling: RoLA scales effective-attention rank past d_model while the
matched-state monolith saturates at d_model. Tests the paper's central claim
across inner kernels and the asymmetry axis.

Setup (route_on='x', d_qk=12, d_v=12, d_model=128, n_heads=4; crossover nc*d_qk =
d_model at nc~11):

RoLA scaling curves — nc in {8,16,32,64,128,256} (rank 96..3072), four variants:
  rola-rla-sym  : linear kernel, normalized (V+1), symmetric (tied) routing
  rola-rla-asym : same but asymmetric (untied) — to show asymmetry buys nothing
                  on a reciprocal recall task like MQAR
  rola-gla-sym  : GLA inner kernel (forget gate routed per-state), no norm, tied
  rola-gdn-sym  : GDN inner kernel (delta rule + gate routed per-state), no norm, tied

Matched-state monoliths (nc=1, big d_qk). d_qk brackets d_model to show the rank
ceiling flattening; capped at per-kernel FLA head-dim limits (RLA ran to 770, GLA
to 320, GDN only to 32 — GDN monolith points are a feasibility probe). We do NOT
run d_qk=1536/3072 monoliths (exceed FLA limits); the monolith is already
rank-saturated by d_qk=128, so the bracket suffices.

VOCAB = 16384 (NOT the 8192 of older sweeps): the MQAR generator requires
vocab_size > input_seq_len, and the hardest slice (kv=2048) is seq=8192. Vocab is
uniform across train+eval so eval tokens are in the trained distribution. Larger
vocab inflates the fp32 LM-head logits ([B,L,vocab]) -> kv=4096/vocab=32768 OOM'd
in cross_entropy, so we cap at kv=2048 (vocab 16384). Not comparable to 8192-vocab
runs (intentional — self-contained RoLA-vs-monolith comparison).

34 runs (24 RoLA scaling + 10 monolith), seed 1337, lr 1e-2. Training stays at
kv<=64 (eval to kv=2048 is extrapolation).
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rla_sweep import wrap_hybrid, D_MODEL
from rola import rola_instance

D_V = 12
D_QK = 12
LR = 1e-2
SEED = 1337
VOCAB_RANK = 16384   # > longest eval seq (8192 at kv=2048); uniform train+eval

# Train at kv<=64 (seq<=256), same as prior sweeps, rebuilt at the larger vocab.
TRAIN_CONFIGS = [
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
# Eval: kv up to 2048 (seq up to 8192). seq = ~4*kv, vocab 16384 > 8192 (OK).
TEST_CONFIGS = [
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=64,    num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=64,    num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=64,    num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=128,   num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=256,   num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=512,   num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=1024,  num_examples=1_000, num_kv_pairs=256),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=2048,  num_examples=500,   num_kv_pairs=512),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=4096,  num_examples=250,   num_kv_pairs=1024),
    MQARConfig(vocab_size=VOCAB_RANK, input_seq_len=8192,  num_examples=128,   num_kv_pairs=2048),
]
# Constant train batch 128 across ALL runs (so the curve is comparable — batch
# size changes optimization, which would confound the rank comparison). The CUDA
# grid limit at high nc (B*n_heads*nc > 65535) is handled INSIDE cla_bench via
# head-chunked kernel launches (_headchunked), so batch need not shrink with nc.
# Test batch sized per-nc, as large as fits (eval-only, no accuracy effect):
# bounded by the seq-8192 logits (~B*0.54 GB at vocab 16384) and, at high nc, the
# virtual-head expansion. Earlier batch-4 was for a dropped nc=512/seq-16384 case
# and made eval ~16x slower than needed. Combined with EVAL_EVERY_N (env, default
# 1) this keeps the heavy multi-slice eval cheap.
def test_batch_for(nc):
    # Eval (no_grad) memory is bounded by the GLA work buffer ~ B*H*nc (a single
    # ~16 GB alloc OOM'd at nc=256/B=8) AND the seq-8192 logits ~B*0.54 GB.
    # nc=256 @ B=4 is proven to fit (an earlier run trained through it); scale up
    # as nc drops. Eval-only -> no accuracy/comparability effect.
    return 4 if nc >= 256 else 8 if nc >= 128 else 16

def make_data(nc):
    return DataConfig(
        train_configs=TRAIN_CONFIGS, test_configs=TEST_CONFIGS,
        batch_size=(128, test_batch_for(nc)),
        cache_dir="/tmp/zoology_cache_rank_v16k",
    )

# (instance, nc, d_qk, tag) tuples
SPECS = []
# RoLA scaling curves. nc=256 is the top (rank 3072); nc=512 dropped — it OOM'd
# on the virtual-head expansion at batch 128 (would need expansion-chunking).
NCS = [8, 16, 32, 64, 128, 256]
for variant in ("rola-rla-sym", "rola-rla-asym", "rola-gla-sym", "rola-gdn-sym"):
    short = variant.replace("rola-", "")
    for nc in NCS:
        SPECS.append((variant, nc, D_QK, f"rola-{short}-nc{nc}-d{D_QK}"))
# Matched-state monoliths (nc=1, big d_qk), feasible-d_qk brackets per kernel
for d in (64, 128, 256, 512, 768):
    SPECS.append(("rola-rla-sym", 1, d, f"mono-rla-d{d}"))
for d in (64, 128, 256):
    SPECS.append(("rola-gla-sym", 1, d, f"mono-gla-d{d}"))
for d in (64, 128):                          # GDN monolith probe (FLA head-dim limit unknown >32)
    SPECS.append(("rola-gdn-sym", 1, d, f"mono-gdn-d{d}"))

configs = []
configs_envs = []
for variant, nc, d_qk, tag in SPECS:
    kw = rola_instance(variant, d_qk=d_qk, d_v=D_V, num_chunks=nc, n_heads=4)
    kernel = dict(name="zoology.mixers.cla.ChunkedLinearAttention", kwargs=kw)
    configs.append(
        TrainConfig(
            data=make_data(nc),
            model=ModelConfig(
                block_type="TransformerBlock",
                sequence_mixer=wrap_hybrid(kernel),
                state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                d_model=D_MODEL, n_layers=2,
                max_position_embeddings=0,
                vocab_size=VOCAB_RANK,
            ),
            logger=LoggerConfig(project_name="rola-rank-scaling", entity=""),
            max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=SEED,
            run_id=f"{tag}_lr{LR:.2e}_s{SEED}",
            early_stopping_threshold=2.0,
            early_stopping_metric="valid/accuracy",
            slice_keys=["num_kv_pairs"],
        )
    )
    configs_envs.append({})

assert len(configs) == 34, len(configs)


def load_configs_and_envs():
    return configs, configs_envs
