"""Train one CLA-GLA model with asym+curriculum, then inspect router peakiness
at end of training.

Reports:
  - Writer/reader router weight std (init vs end)
  - Softmax entropy of writer/reader routing on a batch of real inputs
    (uniform = log(num_chunks); one-hot = 0)
  - Chunk utilization (fraction of chunks that receive any non-trivial weight)
"""
import os, sys, math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/c/Users/Blake/Documents/VSCode/CLA")
sys.path.insert(0, "/home/blake/zoology")

# Same recipe as dot_asym_curr (the winner): w=1.0/r=0.05 init + linear curriculum.
os.environ["WANDB_MODE"] = "offline"
os.environ["MQAR_ROUTER_STD_WRITE"] = "1.0"
os.environ["MQAR_ROUTER_STD_READ"] = "0.05"
os.environ["MQAR_CURR_MODE"] = "linear"
os.environ["MQAR_CURR_W_LR_PHASE1"] = "3.0"
os.environ["MQAR_CURR_W_LR_PHASE2"] = "0.3"
os.environ["MQAR_CURR_R_LR_PHASE1"] = "0.3"
os.environ["MQAR_CURR_R_LR_PHASE2"] = "3.0"

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.model import LanguageModel
from zoology.data.utils import prepare_data
from zoology.train import Trainer
from zoology.logger import WandbLogger

NUM_CHUNKS = 8
VOCAB = 512
SEED = 1337  # this seed groks in the asym+curriculum recipe

def init_router_snapshot(model):
    snap = {}
    for n, p in model.named_parameters():
        if "router" in n and "weight" in n:
            snap[n] = p.std().item()
    return snap

def report_router_stats(model, label):
    print(f"\n=== {label} ===")
    for n, p in model.named_parameters():
        if "router" in n and "weight" in n:
            print(f"  weight std  {n}: {p.std().item():.4f}")

@torch.no_grad()
def report_routing_entropy(model, loader, device):
    """Run one batch through and capture writer/reader softmax distributions."""
    model.eval()
    writer_gates = {}
    reader_gates = {}
    hooks = []
    def make_hook(name, store):
        def hook(module, args, output):
            # output is post-softmax gates of shape [B, L, H, C]; record it
            store[name] = output.detach().float().cpu()
        return hook

    # Patch each writer/reader forward to capture gates. Simplest: monkey-patch.
    from rola import SoftmaxGLAWriter, SoftmaxLinearReader
    captures = {"writer": {}, "reader": {}}
    orig_writer_fwd = SoftmaxGLAWriter.forward
    orig_reader_fwd = SoftmaxLinearReader.forward
    def w_fwd(self, x, q, k, v):
        write_gates = F.softmax(self.router(x).view(x.shape[0], x.shape[1], self.n_heads, self.num_chunks), dim=-1)
        captures["writer"][id(self)] = write_gates.detach().float().cpu()
        return orig_writer_fwd(self, x, q, k, v)
    def r_fwd(self, x, chunk_outputs):
        B, L, H, C, _ = chunk_outputs.shape
        if self._current_epoch < self._hard_uniform_until:
            captures["reader"][id(self)] = (torch.ones(B, L, H, C) / C).cpu()
        else:
            read_gates = F.softmax(self.router(x).view(B, L, H, C), dim=-1)
            captures["reader"][id(self)] = read_gates.detach().float().cpu()
        return orig_reader_fwd(self, x, chunk_outputs)
    SoftmaxGLAWriter.forward = w_fwd
    SoftmaxLinearReader.forward = r_fwd

    batch = next(iter(loader))
    inputs = batch["input_ids"].to(device)
    _ = model(inputs)

    # Restore
    SoftmaxGLAWriter.forward = orig_writer_fwd
    SoftmaxLinearReader.forward = orig_reader_fwd

    print("\n=== ROUTING ENTROPY ON REAL INPUTS ===")
    max_ent = math.log(NUM_CHUNKS)
    print(f"  uniform entropy = log({NUM_CHUNKS}) = {max_ent:.3f}; one-hot = 0")
    for role, store in [("writer", captures["writer"]), ("reader", captures["reader"])]:
        for i, (mod_id, gates) in enumerate(store.items()):
            # entropy per position, then mean
            ent = (-gates * (gates + 1e-12).log()).sum(-1)  # [B, L, H]
            avg_ent = ent.mean().item()
            min_ent = ent.min().item()
            max_obs_ent = ent.max().item()
            # chunk utilization: fraction of chunks with mean gate > 1/(2C)
            mean_per_chunk = gates.mean(dim=(0,1,2))  # [C]
            n_used = (mean_per_chunk > 1.0/(2*NUM_CHUNKS)).sum().item()
            print(f"  {role} layer {i}: entropy avg={avg_ent:.3f} min={min_ent:.3f} max={max_obs_ent:.3f}  "
                  f"chunks used={n_used}/{NUM_CHUNKS}")


def main():
    print("Building model + data...")
    torch.manual_seed(SEED)

    data_cfg = DataConfig(
        train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=30_000,
                                  num_kv_pairs=8, random_non_queries=False)],
        test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=2_000,
                                 num_kv_pairs=8, random_non_queries=False)],
        batch_size=(128, 32),
        cache_dir="/tmp/zoology_cache_stdsweep",
    )
    train_loader, test_loader, slice_keys = prepare_data(data_cfg)

    model_cfg = ModelConfig(
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.cla.ChunkedLinearAttention",
            kwargs=dict(d_qk=7, d_v=8, num_chunks=NUM_CHUNKS, n_heads=4,
                        writer="softmax_gla", reader="softmax_linear", tie_routers=False),
        ),
        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
        d_model=128, n_layers=2, max_position_embeddings=0, vocab_size=VOCAB,
    )
    model = LanguageModel(model_cfg)

    init_snap = init_router_snapshot(model)
    report_router_stats(model, "INIT router weight stds")

    logger = WandbLogger(LoggerConfig(project_name="probe-peakiness", entity=""), TrainConfig(
        data=data_cfg, model=model_cfg, max_epochs=24, learning_rate=2e-3,
        weight_decay=0.0, seed=SEED, run_id="probe_peakiness"))

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        max_epochs=24,
        learning_rate=2e-3,
        weight_decay=0.0,
        early_stopping_metric="valid/accuracy",
        early_stopping_threshold=0.99,
        slice_keys=slice_keys,
        logger=logger,
    )
    trainer.fit()

    report_router_stats(model, "FINAL router weight stds")
    print("\n=== GROWTH RATIO ===")
    for n, p in model.named_parameters():
        if "router" in n and "weight" in n:
            init_std = init_snap.get(n, 0)
            final_std = p.std().item()
            ratio = final_std / init_std if init_std > 0 else float('inf')
            print(f"  {n}: init={init_std:.4f}  final={final_std:.4f}  ratio={ratio:.1f}x")

    report_routing_entropy(model, test_loader, next(model.parameters()).device)


if __name__ == "__main__":
    main()
