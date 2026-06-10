"""Fine-tune CLA-d24-nc64 from a trained RLA-d24 checkpoint.

Tests whether the chunked architecture's value comes purely from the
router learning to organize state, or if joint training of routers +
projections is needed.

Phases:
  A) Train RLA-d24 lr=1e-2 seed=0 from scratch, save state_dict
     (matches the median seed of the main sweep's RLA-d24 lr=1e-2 row)
  B) Load that checkpoint into a fresh CLA-d24-nc64 model. Copy w_q/k/v/o,
     leave routers at their default init. Apply freezing strategy.

Variants:
  v1 router-only: freeze w_q/k/v/o; only writer.W_route + reader.W_route train
  v2 periodic:    routers train at full LR; every N epochs the base model
                   is unfrozen for 1 epoch at lr*base_lr_factor

Usage:
  python run_finetune.py --phase a [--seed 0]
  python run_finetune.py --phase b --variant 1 --seed 1337
  python run_finetune.py --phase b --variant 2 --seed 1337 --N 4 --base_lr_factor 0.1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/c/Users/Blake/Documents/VSCode/CLA")
sys.path.insert(0, "/home/blake/zoology")

# Quiet wandb stdout interception so our prints reach the parent stdout
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")

import torch
import torch.nn as nn
import torch.optim as optim

from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.model import LanguageModel
from zoology.train import Trainer, prepare_data
from zoology.logger import WandbLogger
from zoology.experiments.rla_sweep import (
    data, wrap_hybrid, rla_baseline, cla_rla, D_MODEL, VOCAB,
)
from zoology.utils import set_determinism

CHECKPOINT_DIR = Path("/home/blake/zoology/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CKPT = CHECKPOINT_DIR / "rla_d24_lr1e-2_s0.pt"

RESULTS_PATH = Path("/home/blake/zoology/rla_finetune_results.jsonl")


def resolve_ckpt(ckpt_arg: str) -> Path:
    """Accept either a local path or a gs:// URI. If gs://, download to
    a local cache path and return that. Same URI hits the cache on reuse."""
    if ckpt_arg.startswith("gs://"):
        import hashlib
        h = hashlib.md5(ckpt_arg.encode()).hexdigest()[:12]
        cache_path = CHECKPOINT_DIR / f"_cached_{h}.pt"
        if not cache_path.exists():
            import subprocess
            print(f"[ckpt] downloading {ckpt_arg} → {cache_path}")
            r = subprocess.run(["gsutil", "cp", ckpt_arg, str(cache_path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"gsutil cp failed: {r.stderr[-500:]}")
        return cache_path
    return Path(ckpt_arg)


def build_config(run_id: str, mixer_kwargs: dict, lr: float, seed: int) -> TrainConfig:
    """Mirror the main sweep's config structure."""
    return TrainConfig(
        data=data,
        model=ModelConfig(
            block_type="TransformerBlock",
            sequence_mixer=wrap_hybrid(mixer_kwargs),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            d_model=D_MODEL, n_layers=2,
            max_position_embeddings=0,
            vocab_size=VOCAB,
        ),
        logger=LoggerConfig(project_name="rla-finetune", entity=""),
        max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
        run_id=run_id,
        early_stopping_threshold=0.99,
        early_stopping_metric="valid/accuracy",
        slice_keys=["num_kv_pairs"],
    )


def run_training(config: TrainConfig, freeze_pattern_keep_trainable=None,
                  periodic_unfreeze_N=None, base_lr_factor=None,
                  load_checkpoint_path=None, save_checkpoint_path=None):
    """Build model, optionally load checkpoint + apply freezing, train, save."""
    set_determinism(config.seed)

    logger = WandbLogger(config)
    logger.log_config(config)

    model = LanguageModel(config.model)
    train_dl, test_dl = prepare_data(config.data)
    logger.log_model(model, config=config)

    # Load checkpoint if provided.
    if load_checkpoint_path is not None:
        ckpt = torch.load(load_checkpoint_path, map_location="cpu")
        # strict=False because CLA has W_route keys that aren't in RLA's state_dict
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[ckpt load] missing keys (kept at init): {len(missing)} "
              f"(router params and possibly others)")
        for k in missing[:5]:
            print(f"  missing: {k}")
        print(f"[ckpt load] unexpected keys (ignored): {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"  unexpected: {k}")

    # Freezing strategy.
    if freeze_pattern_keep_trainable is not None:
        # `freeze_pattern_keep_trainable` is a list of substrings — only params
        # whose name contains ANY of these substrings remain trainable.
        for name, p in model.named_parameters():
            keep = any(s in name for s in freeze_pattern_keep_trainable)
            p.requires_grad = keep
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[freeze] trainable params: {n_train:,} / {n_total:,} "
              f"({100*n_train/n_total:.2f}%)")

    task = Trainer(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )

    # Capture per-epoch metrics by wrapping test().
    task._per_epoch = []
    orig_test = task.test
    def wrapped_test(epoch_idx):
        m = orig_test(epoch_idx)
        task._per_epoch.append({"epoch": epoch_idx, **{k: float(v) for k, v in m.items() if not isinstance(v, dict)}})
        return m
    task.test = wrapped_test

    # Variant 2 (periodic unfreeze) requires intercepting the training loop.
    # We monkey-patch task.train_epoch to toggle freezing before each call.
    if periodic_unfreeze_N is not None:
        ROUTER_KEYS = ("writer.W_route", "reader.W_route", "writer.router.weight",
                       "reader.router.weight")
        orig_train_epoch = task.train_epoch

        def patched_train_epoch(epoch_idx):
            # Default: only routers trainable. Every N epochs: unfreeze all
            # with a lower LR on non-routers. Epoch 0 is included so we don't
            # start with the base model frozen if it could benefit from a
            # gentle pull toward the new (chunked) functional form.
            unfreeze_full = (epoch_idx % periodic_unfreeze_N == 0)
            for name, p in task.model.named_parameters():
                is_router = any(k in name for k in ROUTER_KEYS)
                p.requires_grad = is_router or unfreeze_full
            # Adjust optimizer per-group LR (assumes single-group optimizer).
            if unfreeze_full:
                # Lower LR for non-router this epoch — modify the
                # single-group LR temporarily.
                for g in task.optimizer.param_groups:
                    g["lr"] = config.learning_rate * base_lr_factor
                print(f"[v2] epoch {epoch_idx}: UNFREEZE non-routers at "
                      f"lr={config.learning_rate * base_lr_factor:.2e}")
            else:
                for g in task.optimizer.param_groups:
                    g["lr"] = config.learning_rate
            return orig_train_epoch(epoch_idx)

        task.train_epoch = patched_train_epoch

    t0 = time.time()
    task.fit()
    elapsed = time.time() - t0
    logger.finish()

    # Aggregate per-epoch metrics → max_acc + grok_ep + final slice_accs
    metric = config.early_stopping_metric  # "valid/accuracy"
    accs = [(e["epoch"], e.get(metric, 0.0)) for e in task._per_epoch]
    task._max_acc = max((a for _, a in accs), default=0.0)
    task._grok_ep = next((e for e, a in accs if a >= 0.99), None)
    task._epochs_run = max((e for e, _ in accs), default=0)
    # Final-epoch slice_accs
    final_m = task._per_epoch[-1] if task._per_epoch else {}
    task._slice_accs = {k.split("/")[-1].replace("accuracy-", ""): v
                       for k, v in final_m.items()
                       if k.startswith("valid/num_kv_pairs/accuracy-")}

    if save_checkpoint_path is not None:
        torch.save(model.state_dict(), save_checkpoint_path)
        print(f"[ckpt save] saved → {save_checkpoint_path}")

    return model, task, elapsed


def append_result(record: dict):
    """Append a finetune run's metadata + final metrics to results.jsonl."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def phase_a(seed: int = 0, ckpt_path: Path = DEFAULT_CKPT):
    """Train RLA-d24 lr=1e-2, save final state_dict."""
    print(f"=== Phase A: RLA-d24 lr=1e-2 seed={seed} → {ckpt_path} ===")
    config = build_config(f"phase-a-rla-d24-s{seed}", rla_baseline(24),
                          lr=1e-2, seed=seed)
    model, task, elapsed = run_training(config, save_checkpoint_path=ckpt_path)
    # Don't append to results jsonl — this is just to capture weights.
    print(f"Phase A done in {elapsed:.0f}s. Checkpoint: {ckpt_path}")


def phase_b_variant1(seed: int, lr: float, ckpt_path: Path = DEFAULT_CKPT,
                     num_chunks: int = 64):
    """Variant 1: load RLA ckpt into CLA, freeze projections, train routers."""
    print(f"=== Phase B v1: CLA-d24-nc{num_chunks} router-only fine-tune, seed={seed} ===")
    config = build_config(f"phase-b-v1-cla-d24-nc{num_chunks}_lr{lr:.2e}_s{seed}",
                          cla_rla(24, num_chunks), lr=lr, seed=seed)
    model, task, elapsed = run_training(
        config,
        freeze_pattern_keep_trainable=["W_route"],  # only routers trainable
        load_checkpoint_path=ckpt_path,
    )
    # Record result
    final_metrics = getattr(task, "_last_metrics", {})
    rec = {
        "run_id": config.run_id,
        "variant": "v1_router_only",
        "seed": seed, "lr": lr,
        "elapsed": elapsed,
        "ckpt": str(ckpt_path),
        "max_acc": getattr(task, "_max_acc", None),
        "grok_ep": getattr(task, "_grok_ep", None),
        "epochs_run": getattr(task, "_epochs_run", None),
        "slice_accs": getattr(task, "_slice_accs", None),
    }
    append_result(rec)
    print(f"Phase B v1 done in {elapsed:.0f}s.")


def phase_b_variant3(seed: int, lr: float, ckpt_path: Path = DEFAULT_CKPT,
                     num_chunks: int = 64):
    """Variant 3: warm-start from RLA ckpt, no freezing — full training.
    Tests whether the RLA-d24 checkpoint is a better initialization than
    Xavier-init for training CLA-d24-nc{N} to convergence."""
    print(f"=== Phase B v3: CLA-d24-nc{num_chunks} warm-start (no freeze), seed={seed} ===")
    config = build_config(f"phase-b-v3-cla-d24-nc{num_chunks}_lr{lr:.2e}_s{seed}",
                          cla_rla(24, num_chunks), lr=lr, seed=seed)
    model, task, elapsed = run_training(
        config,
        load_checkpoint_path=ckpt_path,
        # No freeze_pattern_keep_trainable → all params trainable
        # No periodic_unfreeze_N → no custom epoch logic
    )
    rec = {
        "run_id": config.run_id,
        "variant": "v3_warmstart_full",
        "seed": seed, "lr": lr,
        "elapsed": elapsed,
        "ckpt": str(ckpt_path),
        "max_acc": getattr(task, "_max_acc", None),
        "grok_ep": getattr(task, "_grok_ep", None),
        "epochs_run": getattr(task, "_epochs_run", None),
        "slice_accs": getattr(task, "_slice_accs", None),
    }
    append_result(rec)
    print(f"Phase B v3 done in {elapsed:.0f}s.")


def phase_b_variant2(seed: int, lr: float, N: int, base_lr_factor: float,
                     ckpt_path: Path = DEFAULT_CKPT, num_chunks: int = 64):
    """Variant 2: routers always train; every N epochs unfreeze all at lr*factor."""
    print(f"=== Phase B v2: CLA-d24-nc{num_chunks} periodic unfreeze "
          f"N={N} lr_factor={base_lr_factor} seed={seed} ===")
    config = build_config(
        f"phase-b-v2-cla-d24-nc{num_chunks}_lr{lr:.2e}_N{N}_f{base_lr_factor}_s{seed}",
        cla_rla(24, num_chunks), lr=lr, seed=seed)
    model, task, elapsed = run_training(
        config,
        # Start with everything trainable; patched train_epoch will set
        # requires_grad per epoch.
        periodic_unfreeze_N=N,
        base_lr_factor=base_lr_factor,
        load_checkpoint_path=ckpt_path,
    )
    rec = {
        "run_id": config.run_id,
        "variant": "v2_periodic",
        "seed": seed, "lr": lr,
        "N": N, "base_lr_factor": base_lr_factor,
        "elapsed": elapsed,
        "ckpt": str(ckpt_path),
        "max_acc": getattr(task, "_max_acc", None),
        "grok_ep": getattr(task, "_grok_ep", None),
        "epochs_run": getattr(task, "_epochs_run", None),
        "slice_accs": getattr(task, "_slice_accs", None),
    }
    append_result(rec)
    print(f"Phase B v2 done in {elapsed:.0f}s.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["a", "b"], required=True)
    p.add_argument("--variant", type=int, choices=[1, 2, 3], default=None,
                   help="Phase B variant (1=router-only, 2=periodic, 3=warmstart-full)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--N", type=int, default=4, help="v2: epochs between unfreezes")
    p.add_argument("--base_lr_factor", type=float, default=0.1,
                   help="v2: LR factor for non-routers when unfrozen")
    p.add_argument("--num_chunks", type=int, default=64,
                   help="Phase B: number of chunks for CLA-d24-nc{num_chunks}")
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT))
    args = p.parse_args()

    if args.phase == "a":
        # Phase A always writes locally (uploading is the caller's job)
        phase_a(seed=args.seed, ckpt_path=Path(args.ckpt_path))
    elif args.phase == "b":
        # Phase B can take gs:// URIs — resolve to local cache
        ckpt_path = resolve_ckpt(args.ckpt_path)
        if args.variant == 1:
            phase_b_variant1(seed=args.seed, lr=args.lr, ckpt_path=ckpt_path,
                             num_chunks=args.num_chunks)
        elif args.variant == 2:
            phase_b_variant2(seed=args.seed, lr=args.lr, N=args.N,
                             base_lr_factor=args.base_lr_factor,
                             ckpt_path=ckpt_path, num_chunks=args.num_chunks)
        elif args.variant == 3:
            phase_b_variant3(seed=args.seed, lr=args.lr, ckpt_path=ckpt_path,
                             num_chunks=args.num_chunks)
        else:
            sys.exit("--phase b requires --variant {1,2}")


if __name__ == "__main__":
    main()
