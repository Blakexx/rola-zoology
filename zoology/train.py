import argparse
import random
from datetime import datetime
from typing import List, Union
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data, prepare_continuous_data
from zoology.config import TrainConfig
from zoology.model import LanguageModel, ContinuousInputModel
from zoology.logger import WandbLogger
from zoology.utils import set_determinism
from zoology.metrics import compute_mse, compute_ce_with_embeddings


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        input_type: str = "discrete",
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_type = input_type
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys
        self.loss_type = loss_type

    def compute_loss(self, inputs, targets):
        if self.input_type == "continuous":
            
            all_embeddings = self.model.backbone.embeddings.word_embeddings.weight
            vocab_size = all_embeddings.shape[0]
            embed_dim = all_embeddings.shape[1]
            value_embeddings = all_embeddings[vocab_size // 2:]  # all values as candidates
            
            outputs = self.model(inputs, return_embeddings=True)
            num_kv_pairs = targets.shape[1]
            outputs = outputs[:, -num_kv_pairs:]
            
            outputs_flat = outputs.reshape(-1, embed_dim)
            targets_flat = targets.reshape(-1)
            
            if self.loss_type == "mse":
                target_embeds = value_embeddings[targets_flat]
                loss, _ = compute_mse(outputs_flat, target_embeds)
            else:  # ce or ce_embed
                loss, _ = compute_ce_with_embeddings(
                    outputs_flat, targets_flat, value_embeddings
                )
            
            logits = outputs_flat @ value_embeddings.T
            preds = (logits).argmax(dim=-1).view(targets.shape)
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs, return_embeddings=False)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "mse":
                embeddings = self.model(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
                logits = embeddings @ self.model.backbone.embeddings.word_embeddings.weight.T
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "ce_embed":
                embeddings = self.model(inputs, return_embeddings=True)
                value_embeddings = self.model.backbone.embeddings.word_embeddings.weight
                flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
                flat_targets = targets.flatten()
                mask = flat_targets != -100
                loss, _ = compute_ce_with_embeddings(
                    flat_embeds[mask], flat_targets[mask], value_embeddings,
                )
                logits = embeddings @ value_embeddings.T
                preds = logits.argmax(dim=-1)
                return loss, preds

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(getattr(self, "_amp", None) == "fp16")):
                loss, preds = self.compute_loss(inputs, targets)

                # Auxiliary losses (discrete mode only)
                if self.input_type == "discrete":
                    auxiliary_loss = []
                    def get_auxiliary_loss(module):
                        if hasattr(module, "get_auxiliary_loss"):
                            auxiliary_loss.append(module.get_auxiliary_loss())
                    self.model.apply(get_auxiliary_loss)
                    if auxiliary_loss:
                        loss = loss + sum(auxiliary_loss)

            scaler = getattr(self, "_scaler", None)
            if scaler is None:
                loss.backward(); self.optimizer.step()
            else:
                scaler.scale(loss).backward(); scaler.step(self.optimizer); scaler.update()
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log({"train/loss": loss.item(), "epoch": epoch_idx})

    def test(self, epoch_idx: int):
        self.model.eval()
        test_loss = 0
        results = []

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(getattr(self, "_amp", None) == "fp16")):
                    loss, preds = self.compute_loss(inputs, targets)
                test_loss += loss / len(self.test_dataloader)
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                iterator.update(1)

            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

            # compute metrics for slices
            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})
        return metrics

    def _gcs_blob(self, uri: str):
        """gs://bucket/path → (storage.Client, bucket, blob). Returns None
        if google-cloud-storage isn't installed (silently no-op)."""
        try:
            from google.cloud import storage
        except ImportError:
            return None
        if not uri.startswith("gs://"):
            return None
        bucket_name, _, blob_name = uri[5:].partition("/")
        client = storage.Client()
        return client.bucket(bucket_name).blob(blob_name)

    def _resume_from_gcs(self):
        """If RESUME_STATE_URI is set + the GCS object exists, download and
        restore optimizer/model/scheduler/epoch. Returns the start epoch
        (next epoch after the saved one), or 0 if no resume. No-op if the
        google-cloud-storage library isn't available."""
        import os
        uri = os.environ.get("RESUME_STATE_URI")
        if not uri:
            return 0
        blob = self._gcs_blob(uri)
        if blob is None:
            print(f"[resume] google-cloud-storage unavailable — skipping", flush=True)
            return 0
        try:
            if not blob.exists():
                print(f"[resume] no checkpoint at {uri} — starting from epoch 0", flush=True)
                return 0
        except Exception as e:
            print(f"[resume] blob.exists() failed: {e} — skipping", flush=True)
            return 0
        local_path = "/tmp/_resume_state.pt"
        try:
            blob.download_to_filename(local_path)
        except Exception as e:
            print(f"[resume] download failed: {e}", flush=True)
            return 0
        try:
            state = torch.load(local_path, map_location="cpu")
        except Exception as e:
            print(f"[resume] torch.load failed: {e} — starting fresh", flush=True)
            return 0
        try:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") and self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            self._per_epoch_resumed = state.get("per_epoch_metrics", [])
            print(f"[resume] restored from {uri}, resuming at epoch {start_epoch}",
                  flush=True)
            return start_epoch
        except Exception as e:
            print(f"[resume] state_dict apply failed: {e} — starting fresh", flush=True)
            return 0


    def _save_to_gcs(self, epoch_idx: int):
        """Save current training state and upload to GCS. Called on SIGTERM
        (Vertex preemption). No-op if google-cloud-storage unavailable."""
        import os
        uri = os.environ.get("RESUME_STATE_URI")
        if not uri:
            return
        local_path = "/tmp/_preempt_state.pt"
        try:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": epoch_idx,
                "per_epoch_metrics": getattr(self, "_per_epoch_history", []),
            }
            torch.save(state, local_path)
        except Exception as e:
            print(f"[preempt] torch.save failed: {e}", flush=True)
            return
        blob = self._gcs_blob(uri)
        if blob is None:
            print(f"[preempt] GCS client unavailable — state saved locally only", flush=True)
            return
        try:
            blob.upload_from_filename(local_path)
            print(f"[preempt] state uploaded to {uri} (epoch={epoch_idx})", flush=True)
        except Exception as e:
            print(f"[preempt] upload failed: {e}", flush=True)

    def _cleanup_gcs_checkpoint(self):
        """Delete the in-progress GCS checkpoint after successful completion."""
        import os
        uri = os.environ.get("RESUME_STATE_URI")
        if not uri:
            return
        blob = self._gcs_blob(uri)
        if blob is None:
            return
        try:
            if blob.exists():
                blob.delete()
        except Exception:
            pass  # cleanup is best-effort

    def fit(self):
        import os, signal
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        # Optional speedups, env-gated (default path unchanged): COMPILE=1 -> torch.compile;
        # AMP=fp16 -> autocast(fp16) + GradScaler. For the precision/compile cost A/B.
        if os.environ.get("COMPILE") == "1":
            self.model = torch.compile(self.model)
            print("[train] torch.compile enabled", flush=True)
        self._amp = os.environ.get("AMP")  # 'fp16' or None
        self._scaler = torch.cuda.amp.GradScaler(enabled=(self._amp == "fp16"))
        if self._amp:
            print(f"[train] AMP autocast enabled: {self._amp}", flush=True)

        # Standard optimizer + cosine LR schedule (default init; no router-init or
        # LR-curriculum regime — default initialization works fine).
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )

        # Resume from GCS if RESUME_STATE_URI is set + ckpt exists.
        start_epoch = self._resume_from_gcs()
        # Track per-epoch metrics across resumes for max_acc / grok_ep history.
        self._per_epoch_history = getattr(self, "_per_epoch_resumed", [])
        # Install SIGTERM handler — Vertex preempts spot VMs with 30s grace.
        def _on_sigterm(signum, frame):
            print(f"[preempt] SIGTERM received at epoch {getattr(self, '_current_epoch_idx', -1)}; saving state...", flush=True)
            # Save epoch_idx-1: in-flight epoch isn't complete, so resume re-runs it.
            ep = getattr(self, "_current_epoch_idx", start_epoch)
            self._save_to_gcs(max(ep - 1, -1))
            import sys
            sys.exit(143)  # 128 + SIGTERM(15) — non-zero so runner marks ok=False and retries
        if os.environ.get("RESUME_STATE_URI"):
            signal.signal(signal.SIGTERM, _on_sigterm)
            print(f"[preempt] SIGTERM handler armed (RESUME_STATE_URI set)", flush=True)

        for epoch_idx in range(start_epoch, self.max_epochs):
            self._current_epoch_idx = epoch_idx
            # Propagate current epoch to modules that use it (e.g. the rank diagnostic
            # throttle in AdditiveKernel).
            for m in self.model.modules():
                if hasattr(m, "_current_epoch"):
                    m._current_epoch = epoch_idx

            self.train_epoch(epoch_idx)
            # Eval cadence: run the (heavy, multi-slice) eval every EVAL_EVERY_N
            # epochs, and always on epoch 0 and the final epoch. Default 1 = every
            # epoch (unchanged). Cuts eval cost on long sweeps; the per-epoch curve
            # is sparser but still captures convergence + a coherent final point.
            eval_every = int(os.environ.get("EVAL_EVERY_N", "1"))
            do_eval = (eval_every <= 1) or (epoch_idx % eval_every == 0) \
                or (epoch_idx == self.max_epochs - 1)
            if do_eval:
                metrics = self.test(epoch_idx)
                # Track per-epoch metrics for resume-aware max_acc accounting.
                try:
                    self._per_epoch_history.append({"epoch": epoch_idx,
                        **{k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}})
                except Exception:
                    pass
                # Best-checkpoint save (model only): enables post-hoc rank/re-eval without
                # re-training. Keeps the single best-by-valid-accuracy state_dict.
                ckpt_path = os.environ.get("BEST_CKPT_PATH")
                if ckpt_path:
                    acc = float(metrics.get("valid/accuracy", 0.0))
                    if acc > getattr(self, "_best_ckpt_acc", -1.0):
                        self._best_ckpt_acc = acc
                        try:
                            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                            torch.save({"model": self.model.state_dict(), "epoch": epoch_idx,
                                        "valid_accuracy": acc}, ckpt_path)
                        except Exception as e:
                            print(f"[ckpt] save failed: {e}", flush=True)

                # early stopping
                if (self.early_stopping_metric is not None) and metrics[
                    self.early_stopping_metric
                ] > self.early_stopping_threshold:
                    print(
                        f"Early stopping triggered at epoch {epoch_idx} with "
                        f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                    )
                    break

            if self.scheduler is not None:
                self.scheduler.step()

            # Periodic checkpoint: save every N epochs regardless of SIGTERM.
            # Robust against SIGKILL or weird preemption modes that skip the
            # 30s SIGTERM grace. Saves AFTER scheduler.step() so resume picks
            # up the post-step LR. Skipped if RESUME_STATE_URI isn't set.
            periodic_n = int(os.environ.get("PERIODIC_CHECKPOINT_EVERY", "4"))
            if periodic_n > 0 and (epoch_idx + 1) % periodic_n == 0:
                self._save_to_gcs(epoch_idx)


        # Successful completion: delete the in-progress GCS checkpoint.
        # (SIGTERM handler exits before reaching here, so its checkpoint is preserved.)
        self._cleanup_gcs_checkpoint()



def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    set_determinism(config.seed)
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)

    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
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
    task.fit()
    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
