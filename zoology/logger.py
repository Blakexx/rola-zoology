from pathlib import Path

import wandb
from torch.nn import Module

from zoology.model import LanguageModel
from zoology.config import LoggerConfig, TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return
        self.no_logger = False
        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name, 
        )
        # wandb.run.log_code(
        #     root=str(Path(__file__).parent.parent),
        #     include_fn=lambda path, root: path.endswith(".py")
        # )

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(
        self, 
        model: LanguageModel,
        config: TrainConfig
    ):
        if self.no_logger:
            return
        
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state_size = model.state_size(sequence_length=max_seq_len)
        # Stdout copy so runners that scrape subprocess stdout can capture these.
        try:
            import json as _json
            print(f"MODEL_STATS_JSON {_json.dumps({'num_parameters': n_params, 'state_size': state_size})}", flush=True)
        except Exception:
            pass

        # Forward FLOPs probe via torch.profiler. ~50ms-1s one-shot per cell at
        # the largest training sequence length and the configured batch size.
        # Cached across runs: identical (architecture × batch × seq_len) tuples
        # reuse the prior measurement via /tmp/cla_flops_cache.json. So each
        # unique cell pays the probe cost once, and the same cell's 20 runs
        # (4 LRs × 5 seeds) reuse the cached number.
        try:
            import os, hashlib, json as _json, torch
            train_batch = (config.data.batch_size if isinstance(config.data.batch_size, int)
                           else config.data.batch_size[0])
            train_seq = max([c.input_seq_len for c in config.data.train_configs])
            # Cache key: hash of (mixer config, d_model, n_layers, batch, train_seq).
            mixer_cfg = config.model.sequence_mixer
            mixer_str = (mixer_cfg.model_dump_json() if hasattr(mixer_cfg, "model_dump_json")
                         else str(mixer_cfg))
            key_blob = _json.dumps({
                "mixer": mixer_str,
                "d_model": config.model.d_model,
                "n_layers": config.model.n_layers,
                "batch": train_batch,
                "seq": train_seq,
            }, sort_keys=True)
            cache_key = hashlib.md5(key_blob.encode()).hexdigest()
            CACHE_FILE = "/tmp/cla_flops_cache.json"
            cache = {}
            if os.path.exists(CACHE_FILE):
                try:
                    cache = _json.load(open(CACHE_FILE))
                except Exception:
                    cache = {}
            # Defensive: some submodules (LayerNorm etc.) may not have been
            # moved by train's .cuda() — re-apply so both (a) the probe
            # forward pass doesn't fail with "weight on cpu, input on cuda"
            # AND (b) cache-hit runs don't pay 4× implicit-copy overhead
            # during training proper. Applies regardless of probe path.
            device = next(model.parameters()).device
            if device.type == "cuda":
                model = model.to(device)
            cached_entry = cache.get(cache_key)
            # Treat old (int-valued) cache entries as stale → re-profile to get
            # per-op breakdown. New entries are dicts {"total": int, "by_op": {...}}.
            if isinstance(cached_entry, dict) and "by_op" in cached_entry:
                total_flops = cached_entry["total"]
                flops_by_op = cached_entry["by_op"]
                cache_hit = True
            else:
                from torch.profiler import profile, ProfilerActivity
                x = torch.randint(0, config.data.train_configs[0].vocab_size,
                                  (train_batch, train_seq), device=device)
                model.eval()
                with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                                              with_flops=True) as prof:
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                flops_by_op = {}
                for e in prof.key_averages():
                    f = getattr(e, "flops", 0) or 0
                    if f > 0:
                        flops_by_op[e.key] = flops_by_op.get(e.key, 0) + f
                total_flops = sum(flops_by_op.values())
                model.train()
                cache[cache_key] = {"total": total_flops, "by_op": flops_by_op,
                                    "batch": train_batch, "seq": train_seq}
                try:
                    _json.dump(cache, open(CACHE_FILE, "w"))
                except Exception:
                    pass
                cache_hit = False
            print(f"FLOPS_JSON {_json.dumps({'forward_flops': total_flops, 'flops_by_op': flops_by_op, 'batch': train_batch, 'seq_len': train_seq, 'cache_hit': cache_hit})}", flush=True)
        except Exception as e:
            try:
                import json as _json, traceback as _tb
                err_info = {
                    'forward_flops': None,
                    'error': f"{type(e).__name__}: {str(e)[:200]}",
                    'tb_tail': _tb.format_exc()[-500:],
                }
                print(f"FLOPS_JSON {_json.dumps(err_info)}", flush=True)
            except Exception:
                pass

        wandb.log({"num_parameters": n_params, "state_size": state_size})
        wandb.watch(model)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        wandb.log(metrics)
    
    def finish(self):
        if self.no_logger:
            return
        self.run.finish()


