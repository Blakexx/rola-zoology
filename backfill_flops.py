"""Backfill `fwd_flops` for rows in an rla_sweep-family results.jsonl that
have `ok=True` but `fwd_flops=None`.

Why: when the FLOPs probe was broken (wandb stdout interception ate the
FLOPS_JSON line), completed runs were written with fwd_flops=None. The
training itself is fine; we just lost the (cheap) FLOPs measurement.

Strategy: for each unique cell (mixer × d_model × n_layers × batch × seq) in
the jsonl that needs FLOPs, look it up in /tmp/cla_flops_cache.json; if not
cached yet, instantiate the model and run a single profiler forward pass to
populate. Then update every row with that cell config in-place.

Usage:
  python backfill_flops.py [results_jsonl]   # default: rla_sweep_results.jsonl

Designed to be re-runnable: if all rows already have FLOPs, exits cleanly.
"""
import argparse
import hashlib
import json
import os
import sys

# Match the subprocess environment so we don't fight wandb stdout capture.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")

sys.path.insert(0, "/mnt/c/Users/Blake/Documents/VSCode/CLA")
sys.path.insert(0, "/home/blake/zoology")

import torch
from zoology.model import LanguageModel
from torch.profiler import profile, ProfilerActivity

CACHE_FILE = "/tmp/cla_flops_cache.json"


def cache_key_for(cfg):
    train_batch = (cfg.data.batch_size if isinstance(cfg.data.batch_size, int)
                   else cfg.data.batch_size[0])
    train_seq = max(c.input_seq_len for c in cfg.data.train_configs)
    mixer_cfg = cfg.model.sequence_mixer
    mixer_str = (mixer_cfg.model_dump_json() if hasattr(mixer_cfg, "model_dump_json")
                 else str(mixer_cfg))
    key_blob = json.dumps({
        "mixer": mixer_str,
        "d_model": cfg.model.d_model,
        "n_layers": cfg.model.n_layers,
        "batch": train_batch,
        "seq": train_seq,
    }, sort_keys=True)
    return hashlib.md5(key_blob.encode()).hexdigest(), train_batch, train_seq


def profile_cell(cfg):
    """Instantiate the model, capture forward FLOPs broken down by op.
    Returns (total, by_op_dict)."""
    train_batch = (cfg.data.batch_size if isinstance(cfg.data.batch_size, int)
                   else cfg.data.batch_size[0])
    train_seq = max(c.input_seq_len for c in cfg.data.train_configs)
    model = LanguageModel(cfg.model).cuda()
    x = torch.randint(0, cfg.data.train_configs[0].vocab_size,
                      (train_batch, train_seq), device="cuda")
    model.eval()
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                                  with_flops=True) as prof:
        _ = model(x)
    torch.cuda.synchronize()
    by_op = {}
    for e in prof.key_averages():
        f = getattr(e, "flops", 0) or 0
        if f > 0:
            by_op[e.key] = by_op.get(e.key, 0) + f
    total = sum(by_op.values())
    del model
    torch.cuda.empty_cache()
    return total, by_op


def load_configs_for(jsonl_path):
    """Infer the sweep module from the path stem and import its configs."""
    name = os.path.basename(jsonl_path).removesuffix("_results.jsonl")
    print(f"Importing zoology.experiments.{name}", flush=True)
    import importlib
    mod = importlib.import_module(f"zoology.experiments.{name}")
    return {c.run_id: c for c in mod.configs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?", default="/home/blake/zoology/rla_sweep_results.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"No such file: {args.jsonl}")
        return 1

    cfg_by_run_id = load_configs_for(args.jsonl)
    print(f"Loaded {len(cfg_by_run_id)} configs from sweep module")

    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE))
        except Exception:
            cache = {}
    print(f"Existing cache entries: {len(cache)}")

    with open(args.jsonl) as f:
        rows = [json.loads(l) for l in f]
    print(f"Read {len(rows)} rows from jsonl")

    # Rows need backfill if (a) no FLOPs at all, OR (b) FLOPs present but no
    # per-op breakdown (older runs from before the per-op feature).
    need = [r for r in rows if r.get("ok") and (
        r.get("fwd_flops") is None or r.get("flops_by_op") is None)]
    print(f"Rows needing FLOPs (or per-op breakdown): {len(need)}")
    if not need:
        print("Nothing to backfill. Exiting.")
        return 0

    # Group needing rows by their cell's cache_key
    unique_cells = {}
    skipped_no_cfg = 0
    for r in need:
        cfg = cfg_by_run_id.get(r["run_id"])
        if cfg is None:
            skipped_no_cfg += 1
            continue
        ck, batch, seq = cache_key_for(cfg)
        if ck not in unique_cells:
            unique_cells[ck] = (cfg, batch, seq)
    print(f"Unique cells to populate: {len(unique_cells)}  (skipped {skipped_no_cfg} rows: no matching cfg)")

    # Profile any not in cache OR in cache as legacy int (no per-op breakdown)
    n_fresh = 0
    for ck, (cfg, batch, seq) in unique_cells.items():
        cell_short = cfg.run_id.rsplit("_lr", 1)[0]
        cached = cache.get(ck)
        is_dict_entry = isinstance(cached, dict) and "by_op" in cached
        if is_dict_entry:
            print(f"  cached {cell_short:36s} fwd_flops={cached['total']/1e9:>7.2f} G  (by_op: {len(cached['by_op'])} ops)")
            continue
        print(f"  profile {cell_short:36s} ...", end=" ", flush=True)
        try:
            total, by_op = profile_cell(cfg)
            cache[ck] = {"total": total, "by_op": by_op, "batch": batch, "seq": seq}
            n_fresh += 1
            print(f"fwd_flops={total/1e9:>7.2f} G  by_op={list(by_op.keys())}")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")

    # Persist updated cache
    json.dump(cache, open(CACHE_FILE, "w"))
    print(f"Cache: {len(cache)} entries ({n_fresh} freshly profiled)")

    # Backfill rows in-place. Treats "needs backfill" as: missing fwd_flops OR
    # missing flops_by_op (older rows captured only total).
    updated = 0
    for r in rows:
        if not r.get("ok"):
            continue
        needs_total = r.get("fwd_flops") is None
        needs_by_op = r.get("flops_by_op") is None
        if not (needs_total or needs_by_op):
            continue
        cfg = cfg_by_run_id.get(r["run_id"])
        if cfg is None:
            continue
        ck, batch, seq = cache_key_for(cfg)
        entry = cache.get(ck)
        if isinstance(entry, dict) and "by_op" in entry:
            r["fwd_flops"] = entry["total"]
            r["flops_by_op"] = entry["by_op"]
            r["flops_batch"] = batch
            r["flops_seq"] = seq
            r["flops_cache_hit"] = True
            updated += 1

    print(f"Updated {updated} rows")

    # Atomic rewrite
    tmp_path = args.jsonl + ".tmp"
    with open(tmp_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, args.jsonl)
    print(f"jsonl rewritten: {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
