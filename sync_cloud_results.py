"""Pull all cloud sweep results to a local cache + summary table.

Run:
    python sync_cloud_results.py
    python sync_cloud_results.py --table-only

Outputs:
    cloud_results_cache.jsonl   — every record from every sweep dir
    cloud_results_summary.md    — per-state breakdown table (markdown)

Implementation: uses google-cloud-storage Python SDK with a thread pool
to parallel-fetch every results.jsonl in the bucket — much faster than
spawning gsutil subprocesses serially.
"""
import os, sys, json, re, argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / "cloud" / ".env"
CACHE = ROOT / "cloud_results_cache.jsonl"
SUMMARY = ROOT / "cloud_results_summary.md"


def load_env():
    env = {}
    if ENV_PATH.exists():
        for line in open(ENV_PATH):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            k, _, v = line.partition('=')
            env[k] = v.strip('"\'')
    return env


def pull_all(bucket_name):
    """Find every results.jsonl under the bucket and fetch them in parallel."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # One server-side list of every blob, filtered to results.jsonl.
    # No fan-out walk; just one paginated request.
    blobs = [b for b in client.list_blobs(bucket)
             if b.name.endswith('results.jsonl') and '/in_progress/' not in b.name]
    print(f"  found {len(blobs)} results.jsonl files", flush=True)

    def fetch(blob):
        try:
            return blob.name, blob.download_as_text()
        except Exception as e:
            return blob.name, f"<error: {e}>"

    records = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for name, text in pool.map(fetch, blobs):
            for line in text.splitlines():
                line = line.strip()
                if line.startswith('{'):
                    records.append((name, line))
    return records


def dedup(records):
    best = {}
    for src, line in records:
        try: r = json.loads(line)
        except: continue
        rid = r.get('run_id')
        if not rid: continue
        r['_src'] = src
        cur = best.get(rid)
        if cur is None:
            best[rid] = r
        elif r.get('ok') and not cur.get('ok'):
            best[rid] = r
        elif r.get('ok') == cur.get('ok'):
            if (r.get('max_acc') or 0) > (cur.get('max_acc') or 0):
                best[rid] = r
    return list(best.values())


# State formula for known cells (the ones we set up by hand).
# Other cells are inferred from the run_id pattern `(rla|cla-rla)-nc?-d?-dv?`.
STATE_OVERRIDES = {
    'cla-rla-d24-nc16': 38400, 'rla-d384-dv24': 38400, 'rla-d98-dv98': 38808,
    'cla-rla-d24-nc64': 153600, 'rla-d1536-dv24': 153600, 'rla-d196-dv196': 154448,
    'cla-rla-d16-nc16': 17408, 'rla-d256-dv16': 17408, 'rla-d66-dv66': 17688,
    'cla-rla-d16-dv8-nc8': 4608, 'rla-d34-dv34': 4760, 'rla-d128-dv8': 4608,
}


def cell_state(cell):
    if cell in STATE_OVERRIDES: return STATE_OVERRIDES[cell]
    # Infer from run_id pattern: e.g. cla-rla-nc16-d16-dv8, rla-d210-dv10
    m = re.match(r'^cla-rla-nc(\d+)-d(\d+)-dv(\d+)$', cell)
    if m:
        nc, d_qk, d_v = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return nc * 4 * d_qk * (d_v + 1)
    m = re.match(r'^rla-d(\d+)-dv(\d+)$', cell)
    if m:
        d_qk, d_v = int(m.group(1)), int(m.group(2))
        return 4 * d_qk * (d_v + 1)
    return 0


VARIANT_RE = re.compile(r'^([\w-]+?)(?:_(norecipe|recipe|linear|mlp-gelu|mlp-relu))?_lr')


def parse_runid(rid):
    m = VARIANT_RE.match(rid)
    if m: return m.group(1), (m.group(2) or 'linear')
    return re.sub(r'_lr.*', '', rid), 'linear'


def render_summary(records):
    rows = []
    for r in records:
        rid = r.get('run_id', '?')
        epoch_ok = (r.get('epochs_run', 0) or 0) >= 20
        if not (r.get('ok') or epoch_ok): continue
        cell, tag = parse_runid(rid)
        sa = r.get('slice_accs') or {}
        def s(k): return sa.get(str(k)) or sa.get(k) or 0.0
        rows.append({
            'cell': cell, 'variant': tag, 'state': cell_state(cell),
            'acc': r.get('max_acc') or 0.0,
            'kv64': s(64), 'kv128': s(128), 'kv256': s(256),
            'grok': r.get('grok_ep'),
            'params': r.get('n_params'),
            'lr': re.search(r'_lr([\de.-]+)', rid).group(1) if re.search(r'_lr([\de.-]+)', rid) else '',
            'seed': re.search(r'_s(\d+)$', rid).group(1) if re.search(r'_s(\d+)$', rid) else '',
        })

    lines = ["# Cloud sweep results — local cache", "",
             f"_{len(rows)} records, sorted within each state group by max_acc descending_", ""]
    by_state = defaultdict(list)
    for row in rows: by_state[row['state']].append(row)
    for state in sorted(by_state):
        lines += [f"## State ≈ {state:,}", ""]
        lines += ["| cell | variant | params | seed | lr | acc | kv=64 | kv=128 | kv=256 | grok_ep |",
                  "|------|---------|--------|------|----|-----|-------|--------|--------|---------|"]
        for r in sorted(by_state[state], key=lambda x: -x['acc']):
            p = f"{r['params']/1e6:.2f}M" if r['params'] else '?'
            lines.append(
                f"| {r['cell']} | {r['variant']} | {p} | {r['seed']} | {r['lr']} | "
                f"{r['acc']:.3f} | {r['kv64']:.2f} | {r['kv128']:.2f} | {r['kv256']:.2f} | "
                f"{r['grok'] if r['grok'] is not None else '—'} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table-only', action='store_true')
    args = ap.parse_args()

    if args.table_only:
        records = []
        if CACHE.exists():
            for line in open(CACHE):
                try: records.append(json.loads(line))
                except: pass
    else:
        env = load_env()
        bucket = env.get('GCS_BUCKET')
        if not bucket:
            print(f"ERROR: GCS_BUCKET not in {ENV_PATH}", file=sys.stderr); sys.exit(1)
        import time
        t0 = time.time()
        print(f"Pulling from gs://{bucket}/ ...", flush=True)
        raw = pull_all(bucket)
        print(f"  {len(raw)} raw records → deduping  ({time.time()-t0:.1f}s)", flush=True)
        records = dedup(raw)
        with open(CACHE, 'w') as f:
            for r in records: f.write(json.dumps(r) + '\n')
        print(f"  wrote {len(records)} records → {CACHE}", flush=True)

    SUMMARY.write_text(render_summary(records))
    print(f"  wrote summary → {SUMMARY}")


if __name__ == '__main__':
    main()
