"""Visualize cloud sweep results from cloud_results_cache.jsonl.

Produces:
  cloud_results_plot.png  — 2-panel scatter (max_acc and kv=256 vs state),
                            colored by routing variant, shaped by arch family.

Usage:
    python plot_results.py
"""
import json, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cloud_results_cache.jsonl"
OUT = ROOT / "cloud_results_plot.png"

STATE_OVERRIDES = {
    'cla-rla-d24-nc16': 38400, 'rla-d384-dv24': 38400, 'rla-d98-dv98': 38808,
    'cla-rla-d16-nc16': 17408, 'rla-d256-dv16': 17408, 'rla-d66-dv66': 17688,
    'cla-rla-d16-dv8-nc8': 4608, 'rla-d34-dv34': 4760, 'rla-d128-dv8': 4608,
}
def cell_state(c):
    if c in STATE_OVERRIDES: return STATE_OVERRIDES[c]
    m = re.match(r'^cla-rla-nc(\d+)-d(\d+)-dv(\d+)$', c)
    if m: return int(m.group(1))*4*int(m.group(2))*(int(m.group(3))+1)
    m = re.match(r'^rla-d(\d+)-dv(\d+)$', c)
    if m: return 4*int(m.group(1))*(int(m.group(2))+1)
    return 0

VARIANT_RE = re.compile(r'^([\w-]+?)(?:_(norecipe|recipe|linear|mlp-gelu|mlp-relu))?_lr')
def parse_runid(rid):
    m = VARIANT_RE.match(rid)
    return (m.group(1), (m.group(2) or 'linear')) if m else (re.sub(r'_lr.*','',rid), 'linear')

def arch_of(cell):
    if not cell.startswith('cla-rla'):
        # RLA: split into wide-symmetric (d_qk == d_v) and wide-asymmetric (d_qk >> d_v)
        m = re.match(r'^rla-d(\d+)-dv(\d+)$', cell)
        if m:
            d_qk, d_v = int(m.group(1)), int(m.group(2))
            return 'RLA-sym' if abs(d_qk - d_v) <= 2 else 'RLA-asym'
        return 'RLA'
    return 'CLA'

# Load
recs = [json.loads(l) for l in open(CACHE)]
points = []
for r in recs:
    if not (r.get('ok') or (r.get('epochs_run', 0) or 0) >= 20): continue
    rid = r.get('run_id', '')
    if rid.endswith('_lr1.00e-04_s1337'): continue
    cell, tag = parse_runid(rid)
    sa = r.get('slice_accs') or {}
    points.append({
        'cell': cell, 'tag': tag, 'arch': arch_of(cell),
        'state': cell_state(cell),
        'acc': r.get('max_acc', 0),
        'kv256': sa.get('256') or sa.get(256, 0) or 0,
        'params': r.get('n_params', 0) or 0,
    })

# Dedup (keep highest acc per (cell, tag))
best = {}
for p in points:
    k = (p['cell'], p['tag'])
    if k not in best or p['acc'] > best[k]['acc']:
        best[k] = p
points = list(best.values())

# Style mapping
COLORS = {
    'linear':   '#1f77b4',
    'mlp-gelu': '#2ca02c',
    'mlp-relu': '#d62728',
    'recipe':   '#ff7f0e',
}
MARKERS = {
    'CLA':       'o',
    'RLA-asym':  's',
    'RLA-sym':   '^',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

for ax, ykey, title in [(ax1, 'acc', 'max_acc (overall)'),
                        (ax2, 'kv256', 'kv=256 (hardest slice)')]:
    for arch, marker in MARKERS.items():
        for tag, color in COLORS.items():
            xs = [p['state'] for p in points if p['arch']==arch and p['tag']==tag]
            ys = [p[ykey] for p in points if p['arch']==arch and p['tag']==tag]
            if not xs: continue
            label = f"{arch} / {tag}" if ax is ax1 else None
            ax.scatter(xs, ys, marker=marker, color=color, s=60, alpha=0.8,
                       edgecolor='black', linewidth=0.5, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('state (log scale)')
    ax.set_ylabel(ykey)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Highlight the matched-state-9.2k group
    ax.axvspan(8900, 9550, alpha=0.05, color='gray')
    # Mark the CLA-wins-RLA point if present
    for p in points:
        if p['cell'] == 'cla-rla-nc18-d10-dv12' and p['tag'] == 'mlp-relu':
            ax.annotate('CLA beats RLA\n(nc18-d10-dv12)', xy=(p['state'], p[ykey]),
                        xytext=(15000, p[ykey]-0.06 if ax is ax1 else p[ykey]-0.18),
                        fontsize=8, ha='center',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))

ax1.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=2)
ax1.set_ylim(0.7, 1.0)
ax2.set_ylim(0.1, 1.0)

plt.suptitle(f"CLA-RLA matched-state sweep ({len(points)} runs, single seed @ lr=1e-2)",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print(f"wrote {OUT}  ({len(points)} points)")
