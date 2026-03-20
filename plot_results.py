#!/usr/bin/env python3
"""Plot TFLOPs data from SpMM sweep and GEMM sweep CSV results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = plt.cm.tab10.colors

# ── Load data ──────────────────────────────────────────────────────────────────
spmm = pd.read_csv('sweep_results.csv')
gemm = pd.read_csv('build/gemm_sweep_results.csv')

# Parse sparsity pattern from the Case column in SpMM data
def parse_pattern(case):
    if 'parametric_row' in case:
        return 'Row'
    elif 'parametric_col' in case:
        return 'Column'
    elif 'parametric_multi_diag' in case:
        return 'Multi-Diagonal'
    elif 'parametric_diag' in case:
        return 'Diagonal'
    elif 'parametric_M' in case:
        return 'Random'
    return 'Unknown'

def parse_density(case):
    m = re.search(r'_d(\d+)', case)
    return int(m.group(1)) if m else None

spmm['Pattern'] = spmm['Case'].apply(parse_pattern)
spmm['Density'] = spmm['Case'].apply(parse_density)

# Parse which dimension varies in GEMM data
def parse_gemm_sweep(row):
    m, k, n = row['M'], row['K'], row['N']
    return f"M={m}, K={k}, N={n}"

gemm['Label'] = gemm.apply(parse_gemm_sweep, axis=1)

# Group GEMM by Registry to identify sweep type
def gemm_sweep_type(group):
    ms = group['M'].unique()
    ks = group['K'].unique()
    ns = group['N'].unique()
    if len(ms) > 1 and len(ks) == 1 and len(ns) == 1:
        return 'Sweep M'
    elif len(ns) > 1 and len(ms) == 1 and len(ks) == 1:
        return 'Sweep N'
    elif len(ks) > 1 and len(ms) == 1 and len(ns) == 1:
        return 'Sweep K'
    elif len(ms) > 1 and len(ks) > 1 and len(ns) > 1:
        if (group['M'] == group['K']).all() and (group['K'] == group['N']).all():
            return 'Sweep All (cubic)'
        return 'Sweep M,N (K varies)'
    return 'Mixed'

gemm_groups = {}
for reg, grp in gemm.groupby('Registry'):
    stype = gemm_sweep_type(grp)
    gemm_groups[reg] = {'type': stype, 'data': grp}

outdir = 'plots'
os.makedirs(outdir, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: SpMM — Avg TFLOPs by Sparsity Pattern (grouped bar, one group per density)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
patterns = ['Row', 'Column', 'Diagonal', 'Multi-Diagonal', 'Random']
densities = sorted(spmm['Density'].unique())

x = np.arange(len(densities))
width = 0.15
for i, pat in enumerate(patterns):
    vals = []
    for d in densities:
        subset = spmm[(spmm['Pattern'] == pat) & (spmm['Density'] == d)]
        vals.append(subset['Avg_TFLOPs'].mean() if len(subset) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=pat, color=COLORS[i])

ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('SpMM: Average TFLOPs by Sparsity Pattern and Density', fontsize=15)
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'{d}%' for d in densities])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/01_spmm_tflops_by_pattern_density.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: SpMM — Avg TFLOPs vs Density, one line per pattern
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for i, pat in enumerate(patterns):
    subset = spmm[spmm['Pattern'] == pat].sort_values('Density')
    ax.plot(subset['Density'], subset['Avg_TFLOPs'], 'o-', label=pat,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('SpMM: TFLOPs Scaling with Density per Pattern', fontsize=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/02_spmm_tflops_vs_density_lines.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: SpMM — Avg Latency (ms) vs Density, one line per pattern
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for i, pat in enumerate(patterns):
    subset = spmm[spmm['Pattern'] == pat].sort_values('Density')
    ax.plot(subset['Density'], subset['Avg_ms'], 's-', label=pat,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('Avg Latency (ms)', fontsize=13)
ax.set_title('SpMM: Kernel Latency vs Density per Pattern', fontsize=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/03_spmm_latency_vs_density.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: SpMM — Avg GB/s vs Density, one line per pattern
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for i, pat in enumerate(patterns):
    subset = spmm[spmm['Pattern'] == pat].sort_values('Density')
    ax.plot(subset['Density'], subset['Avg_GBs'], 'D-', label=pat,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('Avg GB/s', fontsize=13)
ax.set_title('SpMM: Memory Bandwidth vs Density per Pattern', fontsize=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/04_spmm_bandwidth_vs_density.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: SpMM — Heatmap of TFLOPs (Pattern × Density)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
heatdata = np.zeros((len(patterns), len(densities)))
for i, pat in enumerate(patterns):
    for j, d in enumerate(densities):
        subset = spmm[(spmm['Pattern'] == pat) & (spmm['Density'] == d)]
        heatdata[i, j] = subset['Avg_TFLOPs'].mean() if len(subset) > 0 else 0

im = ax.imshow(heatdata, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(densities)))
ax.set_xticklabels([f'{d}%' for d in densities])
ax.set_yticks(range(len(patterns)))
ax.set_yticklabels(patterns)
ax.set_xlabel('Density (%)', fontsize=13)
ax.set_title('SpMM: TFLOPs Heatmap (Pattern × Density)', fontsize=15)
for i in range(len(patterns)):
    for j in range(len(densities)):
        ax.text(j, i, f'{heatdata[i,j]:.1f}', ha='center', va='center',
                fontsize=10, color='black' if heatdata[i,j] < heatdata.max()*0.6 else 'white')
plt.colorbar(im, ax=ax, label='Avg TFLOPs')
fig.tight_layout()
fig.savefig(f'{outdir}/05_spmm_tflops_heatmap.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: SpMM — Radar chart comparing patterns at 25% density
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
d25 = spmm[spmm['Density'] == 25]
metrics = ['Avg_TFLOPs', 'Max_TFLOPs', 'Avg_GBs']
metric_labels = ['Avg TFLOPs', 'Max TFLOPs', 'Avg GB/s']

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for i, pat in enumerate(patterns):
    subset = d25[d25['Pattern'] == pat]
    if len(subset) == 0:
        continue
    vals = [subset[m].values[0] for m in metrics]
    # normalize to max across patterns for this density
    maxvals = [d25[m].max() for m in metrics]
    vals_norm = [v / mx if mx > 0 else 0 for v, mx in zip(vals, maxvals)]
    vals_norm += vals_norm[:1]
    ax.plot(angles, vals_norm, 'o-', label=pat, color=COLORS[i], linewidth=2)
    ax.fill(angles, vals_norm, alpha=0.1, color=COLORS[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_title('SpMM: Pattern Comparison at 25% Density\n(normalized)', fontsize=14, pad=20)
ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)
fig.tight_layout()
fig.savefig(f'{outdir}/06_spmm_radar_25pct.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: SpMM — Efficiency ratio (Avg/Max TFLOPs) by pattern and density
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
spmm['Efficiency'] = spmm['Avg_TFLOPs'] / spmm['Max_TFLOPs'] * 100
for i, pat in enumerate(patterns):
    subset = spmm[spmm['Pattern'] == pat].sort_values('Density')
    ax.plot(subset['Density'], subset['Efficiency'], 'o-', label=pat,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('Efficiency (Avg/Max TFLOPs) %', fontsize=13)
ax.set_title('SpMM: Compute Efficiency by Pattern and Density', fontsize=15)
ax.set_ylim(95, 101)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/07_spmm_efficiency.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: GEMM — TFLOPs per sweep type (subplots)
# ═══════════════════════════════════════════════════════════════════════════════
sweep_types = {}
for reg, info in gemm_groups.items():
    stype = info['type']
    if stype not in sweep_types:
        sweep_types[stype] = []
    sweep_types[stype].append(info['data'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
ax_idx = 0

for stype, dfs in sorted(sweep_types.items()):
    if ax_idx >= len(axes):
        break
    ax = axes[ax_idx]
    for df in dfs:
        # Determine the varying dimension
        if stype == 'Sweep M':
            xvals = df['M']
            xlabel = 'M'
        elif stype == 'Sweep N':
            xvals = df['N']
            xlabel = 'N'
        elif stype == 'Sweep K':
            xvals = df['K']
            xlabel = 'K'
        elif 'cubic' in stype:
            xvals = df['M']
            xlabel = 'M=N=K'
        else:
            xvals = df['M']
            xlabel = 'M'

        ax.plot(xvals, df['Avg_TFLOPs'], 'o-', markersize=8, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Avg TFLOPs', fontsize=12)
    ax.set_title(f'GEMM: {stype}', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.grid(alpha=0.3)
    ax_idx += 1

# Hide unused axes
for i in range(ax_idx, len(axes)):
    axes[i].set_visible(False)

fig.suptitle('GEMM: TFLOPs Across Different Dimension Sweeps', fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(f'{outdir}/08_gemm_tflops_sweeps.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: GEMM — All sweeps overlaid on one plot (varying dim on x-axis)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
sweep_labels = {0: 'Sweep M (K=N=8192)', 1: 'Sweep N (M=K=8192)',
                2: 'Sweep K (M=N=8192)', 3: 'Cubic (M=N=K)', 4: 'Large (M=N=32768, sweep K)'}

for i, (reg, info) in enumerate(sorted(gemm_groups.items())):
    df = info['data'].sort_values(['M', 'K', 'N'])
    stype = info['type']
    if stype == 'Sweep M':
        xvals = df['M']
    elif stype == 'Sweep N':
        xvals = df['N']
    elif stype == 'Sweep K':
        xvals = df['K']
    elif 'cubic' in stype:
        xvals = df['M']
    else:
        xvals = df['K']

    label = sweep_labels.get(reg, f'Registry {reg}')
    ax.plot(xvals, df['Avg_TFLOPs'], 'o-', label=label,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Varying Dimension Size', fontsize=13)
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('GEMM: TFLOPs vs Matrix Dimension (All Sweeps)', fontsize=15)
ax.set_xscale('log', base=2)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/09_gemm_tflops_all_sweeps.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: GEMM — Latency vs dimension size
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
for i, (reg, info) in enumerate(sorted(gemm_groups.items())):
    df = info['data'].sort_values(['M', 'K', 'N'])
    stype = info['type']
    if stype == 'Sweep M':
        xvals = df['M']
    elif stype == 'Sweep N':
        xvals = df['N']
    elif stype == 'Sweep K':
        xvals = df['K']
    elif 'cubic' in stype:
        xvals = df['M']
    else:
        xvals = df['K']

    label = sweep_labels.get(reg, f'Registry {reg}')
    ax.plot(xvals, df['Avg_ms'], 's-', label=label,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Varying Dimension Size', fontsize=13)
ax.set_ylabel('Avg Latency (ms)', fontsize=13)
ax.set_title('GEMM: Latency vs Matrix Dimension (All Sweeps)', fontsize=15)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/10_gemm_latency_all_sweeps.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: GEMM — GB/s (memory bandwidth) vs dimension
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
for i, (reg, info) in enumerate(sorted(gemm_groups.items())):
    df = info['data'].sort_values(['M', 'K', 'N'])
    stype = info['type']
    if stype == 'Sweep M':
        xvals = df['M']
    elif stype == 'Sweep N':
        xvals = df['N']
    elif stype == 'Sweep K':
        xvals = df['K']
    elif 'cubic' in stype:
        xvals = df['M']
    else:
        xvals = df['K']

    label = sweep_labels.get(reg, f'Registry {reg}')
    ax.plot(xvals, df['Avg_GBs'], 'D-', label=label,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Varying Dimension Size', fontsize=13)
ax.set_ylabel('Avg GB/s', fontsize=13)
ax.set_title('GEMM: Memory Bandwidth vs Matrix Dimension', fontsize=15)
ax.set_xscale('log', base=2)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/11_gemm_bandwidth_all_sweeps.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: GEMM — Efficiency (Avg/Max TFLOPs)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
gemm['Efficiency'] = gemm['Avg_TFLOPs'] / gemm['Max_TFLOPs'] * 100
for i, (reg, info) in enumerate(sorted(gemm_groups.items())):
    df = info['data'].sort_values(['M', 'K', 'N'])
    df = df.copy()
    df['Efficiency'] = df['Avg_TFLOPs'] / df['Max_TFLOPs'] * 100
    stype = info['type']
    if stype == 'Sweep M':
        xvals = df['M']
    elif stype == 'Sweep N':
        xvals = df['N']
    elif stype == 'Sweep K':
        xvals = df['K']
    elif 'cubic' in stype:
        xvals = df['M']
    else:
        xvals = df['K']

    label = sweep_labels.get(reg, f'Registry {reg}')
    ax.plot(xvals, df['Efficiency'], 'o-', label=label,
            color=COLORS[i], markersize=8, linewidth=2)

ax.set_xlabel('Varying Dimension Size', fontsize=13)
ax.set_ylabel('Efficiency (Avg/Max TFLOPs) %', fontsize=13)
ax.set_title('GEMM: Compute Efficiency vs Matrix Dimension', fontsize=15)
ax.set_xscale('log', base=2)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/12_gemm_efficiency.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: SpMM vs GEMM — comparison at 8192×8192×8192
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

# GEMM baseline at 8192^3
gemm_8k = gemm[(gemm['M'] == 8192) & (gemm['K'] == 8192) & (gemm['N'] == 8192)]
gemm_avg = gemm_8k['Avg_TFLOPs'].mean()

# SpMM data at each density
spmm_by_pattern_density = spmm.groupby(['Pattern', 'Density'])['Avg_TFLOPs'].mean().reset_index()

bar_data = []
labels = []
colors_list = []
for i, pat in enumerate(patterns):
    for d in densities:
        subset = spmm_by_pattern_density[(spmm_by_pattern_density['Pattern'] == pat) &
                                          (spmm_by_pattern_density['Density'] == d)]
        if len(subset) > 0:
            bar_data.append(subset['Avg_TFLOPs'].values[0])
            labels.append(f'{pat}\n{d}%')
            colors_list.append(COLORS[i])

x = np.arange(len(bar_data))
bars = ax.bar(x, bar_data, color=colors_list, edgecolor='white', linewidth=0.5)
ax.axhline(y=gemm_avg, color='red', linestyle='--', linewidth=2.5,
           label=f'Dense GEMM avg ({gemm_avg:.1f} TFLOPs)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('SpMM vs Dense GEMM at 8192×8192×8192', fontsize=15)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/13_spmm_vs_gemm_comparison.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: SpMM — NNZ_Blocks vs TFLOPs (scatter)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for i, pat in enumerate(patterns):
    subset = spmm[spmm['Pattern'] == pat]
    ax.scatter(subset['NNZ_Blocks'], subset['Avg_TFLOPs'], label=pat,
               color=COLORS[i], s=120, edgecolors='black', linewidth=0.5, zorder=3)

ax.set_xlabel('NNZ Blocks', fontsize=13)
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('SpMM: TFLOPs vs Number of Non-Zero Blocks', fontsize=15)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/14_spmm_tflops_vs_nnz.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: SpMM — Stacked bar: TFLOPs contribution at each density
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(densities))
for i, pat in enumerate(patterns):
    vals = []
    for d in densities:
        subset = spmm[(spmm['Pattern'] == pat) & (spmm['Density'] == d)]
        vals.append(subset['Avg_TFLOPs'].mean() if len(subset) > 0 else 0)
    ax.bar(range(len(densities)), vals, bottom=bottom, label=pat, color=COLORS[i])
    bottom += np.array(vals)

ax.set_xticks(range(len(densities)))
ax.set_xticklabels([f'{d}%' for d in densities])
ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('Cumulative TFLOPs', fontsize=13)
ax.set_title('SpMM: Stacked TFLOPs by Pattern at Each Density', fontsize=15)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/15_spmm_stacked_tflops.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 16: GEMM — Arithmetic Intensity proxy (FLOPs/Byte) vs TFLOPs
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
gemm['FLOP_per_byte'] = gemm['Avg_TFLOPs'] / gemm['Avg_GBs'] * 1000  # TFLOPs/GBs → FLOPs/byte ratio
for i, (reg, info) in enumerate(sorted(gemm_groups.items())):
    df = info['data'].copy()
    df['FLOP_per_byte'] = df['Avg_TFLOPs'] / df['Avg_GBs'] * 1000
    label = sweep_labels.get(reg, f'Registry {reg}')
    ax.scatter(df['FLOP_per_byte'], df['Avg_TFLOPs'], label=label,
               color=COLORS[i], s=120, edgecolors='black', linewidth=0.5, zorder=3)

ax.set_xlabel('Arithmetic Intensity (TFLOPs / GB/s × 1000)', fontsize=12)
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('GEMM: Roofline-Style — TFLOPs vs Arithmetic Intensity', fontsize=15)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/16_gemm_roofline_style.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 17: Summary bar — Best TFLOPs per category
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
summary = {}
for pat in patterns:
    best = spmm[spmm['Pattern'] == pat]['Avg_TFLOPs'].max()
    summary[f'SpMM {pat}'] = best
summary['Dense GEMM (best)'] = gemm['Avg_TFLOPs'].max()
summary['Dense GEMM (avg 8k³)'] = gemm_avg

names = list(summary.keys())
vals = list(summary.values())
bar_colors = [COLORS[i] for i in range(len(patterns))] + ['darkred', 'salmon']
bars = ax.barh(names, vals, color=bar_colors, edgecolor='white')
ax.set_xlabel('Peak Avg TFLOPs', fontsize=13)
ax.set_title('Best Achieved TFLOPs: SpMM Patterns vs Dense GEMM', fontsize=15)
for bar, val in zip(bars, vals):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
            va='center', fontsize=11)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/17_best_tflops_summary.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 18: GEMM — Grouped bar of Avg TFLOPs by Registry (like SpMM Fig 1)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))

registries = sorted(gemm['Registry'].unique())
registry_labels = {reg: sweep_labels.get(reg, f'Registry {reg}') for reg in registries}

# Within each registry, the test cases are the individual matrix sizes
# Group by registry; bars are the individual cases
cases_per_reg = {reg: gemm[gemm['Registry'] == reg].sort_values(['M', 'K', 'N']).reset_index(drop=True)
                 for reg in registries}
max_cases = max(len(df) for df in cases_per_reg.values())

x = np.arange(len(registries))
width = 0.8 / max_cases

for case_idx in range(max_cases):
    vals = []
    for reg in registries:
        df = cases_per_reg[reg]
        if case_idx < len(df):
            vals.append(df.iloc[case_idx]['Avg_TFLOPs'])
        else:
            vals.append(0)
    # Build a label from the first registry that has this case index
    case_label = ''
    for reg in registries:
        df = cases_per_reg[reg]
        if case_idx < len(df):
            row = df.iloc[case_idx]
            case_label = f'{int(row["M"])}×{int(row["N"])}×{int(row["K"])}'
            break
    ax.bar(x + case_idx * width - (max_cases - 1) * width / 2, vals, width,
           label=case_label if case_idx < 6 else '', color=COLORS[case_idx % len(COLORS)],
           edgecolor='white', linewidth=0.5)

ax.set_xlabel('Registry (Sweep Type)', fontsize=13)
ax.set_ylabel('Avg TFLOPs', fontsize=13)
ax.set_title('GEMM: Average TFLOPs by Registry and Test Case', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels([registry_labels[r] for r in registries], fontsize=9, rotation=15, ha='right')
ax.legend(title='Matrix Size (M×N×K)', fontsize=8, title_fontsize=10, ncol=2,
          loc='upper right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/18_gemm_tflops_by_registry_grouped.png', dpi=200)
plt.close(fig)

PEAK_TFLOPS = 330  # RTX 4090 Tensor Core theoretical peak

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 19: SpMM — % of Peak TFLOPs by Sparsity Pattern and Density
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
patterns_no_diag = [p for p in patterns if p != 'Diagonal']
x = np.arange(len(densities))
width = 0.18
for i, pat in enumerate(patterns_no_diag):
    vals = []
    for d in densities:
        subset = spmm[(spmm['Pattern'] == pat) & (spmm['Density'] == d)]
        raw = subset['Avg_TFLOPs'].mean() if len(subset) > 0 else 0
        vals.append(raw / PEAK_TFLOPS * 100)
    bars = ax.bar(x + i * width, vals, width, label=pat, color=COLORS[patterns.index(pat)])
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Density (%)', fontsize=13)
ax.set_ylabel('% of RTX 4090 Peak (330 TFLOPs)', fontsize=13)
ax.set_title('SpMM: Achieved % of Theoretical Peak by Pattern and Density', fontsize=15)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'{d}%' for d in densities])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/19_spmm_pct_peak_by_pattern_density.png', dpi=200)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 20: GEMM — % of Peak TFLOPs by Registry and Test Case
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(registries))
width = 0.8 / max_cases

for case_idx in range(max_cases):
    vals = []
    for reg in registries:
        df = cases_per_reg[reg]
        if case_idx < len(df):
            vals.append(df.iloc[case_idx]['Avg_TFLOPs'] / PEAK_TFLOPS * 100)
        else:
            vals.append(0)
    case_label = ''
    for reg in registries:
        df = cases_per_reg[reg]
        if case_idx < len(df):
            row = df.iloc[case_idx]
            case_label = f'{int(row["M"])}×{int(row["N"])}×{int(row["K"])}'
            break
    bars = ax.bar(x + case_idx * width - (max_cases - 1) * width / 2, vals, width,
           label=case_label if case_idx < 6 else '', color=COLORS[case_idx % len(COLORS)],
           edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=6)

ax.set_xlabel('Registry (Sweep Type)', fontsize=13)
ax.set_ylabel('% of RTX 4090 Peak (330 TFLOPs)', fontsize=13)
ax.set_title('GEMM: Achieved % of Theoretical Peak by Registry and Test Case', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels([registry_labels[r] for r in registries], fontsize=9, rotation=15, ha='right')
ax.legend(title='Matrix Size (M×N×K)', fontsize=8, title_fontsize=10, ncol=2,
          loc='upper right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{outdir}/20_gemm_pct_peak_by_registry_grouped.png', dpi=200)
plt.close(fig)

print(f"✓ Generated 20 figures in '{outdir}/' directory")
