"""E05: figure production for the paper.

Figure list (outputs results/figures/, PDF+PNG, English labels):
  fig1_multi_property.pdf    multi-property: RF/MLP test MAE across 15 targets
  fig2_factors_matrix.pdf    determinant matrix (target x gini/delta%/LoEO ratio)
  fig3_five_series.pdf       charge dominance across five series (Gini +- CI, ablation delta%)
  fig4_transfer_matrix.pdf   zero-shot transfer matrix heatmap (5x5)
  fig5_finetune_curve.pdf    fine-tuning sample efficiency (4 pairs x 3 budgets x 2 arms)
  fig6_design_bench.pdf      inverse-design hit rate of four methods
Figures with missing input data are skipped automatically (prints skip).
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RESULTS, EV_PER_HA

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'STIXGeneral', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 0.8

FIGDIR = RESULTS / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)
SERIES = ['3d', '4d', '5d', '4f', '5f']
S_COLOR = {'3d': '#4C72B0', '4d': '#55A868', '5d': '#8172B2', '4f': '#C44E52', '5f': '#937860'}


def load_csv(path):
    if not Path(path).exists():
        print(f'  skip: {path} not found')
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.pdf/png', flush=True)


def fig1():
    rows = load_csv(RESULTS / 'E02' / 'full_targets_test_both.csv') or \
        load_csv(RESULTS / 'E02' / 'full_targets_test_rf.csv')
    mlp = load_csv(RESULTS / 'E02' / 'full_targets_test_mlp.csv')
    if rows is None or mlp is None:
        return
    mlp_by = {r['target']: r for r in mlp}
    names = [r['target'] for r in rows]
    rf_v = [float(r['rf_test']) for r in rows]
    mlp_v = [float(mlp_by[n]['mlp_test']) if mlp_by[n]['mlp_test'] else np.nan for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    ax.barh(y + 0.2, rf_v, height=0.38, color='#4C72B0', label='RF')
    ax.barh(y - 0.2, mlp_v, height=0.38, color='#C44E52', label='MLP (GPU)')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Test MAE (native units, log scale)', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.invert_yaxis()
    save(fig, 'fig1_multi_property')


def fig2():
    dm = load_csv(RESULTS / 'E02b' / 'decision_matrix.csv')
    if dm is None:
        return
    names = [r['target'] for r in dm]
    gini = np.array([float(r['gini']) for r in dm])
    dlt = np.array([float(r['delta_pct']) for r in dm])
    loeo = np.array([float(r['loeo_mean']) for r in dm])
    cv = {r['target']: float(r['rf_cv_mean']) for r in (load_csv(RESULTS / 'E02' / 'full_targets_cv_rf.csv') or [])}
    ratio = np.array([loeo[i] / max(gini[i] * 0 + 1e-12 + cv.get(names[i], np.nan), 1e-12)
                      if cv.get(names[i]) else np.nan for i in range(len(names))])
    M = np.vstack([gini, np.clip(dlt / 100, 0, 6) / 6, np.clip(ratio, 0, 2) / 2]).T
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['charge Gini', 'ablation Δ%', 'LoEO/CV'], fontsize=9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label='normalized (see caption)')
    save(fig, 'fig2_factors_matrix')


def fig3():
    ci = load_csv(RESULTS / 'E03' / 'gini_ci.csv')
    ab = load_csv(RESULTS / 'E03' / 'ablation_series.csv')
    if ci is None or ab is None:
        return
    ab_by = {r['series']: r for r in ab}
    x = np.arange(len(SERIES))
    g = np.array([float(r['gini_mean']) for r in ci])
    lo = np.array([float(r['ci_lo']) for r in ci])
    hi = np.array([float(r['ci_hi']) for r in ci])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    for i, s in enumerate(SERIES):
        ax.errorbar(x[i], g[i], yerr=[[g[i] - lo[i]], [hi[i] - g[i]]], fmt='o',
                    color=S_COLOR[s], capsize=3, ms=6)
    ax.set_xticks(x); ax.set_xticklabels(SERIES)
    ax.set_ylim(0.78, 1.0)
    ax.set_ylabel('charge Gini (homo)', fontsize=9)
    ax.axhline(0.95, ls=':', c='gray', lw=0.7)
    ax = axes[1]
    av = [float(ab_by[s]['delta_pct']) for s in SERIES]
    ax.bar(x, av, color=[S_COLOR[s] for s in SERIES], width=0.62)
    ax.set_xticks(x); ax.set_xticklabels(SERIES)
    ax.set_ylabel('MAE increase when charge removed (%)', fontsize=9)
    save(fig, 'fig3_five_series')


def fig4():
    zs = load_csv(RESULTS / 'E03' / 'transfer_zeroshot.csv')
    if zs is None:
        return
    M = np.full((5, 5), np.nan)
    for r in zs:
        M[SERIES.index(r['src']), SERIES.index(r['tgt'])] = float(r['zeroshot_mae_eV'])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(np.log10(M), cmap='RdYlGn_r', vmin=np.log10(0.2), vmax=np.log10(6))
    for i in range(5):
        for j in range(5):
            v = M[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5)
    ax.set_xticks(range(5)); ax.set_xticklabels([f'{s} (tgt)' for s in SERIES], fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels([f'{s} (src)' for s in SERIES], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label='log10 zero-shot MAE (eV)')
    save(fig, 'fig4_transfer_matrix')


def fig5():
    cur = load_csv(RESULTS / 'E03' / 'transfer_curve.csv')
    if cur is None:
        return
    pairs = sorted(set((r['src'], r['tgt']) for r in cur))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    budgets = [25, 100, 500]
    for src, tgt in pairs:
        sel = [r for r in cur if r['src'] == src and r['tgt'] == tgt]
        if not sel:
            continue
        sc = [np.mean([float(r['scratch_mae']) for r in sel if int(r['n']) == n]) * EV_PER_HA
              for n in budgets]
        ft = [np.mean([float(r['finetune_mae']) for r in sel if int(r['n']) == n]) * EV_PER_HA
              for n in budgets]
        ax.plot(budgets, sc, ls='--', marker='o', ms=4, lw=1,
                label=f'{src}->{tgt} scratch')
        ax.plot(budgets, ft, ls='-', marker='s', ms=4, lw=1.4,
                label=f'{src}->{tgt} finetune')
    ax.set_xscale('log')
    ax.set_xlabel('target-domain training size', fontsize=9)
    ax.set_ylabel('target test MAE (eV)', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    save(fig, 'fig5_finetune_curve')


def fig6():
    bench = load_csv(RESULTS / 'E04' / 'design_benchmark.csv')
    if bench is None:
        return
    targets = sorted(set(r['target'] for r in bench))
    methods = ['gradient', 'random', 'ga', 'greedy']
    x = np.arange(len(targets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for k, m in enumerate(methods):
        hr = [float([r for r in bench if r['target'] == t and r['method'] == m][0]['hit_rate'])
              for t in targets]
        ax.bar(x + (k - 1.5) * w, hr, width=w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=8)
    ax.set_ylabel('hit rate (|Δ|/σ < 0.1)', fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=4)
    save(fig, 'fig6_design_bench')


if __name__ == '__main__':
    print('fig1...'); fig1()
    print('fig2...'); fig2()
    print('fig3...'); fig3()
    print('fig4...'); fig4()
    print('fig5...'); fig5()
    print('fig6...'); fig6()
    print('DONE E05')
