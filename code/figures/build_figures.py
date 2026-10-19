"""Rebuild changed scientific figures only from completed result records."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, r'home\.agents\skills\sci-figures')
from figstyle import apply_house_style

plt = apply_house_style()
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.grid': False})

WORK = Path(r'workspace')
OLD = Path(r'legacy_project\submission_package\ijismd_submission_ready_v14_reviewer_aligned_20261018\02_figures_TIF')
OUT = WORK / 'manuscript/figures_v15'
RESULTS = OLD.parents[2] / 'results'
COLORS = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#555555']
MANIFEST = []


def save(fig, number, title, sources):
    OUT.mkdir(exist_ok=True, parents=True)
    stem = OUT / f'Figure_{number:02d}_{title}'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    fig.savefig(stem.with_suffix('.TIF'), dpi=900,
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / f'preview_{number:02d}.png', dpi=150)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    bad = []
    for text in fig.findobj(match=plt.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            bad.append(text.get_text())
    assert not bad, (number, bad)
    MANIFEST.append({'number': number, 'stem': str(stem),
                     'sources': [str(x) for x in sources], 'outside_canvas': bad,
                     'size_inches': fig.get_size_inches().tolist(), 'dpi': 900})
    plt.close(fig)


def study():
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.subplots_adjust(left=.025, right=.975, bottom=.02, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    boxes = [
        (.04, .80, .92, .16, 'Source and role audit',
         'Provenance, predictor groups, conflicts and label support'),
        (.04, .55, .44, .17, 'Local utility',
         'Weighted flat, calibrated blend, stacker\nand real taxonomy blend'),
        (.52, .55, .44, .17, 'Assignment controls',
         'Wrong, random, feature, language\nand standard informed maps'),
        (.04, .28, .44, .18, 'Robustness',
         'Paired uncertainty and predictor projection\nSeparate regrouped confirmation'),
        (.52, .28, .44, .18, 'Transfer boundaries',
         'IoTID20, official archive and NF ToN\nGrouped and genuine chronological roles'),
        (.04, .025, .92, .16, 'Local adoption decision',
         'Effects, calibration, alert budgets and workstation costs\nProduction and physical edge validation remain necessary'),
    ]
    for i, (x, y, w, h, heading, body) in enumerate(boxes):
        ax.add_patch(Rectangle((x, y), w, h, facecolor='#F6F6F6',
                               edgecolor=COLORS[i % 3], linewidth=1.0))
        ax.text(x + w/2, y + h*.72, heading, ha='center', va='center', weight='bold', fontsize=9)
        ax.text(x + w/2, y + h*.34, body, ha='center', va='center', fontsize=8.2, linespacing=1.35)
    for a, b in [((.26, .80), (.26, .73)), ((.74, .80), (.74, .73)),
                 ((.26, .55), (.26, .47)), ((.74, .55), (.74, .47)),
                 ((.26, .28), (.26, .19)), ((.74, .28), (.74, .19))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                    color='#555555', linewidth=1.1))
    save(fig, 1, 'Study_Design', [WORK / 'manuscript/REVISION_CONTRACT.md'])


def random_reference():
    root = WORK / 'pseudo_null/expanded_seed42'
    assert (root / 'COMPLETED.json').exists()
    data = pd.read_csv(root / 'pseudo_taxonomy_metrics.csv')
    summary = pd.read_csv(root / 'expanded_null_summary.csv').set_index('metric')
    assert len(data) == 19
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    fig.subplots_adjust(left=.10, right=.98, bottom=.23, top=.88, wspace=.35)
    for ax, metric, label, color in zip(axes, ['f1_macro', 'rare_recall'],
                                         ['Macro F1', 'Rare recall'], COLORS[:2]):
        values = data[metric].sort_values().to_numpy()
        real = float(summary.loc[metric, 'real'])
        ax.scatter(np.arange(1, 20), values, s=24, color=color, zorder=3, label='Random maps')
        ax.axhline(real, color='#222222', linestyle='--', linewidth=1.2, label='Real map')
        ax.set(xlabel='Random map rank', ylabel=label, xticks=[1, 5, 10, 15, 19])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E2E2E2', linewidth=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=2, frameon=False)
    save(fig, 4, 'Pseudo_Mappings', [root / 'pseudo_taxonomy_metrics.csv', root / 'expanded_null_summary.csv'])


def mapping_controls():
    core_source = RESULTS / 'ijismd_v10_ciciot_grouped_core_6split/nested_split_results.csv'
    attack_source = RESULTS / 'ijismd_v10_attack_alignment_control_6split_corrected/model_results.csv'
    core = pd.read_csv(core_source)
    attack = pd.read_csv(attack_source)
    configs = [
        (core, ['flat_weighted_xgboost', 'equal_flat_taxonomy_blend',
                'calibrated_cyclic_taxonomy_control', 'calibrated_support_matched_control'],
         ['Flat', 'Real', 'Cyclic', 'Support'], 'Original grouping'),
        (attack, ['flat_weighted_xgboost', 'original_taxonomy_equal_blend',
                  'attack_aligned_equal_blend', 'attack_aligned_cyclic_control'],
         ['Flat', 'Original', 'ATT&CK', 'Wrong'], 'ATT&CK grouping'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.88, wspace=.32)
    for ax, (frame, methods, labels, title) in zip(axes, configs):
        ax.set_title(title)
        x = np.arange(4)
        for offset, metric, name, color, hatch in [
            (-.18, 'f1_macro', 'Macro F1', COLORS[0], None),
            (.18, 'rare_recall', 'Rare recall', COLORS[1], '//'),
        ]:
            values = [frame.loc[frame.run_id.eq(m), metric] for m in methods]
            assert all(len(v) == 6 for v in values)
            ax.bar(x + offset, [v.mean() for v in values], width=.34,
                   yerr=[v.std(ddof=1) for v in values], capsize=2,
                   color=color, hatch=hatch, label=name)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel='Score')
        ax.grid(axis='y', color='#E2E2E2', linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=2, frameon=False)
    save(fig, 3, 'Mapping_Controls', [core_source, attack_source])


def language_prior():
    assets = RESULTS / 'jit_llm_taxonomy_assets'
    ledger_root = RESULTS / 'jit_frozen_evidence_ledger_v7_ab'
    matrix_source = assets / 'affinity_matrices.npz'
    bootstrap_source = ledger_root / 'tables/llm_semantics_paired_bootstrap.csv'
    description_source = assets / 'descriptions.json'
    descriptions = json.loads(description_source.read_text(encoding='utf-8'))
    with np.load(matrix_source) as matrices:
        affinity = matrices['S_mpnet_tau_0.5']
    fine = descriptions['fine_labels']
    assert affinity.shape == (34, 8)
    short = [s.replace('_Attack', '').replace('_Malware', '').replace('_Flood', '')
             .replace('_flood', '').replace('BenignTraffic', 'Benign')
             .replace('DictionaryBruteForce', 'Brute force').replace('Fragmentation', 'frag.')
             .replace('SynonymousIP', 'Synonym IP').replace('HostDiscovery', 'Hosts')
             .replace('VulnerabilityScan', 'Vulnerability').replace('_', ' ').replace('-', ' ')
             for s in fine]
    families = ['Benign', 'Brute force', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']
    assert len(set(short)) == 34
    bootstrap = pd.read_csv(bootstrap_source)
    ledger = json.loads((ledger_root / 'evidence_ab_ledger.json').read_text())
    primary = ledger['experiment_b']['primary_run_id']
    fig = plt.figure(figsize=(6.4, 5.7))
    ax = fig.add_axes([.22, .16, .36, .77])
    image = ax.imshow(affinity, aspect='auto', cmap='cividis', vmin=0, vmax=1)
    ax.set(xticks=np.arange(8), xticklabels=families,
           yticks=np.arange(34), yticklabels=short, title='Description affinities')
    ax.tick_params(axis='y', labelsize=7, length=0)
    ax.tick_params(axis='x', labelsize=7, rotation=55, length=0)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    cax = fig.add_axes([.59, .16, .018, .77])
    fig.colorbar(image, cax=cax, ticks=[0, .5, 1])
    ax = fig.add_axes([.75, .38, .22, .38])
    comparators = ['flat_weighted_xgboost', 'llm_mpnet_cyclic_tau_0.5',
                   'llm_mpnet_name_tau_0.5']
    values = [bootstrap.loc[bootstrap.primary.eq(primary) & bootstrap.comparator.eq(m),
                            'macro_f1_delta_point'] for m in comparators]
    assert all(len(v) == 6 for v in values)
    ax.bar(np.arange(3), [v.mean() for v in values], yerr=[v.std(ddof=1) for v in values],
           color=COLORS[0], capsize=3, width=.6)
    ax.set(xticks=np.arange(3), xticklabels=['Flat', 'Cyclic', 'Names'],
           ylabel='Change in Macro F1', title='Prior contrasts')
    ax.axhline(0, color='#333333', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E2E2E2', linewidth=.5)
    ax.set_axisbelow(True)
    save(fig, 5, 'Language_Prior', [matrix_source, description_source, bootstrap_source])


def unknown():
    source = WORK / 'extensions/lofo_results.csv'
    data = pd.read_csv(source)
    assert data.groupby('method').size().eq(21).all()
    families = ['BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-based']
    displays = ['Brute force', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    fig.subplots_adjust(left=.10, right=.98, top=.98, bottom=.32)
    methods = ['flat_xgb', 'temperature_flat_blend', 'newton_flat_stacker', 'taxonomy_blend', 'cyclic_blend']
    names = ['Flat XGBoost', 'Calibrated flat', 'Flat stacker', 'Real blend', 'Cyclic blend']
    for method, name, color, marker, line in zip(methods, names, COLORS,
                                                ['o', 's', '^', 'D', 'x'], ['-', '--', ':', '-.', '--']):
        part = data[data.method.eq(method)].groupby('case')['msp_unknown_auroc'].mean().reindex(families)
        assert part.notna().all()
        ax.plot(np.arange(7), part.values, color=color, marker=marker,
                linestyle=line, linewidth=1.2, markersize=4, label=name)
    ax.set(xticks=np.arange(7), xticklabels=displays, ylabel='Unknown family AUROC', ylim=(0, 1))
    ax.tick_params(axis='x', rotation=25)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E2E2E2', linewidth=.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3, frameon=False)
    save(fig, 9, 'Unknown_Flagging', [source])


def capture():
    source = WORK / 'extensions/capture_windows_results.csv'
    data = pd.read_csv(source)
    assert data.window.nunique() == 4
    windows = sorted(data.window.unique())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    fig.subplots_adjust(left=.09, right=.98, bottom=.28, top=.95, wspace=.33)
    assert 'legacy_flat_xgb' not in set(data.method)
    methods = ['flat_xgb', 'temperature_flat_blend', 'newton_flat_stacker', 'taxonomy_blend', 'cyclic_blend']
    names = ['Flat XGBoost', 'Calibrated flat', 'Flat stacker', 'Real blend', 'Cyclic blend']
    for ax, metric, ylabel in zip(axes, ['macro_f1', 'rare_recall'], ['Macro F1', 'Rare recall']):
        for method, name, color, marker, line in zip(methods, names, COLORS,
                                                    ['o', 's', '^', 'D', 'x'], ['-', '--', ':', '-.', '--']):
            part = data[data.method.eq(method)].set_index('window').reindex(windows)
            assert part[metric].notna().all(), (method, metric)
            ax.plot(np.arange(4), part[metric].values, color=color, marker=marker,
                    linestyle=line, linewidth=1.2, markersize=4, label=name)
        ax.set(xticks=np.arange(4), xticklabels=['55 to 56', '57 to 58', '59 to 60', '61 to 63'],
               xlabel='Source CSV range', ylabel=ylabel)
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E2E2E2', linewidth=.5)
    axes[0].set(ylim=(.61, .645), yticks=[.61, .62, .63, .64])
    axes[1].set(ylim=(.515, .56), yticks=[.52, .53, .54, .55, .56])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=3, frameon=False)
    save(fig, 10, 'Capture_Order', [source])


def retained():
    OUT.mkdir(exist_ok=True, parents=True)
    for path in sorted(OLD.glob('Figure_*.TIF')):
        number = int(path.name.split('_')[1])
        if number in (1, 3, 4, 5, 9, 10):
            continue
        target = OUT / path.name
        shutil.copy2(path, target)
        with Image.open(path) as image:
            image.thumbnail((1800, 1200))
            image.convert('RGB').save(OUT / f'preview_{number:02d}.png')
        MANIFEST.append({'number': number, 'retained_source': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-only', action='store_true')
    args = parser.parse_args()
    study()
    mapping_controls()
    language_prior()
    retained()
    if (WORK / 'pseudo_null/expanded_seed42/COMPLETED.json').exists():
        random_reference()
    if not args.study_only:
        unknown()
        capture()
        assert len(MANIFEST) == 11
    (OUT / 'figure_manifest.json').write_text(json.dumps(MANIFEST, indent=2))


if __name__ == '__main__':
    main()
