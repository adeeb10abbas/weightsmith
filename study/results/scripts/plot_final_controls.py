#!/usr/bin/env python3
"""Present collector-verified final controls, keeping versions and shared data explicit.

This presentation adapter reads derived publication JSON only. It neither scores
episodes nor verifies raw archives: collect_experiment_results.py must do that
first. The frozen plans currently contain zero edited incumbents. Missing final
archives stay pending, never zero-valued observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
SPECS = (
    ('v2', 'identity', 'Fixed16 v2', 'Identity', 'no_edit', 128, False),
    ('v1p1', 'identity', 'Original v1.1', 'Identity', 'no_edit', 128, False),
    ('v1p1', 'gripper-control', 'Original v1.1', 'Gripper 0.4', 'runtime_control', 128, False),
    ('v1p1', 'base-repeat', 'Original v1.1', 'Extra base repeat', 'base_repeat', 64, True),
)
NOTE = ('Versions are separate. Intervals are nominal paired 95% bootstrap intervals; '
        'dependence and fixed execution order limit inference. The extra v1.1 repeat '
        'reuses the identity comparison\'s base rows, so those contrasts are correlated. '
        'Shared editor labels do not add observations. No rejected edit is scored here.')


def build_rows(data):
    finals = {'v1p1': data.get('final_scoring', {}),
              'v2': data.get('followup_v2', {}).get('final_scoring', {})}
    for version, final in finals.items():
        allowed = {s[1] for s in SPECS if s[0] == version}
        planned = final.get('plan', {}).get('comparisons', [])
        if {p['id'] for p in planned} - allowed or set(final.get('comparisons', {})) - allowed:
            raise ValueError('This control-only view cannot silently omit an edited or unknown comparison')
        if planned and {p['id'] for p in planned} != allowed:
            raise ValueError('Prepared control plan differs from the completed-search plan')
    rows = []
    for version, cid, label, name, kind, additional, reused in SPECS:
        final = finals[version]
        a = final.get('comparisons', {}).get(cid)
        row = {'version': version, 'version_label': label, 'comparison_id': cid,
               'label': name, 'status': 'pending', 'additional_episodes': additional,
               'reference_reused': reused, 'shared_roles': []}
        if a is not None:
            if (a.get('status') != 'complete' or a.get('comparison_status') != 'complete'
                    or a.get('comparison_id') != cid or a.get('paired_n') != 64
                    or a.get('expected_n') != 64 or a.get('intervention_kind') != kind
                    or a.get('changed_elements') != 0):
                raise ValueError('Incomplete or wrongly attributed final control')
            roles = [k for k, v in final.get('role_mapping', {}).items() if v['comparison_id'] == cid]
            if a.get('shared_roles') != roles:
                raise ValueError('Shared labels differ from the verified role mapping')
            counts = {k: a[k] for k in ('base_success', 'candidate_success', 'fixed',
                                       'broken', 'both_success', 'both_failure')}
            if any(type(v) is not int or not 0 <= v <= 64 for v in counts.values()):
                raise ValueError('Invalid paired counts')
            if (counts['base_success'] != counts['both_success'] + counts['broken']
                    or counts['candidate_success'] != counts['both_success'] + counts['fixed']
                    or sum(counts[k] for k in ('fixed', 'broken', 'both_success', 'both_failure')) != 64):
                raise ValueError('Paired count identities fail')
            delta = a['success_difference']; ci = a['paired_bootstrap_95']
            if (not math.isfinite(delta) or delta != (counts['fixed'] - counts['broken']) / 64
                    or len(ci) != 2 or not all(math.isfinite(x) and -1 <= x <= 1 for x in ci)
                    or ci[0] > ci[1]):
                raise ValueError('Invalid paired difference or interval')
            row.update(status='complete', n=64, shared_roles=roles, **counts,
                       difference=delta, interval_95=list(ci))
        rows.append(row)
    return rows


def table(rows, latex=False):
    if latex:
        lines = [r'\begin{tabularx}{\linewidth}{@{}llrrrX@{}}', r'\toprule',
                 r'Runtime & Control & Base / candidate & F / B & New episodes & $\Delta$ [95\% interval] \\', r'\midrule']
    else:
        lines = ['| Runtime | Control | Base / candidate | F / B | New episodes | Difference [nominal 95% interval] |',
                 '|---|---|---:|---:|---:|---:|']
    for row in rows:
        completed = row['status'] == 'complete'
        scores = f"{row['base_success']}/64 / {row['candidate_success']}/64" if completed else 'Pending'
        transitions = f"{row['fixed']} / {row['broken']}" if completed else '--'
        effect = f"{row['difference']:+.3f} [{row['interval_95'][0]:+.3f}, {row['interval_95'][1]:+.3f}]" if completed else '--'
        count = str(row['additional_episodes']) if completed else f"{row['additional_episodes']} planned"
        values = [row['version_label'], row['label'] + ('*' if row['reference_reused'] else ''),
                  scores, transitions, count, effect]
        lines.append(' & '.join(values) + r' \\' if latex else '| ' + ' | '.join(values) + ' |')
    if latex:
        lines += [r'\bottomrule', r'\end{tabularx}']
    else:
        lines += ['', '*Extra repeat reuses the original-runtime identity reference; its 64 additional episodes are not a fresh 128-episode comparison.', '', NOTE]
    return '\n'.join(lines) + '\n'


def compact_latex(rows):
    labels = ('Fixed16 identity', 'Original identity', 'Original gripper 0.4', 'Original repeat*')
    lines = [r'\begin{tabularx}{\linewidth}{@{}Xrr@{}}', r'\toprule',
             r'Control & Base / candidate & F / B \\', r'\midrule']
    for label, row in zip(labels, rows):
        complete = row['status'] == 'complete'
        scores = f"{row['base_success']} / {row['candidate_success']}" if complete else r'\textit{Pending}'
        transitions = f"{row['fixed']} / {row['broken']}" if complete else '--'
        lines.append(f'{label} & {scores} & {transitions} ' + chr(92)*2)
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabularx}']) + '\n'


def render(rows, output, formats=('png',)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7, 2.55), layout='constrained')
    ys = [0, 1.5, 2.5, 3.5]
    for y, row in zip(ys, rows):
        if row['status'] == 'complete':
            color = '#24465A' if row['comparison_id'] != 'gripper-control' else '#A64B2A'
            lo, hi = row['interval_95']
            ax.hlines(y, lo, hi, color=color, lw=1.3)
            ax.plot([lo, hi], [y, y], '|', color=color, ms=7)
            ax.plot(row['difference'], y, 's', color=color, ms=5)
        else:
            ax.text(.99, y, 'Pending verified archive', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', color='#72787D', fontsize=8)
    labels = [r['version_label'] + ' — ' + r['label'] + ('*' if r['reference_reused'] else '') for r in rows]
    ax.set(yticks=ys, yticklabels=labels, ylim=(4.05, -.5), xlim=(-1.04, 1.04),
           xlabel='Candidate − reference success difference (percentage points)')
    ax.set_xticks([-1, -.5, 0, .5, 1]); ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{100*x:g}'))
    ax.axvline(0, color='#72787D', lw=.8); ax.axhline(.75, color='#DDDDDD', lw=.8)
    ax.grid(axis='x', alpha=.14)
    for suffix in formats:
        fig.savefig(output / ('final_controls.' + suffix), dpi=220, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PAPER / 'data/experiment_results.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--formats', nargs='+', choices=('png', 'svg', 'pdf'), default=['png'])
    args = parser.parse_args()
    raw = args.input.read_bytes(); rows = build_rows(json.loads(raw))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    view = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'source': str(args.input),
            'rows': rows, 'interpretation': NOTE}
    (args.output_dir / 'final_controls.json').write_text(json.dumps(view, indent=2) + '\n')
    (args.output_dir / 'final_controls.md').write_text(table(rows))
    (args.output_dir / 'final_controls.tex').write_text(table(rows, latex=True))
    (args.output_dir / 'final_controls_compact.tex').write_text(compact_latex(rows))
    render(rows, args.output_dir, args.formats)
    print(f"Presented {sum(r['status'] == 'complete' for r in rows)}/4 verified controls; pending rows have no points.")


if __name__ == '__main__':
    main()
