"""
plot_all.py
Comprehensive training visualisation for Vanilla MTL experiments.

Usage:
    python plot_all.py <training_history.json> [--output_dir <folder>]

Figures generated
-----------------
1.  overall_train_val.png              — avg train loss, avg val loss, combined grad score
2.  overall_train_val_vs_<metric>.png  — avg train + val loss vs each gradient metric
3.  per_task_train_val.png             — per-task train + val loss (2×2 grid)
4.  per_task_val_acc.png               — per-task validation accuracy (2×2 grid)
5.  per_task_with_grad_<task>.png      — per-task train/val + combined grad score
6.  grad_feature_<metric>_overall.png  — each gradient feature over time
7.  grad_feature_<metric>_per_task.png — each gradient feature vs per-task val loss
8.  severity_over_snr_overall.png      — score = severity / SNR vs avg val loss
9.  severity_over_snr_per_task.png     — score vs per-task val loss (2×2 grid)
10. severity_over_snr_argmin.png       — annotated argmin(score) vs argmin(val_loss)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="Path to training history JSON")
parser.add_argument("--output_dir", default=".", help="Folder to save plots (created if needed)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def save(fname):
    path = os.path.join(args.output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

with open(args.json_file) as f:
    data = json.load(f)

# ── Constants ──────────────────────────────────────────────────────────────────
TASKS        = ['yelp', 'qnli', 'qqp', 'mnli']
TASK_COLORS  = {'yelp': '#e74c3c', 'qnli': '#3498db', 'qqp': '#2ecc71', 'mnli': '#f39c12'}
GRAD_KEYS    = ['conflict_rate', 'conflict_severity', 'gradient_variance',
                'grad_norm_ratio', 'grad_snr']
GRAD_LABELS  = {
    'conflict_rate':          'Conflict Rate',
    'conflict_severity':      'Conflict Severity',
    'gradient_variance':      'Gradient Variance',
    'grad_norm_ratio':        'Grad Norm Ratio',
    'grad_snr':               'Grad SNR',
    'combined_gradient_score':'Combined Grad Score',
}
GRAD_COLORS  = {
    'conflict_rate':           '#c0392b',
    'conflict_severity':       '#e67e22',
    'gradient_variance':       '#8e44ad',
    'grad_norm_ratio':         '#16a085',
    'grad_snr':                '#2980b9',
    'combined_gradient_score': '#2c3e50',
}

# Color for the new severity/SNR score — a fresh hue distinct from existing ones
SCORE_COLOR  = '#d62728'   # crimson
SCORE_LABEL  = 'Severity / SNR'

# ── Build flat timeline from mid-epoch checks ──────────────────────────────────
steps            = []
avg_val_loss     = []
combined_score   = []
grad_data        = {k: [] for k in GRAD_KEYS}
per_task_train   = {t: [] for t in TASKS}
per_task_val     = {t: [] for t in TASKS}
per_task_acc     = {t: [] for t in TASKS}

for epoch_data in data:
    for chk in epoch_data['mid_epoch_checks']:
        steps.append(chk['global_step'])
        avg_val_loss.append(chk['avg_val_loss'])
        combined_score.append(chk['combined_gradient_score'])
        for k in GRAD_KEYS:
            grad_data[k].append(chk[k])
        for t in TASKS:
            per_task_train[t].append(chk['train_loss'][t])
            per_task_val[t].append(chk['per_task_val'][t]['loss'])
            per_task_acc[t].append(chk['per_task_val'][t]['acc'])

# avg train loss across tasks
avg_train_loss = [
    np.mean([per_task_train[t][i] for t in TASKS])
    for i in range(len(steps))
]

steps          = np.array(steps)
avg_val_loss   = np.array(avg_val_loss)
combined_score = np.array(combined_score)

# Freeze steps
freeze_steps = {}
for epoch_data in data:
    for chk in epoch_data['mid_epoch_checks']:
        for t in TASKS:
            if t in chk['frozen_tasks'] and t not in freeze_steps:
                freeze_steps[t] = chk['global_step']

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_ax(ax, ylabel=None, legend=True, loc='best'):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax.set_xlabel('Global Step', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if legend:
        ax.legend(fontsize=8, loc=loc)

def add_freeze_lines(ax, tasks=None):
    shown = tasks or list(freeze_steps.keys())
    for t in shown:
        if t in freeze_steps:
            ax.axvline(freeze_steps[t], color=TASK_COLORS[t],
                       lw=1.5, ls=':', alpha=0.85)
            ylim = ax.get_ylim()
            ax.text(freeze_steps[t] + 200,
                    ylim[0] + (ylim[1]-ylim[0]) * 0.03,
                    f'{t}\nfrozen', fontsize=6.5,
                    color=TASK_COLORS[t], va='bottom')

def norm01(arr):
    arr = np.array(arr)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)

def smooth(x, w=5):
    """Centered moving average. Window size in checks."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - w // 2), min(len(x), i + w // 2 + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overall: avg train loss, avg val loss, combined grad score
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()

ax1.plot(steps, avg_train_loss, color='steelblue', lw=1.8,
         marker='.', ms=3, label='Avg Train Loss')
ax1.plot(steps, avg_val_loss,   color='royalblue', lw=2,
         marker='.', ms=3, ls='--', label='Avg Val Loss')
ax2.plot(steps, combined_score, color=GRAD_COLORS['combined_gradient_score'],
         lw=1.8, marker='.', ms=3, ls='-.', label='Combined Grad Score')

ax1.set_ylabel('Loss', color='steelblue', fontsize=9)
ax2.set_ylabel('Combined Gradient Score', color=GRAD_COLORS['combined_gradient_score'], fontsize=9)
ax1.tick_params(axis='y', colors='steelblue')
ax2.tick_params(axis='y', colors=GRAD_COLORS['combined_gradient_score'])
ax1.set_title('Overall: Avg Train Loss · Avg Val Loss · Combined Gradient Score',
              fontsize=11, fontweight='bold')

add_freeze_lines(ax1)
fmt_ax(ax1, legend=False)

lines  = ax1.get_lines() + ax2.get_lines()
freeze_legend = [Line2D([0],[0], color=TASK_COLORS[t], ls=':', lw=1.5,
                         label=f'{t.upper()} frozen')
                 for t in freeze_steps]
ax1.legend(handles=lines + freeze_legend, fontsize=7, loc='upper right', ncol=2)

plt.tight_layout()
save('overall_train_val.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1b — Overall avg train + val loss vs EACH gradient metric (separate files)
# ══════════════════════════════════════════════════════════════════════════════
all_grad_keys_combined = GRAD_KEYS + ['combined_gradient_score']

for gk in all_grad_keys_combined:
    g = combined_score if gk == 'combined_gradient_score' else np.array(grad_data[gk])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = ax.twinx()

    ax.plot(steps, avg_train_loss, color='steelblue', lw=1.8,
            marker='.', ms=3, label='Avg Train Loss')
    ax.plot(steps, avg_val_loss,   color='royalblue', lw=2,
            marker='.', ms=3, ls='--', label='Avg Val Loss')
    ax2.plot(steps, g, color=GRAD_COLORS[gk], lw=1.8,
             marker='.', ms=3, ls='-.', alpha=0.9, label=GRAD_LABELS[gk])

    ax.set_ylabel('Loss', color='steelblue', fontsize=9)
    ax2.set_ylabel(GRAD_LABELS[gk], color=GRAD_COLORS[gk], fontsize=9)
    ax.tick_params(axis='y', colors='steelblue')
    ax2.tick_params(axis='y', colors=GRAD_COLORS[gk])
    ax.set_title(f'Overall Train & Val Loss vs {GRAD_LABELS[gk]}',
                 fontsize=11, fontweight='bold')

    add_freeze_lines(ax)
    lines = ax.get_lines() + ax2.get_lines()
    freeze_legend = [Line2D([0],[0], color=TASK_COLORS[t], ls=':', lw=1.5,
                             label=f'{t.upper()} frozen') for t in freeze_steps]
    ax.legend(handles=lines + freeze_legend, fontsize=7,
              loc='upper right', ncol=2)
    fmt_ax(ax, legend=False)

    plt.tight_layout()
    fname = f'overall_train_val_vs_{gk}.png'
    save(fname)

# ── Figure 2 — Per-task train + val loss (2×2 grid)
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
fig.suptitle('Per-Task Train & Validation Loss', fontsize=12, fontweight='bold')

for ax, t in zip(axes.flat, TASKS):
    ax.plot(steps, per_task_train[t], color=TASK_COLORS[t],
            lw=1.8, marker='.', ms=3, label='Train Loss')
    ax.plot(steps, per_task_val[t],   color=TASK_COLORS[t],
            lw=2,   marker='.', ms=3, ls='--', alpha=0.7, label='Val Loss')
    add_freeze_lines(ax, tasks=[t])
    ax.set_title(t.upper(), fontsize=10, fontweight='bold', color=TASK_COLORS[t])
    fmt_ax(ax, ylabel='Loss')

plt.tight_layout()
save('per_task_train_val.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-task validation accuracy (2×2 grid)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
fig.suptitle('Per-Task Validation Accuracy', fontsize=12, fontweight='bold')

for ax, t in zip(axes.flat, TASKS):
    ax.plot(steps, per_task_acc[t], color=TASK_COLORS[t],
            lw=2, marker='.', ms=3)
    add_freeze_lines(ax, tasks=[t])
    ax.set_title(t.upper(), fontsize=10, fontweight='bold', color=TASK_COLORS[t])
    fmt_ax(ax, ylabel='Accuracy', legend=False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
save('per_task_val_acc.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-task: train loss, val loss, val acc + combined grad score (separate)
# ══════════════════════════════════════════════════════════════════════════════
for t in TASKS:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = ax.twinx()

    ax.plot(steps, per_task_train[t], color=TASK_COLORS[t],
            lw=1.8, marker='.', ms=3, label='Train Loss')
    ax.plot(steps, per_task_val[t],   color=TASK_COLORS[t],
            lw=2,   marker='.', ms=3, ls='--', alpha=0.7, label='Val Loss')

    acc_scaled = norm01(per_task_acc[t]) * (
        max(max(per_task_train[t]), max(per_task_val[t])) -
        min(min(per_task_train[t]), min(per_task_val[t]))
    ) + min(min(per_task_train[t]), min(per_task_val[t]))
    ax.plot(steps, acc_scaled, color=TASK_COLORS[t],
            lw=1.2, ls=':', ms=2, marker='.', alpha=0.5, label='Val Acc (scaled)')

    ax2.plot(steps, combined_score, color=GRAD_COLORS['combined_gradient_score'],
             lw=1.5, ls='-.', marker='.', ms=2, alpha=0.85, label='Combined Grad Score')

    add_freeze_lines(ax, tasks=[t])
    ax.set_ylabel('Loss', color=TASK_COLORS[t], fontsize=9)
    ax2.set_ylabel('Grad Score', color=GRAD_COLORS['combined_gradient_score'], fontsize=8)
    ax.tick_params(axis='y', colors=TASK_COLORS[t])
    ax2.tick_params(axis='y', colors=GRAD_COLORS['combined_gradient_score'])
    ax.set_title(f'{t.upper()} — Train · Val Loss · Accuracy + Combined Gradient Score',
                 fontsize=11, fontweight='bold', color=TASK_COLORS[t])

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(handles=lines, fontsize=7, loc='upper right')
    fmt_ax(ax, legend=False)

    plt.tight_layout()
    fname = f'per_task_with_grad_{t}.png'
    save(fname)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Each gradient feature overall (separate file per feature)
# ══════════════════════════════════════════════════════════════════════════════
for gk in all_grad_keys_combined:
    g = combined_score if gk == 'combined_gradient_score' else np.array(grad_data[gk])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, g, color=GRAD_COLORS[gk], lw=2, marker='.', ms=3)
    ax.fill_between(steps, g, alpha=0.12, color=GRAD_COLORS[gk])
    add_freeze_lines(ax)
    ax.set_ylabel(GRAD_LABELS[gk], fontsize=9)
    ax.set_title(f'{GRAD_LABELS[gk]} Over Training',
                 fontsize=11, fontweight='bold')

    freeze_legend = [Line2D([0],[0], color=TASK_COLORS[t], ls=':', lw=1.5,
                             label=f'{t.upper()} frozen') for t in freeze_steps]
    ax.legend(handles=freeze_legend, fontsize=7, loc='upper right')
    fmt_ax(ax, legend=False)

    plt.tight_layout()
    fname = f'grad_feature_{gk}_overall.png'
    save(fname)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Each gradient feature vs per-task val loss (one file per feature)
# ══════════════════════════════════════════════════════════════════════════════
all_grad_keys = GRAD_KEYS + ['combined_gradient_score']

for gk in all_grad_keys:
    g = np.array(grad_data[gk] if gk in grad_data else combined_score)
    if gk == 'combined_gradient_score':
        g = combined_score

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(f'{GRAD_LABELS[gk]} vs Per-Task Val Loss',
                 fontsize=12, fontweight='bold')

    for ax, t in zip(axes.flat, TASKS):
        ax2 = ax.twinx()

        vl = np.array(per_task_val[t])
        ax.plot(steps, vl, color=TASK_COLORS[t], lw=2,
                marker='.', ms=3, label='Val Loss')
        ax.set_ylabel('Val Loss', color=TASK_COLORS[t], fontsize=9)
        ax.tick_params(axis='y', colors=TASK_COLORS[t])

        ax2.plot(steps, g, color=GRAD_COLORS[gk], lw=1.8,
                 marker='.', ms=3, ls='--', alpha=0.85,
                 label=GRAD_LABELS[gk])
        ax2.set_ylabel(GRAD_LABELS[gk], color=GRAD_COLORS[gk], fontsize=9)
        ax2.tick_params(axis='y', colors=GRAD_COLORS[gk])

        add_freeze_lines(ax, tasks=[t])
        ax.set_title(t.upper(), fontsize=10, fontweight='bold',
                     color=TASK_COLORS[t])

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(handles=lines, fontsize=7, loc='upper right')
        fmt_ax(ax, legend=False)

    plt.tight_layout()
    fname = f'grad_feature_{gk}_per_task.png'
    save(fname)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Severity / SNR score
#   score(t) = smooth(conflict_severity) / smooth(grad_snr)
#   Higher = worse (bad / good). Should U-shape near argmin(val_loss).
# ══════════════════════════════════════════════════════════════════════════════
SMOOTH_W = 5  # centered moving average window (in checks)

sev_arr = np.array(grad_data['conflict_severity'])
snr_arr = np.array(grad_data['grad_snr'])

sev_s   = smooth(sev_arr, SMOOTH_W)
snr_s   = smooth(snr_arr, SMOOTH_W)
val_s   = smooth(avg_val_loss, SMOOTH_W)
score   = sev_s / snr_s

# Argmin search ignoring early warmup (first 20% of run is initialization noise)
warmup = max(2, len(steps) // 5)
score_search = score.copy(); score_search[:warmup] = np.inf
val_search   = avg_val_loss.copy(); val_search[:warmup] = np.inf
i_score = int(np.argmin(score_search))
i_val   = int(np.argmin(val_search))

# ─── 7a. Overall: avg val loss vs severity/SNR score ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax2 = ax.twinx()

ax.plot(steps, avg_val_loss, color='royalblue', lw=2,
        marker='.', ms=3, ls='--', alpha=0.5, label='Avg Val Loss (raw)')
val_smooth_line, = ax.plot(steps, val_s, color='royalblue', lw=2.2,
        marker='.', ms=3, label=f'Avg Val Loss (smoothed w={SMOOTH_W})')

score_line, = ax2.plot(steps, score, color=SCORE_COLOR, lw=2.2, marker='.', ms=3,
         label=f'{SCORE_LABEL} (smoothed)')

# Vertical lines at the two minima — kept off the auto legend
ax.axvline(steps[i_val],   color='royalblue', lw=1.5, ls='-.', alpha=0.8,
           label='_nolegend_')
ax.axvline(steps[i_score], color=SCORE_COLOR,  lw=1.5, ls='-.', alpha=0.8,
           label='_nolegend_')

ax.set_ylabel('Avg Val Loss', color='royalblue', fontsize=9)
ax2.set_ylabel(SCORE_LABEL, color=SCORE_COLOR, fontsize=9)
ax.tick_params(axis='y', colors='royalblue')
ax2.tick_params(axis='y', colors=SCORE_COLOR)
ax.set_title(f'Overall Val Loss vs {SCORE_LABEL}  '
             f'(argmin(val)=step {steps[i_val]}, argmin(score)=step {steps[i_score]})',
             fontsize=11, fontweight='bold')

add_freeze_lines(ax)
# raw val loss line is still in get_lines(); we want it shown, so grab it explicitly
raw_val_line = ax.get_lines()[0]
extra = [
    Line2D([0],[0], color='royalblue', ls='-.', lw=1.5, label=f'argmin(val) @ {steps[i_val]}'),
    Line2D([0],[0], color=SCORE_COLOR, ls='-.', lw=1.5, label=f'argmin(score) @ {steps[i_score]}'),
]
freeze_legend = [Line2D([0],[0], color=TASK_COLORS[t], ls=':', lw=1.5,
                         label=f'{t.upper()} frozen') for t in freeze_steps]
ax.legend(handles=[raw_val_line, val_smooth_line, score_line] + extra + freeze_legend,
          fontsize=7, loc='upper right', ncol=2)
fmt_ax(ax, legend=False)

plt.tight_layout()
save('severity_over_snr_overall.png')

# ─── 7b. Per-task: each task's val loss vs the same overall score ──────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
fig.suptitle(f'{SCORE_LABEL} vs Per-Task Val Loss', fontsize=12, fontweight='bold')

for ax, t in zip(axes.flat, TASKS):
    ax2 = ax.twinx()

    val_line, = ax.plot(steps, per_task_val[t], color=TASK_COLORS[t], lw=2,
            marker='.', ms=3, label='Val Loss')
    score_line, = ax2.plot(steps, score, color=SCORE_COLOR, lw=1.8,
             marker='.', ms=3, ls='--', alpha=0.85, label=SCORE_LABEL)

    ax.axvline(steps[i_score], color=SCORE_COLOR, lw=1.2, ls='-.', alpha=0.7,
               label='_nolegend_')

    ax.set_ylabel('Val Loss', color=TASK_COLORS[t], fontsize=9)
    ax2.set_ylabel(SCORE_LABEL, color=SCORE_COLOR, fontsize=9)
    ax.tick_params(axis='y', colors=TASK_COLORS[t])
    ax2.tick_params(axis='y', colors=SCORE_COLOR)

    add_freeze_lines(ax, tasks=[t])
    ax.set_title(t.upper(), fontsize=10, fontweight='bold', color=TASK_COLORS[t])

    ax.legend(handles=[val_line, score_line], fontsize=7, loc='upper right')
    fmt_ax(ax, legend=False)

plt.tight_layout()
save('severity_over_snr_per_task.png')

# ─── 7c. Annotated argmin comparison (for the report headline figure) ──────────
fig, ax = plt.subplots(figsize=(12, 4.5))
ax2 = ax.twinx()

# Use _nolegend_ on auxiliary artists so the manual legend below stays clean
val_line,   = ax.plot(steps, val_s, color='royalblue', lw=2.2,
                      label='Avg Val Loss (smoothed)')
score_line, = ax2.plot(steps, score, color=SCORE_COLOR, lw=2.2,
                       label=f'{SCORE_LABEL} (smoothed)')

# Mark the two minima with markers + vertical lines (excluded from legend)
ax.plot(steps[i_val], val_s[i_val], 'o', color='royalblue', ms=10,
        markeredgecolor='white', markeredgewidth=1.5, zorder=5,
        label='_nolegend_')
ax2.plot(steps[i_score], score[i_score], 'o', color=SCORE_COLOR, ms=10,
         markeredgecolor='white', markeredgewidth=1.5, zorder=5,
         label='_nolegend_')

ax.axvline(steps[i_val],   color='royalblue', lw=1.2, ls='-.', alpha=0.7,
           label='_nolegend_')
ax.axvline(steps[i_score], color=SCORE_COLOR,  lw=1.2, ls='-.', alpha=0.7,
           label='_nolegend_')

# Stats annotation
gap = steps[i_score] - steps[i_val]
val_pen = avg_val_loss[i_score] - avg_val_loss[i_val]
stats_txt = (f'argmin(val_loss) = step {steps[i_val]}  (val={avg_val_loss[i_val]:.4f})\n'
             f'argmin(score)    = step {steps[i_score]}  (val={avg_val_loss[i_score]:.4f})\n'
             f'step gap = {gap:+d}     val penalty = {val_pen:+.4f}')
ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#888', alpha=0.9))

ax.set_ylabel('Avg Val Loss', color='royalblue', fontsize=9)
ax2.set_ylabel(SCORE_LABEL, color=SCORE_COLOR, fontsize=9)
ax.tick_params(axis='y', colors='royalblue')
ax2.tick_params(axis='y', colors=SCORE_COLOR)
ax.set_title(f'Early-stopping signal: {SCORE_LABEL} tracks argmin(val_loss)',
             fontsize=11, fontweight='bold')

add_freeze_lines(ax)
ax.legend(handles=[val_line, score_line], fontsize=8, loc='upper right')
fmt_ax(ax, legend=False)

plt.tight_layout()
save('severity_over_snr_argmin.png')

print("\nAll plots saved.")