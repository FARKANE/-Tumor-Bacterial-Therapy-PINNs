"""
=============================================================================
POST-PROCESSING & VISUALIZATION FOR ABLATION STUDIES
=============================================================================
Loads all saved models from completed ablation runs and generates
publication-quality figures with domain-wide metrics vs time.

Figures generated per study:
  Fig A — Domain sup/max of each variable vs time (one curve per experiment)
  Fig B — Domain mean of each variable vs time
  Fig C — Domain min of each variable vs time
  Fig D — Tumor suppression summary: T_max(t), T_mean(t), T_min(t) overlay
  Fig E — Therapy efficiency: B/T ratio (mean) vs time
  Fig F — Spatial spread: std(T), std(B), std(S) vs time
  Fig G — Phase portrait: T_mean vs S_mean trajectory
  Fig H — Terminal state comparison bar chart (all 5 variables)
  Fig I — Heatmap: terminal T*(30) and S*(30) for each experiment
  Fig J — Combined summary: T_max overlay + O_mean overlay

Usage (in Colab):
  1. Run the ablation_study.py first (to train and save models)
  2. Then run this script — it loads all saved .pt files automatically
  3. Edit STUDIES_TO_PLOT and ABLATION_DIR at the bottom
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.gridspec import GridSpec
import os
import json
import glob
import copy
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# =============================================================================
# RE-IMPORT CLASSES (needed to reload models)
# =============================================================================

class QuadGeometry:
    def __init__(self, x_min=0.0, x_max=6.0, y_min=0.0, y_max=6.0, nx=60, ny=60):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.nx, self.ny = nx, ny
        self.nodes, self.elements, self.boundary_edges = self._build_mesh()
        self.n_nodes = len(self.nodes)
        self.x = self.nodes[:, 0]
        self.y = self.nodes[:, 1]
        self.boundary_node_indices = np.unique(self.boundary_edges[:, :2].flatten())

    def _build_mesh(self):
        nx, ny = self.nx, self.ny
        xs = np.linspace(self.x_min, self.x_max, nx + 1)
        ys = np.linspace(self.y_min, self.y_max, ny + 1)
        XX, YY = np.meshgrid(xs, ys)
        nodes = np.column_stack([XX.ravel(), YY.ravel()])
        def nid(i, j): return j * (nx + 1) + i
        tris = []
        for j in range(ny):
            for i in range(nx):
                a, b, c, d = nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)
                tris.append([a, b, c]); tris.append([a, c, d])
        elements = np.array(tris, dtype=int)
        bedges = []
        for i in range(nx): bedges.append([nid(i,0), nid(i+1,0)])
        for j in range(ny): bedges.append([nid(nx,j), nid(nx,j+1)])
        for i in range(nx-1,-1,-1): bedges.append([nid(i+1,ny), nid(i,ny)])
        for j in range(ny-1,-1,-1): bedges.append([nid(0,j+1), nid(0,j)])
        return nodes, elements, np.array(bedges, dtype=int)


class PINN_Net(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        act_map = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'gelu': nn.GELU(),
                   'silu': nn.SiLU(), 'sigmoid': nn.Sigmoid()}
        self.activation = act_map.get(activation, nn.Tanh())
        self.linear_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight); nn.init.zeros_(lin.bias)
            self.linear_layers.append(lin)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        for layer in self.linear_layers[:-1]:
            inp = self.activation(layer(inp))
        return self.linear_layers[-1](inp)


# =============================================================================
# LOAD MODEL UTILITY
# =============================================================================

def load_model(path, geo, device):
    """Load a saved AblationPINN checkpoint and return the model + params."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PINN_Net(ckpt['layers'], ckpt.get('activation', 'tanh')).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt.get('params', {}), ckpt.get('layers', [])


def discover_models(study_dir):
    """Find all .pt model files in a study's models/ directory."""
    models_dir = os.path.join(study_dir, 'models')
    if not os.path.isdir(models_dir):
        return []
    files = sorted(glob.glob(os.path.join(models_dir, '*.pt')))
    return files


def label_from_path(path, study_name):
    """Extract a human-readable label from a model filename."""
    fname = os.path.splitext(os.path.basename(path))[0]
    label = fname.replace(f'{study_name}_', '').replace('_', ' ')
    return label


# =============================================================================
# COMPUTE DOMAIN-WIDE STATISTICS VS TIME
# =============================================================================

def compute_time_statistics(model, geo, device, times=None, t_final=30.0):
    """
    Evaluate model at all mesh nodes for each time in `times`.
    Returns dict with keys: 'times', and for each variable (T,B,O,I,S):
      '{var}_max', '{var}_mean', '{var}_min', '{var}_std'
    """
    if times is None:
        times = np.linspace(0, t_final, 61)

    xm = torch.FloatTensor(geo.x).unsqueeze(1).to(device)
    ym = torch.FloatTensor(geo.y).unsqueeze(1).to(device)

    var_names = ['T', 'B', 'O', 'I', 'S']
    stats = {'times': times}
    for v in var_names:
        stats[f'{v}_max'] = []
        stats[f'{v}_mean'] = []
        stats[f'{v}_min'] = []
        stats[f'{v}_std'] = []

    model.eval()
    with torch.no_grad():
        for t_val in times:
            tm = torch.full_like(xm, t_val)
            out = model(xm, ym, tm).cpu().numpy()  # (n_nodes, 5)
            for j, v in enumerate(var_names):
                col = out[:, j]
                stats[f'{v}_max'].append(float(col.max()))
                stats[f'{v}_mean'].append(float(col.mean()))
                stats[f'{v}_min'].append(float(col.min()))
                stats[f'{v}_std'].append(float(col.std()))

    # convert to numpy
    for k in stats:
        if k != 'times':
            stats[k] = np.array(stats[k])
    stats['times'] = np.array(times)

    return stats


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

VAR_NAMES = ['T', 'B', 'O', 'I', 'S']
VAR_LABELS = ['Tumor $T$', 'Bacteria $B$', 'Oxygen $O$',
              'Cytokines $I$', 'Signal $S$']
VAR_COLORS_BASE = ['#d32f2f', '#1565c0', '#2e7d32', '#e65100', '#6a1b9a']

# Consistent color palette for experiments within a study
def get_experiment_colors(n):
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def style_ax(ax, xlabel='Time (days)', ylabel='', title='', legend=True):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=11, weight='bold')
    ax.grid(True, alpha=0.25, linestyle='-')
    ax.tick_params(labelsize=9)
    if legend:
        ax.legend(fontsize=7, loc='best', framealpha=0.9)


# =============================================================================
# FIGURE A — sup/max of each variable vs time
# =============================================================================

def plot_fig_A_max(study_title, labels, all_stats, save_dir):
    """5 subplots: for each variable, plot max_Omega phi(t,x) vs t."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
    colors = get_experiment_colors(len(labels))

    for j, (vname, vlabel) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        ax = axes[j]
        for i, (label, stats) in enumerate(zip(labels, all_stats)):
            lw = 2.5 if 'baseline' in label.lower() else 1.3
            ls = '-' if 'baseline' in label.lower() else '--'
            ax.plot(stats['times'], stats[f'{vname}_max'],
                    color=colors[i], lw=lw, ls=ls, label=label, alpha=0.9)
        style_ax(ax, ylabel=f'$\\sup_\\Omega\\, {vname}(t,\\mathbf{{x}})$',
                 title=vlabel)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_A_domain_max.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig A → {path}")


# =============================================================================
# FIGURE B — mean of each variable vs time
# =============================================================================

def plot_fig_B_mean(study_title, labels, all_stats, save_dir):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
    fig.suptitle(f'{study_title} — Domain Mean vs Time', fontsize=14, weight='bold', y=1.03)
    colors = get_experiment_colors(len(labels))

    for j, (vname, vlabel) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        ax = axes[j]
        for i, (label, stats) in enumerate(zip(labels, all_stats)):
            lw = 2.5 if 'baseline' in label.lower() else 1.3
            ls = '-' if 'baseline' in label.lower() else '--'
            ax.plot(stats['times'], stats[f'{vname}_mean'],
                    color=colors[i], lw=lw, ls=ls, label=label, alpha=0.9)
        style_ax(ax, ylabel=f'$\\overline{{{vname}}}(t)$', title=vlabel)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_B_domain_mean.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig B → {path}")


# =============================================================================
# FIGURE C — min of each variable vs time
# =============================================================================

def plot_fig_C_min(study_title, labels, all_stats, save_dir):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
    fig.suptitle(f'{study_title} — Domain Minimum vs Time', fontsize=14, weight='bold', y=1.03)
    colors = get_experiment_colors(len(labels))

    for j, (vname, vlabel) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        ax = axes[j]
        for i, (label, stats) in enumerate(zip(labels, all_stats)):
            lw = 2.5 if 'baseline' in label.lower() else 1.3
            ls = '-' if 'baseline' in label.lower() else '--'
            ax.plot(stats['times'], stats[f'{vname}_min'],
                    color=colors[i], lw=lw, ls=ls, label=label, alpha=0.9)
        style_ax(ax, ylabel=f'$\\inf_\\Omega\\, {vname}(t,\\mathbf{{x}})$',
                 title=vlabel)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_C_domain_min.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig C → {path}")


# =============================================================================
# FIGURE D — Tumor suppression: max, mean, min of T on single plot per experiment
# =============================================================================

def plot_fig_D_tumor_envelope(study_title, labels, all_stats, save_dir):
    """For each experiment: T_max, T_mean, T_min as envelope."""
    n = len(labels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(f'{study_title} — Tumor Suppression Envelope',
                 fontsize=14, weight='bold', y=1.02)

    for idx, (label, stats) in enumerate(zip(labels, all_stats)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        t = stats['times']
        ax.fill_between(t, stats['T_min'], stats['T_max'],
                        color='#ef9a9a', alpha=0.35, label='[min, max]')
        ax.plot(t, stats['T_max'], color='#c62828', lw=1.5, ls='--',
                label='$\\sup_\\Omega T$')
        ax.plot(t, stats['T_mean'], color='#d32f2f', lw=2.5,
                label='$\\overline{T}$')
        ax.plot(t, stats['T_min'], color='#e57373', lw=1.5, ls=':',
                label='$\\inf_\\Omega T$')
        style_ax(ax, ylabel='Tumor density $T$', title=label)

    # hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_D_tumor_envelope.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig D → {path}")


# =============================================================================
# FIGURE E — Therapy efficiency: B_mean / T_mean ratio vs time
# =============================================================================

def plot_fig_E_BT_ratio(study_title, labels, all_stats, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_experiment_colors(len(labels))

    for i, (label, stats) in enumerate(zip(labels, all_stats)):
        T_mean = np.array(stats['T_mean'])
        B_mean = np.array(stats['B_mean'])
        ratio = B_mean / (T_mean + 1e-10)
        lw = 2.5 if 'baseline' in label.lower() else 1.3
        ls = '-' if 'baseline' in label.lower() else '--'
        ax.plot(stats['times'], ratio, color=colors[i], lw=lw, ls=ls,
                label=label, alpha=0.9)

    style_ax(ax, ylabel='$\\overline{B}(t) \\,/\\, \\overline{T}(t)$',
             title=f'{study_title} — Bacteria-to-Tumor Ratio (Therapy Efficiency)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_E_BT_ratio.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig E → {path}")


# =============================================================================
# FIGURE F — Spatial heterogeneity: std of T, B, S vs time
# =============================================================================

def plot_fig_F_spatial_std(study_title, labels, all_stats, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{study_title} — Spatial Heterogeneity (Std Dev)',
                 fontsize=14, weight='bold', y=1.03)
    colors = get_experiment_colors(len(labels))
    plot_vars = [('T', 'Tumor $T$'), ('B', 'Bacteria $B$'), ('S', 'Signal $S$')]

    for ax, (vname, vlabel) in zip(axes, plot_vars):
        for i, (label, stats) in enumerate(zip(labels, all_stats)):
            lw = 2.5 if 'baseline' in label.lower() else 1.3
            ls = '-' if 'baseline' in label.lower() else '--'
            ax.plot(stats['times'], stats[f'{vname}_std'],
                    color=colors[i], lw=lw, ls=ls, label=label, alpha=0.9)
        style_ax(ax, ylabel=f'$\\mathrm{{std}}_\\Omega({vname})$', title=vlabel)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_F_spatial_std.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig F → {path}")


# =============================================================================
# FIGURE G — Phase portrait: T_mean vs S_mean trajectory
# =============================================================================

def plot_fig_G_phase_portrait(study_title, labels, all_stats, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f'{study_title} — Phase Portraits',
                 fontsize=14, weight='bold', y=1.03)
    colors = get_experiment_colors(len(labels))

    # T_mean vs S_mean
    ax = axes[0]
    for i, (label, stats) in enumerate(zip(labels, all_stats)):
        lw = 2.5 if 'baseline' in label.lower() else 1.3
        ax.plot(stats['T_mean'], stats['S_mean'],
                color=colors[i], lw=lw, label=label, alpha=0.85)
        ax.plot(stats['T_mean'][0], stats['S_mean'][0], 'o',
                color=colors[i], ms=6)
        ax.plot(stats['T_mean'][-1], stats['S_mean'][-1], 's',
                color=colors[i], ms=6)
    style_ax(ax, xlabel='$\\overline{T}(t)$', ylabel='$\\overline{S}(t)$',
             title='$\\overline{T}$ vs $\\overline{S}$ trajectory')

    # T_mean vs B_mean
    ax = axes[1]
    for i, (label, stats) in enumerate(zip(labels, all_stats)):
        lw = 2.5 if 'baseline' in label.lower() else 1.3
        ax.plot(stats['T_mean'], stats['B_mean'],
                color=colors[i], lw=lw, label=label, alpha=0.85)
        ax.plot(stats['T_mean'][0], stats['B_mean'][0], 'o',
                color=colors[i], ms=6)
        ax.plot(stats['T_mean'][-1], stats['B_mean'][-1], 's',
                color=colors[i], ms=6)
    style_ax(ax, xlabel='$\\overline{T}(t)$', ylabel='$\\overline{B}(t)$',
             title='$\\overline{T}$ vs $\\overline{B}$ trajectory')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_G_phase_portrait.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig G → {path}")


# =============================================================================
# FIGURE H — Terminal state bar chart (all 5 variables)
# =============================================================================

def plot_fig_H_terminal_bars(study_title, labels, all_stats, save_dir):
    var_labels_short = ['$\\bar{T}^*$', '$\\bar{B}^*$', '$\\bar{O}^*$',
                        '$\\bar{I}^*$', '$\\bar{S}^*$']

    n_exp = len(labels)
    n_var = 5
    x = np.arange(n_exp)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(12, 2 * n_exp), 6))
    for j in range(n_var):
        vals = [stats[f'{VAR_NAMES[j]}_mean'][-1] for stats in all_stats]
        ax.bar(x + (j - 2) * width, vals, width,
               label=var_labels_short[j], color=VAR_COLORS_BASE[j], alpha=0.85,
               edgecolor='k', lw=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Domain-averaged value at $t = 30$ days', fontsize=11)
    ax.set_title(f'{study_title} — Terminal Equilibrium State', fontsize=13, weight='bold')
    ax.legend(ncol=5, fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_H_terminal_bars.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig H → {path}")


# =============================================================================
# FIGURE I — Heatmap: T_max(t) per experiment (rows = experiments, cols = time)
# =============================================================================

def plot_fig_I_heatmap(study_title, labels, all_stats, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, max(3, 0.6 * len(labels) + 1)))
    fig.suptitle(f'{study_title} — Spatio-temporal Heatmap',
                 fontsize=13, weight='bold', y=1.05)

    times = all_stats[0]['times']

    for ax, vname, cmap_name, title in [
        (axes[0], 'T', 'Reds', 'Tumor $\\sup_\\Omega T(t)$'),
        (axes[1], 'S', 'Purples', 'Signal $\\sup_\\Omega S(t)$')]:

        matrix = np.array([s[f'{vname}_max'] for s in all_stats])
        im = ax.imshow(matrix, aspect='auto', cmap=cmap_name,
                       extent=[times[0], times[-1], len(labels) - 0.5, -0.5])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_I_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig I → {path}")


# =============================================================================
# FIGURE J — Combined summary: T_max overlay + O_mean overlay
# =============================================================================

def plot_fig_J_combined_summary(study_title, labels, all_stats, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'{study_title} — Combined Summary',
                 fontsize=15, weight='bold', y=1.02)
    colors = get_experiment_colors(len(labels))

    plot_configs = [
        (axes[0, 0], 'T_max',  '$\\sup_\\Omega T(t)$',  'Tumor Peak'),
        (axes[0, 1], 'T_mean', '$\\overline{T}(t)$',    'Tumor Mean'),
        (axes[0, 2], 'B_mean', '$\\overline{B}(t)$',    'Bacteria Mean'),
        (axes[1, 0], 'O_mean', '$\\overline{O}(t)$',    'Oxygen Mean'),
        (axes[1, 1], 'S_mean', '$\\overline{S}(t)$',    'Signal Mean'),
        (axes[1, 2], 'I_mean', '$\\overline{I}(t)$',    'Cytokines Mean'),
    ]

    for ax, key, ylabel, title in plot_configs:
        for i, (label, stats) in enumerate(zip(labels, all_stats)):
            lw = 2.8 if 'baseline' in label.lower() else 1.3
            ls = '-' if 'baseline' in label.lower() else '--'
            ax.plot(stats['times'], stats[key], color=colors[i], lw=lw, ls=ls,
                    label=label, alpha=0.9)
        style_ax(ax, ylabel=ylabel, title=title)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_J_combined_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig J → {path}")


# =============================================================================
# MASTER: Process one study
# =============================================================================

def process_study(study_name, study_dir, geo, times, output_subdir='figures'):
    """Load all models for a study and generate all figures."""
    fig_dir = os.path.join(study_dir, output_subdir)
    os.makedirs(fig_dir, exist_ok=True)

    # ── discover and load models ──
    model_paths = discover_models(study_dir)
    if not model_paths:
        print(f"  No models found in {study_dir}/models/  — skipping.")
        return

    # ── try to read study title from CSV ──
    csv_files = glob.glob(os.path.join(study_dir, '*.csv'))
    study_title = study_name
    if csv_files:
        import csv as csv_mod
        with open(csv_files[0], 'r') as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
            study_title = study_name.replace('_', ' ')

    print(f"\n{'=' * 70}")
    print(f"  Processing: {study_title}")
    print(f"  Models found: {len(model_paths)}")
    print(f"  Time points: {len(times)}")
    print(f"  Output: {fig_dir}")
    print(f"{'=' * 70}")

    labels = []
    all_stats = []

    for mp in model_paths:
        label = label_from_path(mp, study_name)
        print(f"  Loading {label} ...")
        try:
            model, params, layers = load_model(mp, geo, device)
            stats = compute_time_statistics(model, geo, device, times=times)
            labels.append(label)
            all_stats.append(stats)
            print(f"    T_max(0)={stats['T_max'][0]:.3f}  "
                  f"T_mean(30)={stats['T_mean'][-1]:.4f}  "
                  f"S_mean(30)={stats['S_mean'][-1]:.4f}")
        except Exception as e:
            print(f"    ERROR loading {mp}: {e}")
            continue

    if len(labels) < 1:
        print("  No valid models loaded — skipping plots.")
        return

    # ── generate all figures ──
    print(f"\n  Generating figures ...")
    plot_fig_A_max(study_title, labels, all_stats, fig_dir)
    plot_fig_B_mean(study_title, labels, all_stats, fig_dir)
    plot_fig_C_min(study_title, labels, all_stats, fig_dir)
    plot_fig_D_tumor_envelope(study_title, labels, all_stats, fig_dir)
    plot_fig_E_BT_ratio(study_title, labels, all_stats, fig_dir)
    plot_fig_F_spatial_std(study_title, labels, all_stats, fig_dir)
    plot_fig_G_phase_portrait(study_title, labels, all_stats, fig_dir)
    plot_fig_H_terminal_bars(study_title, labels, all_stats, fig_dir)
    plot_fig_I_heatmap(study_title, labels, all_stats, fig_dir)
    plot_fig_J_combined_summary(study_title, labels, all_stats, fig_dir)

    # ── save statistics to JSON for later use ──
    stats_path = os.path.join(fig_dir, 'time_statistics.json')
    save_stats = {}
    for label, stats in zip(labels, all_stats):
        save_stats[label] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                             for k, v in stats.items()}
    with open(stats_path, 'w') as f:
        json.dump(save_stats, f, indent=2)
    print(f"  Stats JSON → {stats_path}")

    print(f"\n  Done: {len(labels)} experiments × 10 figures = "
          f"{len(labels) * 10} curves plotted.")
    return labels, all_stats


# =============================================================================
#
# CONFIGURATION — EDIT HERE THEN RUN
#
# =============================================================================

# ─── Which studies to process ──────────────────────────────────────
# Must match directory names in ABLATION_DIR
STUDIES_TO_PLOT = ['S3_beta_I']
# STUDIES_TO_PLOT = ['S1_alpha_S', 'S2_gamma_vas', 'S3_beta_I']
# STUDIES_TO_PLOT = ['all']   # processes all found studies

# ─── Paths ─────────────────────────────────────────────────────────
ABLATION_DIR = 'ablation_results'      # where the studies were saved
NX = 60                                # must match training mesh

# ─── Time resolution for evaluation ───────────────────────────────
TIMES = np.linspace(0, 30, 121)        # 121 points = every 0.25 days

# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':

    geo = QuadGeometry(x_min=0.0, x_max=6.0, y_min=0.0, y_max=6.0, nx=NX, ny=NX)

    # Discover available studies
    if 'all' in STUDIES_TO_PLOT:
        available = [d for d in os.listdir(ABLATION_DIR)
                     if os.path.isdir(os.path.join(ABLATION_DIR, d))
                     and os.path.isdir(os.path.join(ABLATION_DIR, d, 'models'))]
        studies_to_plot = sorted(available)
    else:
        studies_to_plot = STUDIES_TO_PLOT

    print(f"\n{'#' * 70}")
    print(f"  POST-PROCESSING VISUALIZATION")
    print(f"  Studies: {studies_to_plot}")
    print(f"  Mesh: {NX}x{NX} = {geo.n_nodes} nodes")
    print(f"  Time points: {len(TIMES)}")
    print(f"{'#' * 70}")

    from datetime import datetime
    t0 = datetime.now()

    for study_name in studies_to_plot:
        study_dir = os.path.join(ABLATION_DIR, study_name)
        if not os.path.isdir(study_dir):
            print(f"\n  Study dir not found: {study_dir} — skipping.")
            continue
        process_study(study_name, study_dir, geo, TIMES)

    total = (datetime.now() - t0).total_seconds() / 60
    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — {total:.1f} min")
    print(f"  Figures in: {ABLATION_DIR}/<study>/figures/")
    print(f"{'#' * 70}")
