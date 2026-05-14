"""
=============================================================================
PINN vs FEM Comparison — Biological-Scale MMS
=============================================================================
Run this AFTER training (so pinn_bio_mms/pinn_bio_mms_model.pt exists)
and AFTER unzipping David's FEM data into ./Data/

Expected layout:
  ./pinn_bio_mms/pinn_bio_mms_model.pt   (your trained PINN)
  ./Data/Solution_exacte/{Tex,Bex,Oex,Iex,Sex}/<time>     (David's exact)
  ./Data/Solution_approchée/{Tap,Bap,Oap,Iap,Sap}/<time>  (David's FEM)

What this script does:
  1. Reloads the trained PINN model.
  2. Identifies the FEM grid (91×91) and time points from the Data folder.
  3. Evaluates PINN on the SAME grid at the SAME times.
  4. Verifies David's "exact" solution matches our analytical MMS.
  5. Computes relative AND absolute L2 errors for PINN and FEM uniformly.
  6. Produces 3 publication-ready figures + a results table.
=============================================================================
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ---- paths ------------------------------------------------------------------
PINN_DIR = 'pinn_bio_mms'
PINN_MODEL_PATH = os.path.join(PINN_DIR, 'pinn_bio_mms_model.pt')
DATA_DIR = 'Data'
EX_DIR = os.path.join(DATA_DIR, 'Solution_exacte')
OUT_DIR = 'comparison_results'
os.makedirs(OUT_DIR, exist_ok=True)

# Find the "Solution_approchée" folder (handles unicode encoding quirks)
ap_candidates = [d for d in os.listdir(DATA_DIR)
                 if d.startswith('Solution_approch') and d != 'Solution_exacte']
if not ap_candidates:
    raise FileNotFoundError(
        f"Could not find Solution_approch* in {DATA_DIR}. Got: {os.listdir(DATA_DIR)}")
AP_DIR = os.path.join(DATA_DIR, ap_candidates[0])
print(f"PINN model : {PINN_MODEL_PATH}")
print(f"FEM exact  : {EX_DIR}")
print(f"FEM approx : {AP_DIR}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device     : {device}")


# =============================================================================
# CLASSES — duplicated from the corrected training file so this script is
# fully self-contained.
# =============================================================================

class BioMMS:
    def __init__(self, L=1.0):
        self.L = L
        self.k = 2.0 * np.pi**2 / L**2

    def phi_t(self, x, y):
        return torch.cos(np.pi * x / self.L) * torch.cos(np.pi * y / self.L)

    def phi_n(self, x, y):
        return np.cos(np.pi * x / self.L) * np.cos(np.pi * y / self.L)

    def exact_n(self, x, y, t):
        p = self.phi_n(x, y)
        T = (0.4 + 0.2 * p) * np.exp(-0.3 * t)
        B = (0.08 + 0.03 * p) * (1.0 - np.exp(-0.5 * t))
        O = 0.2 + 0.02 * p * np.exp(-t)
        I = (0.02 + 0.008 * p) * (1.0 - np.exp(-0.3 * t))
        S = (0.05 + 0.02 * p) * (1.0 - np.exp(-0.5 * t))
        return T, B, O, I, S

    def ic_t(self, x, y):
        p = self.phi_t(x, y)
        T0 = 0.4 + 0.2 * p
        B0 = torch.zeros_like(p)
        O0 = 0.2 + 0.02 * p
        I0 = torch.zeros_like(p)
        S0 = torch.zeros_like(p)
        return torch.cat([T0, B0, O0, I0, S0], dim=1)


class PINN_HardIC(nn.Module):
    def __init__(self, layers, mms):
        super().__init__()
        self.act = nn.Tanh()
        self.lins = nn.ModuleList()
        for i in range(len(layers) - 1):
            l = nn.Linear(layers[i], layers[i + 1])
            self.lins.append(l)
        self.mms = mms

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        for l in self.lins[:-1]:
            inp = self.act(l(inp))
        nn_out = self.lins[-1](inp)
        return self.mms.ic_t(x, y) + t * nn_out


# =============================================================================
# LOAD TRAINED PINN
# =============================================================================

print("\nLoading PINN model...")
ckpt = torch.load(PINN_MODEL_PATH, map_location=device, weights_only=False)
LAYERS = ckpt['layers']
mms = BioMMS(L=1.0)
model = PINN_HardIC(LAYERS, mms).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Layers: {LAYERS}")
print(f"  Params: {sum(p.numel() for p in model.parameters())}")
print(f"  Train wall time: {ckpt.get('total_time', 0):.0f}s")


def predict_pinn(x_flat, y_flat, t_val):
    """Evaluate PINN at 1D arrays of (x,y) for a single time t."""
    with torch.no_grad():
        xt = torch.as_tensor(x_flat, dtype=torch.float32,
                             device=device).unsqueeze(1)
        yt = torch.as_tensor(y_flat, dtype=torch.float32,
                             device=device).unsqueeze(1)
        tt = torch.full((len(x_flat), 1), float(t_val),
                        dtype=torch.float32, device=device)
        return model(xt, yt, tt).cpu().numpy()


# =============================================================================
# IDENTIFY FEM GRID & TIME POINTS
# =============================================================================

# Sort time files from the Tap folder by numerical value
tap_dir = os.path.join(AP_DIR, 'Tap')
time_entries = sorted([(float(f), f) for f in os.listdir(tap_dir)])
time_vals = [t for t, _ in time_entries]
time_names = [n for _, n in time_entries]
print(f"\nFEM timesteps: {len(time_entries)} "
      f"(t in [{time_vals[0]:.2f}, {time_vals[-1]:.2f}])")

# Read one file to learn the grid
sample = np.loadtxt(os.path.join(tap_dir, time_names[0]))
N_grid = sample.shape[0]
nx = int(np.sqrt(N_grid))
assert nx * nx == N_grid, f"FEM grid is not square: {N_grid} points"
h = 1.0 / (nx - 1)
xs_fem = sample[:, 0]
ys_fem = sample[:, 1]
print(f"FEM grid: {nx}×{nx} = {N_grid} points (h = {h:.6f})")


# =============================================================================
# VERIFY FEM "EXACT" MATCHES OUR MMS
# =============================================================================

print("\nVerifying FEM exact solution matches analytical MMS...")
max_diff = 0.0
for t_val, t_name in time_entries[::20]:  # spot-check every 20th step
    for vname, ex_sub, idx in [('T','Tex',0),('B','Bex',1),('O','Oex',2),
                                ('I','Iex',3),('S','Sex',4)]:
        d = np.loadtxt(os.path.join(EX_DIR, ex_sub, t_name))
        analytic = mms.exact_n(d[:, 0], d[:, 1], t_val)[idx]
        diff = np.abs(d[:, 2] - analytic).max()
        max_diff = max(max_diff, diff)
print(f"  max |FEM_exact - analytical| = {max_diff:.2e}  (should be ≪ 1)")


# =============================================================================
# EVALUATE PINN ON FEM GRID, COMPUTE ERRORS
# =============================================================================

print("\nEvaluating PINN on FEM grid + computing errors...")
vn = ['T', 'B', 'O', 'I', 'S']
ap_sub_map = {'T':'Tap','B':'Bap','O':'Oap','I':'Iap','S':'Sap'}
ex_sub_map = {'T':'Tex','B':'Bex','O':'Oex','I':'Iex','S':'Sex'}

# Storage
pinn_rel = {v: [] for v in vn}; pinn_abs = {v: [] for v in vn}
fem_rel  = {v: [] for v in vn}; fem_abs  = {v: [] for v in vn}

for k, (t_val, t_name) in enumerate(time_entries):
    # Reference exact: load David's file (already validated against analytical)
    ex_data = {}
    for v in vn:
        d = np.loadtxt(os.path.join(EX_DIR, ex_sub_map[v], t_name))
        ex_data[v] = d[:, 2]
        if v == 'T':
            x_ref, y_ref = d[:, 0], d[:, 1]

    # PINN prediction at FEM grid points
    pinn_pred = predict_pinn(x_ref, y_ref, t_val)  # shape (N, 5)

    # FEM prediction
    fem_data = {}
    for v in vn:
        d = np.loadtxt(os.path.join(AP_DIR, ap_sub_map[v], t_name))
        fem_data[v] = d[:, 2]

    # Errors
    for j, v in enumerate(vn):
        ue = ex_data[v]
        nrm = np.sqrt(h*h*np.sum(ue**2))

        err_p = pinn_pred[:, j] - ue
        L2_p = np.sqrt(h*h*np.sum(err_p**2))
        pinn_abs[v].append(L2_p)
        pinn_rel[v].append(L2_p / nrm if nrm > 1e-10 else float('nan'))

        err_f = fem_data[v] - ue
        L2_f = np.sqrt(h*h*np.sum(err_f**2))
        fem_abs[v].append(L2_f)
        fem_rel[v].append(L2_f / nrm if nrm > 1e-10 else float('nan'))

    if (k + 1) % 25 == 0:
        print(f"  {k+1:>3}/{len(time_entries)} times processed")


# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

print(f"\n{'='*72}")
print(f"{'PINN vs FEM ERROR SUMMARY (max over t in [0,1], grid '+str(nx)+'×'+str(nx)+')':<72}")
print(f"{'='*72}")
print(f"{'Var':<5}"
      f"{'PINN rel':>14}{'FEM rel':>14}{'ratio':>8}"
      f"{'PINN abs':>14}{'FEM abs':>14}{'ratio':>8}")
print('-' * 78)
ta = np.array(time_vals)

def maxat(arr, t, vname):
    a = np.array(arr)
    if vname in ('B', 'I', 'S'):
        m = t >= 0.1
        i = np.nanargmax(a[m]); return a[m][i], t[m][i]
    i = np.nanargmax(a); return a[i], t[i]

for v in vn:
    pr, prt = maxat(pinn_rel[v], ta, v)
    fr, frt = maxat(fem_rel[v], ta, v)
    pa, pat = maxat(pinn_abs[v], ta, v)
    fa, fat = maxat(fem_abs[v], ta, v)
    rel_ratio = fr / pr if pr > 0 else float('nan')
    abs_ratio = fa / pa if pa > 0 else float('nan')
    print(f"{v:<5}"
          f"{pr:>14.3e}{fr:>14.3e}{rel_ratio:>8.2f}×"
          f"{pa:>14.3e}{fa:>14.3e}{abs_ratio:>8.2f}×")
print(f"{'='*78}")
print("ratio > 1 means PINN is more accurate on that variable (FEM error / PINN error).")
print("For B, I, S the relative L2 is reported from t >= 0.1 only "
      "(initial condition is zero).")


# =============================================================================
# SAVE NUMERICAL RESULTS
# =============================================================================
np.savez(os.path.join(OUT_DIR, 'errors.npz'),
         times=ta, **{f'pinn_rel_{v}': pinn_rel[v] for v in vn},
         **{f'pinn_abs_{v}': pinn_abs[v] for v in vn},
         **{f'fem_rel_{v}': fem_rel[v] for v in vn},
         **{f'fem_abs_{v}': fem_abs[v] for v in vn})


# =============================================================================
# FIGURE 1 — Error vs time, 2 rows × 5 cols (relative + absolute)
# =============================================================================

print("\nGenerating figures...")
vc_pinn = '#1565c0'   # blue
vc_fem  = '#d32f2f'   # red
vlabels = {'T':'$T$','B':'$B$','O':'$O$','I':'$I$','S':'$S$'}

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle('PINN vs FEM Errors  —  Biological-Scale MMS  (grid '
             f'{nx}×{nx})', fontsize=15, weight='bold', y=1.00)

for j, v in enumerate(vn):
    # relative
    ax = axes[0, j]
    if v in ('B', 'I', 'S'):
        m = ta >= 0.1
        ax.semilogy(ta[m], np.array(pinn_rel[v])[m],
                    '-', color=vc_pinn, lw=2.0, label='PINN')
        ax.semilogy(ta[m], np.array(fem_rel[v])[m],
                    '--', color=vc_fem, lw=2.0, label='FEM')
        ax.text(0.04, 0.04, '$t \\geq 0.1$', transform=ax.transAxes,
                fontsize=8, color='gray')
    else:
        ax.semilogy(ta, pinn_rel[v], '-', color=vc_pinn, lw=2.0, label='PINN')
        ax.semilogy(ta, fem_rel[v], '--', color=vc_fem, lw=2.0, label='FEM')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Relative $L^2$ error')
    ax.set_title(f'{vlabels[v]}  (relative)', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, which='both')
    if j == 0:
        ax.legend(loc='best', fontsize=10)

    # absolute
    ax = axes[1, j]
    ax.semilogy(ta, pinn_abs[v], '-', color=vc_pinn, lw=2.0, label='PINN')
    ax.semilogy(ta, fem_abs[v], '--', color=vc_fem, lw=2.0, label='FEM')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Absolute $L^2$ error')
    ax.set_title(f'{vlabels[v]}  (absolute)', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, which='both')
    if j == 0:
        ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pinn_vs_fem_errors.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'pinn_vs_fem_errors.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUT_DIR}/pinn_vs_fem_errors.pdf")


# =============================================================================
# FIGURE 2 — Snapshot at t = 0.5: Exact / PINN / FEM / |PINN-ex| / |FEM-ex|
# =============================================================================

t_snap = 0.5
# Find closest available time
i_snap = int(np.argmin(np.abs(ta - t_snap)))
t_snap_actual = ta[i_snap]
t_name_snap = time_names[i_snap]
print(f"  Snapshot time: t = {t_snap_actual:.2f} (file '{t_name_snap}')")

X = x_ref.reshape(nx, nx); Y = y_ref.reshape(nx, nx)
ex_snap = {}
for v in vn:
    d = np.loadtxt(os.path.join(EX_DIR, ex_sub_map[v], t_name_snap))
    ex_snap[v] = d[:, 2].reshape(nx, nx)
pinn_snap = predict_pinn(x_ref, y_ref, t_snap_actual)
fem_snap = {}
for v in vn:
    d = np.loadtxt(os.path.join(AP_DIR, ap_sub_map[v], t_name_snap))
    fem_snap[v] = d[:, 2].reshape(nx, nx)

# Two rows of error fields (PINN top, FEM bottom). Same colour-scale per
# variable column so the eye can compare directly. Lighter and clearer than
# showing the exact + both predictions, which only carry redundant info.
fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
fig.suptitle(f'Pointwise error  $|u - u_{{\\mathrm{{exact}}}}|$  '
             f'at $t = {t_snap_actual:.2f}$  (grid {nx}$\\times${nx})',
             fontsize=15, weight='bold', y=1.00)

for j, v in enumerate(vn):
    ue = ex_snap[v]
    up = pinn_snap[:, j].reshape(nx, nx)
    uf = fem_snap[v]
    err_p = np.abs(up - ue)
    err_f = np.abs(uf - ue)
    err_hi = max(err_p.max(), err_f.max(), 1e-12)

    im0 = axes[0, j].pcolormesh(X, Y, err_p, cmap='inferno',
                                shading='auto', vmin=0, vmax=err_hi)
    plt.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)
    axes[0, j].set_aspect('equal')
    axes[0, j].set_title(f'PINN  {vlabels[v]}', fontsize=12, weight='bold')
    axes[0, j].set_xticks([0, 0.5, 1]); axes[0, j].set_yticks([0, 0.5, 1])

    im1 = axes[1, j].pcolormesh(X, Y, err_f, cmap='inferno',
                                shading='auto', vmin=0, vmax=err_hi)
    plt.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)
    axes[1, j].set_aspect('equal')
    axes[1, j].set_title(f'FEM  {vlabels[v]}', fontsize=12, weight='bold')
    axes[1, j].set_xticks([0, 0.5, 1]); axes[1, j].set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'snapshot_t0.5.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'snapshot_t0.5.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUT_DIR}/snapshot_t0.5.pdf")


# =============================================================================
# FIGURE 3 — Compact summary bar chart for the paper
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
xpos = np.arange(len(vn))
width = 0.35

# Use the max value reported in the table above
def maxv_arr(d, vlist, t, _filter=True):
    out = []
    for v in vlist:
        a = np.array(d[v])
        if _filter and v in ('B','I','S'):
            m = t >= 0.1
            out.append(np.nanmax(a[m]))
        else:
            out.append(np.nanmax(a))
    return np.array(out)

pinn_relmax = maxv_arr(pinn_rel, vn, ta, True)
fem_relmax  = maxv_arr(fem_rel,  vn, ta, True)
pinn_absmax = maxv_arr(pinn_abs, vn, ta, False)
fem_absmax  = maxv_arr(fem_abs,  vn, ta, False)

axes[0].bar(xpos - width/2, pinn_relmax, width, color=vc_pinn, label='PINN')
axes[0].bar(xpos + width/2, fem_relmax, width, color=vc_fem, label='FEM')
axes[0].set_yscale('log')
axes[0].set_xticks(xpos); axes[0].set_xticklabels([vlabels[v] for v in vn])
axes[0].set_ylabel('Max relative $L^2$ error')
axes[0].set_title('Relative $L^2$ (max over $t$)', fontsize=12, weight='bold')
axes[0].grid(True, alpha=0.3, axis='y', which='both')
axes[0].legend()

axes[1].bar(xpos - width/2, pinn_absmax, width, color=vc_pinn, label='PINN')
axes[1].bar(xpos + width/2, fem_absmax, width, color=vc_fem, label='FEM')
axes[1].set_yscale('log')
axes[1].set_xticks(xpos); axes[1].set_xticklabels([vlabels[v] for v in vn])
axes[1].set_ylabel('Max absolute $L^2$ error')
axes[1].set_title('Absolute $L^2$ (max over $t$)', fontsize=12, weight='bold')
axes[1].grid(True, alpha=0.3, axis='y', which='both')
axes[1].legend()

fig.suptitle('PINN vs FEM — Summary', fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'summary_bars.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'summary_bars.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {OUT_DIR}/summary_bars.pdf")


# =============================================================================
# WRITE LATEX-READY RESULTS TABLE
# =============================================================================

tex_path = os.path.join(OUT_DIR, 'results_table.tex')
with open(tex_path, 'w') as f:
    f.write(r"""\begin{table}[h]
\centering
\caption{PINN vs FEM: maximum $L^2$ errors on the """ + f"{nx}$\\times${nx}" +
            r""" evaluation grid over $t \in [0,1]$ (relative errors for $B,I,S$ from $t \ge 0.1$).}
\label{tab:pinn_vs_fem}
\begin{tabular}{lcccc}
\toprule
Variable & PINN rel.\ $L^2$ & FEM rel.\ $L^2$ & PINN abs.\ $L^2$ & FEM abs.\ $L^2$ \\
\midrule
""")
    for v, prv, frv, pav, fav in zip(vn, pinn_relmax, fem_relmax,
                                       pinn_absmax, fem_absmax):
        f.write(f"  ${v}$ & {prv:.2e} & {frv:.2e} & {pav:.2e} & {fav:.2e} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
print(f"  -> {tex_path}")

print(f"\nDone. All comparison artifacts in {OUT_DIR}/")
