# FEM vs PINN Comparison — Method of Manufactured Solutions

This subfolder accompanies Section 6.5 of the paper. It benchmarks the
PINN framework against a $\mathbb{P}_1$ finite-element method on a test
case where the exact solution is known, using the Method of Manufactured
Solutions (Roache, 2002).

The PINN training in this subfolder is **independent** of the main
biological problem in `pinn_solver.py`: it solves the modified PDE
system that the manufactured solution satisfies, on the unit square
$\Omega = [0,1]^2$ with homogeneous Neumann boundary conditions, and
compares its output against a reference FEM solution on the same
$91 \times 91$ evaluation grid.

## Files

| File | Purpose |
|---|---|
| `train_pinn_mms.py` | Trains the PINN on the manufactured solution. Adam (45,000 epochs) + L-BFGS (5,000 iters), $\sim$2.4 h on a Tesla T4 GPU. Saves the trained model to `pinn_bio_mms/pinn_bio_mms_model.pt`. |
| `compare_pinn_fem.py` | Loads the trained PINN and the FEM data, computes PINN and FEM errors against the exact MMS on a common grid, writes figures + a LaTeX table to `comparison_results/`. |
| `pinn_bio_mms_model.pt` | Trained PINN weights used for the published results (~200 KB). Committed to the repo so the comparison is reproducible from a clone. |
| `PINN_vs_FEM_SelfBootstrap.ipynb` | **Recommended Colab notebook.** Self-bootstrapping: clones this repo and downloads the FEM `Data.zip` from a GitHub Release asset; no Drive setup needed. Click "Open in Colab" and "Run all". |
| `PINN_vs_FEM_Comparison.ipynb` | Alternative Colab notebook for paper authors. Reads `pinn_bio_mms_model.pt` and `Data.zip` from `MyDrive/PINN_FEM/`. |
| `requirements.txt` | Minimal Python dependencies for this subfolder. |

## Manufactured solution

With $\varphi(x, y) = \cos(\pi x)\cos(\pi y)$ (satisfies $\partial_n \varphi = 0$
on $\partial \Omega$):

```
T_ex(t, x, y) = (0.40 + 0.20 φ) · exp(-0.3 t)
B_ex(t, x, y) = (0.08 + 0.03 φ) · (1 - exp(-0.5 t))
O_ex(t, x, y) =  0.20 + 0.02 φ · exp(-t)
I_ex(t, x, y) = (0.020 + 0.008 φ) · (1 - exp(-0.3 t))
S_ex(t, x, y) = (0.050 + 0.020 φ) · (1 - exp(-0.5 t))
```

The five source terms $f_T, f_B, f_O, f_I, f_S$ that make this an exact
solution of the original system are given in Appendix A of the paper and
are computed in `train_pinn_mms.py` (class `BioMMS.source_terms`).

## Reproducibility — 3-step workflow

### 1. Train the PINN

```bash
python train_pinn_mms.py
```

Outputs:
- `pinn_bio_mms/pinn_bio_mms_model.pt`  — trained network weights
- `pinn_bio_mms/training_history.png`   — Adam + L-BFGS loss curves
- `pinn_bio_mms/pinn_errors.pdf`        — PINN errors on a 61×61 grid
- `pinn_bio_mms/exact_solution/`        — exact MMS field at 100 times,
   to be sent to the FEM solver as the reference target

A GPU is strongly recommended ($\sim$2.4 h on T4; CPU would take days).

### 2. Obtain the FEM reference solution

Solve the same modified system with $\mathbb{P}_1$ elements on a
$91 \times 91$ grid over $t \in [0, 1]$, using the source terms from
`pinn_bio_mms/exact_solution/`. Save the output in the layout

```
Data/
├── Solution_exacte/{Tex,Bex,Oex,Iex,Sex}/<t>     # analytical exact
└── Solution_approchée/{Tap,Bap,Oap,Iap,Sap}/<t>  # FEM approximation
```

where `<t>` is the time label (`0.01`, `0.02`, ..., `0.99`, `1`) and
each file is a 3-column text grid `x y u`.

Any FEM/FreeFEM/FEniCS solver capable of integrating the modified system
will do. The reference data used in the paper was produced with a
first-order semi-implicit scheme ($\Delta t = 10^{-2}$, diffusion
implicit, reactions explicit).

### 3. Run the comparison

```bash
python compare_pinn_fem.py
```

Outputs in `comparison_results/`:
- `pinn_vs_fem_errors.pdf` — relative + absolute $L^2$ error vs $t$
- `snapshot_t0.5.pdf`      — pointwise error fields at $t = 0.5$
- `summary_bars.pdf`       — max errors per variable
- `results_table.tex`      — LaTeX table for the paper
- `errors.npz`             — raw error arrays for further analysis

## Expected results

On the $91 \times 91$ evaluation grid, max $L^2$ errors over $t \in [0,1]$
(see `comparison_results/results_table.tex`):

| Variable | PINN rel. $L^2$ | FEM rel. $L^2$ | PINN abs. $L^2$ | FEM abs. $L^2$ |
|---|---|---|---|---|
| $T$ | $2.05 \times 10^{-4}$ | $5.80 \times 10^{-3}$ | $6.34 \times 10^{-5}$ | $1.79 \times 10^{-3}$ |
| $B$ | $8.67 \times 10^{-4}$ | $2.87 \times 10^{-3}$ | $1.68 \times 10^{-5}$ | $9.30 \times 10^{-5}$ |
| $O$ | $8.81 \times 10^{-4}$ | $1.73 \times 10^{-4}$ | $1.78 \times 10^{-4}$ | $3.50 \times 10^{-5}$ |
| $I$ | $1.48 \times 10^{-2}$ | $3.41 \times 10^{-2}$ | $7.89 \times 10^{-5}$ | $1.83 \times 10^{-4}$ |
| $S$ | $2.04 \times 10^{-3}$ | $4.09 \times 10^{-3}$ | $3.73 \times 10^{-5}$ | $8.30 \times 10^{-5}$ |

The PINN is more accurate than FEM on $T, B, I, S$ (factors of $28$,
$3.3$, $2.3$, $2.0$ respectively); FEM is more accurate on $O$ (factor
of $5$). See Section 6.5 of the paper for the interpretation.

## Running on Google Colab

### Option A — Self-bootstrap (recommended for reviewers)

Open `PINN_vs_FEM_SelfBootstrap.ipynb` directly from this repo in Colab — click "Open in Colab" or use the URL:

```
https://colab.research.google.com/github/FARKANE/-Tumor-Bacterial-Therapy-PINNs/blob/main/fem_comparison/PINN_vs_FEM_SelfBootstrap.ipynb
```

Then click **Runtime → Run all**. The notebook will:
- clone this repo,
- download `Data.zip` from the [latest release](https://github.com/FARKANE/-Tumor-Bacterial-Therapy-PINNs/releases) asset,
- load the trained PINN from the cloned repo,
- run the full comparison and produce all figures + the LaTeX table,
- offer a one-click download of the results as a zip.

No Drive setup, no manual uploads, GPU optional ($\sim$3 minutes on CPU).

### Option B — Drive-backed (for paper authors)

Open `PINN_vs_FEM_Comparison.ipynb`. Put `pinn_bio_mms_model.pt` and `Data.zip` into `MyDrive/PINN_FEM/`. Run all cells; outputs are auto-copied to `MyDrive/PINN_FEM/comparison_results/`. Useful when you are iterating on figures and want outputs to persist across sessions.

## License

MIT, same as the parent repository.
