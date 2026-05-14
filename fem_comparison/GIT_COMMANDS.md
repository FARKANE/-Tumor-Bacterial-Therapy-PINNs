# Step-by-step: add the FEM comparison to the repo

## 0. Prerequisites

Working clone of the repo on your machine:

```bash
git clone https://github.com/FARKANE/-Tumor-Bacterial-Therapy-PINNs.git
cd -Tumor-Bacterial-Therapy-PINNs
```

## 1. Drop the new subfolder in

Copy the entire `fem_comparison/` folder I sent you into the repo root.
After this step the directory tree should look like:

```
-Tumor-Bacterial-Therapy-PINNs/
├── .gitignore
├── README.md
├── ablation_study.py
├── pinn_solver.py
├── postprocessing.py
├── requirements.txt
└── fem_comparison/                <- new
    ├── .gitignore
    ├── README.md
    ├── PINN_vs_FEM_Comparison.ipynb
    ├── _README_root_snippet.md
    ├── compare_pinn_fem.py
    ├── requirements.txt
    └── train_pinn_mms.py
```

## 2. Patch the root README

Open `README.md`. Find the line that starts the citation section
(`## Citation`). Just **before** that line, paste the contents of
`fem_comparison/_README_root_snippet.md` (skip the first three comment
lines — paste only the `## Comparison with FEM` section and what
follows). Then delete `fem_comparison/_README_root_snippet.md`:

```bash
rm fem_comparison/_README_root_snippet.md
```

## 3. Stage and commit

```bash
git add fem_comparison/
git add README.md
git status            # confirm what you're about to commit

git commit -m "Add FEM comparison (Section 6.5): MMS test, PINN training, plotting, Colab notebook"
```

## 4. Commit the trained PINN model

The self-bootstrap notebook expects `fem_comparison/pinn_bio_mms_model.pt`
in the repo (it's only ~200 KB). Drop the model file into the folder and
commit:

```bash
cp /path/to/pinn_bio_mms_model.pt fem_comparison/
# Remove that filename from .gitignore if present:
sed -i 's|^pinn_bio_mms/$||' fem_comparison/.gitignore

git add fem_comparison/pinn_bio_mms_model.pt
git commit -m "fem_comparison: add trained PINN checkpoint (paper results)"
```

## 5. Create a GitHub Release with Data.zip

The FEM reference data (~50 MB) is too big to commit. Upload it as a
release asset instead:

```bash
git tag -a v1.0-paper-results -m "Code and data for the published paper"
git push origin v1.0-paper-results
```

Then on GitHub:
1. Go to **Releases → Draft a new release**.
2. Choose the tag `v1.0-paper-results`.
3. Set a title (e.g. "Paper results — Section 6.5 (FEM comparison)").
4. **Attach `Data.zip`** by dragging it into the "binary assets" area.
5. Publish.

After this, the self-bootstrap notebook will resolve the asset at
`https://github.com/FARKANE/-Tumor-Bacterial-Therapy-PINNs/releases/download/v1.0-paper-results/Data.zip`
automatically. Test it: open
`fem_comparison/PINN_vs_FEM_SelfBootstrap.ipynb` in Colab via
**Open in Colab** and run all cells.

If you use a different tag name, update `RELEASE_TAG` in Cell 1 of the
notebook accordingly.

## 6. (Optional) Commit the published figures

If you want the comparison figures to ship with the repo (so reviewers
can see the published results without rerunning anything):

```bash
mkdir -p fem_comparison/paper_figures
cp comparison_results/pinn_vs_fem_errors.pdf  fem_comparison/paper_figures/
cp comparison_results/snapshot_t0.5.pdf       fem_comparison/paper_figures/
cp comparison_results/summary_bars.pdf        fem_comparison/paper_figures/
cp comparison_results/solution_exact_pinn_fem_t0.50.pdf  fem_comparison/paper_figures/
cp comparison_results/results_table.tex       fem_comparison/paper_figures/

git add fem_comparison/paper_figures/
git commit -m "fem_comparison: include published figures + LaTeX table"
```

## 7. Push everything

```bash
git push origin main
```

## 8. (Optional) Verify the self-bootstrap notebook works

Open the notebook in Colab using the link in `fem_comparison/README.md`,
click **Run all**, and watch it pull everything from GitHub
automatically. If something fails:
- "release asset 404" → step 5 was not completed, or the tag name differs from `RELEASE_TAG` in Cell 1.
- "trained PINN not found" → step 4 was not completed.
- Anything else → the cell 2 sanity check will tell you which expected file is missing.

## 9. (Optional) Tag a follow-up release

If you later update the comparison code or retrain with a better model,
bump the release tag (`v1.1-paper-results`, etc.) and re-upload assets.
Update `RELEASE_TAG` in the notebook's Cell 1 to keep the
"Open in Colab" link pointed at the new release.

## Notes

- The `.gitignore` inside `fem_comparison/` excludes `pinn_bio_mms/`
  (the training output directory), `comparison_results/`, `Data/`, and
  `Data.zip`. The trained model file `pinn_bio_mms_model.pt` is
  **committed directly** to `fem_comparison/`; the training directory
  `pinn_bio_mms/` (used during training only) is ignored.
- If you change the figure filenames in the LaTeX section
  (Section 6.5), keep them in sync with the names produced by
  `compare_pinn_fem.py` and the self-bootstrap notebook.
