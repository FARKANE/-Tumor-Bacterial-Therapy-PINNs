# Deep Learning for Tumor-Bacteria PDE Systems: A Physics-Informed Neural Network Approach

This repository contains the code accompanying the paper by **A. Farkane \& C. Lassounon**.

## Problem Description

We solve a five-species reaction-diffusion PDE system modeling breast tumor–bacteria interactions using Physics-Informed Neural Networks (PINNs). The species are:

|Variable|Description|
|-|-|
|**T**|Tumor cell density|
|**B**|Bacteria density|
|**O**|Oxygen concentration|
|**I**|Cytokine concentration|
|**S**|Signaling molecule concentration|

The system is defined on a 2D quadrangle domain with Neumann (zero-flux) boundary conditions and Gaussian-type initial conditions with stochastic perturbations.

## Repository Structure

```
├── src/
│   ├── pinn\_solver.py          # Main PINN solver (mesh, network, training, visualization)
│   ├── ablation\_study.py       # 9 sensitivity studies (S1–S9) with model checkpointing
│   └── postprocessing.py       # Load saved models \& generate publication figures (A–J)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/tumor-bacteria-pinn.git
cd tumor-bacteria-pinn
pip install -r requirements.txt
```

### Training

```bash
python src/pinn\_solver.py
```

This will:

1. Build a 60×60 triangulated quadrangle mesh on \[0, 6]²
2. Train the PINN for 8000 epochs (early stopping with patience 2000)
3. Save checkpoints to `checkpoints\_quad/`
4. Generate result plots (`results\_quad.png`, `training\_history\_quad.png`, `initial\_conditions\_quad.png`)

### Ablation Studies

```bash
python src/ablation\_study.py
```

### Post-Processing \& Visualization

After ablation training completes, load saved models and generate publication figures (A–J):

```bash
python src/postprocessing.py
```

Generates 10 figure types per study: domain max/mean/min vs time, tumor suppression envelope, B/T therapy ratio, spatial heterogeneity, phase portraits, terminal bar charts, heatmaps, and combined summaries.

## Method

* **Architecture**: Fully connected network \[3 → 64 → 64 → 64 → 64 → 5] with Tanh activation
* **Loss**: Weighted sum of PDE residual (λ\_pde=1), initial condition (λ\_ic=50), and Neumann BC (λ\_bc=1) losses
* **Optimizer**: Adam with ReduceLROnPlateau scheduler
* **Domain discretization**: Structured triangular mesh via quad splitting

## Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0 (CUDA recommended)
* NumPy, Matplotlib

## Citation

If you use this code, please cite:

```bibtex
@article{farkane2026tumor,
  title={Mathematical Modeling of Cancer–Bacterial Therapy: Analysis and

Numerical Simulation via Physics-Informed Neural Networks (PINNs)},
  author={Farkane, Ayoub and Lassounon, David.},
  year={2026}
}
```

## License

MIT License

