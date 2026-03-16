"""
=============================================================================
ABLATION & SENSITIVITY STUDY FOR TUMOR-BACTERIA PINN
=============================================================================
Adapted for Google Colab — no argparse, direct configuration at the bottom.
Each experiment's best model is saved for later analysis.

Studies:
  S1: Signal cytotoxicity  α_S
  S2: Vascularization      γ_vas
  S3: Immune clearance     β_I
  S4: Signal diffusion     D_S
  S5: Hypoxia sensitivity  K_H
  S6: Tumor growth rate    ρ_T
  S7: Network architecture (depth/width)
  S8: Collocation points   (N_pde, N_ic, N_bc)
  S9: Loss weights         (λ_pde, λ_ic, λ_bc)

Usage (in Colab):
  1. Run this entire cell
  2. Edit STUDIES_TO_RUN and N_EPOCHS at the bottom
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import json
import csv
import copy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =============================================================================
# GEOMETRY
# =============================================================================

class QuadGeometry:
    """Rectangular domain [x_min, x_max] × [y_min, y_max] with triangular mesh."""

    def __init__(self, x_min=0.0, x_max=6.0, y_min=0.0, y_max=6.0, nx=40, ny=40):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.nx, self.ny = nx, ny
        self.nodes, self.elements, self.boundary_edges = self._build_mesh()
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        self.x = self.nodes[:, 0]
        self.y = self.nodes[:, 1]
        self.boundary_node_indices = np.unique(self.boundary_edges[:, :2].flatten())
        print(f"  Quad domain [{x_min},{x_max}]x[{y_min},{y_max}]  "
              f"{self.n_nodes} nodes  {self.n_elements} triangles")

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
                tris.append([a, b, c])
                tris.append([a, c, d])
        elements = np.array(tris, dtype=int)
        bedges = []
        for i in range(nx): bedges.append([nid(i,0), nid(i+1,0)])
        for j in range(ny): bedges.append([nid(nx,j), nid(nx,j+1)])
        for i in range(nx-1,-1,-1): bedges.append([nid(i+1,ny), nid(i,ny)])
        for j in range(ny-1,-1,-1): bedges.append([nid(0,j+1), nid(0,j)])
        return nodes, elements, np.array(bedges, dtype=int)

    def sample_points(self, n):
        if n <= self.n_nodes:
            idx = np.random.choice(self.n_nodes, n, replace=False)
            return torch.FloatTensor(self.nodes[idx])
        all_n = torch.FloatTensor(self.nodes)
        xr = torch.rand(n - self.n_nodes) * (self.x_max - self.x_min) + self.x_min
        yr = torch.rand(n - self.n_nodes) * (self.y_max - self.y_min) + self.y_min
        return torch.cat([all_n, torch.stack([xr, yr], dim=1)], dim=0)

    def sample_boundary(self, n):
        bc = self.nodes[self.boundary_node_indices]
        idx = np.random.choice(len(bc), n, replace=(n > len(bc)))
        return torch.FloatTensor(bc[idx])


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class PINN_Net(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        act_map = {
            'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'gelu': nn.GELU(),
            'silu': nn.SiLU(), 'sigmoid': nn.Sigmoid(),
        }
        self.activation = act_map.get(activation, nn.Tanh())
        self.linear_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.linear_layers.append(lin)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        for layer in self.linear_layers[:-1]:
            inp = self.activation(layer(inp))
        return self.linear_layers[-1](inp)


# =============================================================================
# PINN SOLVER (streamlined for ablation)
# =============================================================================

class AblationPINN:
    """Streamlined PINN for ablation runs. Saves best model."""

    def __init__(self, layers, params, geometry, device,
                 lambda_pde=1.0, lambda_ic=50.0, lambda_bc=1.0,
                 lr=1e-3, activation='tanh'):
        self.device = device
        self.geometry = geometry
        self.params = params
        self.layers = layers
        self.activation_name = activation
        self.model = PINN_Net(layers, activation).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500)
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # ── collocation data ──────────────────────────────────────────
    def generate_data(self, n_pde=10000, n_ic=600, n_bc=400,
                      t_domain=[0, 30]):
        t_min, t_max = t_domain
        self.xy_pde = self.geometry.sample_points(n_pde).to(self.device)
        self.t_pde = (torch.rand(n_pde, 1) * (t_max - t_min) + t_min).to(self.device)
        self.xy_ic = self.geometry.sample_points(n_ic).to(self.device)
        self.t_ic = torch.zeros(n_ic, 1).to(self.device)
        self.xy_bc = self.geometry.sample_boundary(n_bc).to(self.device)
        self.t_bc = (torch.rand(n_bc, 1) * (t_max - t_min) + t_min).to(self.device)
        # validation set (20%)
        nv_pde = max(int(n_pde * 0.2), 100)
        nv_ic = max(int(n_ic * 0.2), 20)
        nv_bc = max(int(n_bc * 0.2), 20)
        self.xy_pde_val = self.geometry.sample_points(nv_pde).to(self.device)
        self.t_pde_val = (torch.rand(nv_pde, 1) * (t_max - t_min) + t_min).to(self.device)
        self.xy_ic_val = self.geometry.sample_points(nv_ic).to(self.device)
        self.t_ic_val = torch.zeros(nv_ic, 1).to(self.device)
        self.xy_bc_val = self.geometry.sample_boundary(nv_bc).to(self.device)
        self.t_bc_val = (torch.rand(nv_bc, 1) * (t_max - t_min) + t_min).to(self.device)

    # ── PDE residuals ─────────────────────────────────────────────
    def _residuals(self, xy, t_in):
        x = xy[:, 0:1].clone().detach().requires_grad_(True)
        y = xy[:, 1:2].clone().detach().requires_grad_(True)
        t = t_in.clone().detach().requires_grad_(True)

        out = self.model(x, y, t)
        T, B, O, I, S = [out[:, i:i + 1] for i in range(5)]

        def g1(u, v):
            return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]

        def lap(u):
            ux = g1(u, x); uy = g1(u, y)
            return g1(ux, x) + g1(uy, y), ux, uy

        T_l, Tx, Ty = lap(T); B_l, Bx, By = lap(B)
        O_l, Ox, Oy = lap(O); I_l, Ix, Iy = lap(I); S_l, Sx, Sy = lap(S)
        Tt = g1(T, t); Bt = g1(B, t); Ot = g1(O, t)
        It = g1(I, t); St = g1(S, t)

        p = self.params
        fT = (Tt - p['D_T'] * T_l - p['rho_T'] * T * (1 - T / p['theta'])
              + p['delta_T'] * T + p['alpha_S'] * S * T)
        fB = (Bt - p['D_B'] * B_l
              - p['rho_B'] * B * (p['K_H'] / (p['K_H'] + O + 1e-8))
              + p['delta_B'] * B + p['beta_I'] * I * B)
        fO = (Ot - p['D_O'] * O_l + p['gamma_T'] * T * O
              - p['gamma_E'] * (p['O_ext'] - O))
        fI = It - p['D_I'] * I_l - p['beta_T'] * T + p['delta_I'] * I
        fS = St - p['D_S'] * S_l - p['beta_B_signal'] * B + p['delta_S'] * S

        grads = [torch.cat([Tx, Ty], 1), torch.cat([Bx, By], 1),
                 torch.cat([Ox, Oy], 1), torch.cat([Ix, Iy], 1),
                 torch.cat([Sx, Sy], 1)]
        return fT, fB, fO, fI, fS, grads

    # ── initial conditions ────────────────────────────────────────
    def _ic(self, xy):
        x = xy[:, 0:1]; y = xy[:, 1:2]
        g = self.geometry; p = self.params
        cx = (g.x_max + g.x_min) / 2; cy = (g.y_max + g.y_min) / 2
        ds = max(g.x_max - g.x_min, g.y_max - g.y_min)
        tx = p.get('tumor_x', cx); ty = p.get('tumor_y', cy)
        ts = p.get('tumor_size', ds / 5)
        T0 = torch.exp(-((x - tx) ** 2 + (y - ty) ** 2) / (2 * ts ** 2))
        bx = p.get('bacteria_x', g.x_min + 0.3 * (g.x_max - g.x_min))
        by = p.get('bacteria_y', g.y_min + 0.6 * (g.y_max - g.y_min))
        bs = p.get('bacteria_spread', ds / 8)
        B0 = 0.3 * torch.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * bs ** 2))
        O0 = p['O_ext'] * torch.ones_like(x)
        I0 = 0.01 * torch.ones_like(x)
        S0 = torch.zeros_like(x)
        return T0, B0, O0, I0, S0

    # ── loss ──────────────────────────────────────────────────────
    def _compute_loss(self, xy_p, t_p, xy_i, t_i, xy_b, t_b):
        fT, fB, fO, fI, fS, _ = self._residuals(xy_p, t_p)
        loss_pde = sum(torch.mean(f ** 2) for f in [fT, fB, fO, fI, fS])

        out_ic = self.model(xy_i[:, 0:1], xy_i[:, 1:2], t_i)
        T0, B0, O0, I0, S0 = self._ic(xy_i)
        loss_ic = sum(torch.mean((out_ic[:, j:j + 1] - ref) ** 2)
                      for j, ref in enumerate([T0, B0, O0, I0, S0]))

        _, _, _, _, _, grads = self._residuals(xy_b, t_b)
        loss_bc = sum(torch.mean(g ** 2) for g in grads)

        total = (self.lambda_pde * loss_pde
                 + self.lambda_ic * loss_ic
                 + self.lambda_bc * loss_bc)
        return total, loss_pde.item(), loss_ic.item(), loss_bc.item()

    # ── train ─────────────────────────────────────────────────────
    def train(self, n_epochs=6000, patience=1500, print_every=20):
        best_val = float('inf')
        best_state = None
        best_epoch = 0
        pat_counter = 0

        history = {k: [] for k in ['train_total', 'val_total',
                                    'train_pde', 'train_ic', 'train_bc',
                                    'val_pde', 'val_ic', 'val_bc']}

        for ep in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            loss, lp, li, lb = self._compute_loss(
                self.xy_pde, self.t_pde, self.xy_ic, self.t_ic,
                self.xy_bc, self.t_bc)
            loss.backward()
            self.optimizer.step()

            vloss, vp, vi, vb = self._compute_loss(
                self.xy_pde_val, self.t_pde_val,
                self.xy_ic_val, self.t_ic_val,
                self.xy_bc_val, self.t_bc_val)
            vl = vloss.item()
            self.scheduler.step(vl)

            history['train_total'].append(loss.item())
            history['val_total'].append(vl)
            history['train_pde'].append(lp); history['train_ic'].append(li)
            history['train_bc'].append(lb)
            history['val_pde'].append(vp); history['val_ic'].append(vi)
            history['val_bc'].append(vb)

            if vl < best_val:
                best_val = vl
                best_epoch = ep + 1
                best_state = copy.deepcopy(self.model.state_dict())
                pat_counter = 0
            else:
                pat_counter += 1

            if (ep + 1) % print_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"    Ep {ep+1:>5}/{n_epochs}  Train {loss.item():.3e}  "
                      f"Val {vl:.3e}  Best {best_val:.3e}@{best_epoch}  "
                      f"LR {lr:.1e}  Pat {pat_counter}/{patience}")

            if pat_counter >= patience:
                print(f"    Early stop @ epoch {ep+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_val, best_epoch, history

    # ── save / load model ─────────────────────────────────────────
    def save_model(self, path):
        """Save full checkpoint: model + config to recreate it."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'layers': self.layers,
            'activation': self.activation_name,
            'params': self.params,
            'lambda_pde': self.lambda_pde,
            'lambda_ic': self.lambda_ic,
            'lambda_bc': self.lambda_bc,
            'n_params': self.n_params,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_model(path, geometry, device):
        """Recreate a full AblationPINN from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        pinn = AblationPINN(
            layers=checkpoint['layers'],
            params=checkpoint['params'],
            geometry=geometry,
            device=device,
            lambda_pde=checkpoint['lambda_pde'],
            lambda_ic=checkpoint['lambda_ic'],
            lambda_bc=checkpoint['lambda_bc'],
            activation=checkpoint['activation'])
        pinn.model.load_state_dict(checkpoint['model_state_dict'])
        pinn.model.eval()
        print(f"  Loaded model from {path}  ({checkpoint['n_params']} params)")
        return pinn

    # ── evaluate terminal state ───────────────────────────────────
    def evaluate_terminal(self, t_final=30.0):
        self.model.eval()
        xm = torch.FloatTensor(self.geometry.x).unsqueeze(1).to(self.device)
        ym = torch.FloatTensor(self.geometry.y).unsqueeze(1).to(self.device)
        tm = torch.full_like(xm, t_final)
        with torch.no_grad():
            out = self.model(xm, ym, tm).cpu().numpy()
        means = out.mean(axis=0)
        maxs = out.max(axis=0)
        return {
            'T_mean': float(means[0]), 'B_mean': float(means[1]),
            'O_mean': float(means[2]), 'I_mean': float(means[3]),
            'S_mean': float(means[4]),
            'T_max': float(maxs[0]), 'B_max': float(maxs[1]),
            'O_max': float(maxs[2]), 'I_max': float(maxs[3]),
            'S_max': float(maxs[4]),
        }

    def evaluate_at_probe(self, x_probe=3.0, y_probe=3.0, times=None):
        if times is None:
            times = np.linspace(0, 30, 61)
        self.model.eval()
        results = []
        for t_val in times:
            xp = torch.FloatTensor([[x_probe]]).to(self.device)
            yp = torch.FloatTensor([[y_probe]]).to(self.device)
            tp = torch.FloatTensor([[t_val]]).to(self.device)
            with torch.no_grad():
                out = self.model(xp, yp, tp).cpu().numpy()[0]
            results.append([t_val] + list(out))
        return np.array(results)

    def predict_field(self, t_val):
        """Return full spatial field at time t_val. Shape (n_nodes, 5)."""
        self.model.eval()
        xm = torch.FloatTensor(self.geometry.x).unsqueeze(1).to(self.device)
        ym = torch.FloatTensor(self.geometry.y).unsqueeze(1).to(self.device)
        tm = torch.full_like(xm, t_val)
        with torch.no_grad():
            return self.model(xm, ym, tm).cpu().numpy()


# =============================================================================
# BASELINE PARAMETERS
# =============================================================================

def get_baseline_params(geo):
    return {
        'D_T': 0.01, 'D_B': 0.1, 'D_O': 1.0, 'D_I': 0.5, 'D_S': 0.3,
        'rho_T': 0.3, 'theta': 1.0, 'delta_T': 0.05, 'alpha_S': 0.2,
        'rho_B': 0.5, 'K_H': 0.1, 'delta_B': 0.1, 'beta_I': 0.3,
        'gamma_T': 0.2, 'gamma_E': 0.5, 'O_ext': 0.2,
        'beta_T': 0.1, 'delta_I': 0.2,
        'beta_B_signal': 0.4, 'delta_S': 0.3,
        'tumor_x': (geo.x_max + geo.x_min) / 2,
        'tumor_y': (geo.y_max + geo.y_min) / 2,
        'tumor_size': (geo.x_max - geo.x_min) / 5,
        'bacteria_x': geo.x_min + 0.3 * (geo.x_max - geo.x_min),
        'bacteria_y': geo.y_min + 0.6 * (geo.y_max - geo.y_min),
        'bacteria_spread': (geo.x_max - geo.x_min) / 8,
    }

BASELINE_LAYERS = [3, 64, 64, 64, 64, 5]
BASELINE_COLLOC = {'n_pde': 10000, 'n_ic': 600, 'n_bc': 400}
BASELINE_LAMBDAS = {'lambda_pde': 1.0, 'lambda_ic': 50.0, 'lambda_bc': 1.0}


# =============================================================================
# STUDY DEFINITIONS
# =============================================================================

def define_studies():
    studies = {}

    studies['S1_alpha_S'] = {
        'title': 'S1: Signal Cytotoxicity alpha_S',
        'param_name': 'alpha_S',
        'param_symbol': r'$\alpha_S$',
        'experiments': [
            ('0.01', {'alpha_S': 0.01}),
            ('0.05', {'alpha_S': 0.05}),
            ('0.10', {'alpha_S': 0.10}),
            ('0.20 (baseline)', {'alpha_S': 0.20}),
            ('0.50', {'alpha_S': 0.50}),
            ('1.00', {'alpha_S': 1.00}),
        ]
    }

    studies['S2_gamma_vas'] = {
        'title': 'S2: Vascularization gamma_vas',
        'param_name': 'gamma_E',
        'param_symbol': r'$\gamma_{\mathrm{vas}}$',
        'experiments': [
            ('0.05', {'gamma_E': 0.05}),
            ('0.10', {'gamma_E': 0.10}),
            ('0.50 (baseline)', {'gamma_E': 0.50}),
            ('1.00', {'gamma_E': 1.00}),
            ('2.00', {'gamma_E': 2.00}),
        ]
    }

    studies['S3_beta_I'] = {
        'title': 'S3: Immune Clearance beta_I',
        'param_name': 'beta_I',
        'param_symbol': r'$\beta_I$',
        'experiments': [
            ('0.05', {'beta_I': 0.05}),
            ('0.10', {'beta_I': 0.10}),
            ('0.30 (baseline)', {'beta_I': 0.30}),
            ('0.50', {'beta_I': 0.50}),
            ('1.00', {'beta_I': 1.00}),
        ]
    }

    studies['S4_D_S'] = {
        'title': 'S4: Signal Diffusivity D_S',
        'param_name': 'D_S',
        'param_symbol': r'$D_S$',
        'experiments': [
            ('0.05', {'D_S': 0.05}),
            ('0.10', {'D_S': 0.10}),
            ('0.30 (baseline)', {'D_S': 0.30}),
            ('0.50', {'D_S': 0.50}),
            ('1.00', {'D_S': 1.00}),
        ]
    }

    studies['S5_K_H'] = {
        'title': 'S5: Hypoxia Sensitivity K_H',
        'param_name': 'K_H',
        'param_symbol': r'$K_H$',
        'experiments': [
            ('0.01', {'K_H': 0.01}),
            ('0.05', {'K_H': 0.05}),
            ('0.10 (baseline)', {'K_H': 0.10}),
            ('0.50', {'K_H': 0.50}),
            ('1.00', {'K_H': 1.00}),
        ]
    }

    studies['S6_rho_T'] = {
        'title': 'S6: Tumor Growth Rate rho_T',
        'param_name': 'rho_T',
        'param_symbol': r'$\rho_T$',
        'experiments': [
            ('0.10', {'rho_T': 0.10}),
            ('0.20', {'rho_T': 0.20}),
            ('0.30 (baseline)', {'rho_T': 0.30}),
            ('0.50', {'rho_T': 0.50}),
            ('0.80', {'rho_T': 0.80}),
        ]
    }

    studies['S7_architecture'] = {
        'title': 'S7: Network Architecture',
        'param_name': 'architecture',
        'param_symbol': 'Architecture',
        'experiments': [
            ('2x32', {'_layers': [3, 32, 32, 5]}),
            ('2x128', {'_layers': [3, 128, 128, 5]}),
            ('4x32', {'_layers': [3, 32, 32, 32, 32, 5]}),
            ('4x64 (baseline)', {'_layers': [3, 64, 64, 64, 64, 5]}),
            ('4x128', {'_layers': [3, 128, 128, 128, 128, 5]}),
            ('6x64', {'_layers': [3, 64, 64, 64, 64, 64, 64, 5]}),
        ]
    }

    studies['S8_collocation'] = {
        'title': 'S8: Number of Collocation Points',
        'param_name': 'N_pde',
        'param_symbol': r'$N_{\mathrm{pde}}$',
        'experiments': [
            ('2k/200/100', {'_colloc': {'n_pde': 2000, 'n_ic': 200, 'n_bc': 100}}),
            ('5k/300/200', {'_colloc': {'n_pde': 5000, 'n_ic': 300, 'n_bc': 200}}),
            ('10k/600/400 (baseline)', {'_colloc': {'n_pde': 10000, 'n_ic': 600, 'n_bc': 400}}),
            ('20k/1.2k/800', {'_colloc': {'n_pde': 20000, 'n_ic': 1200, 'n_bc': 800}}),
            ('40k/2.4k/1.6k', {'_colloc': {'n_pde': 40000, 'n_ic': 2400, 'n_bc': 1600}}),
        ]
    }

    studies['S9_loss_weights'] = {
        'title': 'S9: Loss Weights (lambda_pde, lambda_ic, lambda_bc)',
        'param_name': 'lambda_ic',
        'param_symbol': r'$\lambda_{\mathrm{ic}}$',
        'experiments': [
            ('1/1/1 (equal)', {'_lambdas': {'lambda_pde': 1., 'lambda_ic': 1., 'lambda_bc': 1.}}),
            ('1/10/1', {'_lambdas': {'lambda_pde': 1., 'lambda_ic': 10., 'lambda_bc': 1.}}),
            ('1/50/1 (baseline)', {'_lambdas': {'lambda_pde': 1., 'lambda_ic': 50., 'lambda_bc': 1.}}),
            ('1/100/1', {'_lambdas': {'lambda_pde': 1., 'lambda_ic': 100., 'lambda_bc': 1.}}),
            ('1/50/10', {'_lambdas': {'lambda_pde': 1., 'lambda_ic': 50., 'lambda_bc': 10.}}),
            ('10/50/1', {'_lambdas': {'lambda_pde': 10., 'lambda_ic': 50., 'lambda_bc': 1.}}),
        ]
    }

    return studies


# =============================================================================
# RUN SINGLE EXPERIMENT
# =============================================================================

def run_experiment(geo, base_params, layers, colloc, lambdas, activation,
                   n_epochs, patience, overrides, save_path=None, verbose=True):
    """
    Run one PINN training.
    If save_path is provided, the best model is saved there.
    Returns: result dict, history dict, probe data array, pinn object.
    """
    params = copy.deepcopy(base_params)
    my_layers = list(layers)
    my_colloc = dict(colloc)
    my_lambdas = dict(lambdas)
    my_activation = activation

    for k, v in overrides.items():
        if k == '_layers':
            my_layers = v
        elif k == '_colloc':
            my_colloc.update(v)
        elif k == '_lambdas':
            my_lambdas.update(v)
        elif k == '_activation':
            my_activation = v
        else:
            params[k] = v

    pinn = AblationPINN(
        my_layers, params, geo, device,
        lambda_pde=my_lambdas['lambda_pde'],
        lambda_ic=my_lambdas['lambda_ic'],
        lambda_bc=my_lambdas['lambda_bc'],
        activation=my_activation)

    pinn.generate_data(**my_colloc)

    t0 = datetime.now()
    pe = max(n_epochs // 4, 1) if verbose else n_epochs + 1
    best_val, best_ep, history = pinn.train(
        n_epochs=n_epochs, patience=patience, print_every=pe)
    train_time = (datetime.now() - t0).total_seconds()

    # ── save best model ──
    if save_path is not None:
        pinn.save_model(save_path)
        print(f"    Model saved → {save_path}")

    terminal = pinn.evaluate_terminal(t_final=30.0)
    probe_data = pinn.evaluate_at_probe(x_probe=3.0, y_probe=3.0)

    result = {
        'best_val_loss': best_val,
        'best_epoch': best_ep,
        'train_time_s': train_time,
        'n_params': pinn.n_params,
        'model_path': save_path if save_path else '',
        **terminal,
        'final_train_pde': history['train_pde'][-1] if history['train_pde'] else None,
        'final_train_ic': history['train_ic'][-1] if history['train_ic'] else None,
        'final_train_bc': history['train_bc'][-1] if history['train_bc'] else None,
    }
    return result, history, probe_data, pinn


# =============================================================================
# RUN A FULL STUDY
# =============================================================================

def run_study(study_name, study_def, geo, base_params, n_epochs, patience,
              output_dir='ablation_results'):
    os.makedirs(output_dir, exist_ok=True)
    study_dir = os.path.join(output_dir, study_name)
    models_dir = os.path.join(study_dir, 'models')
    os.makedirs(study_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  STUDY: {study_def['title']}")
    print(f"  {len(study_def['experiments'])} experiments x {n_epochs} epochs")
    print(f"  Models dir: {models_dir}")
    print(f"{'=' * 70}")

    all_results = []
    all_probes = {}

    for i, (label, overrides) in enumerate(study_def['experiments']):
        print(f"\n  [{i + 1}/{len(study_def['experiments'])}] {label}")
        print(f"  {'─' * 50}")

        torch.manual_seed(42)
        np.random.seed(42)

        # Clean label for filename
        safe_label = label.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
        model_path = os.path.join(models_dir, f'{study_name}_{safe_label}.pt')

        result, history, probe_data, pinn = run_experiment(
            geo, base_params,
            layers=BASELINE_LAYERS, colloc=BASELINE_COLLOC,
            lambdas=BASELINE_LAMBDAS, activation='tanh',
            n_epochs=n_epochs, patience=patience,
            overrides=overrides, save_path=model_path)

        result['label'] = label
        result['overrides'] = str(overrides)
        all_results.append(result)
        all_probes[label] = probe_data

        print(f"  => Val: {result['best_val_loss']:.3e}  "
              f"T*={result['T_mean']:.4f}  B*={result['B_mean']:.4f}  "
              f"O*={result['O_mean']:.4f}  S*={result['S_mean']:.4f}  "
              f"({result['train_time_s']:.0f}s)")

    # ── save CSV ──
    csv_path = os.path.join(study_dir, f'{study_name}_results.csv')
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  CSV → {csv_path}")

    # ── plots ──
    _plot_terminal_vs_param(study_def, all_results, study_dir, study_name)
    _plot_probe_timeseries(study_def, all_probes, study_dir, study_name)
    _plot_val_loss_bar(study_def, all_results, study_dir, study_name)
    _plot_all_terminal(study_def, all_results, study_dir, study_name)

    # ── print model inventory ──
    print(f"\n  Saved models:")
    for r in all_results:
        mp = r.get('model_path', '')
        if mp and os.path.exists(mp):
            size_mb = os.path.getsize(mp) / 1e6
            print(f"    {r['label']:30s}  {size_mb:.2f} MB  → {mp}")

    return all_results


# =============================================================================
# LOAD UTILITY — reload any experiment's model
# =============================================================================

def load_experiment(study_name, label, geo, output_dir='ablation_results'):
    """
    Reload a saved model from a completed ablation experiment.

    Usage:
        pinn = load_experiment('S1_alpha_S', '0.50', geo)
        field = pinn.predict_field(t_val=15.0)
        probe = pinn.evaluate_at_probe(x_probe=3.0, y_probe=3.0)
    """
    safe_label = label.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
    model_path = os.path.join(output_dir, study_name, 'models',
                              f'{study_name}_{safe_label}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return AblationPINN.load_model(model_path, geo, device)


# =============================================================================
# PLOTTING
# =============================================================================

def _plot_terminal_vs_param(study_def, results, study_dir, study_name):
    labels = [r['label'] for r in results]
    T_vals = [r['T_mean'] for r in results]
    B_vals = [r['B_mean'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#d32f2f' if 'baseline' in l else '#1976d2' for l in labels]
    ax1.bar(range(len(labels)), T_vals, color=colors, alpha=0.85, edgecolor='k', lw=0.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Mean T(t=30)', fontsize=12)
    ax1.set_title(f'{study_def["title"]}\nTerminal Tumor Density', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    colors2 = ['#1565c0' if 'baseline' in l else '#42a5f5' for l in labels]
    ax2.bar(range(len(labels)), B_vals, color=colors2, alpha=0.85, edgecolor='k', lw=0.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Mean B(t=30)', fontsize=12)
    ax2.set_title(f'{study_def["title"]}\nTerminal Bacteria Density', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(study_dir, f'{study_name}_terminal.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {path}")


def _plot_probe_timeseries(study_def, all_probes, study_dir, study_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    var_names = ['Tumor T', 'Bacteria B', 'Signal S']
    var_idx = [1, 2, 5]

    cmap = plt.cm.viridis
    n_exp = len(all_probes)

    for ax, vname, vidx in zip(axes, var_names, var_idx):
        for i, (label, data) in enumerate(all_probes.items()):
            color = cmap(i / max(n_exp - 1, 1))
            lw = 2.5 if 'baseline' in label else 1.5
            ls = '-' if 'baseline' in label else '--'
            ax.plot(data[:, 0], data[:, vidx], color=color, lw=lw, ls=ls,
                    label=label, alpha=0.9)
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel(vname, fontsize=11)
        ax.set_title(f'{vname} at probe (3,3)', fontsize=11)
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)

    fig.suptitle(study_def['title'], fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(study_dir, f'{study_name}_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {path}")


def _plot_val_loss_bar(study_def, results, study_dir, study_name):
    labels = [r['label'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    pde_losses = [r['final_train_pde'] or 0 for r in results]
    ic_losses = [r['final_train_ic'] or 0 for r in results]
    bc_losses = [r['final_train_bc'] or 0 for r in results]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5 * width, val_losses, width, label='Val Total', color='#d32f2f', alpha=0.8)
    ax.bar(x - 0.5 * width, pde_losses, width, label='PDE', color='#1976d2', alpha=0.8)
    ax.bar(x + 0.5 * width, ic_losses, width, label='IC', color='#388e3c', alpha=0.8)
    ax.bar(x + 1.5 * width, bc_losses, width, label='BC', color='#f57c00', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_yscale('log')
    ax.set_title(f'{study_def["title"]} — Loss Comparison', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(study_dir, f'{study_name}_losses.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {path}")


def _plot_all_terminal(study_def, results, study_dir, study_name):
    labels = [r['label'] for r in results]
    vars_ = ['T_mean', 'B_mean', 'O_mean', 'I_mean', 'S_mean']
    var_labels = ['T*', 'B*', 'O*', 'I*', 'S*']
    colors = ['#d32f2f', '#1976d2', '#388e3c', '#f57c00', '#7b1fa2']

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, (var, vlbl, col) in enumerate(zip(vars_, var_labels, colors)):
        vals = [r[var] for r in results]
        ax.bar(x + (j - 2) * width, vals, width, label=vlbl, color=col, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Domain-averaged value at t=30', fontsize=11)
    ax.set_title(f'{study_def["title"]} — Terminal State', fontsize=12)
    ax.legend(ncol=5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(study_dir, f'{study_name}_all_terminal.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {path}")


# =============================================================================
# LATEX TABLE GENERATOR
# =============================================================================

def generate_latex_table(study_name, study_def, results, output_dir):
    study_dir = os.path.join(output_dir, study_name)
    os.makedirs(study_dir, exist_ok=True)
    path = os.path.join(study_dir, f'{study_name}_table.tex')

    with open(path, 'w') as f:
        f.write(f"% Auto-generated ablation table: {study_name}\n")
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Sensitivity study: " + study_def['title'] + "}\n")
        f.write(r"\label{tab:" + study_name + "}\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{l c c c c c c c c}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(
            study_def['param_symbol']
            + r" & $\mathcal{L}_{\mathrm{val}}^{\mathrm{best}}$"
            + r" & $\bar{T}^*$ & $\bar{B}^*$ & $\bar{O}^*$"
            + r" & $\bar{I}^*$ & $\bar{S}^*$ & Epoch & Time (s) \\" + "\n")
        f.write(r"\hline" + "\n")

        for r in results:
            label_tex = r['label'].replace('(baseline)', r'\textbf{(base.)}')
            f.write(
                f"  {label_tex} & "
                f"{r['best_val_loss']:.2e} & "
                f"{r['T_mean']:.4f} & "
                f"{r['B_mean']:.4f} & "
                f"{r['O_mean']:.4f} & "
                f"{r['I_mean']:.4f} & "
                f"{r['S_mean']:.4f} & "
                f"{r['best_epoch']} & "
                f"{r['train_time_s']:.0f} \\\\\n")

        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"  LaTeX → {path}")


# =============================================================================
#
# EDIT THE SETTINGS BELOW, THEN RUN THIS CELL
#
# =============================================================================

# ─── Which studies to run ──────────────────────────────────────────
# Options: 'S1_alpha_S', 'S2_gamma_vas', 'S3_beta_I', 'S4_D_S',
#          'S5_K_H', 'S6_rho_T', 'S7_architecture', 'S8_collocation',
#          'S9_loss_weights', or 'all'
STUDIES_TO_RUN = ['all']

# ─── Training settings ─────────────────────────────────────────────
N_EPOCHS = 6000       # max epochs per experiment
PATIENCE = 1500       # early stopping patience
NX = 60               # mesh resolution (60 → 3721 nodes)
OUTPUT_DIR = 'ablation_results'

# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':

    geo = QuadGeometry(x_min=0.0, x_max=6.0, y_min=0.0, y_max=6.0, nx=NX, ny=NX)
    base_params = get_baseline_params(geo)
    all_studies = define_studies()

    if 'all' in STUDIES_TO_RUN:
        studies_to_run = list(all_studies.keys())
    else:
        studies_to_run = STUDIES_TO_RUN

    print(f"\n{'#' * 70}")
    print(f"  ABLATION STUDY SUITE")
    print(f"  Studies : {studies_to_run}")
    print(f"  Epochs  : {N_EPOCHS}")
    print(f"  Patience: {PATIENCE}")
    print(f"  Mesh    : {NX}x{NX} = {geo.n_nodes} nodes")
    print(f"  Device  : {device}")
    print(f"{'#' * 70}")

    global_start = datetime.now()
    all_study_results = {}

    for study_name in studies_to_run:
        if study_name not in all_studies:
            print(f"\n  WARNING: Unknown study '{study_name}' — skipping.")
            continue

        study_def = all_studies[study_name]
        results = run_study(
            study_name, study_def, geo, base_params,
            n_epochs=N_EPOCHS, patience=PATIENCE, output_dir=OUTPUT_DIR)
        all_study_results[study_name] = results
        generate_latex_table(study_name, study_def, results, OUTPUT_DIR)

    # ── Global summary ──
    total_time = (datetime.now() - global_start).total_seconds() / 60

    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    summary = {}
    for sname, results in all_study_results.items():
        summary[sname] = [{k: v for k, v in r.items() if k != 'overrides'}
                          for r in results]
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — {total_time:.1f} min total")
    print(f"  Results : {OUTPUT_DIR}/")
    print(f"  Summary : {summary_path}")
    print(f"{'#' * 70}")
