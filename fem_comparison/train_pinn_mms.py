"""
=============================================================================
PINN vs FEM — MANUFACTURED SOLUTION (Biological Scale)  
=============================================================================

Manufactured solution:
  φ(x,y) = cos(πx)cos(πy),   Δφ = -2π² φ
  T_ex = (0.4 + 0.2φ) e^{-0.3t}        ∈ [0.2, 0.6]
  B_ex = (0.08 + 0.03φ)(1 - e^{-0.5t})  ∈ [0, 0.11]
  O_ex = 0.2 + 0.02φ e^{-t}             ∈ [0.18, 0.22]
  I_ex = (0.02 + 0.008φ)(1 - e^{-0.3t}) ∈ [0, 0.028]
  S_ex = (0.05 + 0.02φ)(1 - e^{-0.5t})  ∈ [0, 0.07]
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, copy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# MANUFACTURED SOLUTION (Biological Scale)
# =============================================================================

class BioMMS:
    """Manufactured solution with values in [0,1] — biological regime."""

    def __init__(self, L=1.0):
        self.L = L
        self.k = 2.0 * np.pi**2 / L**2  # = 2π²  (since Δφ = -2π² φ for L=1)

    def phi_t(self, x, y):
        return torch.cos(np.pi * x / self.L) * torch.cos(np.pi * y / self.L)

    def phi_n(self, x, y):
        return np.cos(np.pi * x / self.L) * np.cos(np.pi * y / self.L)

    def exact_t(self, x, y, t):
        """Exact solution (torch tensors, for training)."""
        p = self.phi_t(x, y)
        T = (0.4 + 0.2 * p) * torch.exp(-0.3 * t)
        B = (0.08 + 0.03 * p) * (1.0 - torch.exp(-0.5 * t))
        O = 0.2 + 0.02 * p * torch.exp(-t)
        I = (0.02 + 0.008 * p) * (1.0 - torch.exp(-0.3 * t))
        S = (0.05 + 0.02 * p) * (1.0 - torch.exp(-0.5 * t))
        return T, B, O, I, S

    def exact_n(self, x, y, t):
        """Exact solution (numpy, for evaluation)."""
        p = self.phi_n(x, y)
        T = (0.4 + 0.2 * p) * np.exp(-0.3 * t)
        B = (0.08 + 0.03 * p) * (1.0 - np.exp(-0.5 * t))
        O = 0.2 + 0.02 * p * np.exp(-t)
        I = (0.02 + 0.008 * p) * (1.0 - np.exp(-0.3 * t))
        S = (0.05 + 0.02 * p) * (1.0 - np.exp(-0.5 * t))
        return T, B, O, I, S

    def ic_t(self, x, y):
        """Initial conditions at t=0 (torch). Same as exact_t at t=0."""
        p = self.phi_t(x, y)
        T0 = 0.4 + 0.2 * p           # T(0)
        B0 = torch.zeros_like(p)     # B(0) = 0
        O0 = 0.2 + 0.02 * p          # O(0)
        I0 = torch.zeros_like(p)     # I(0) = 0
        S0 = torch.zeros_like(p)     # S(0) = 0
        return torch.cat([T0, B0, O0, I0, S0], dim=1)

    def source_terms(self, x, y, t, pr):
        """Source terms so that exact solution satisfies the modified PDE."""
        p = self.phi_t(x, y)
        k = self.k
        Te, Be, Oe, Ie, Se = self.exact_t(x, y, t)

        # Time derivatives ∂_t u_ex
        dTdt = -0.3 * (0.4 + 0.2 * p) * torch.exp(-0.3 * t)
        dBdt =  0.5 * (0.08 + 0.03 * p) * torch.exp(-0.5 * t)
        dOdt = -0.02 * p * torch.exp(-t)
        dIdt =  0.3 * (0.02 + 0.008 * p) * torch.exp(-0.3 * t)
        dSdt =  0.5 * (0.05 + 0.02 * p) * torch.exp(-0.5 * t)

        # Laplacians: Δ(a + b·φ) = -b·k·φ  with k = 2π²
        lapT = -0.2   * k * p * torch.exp(-0.3 * t)
        lapB = -0.03  * k * p * (1.0 - torch.exp(-0.5 * t))
        lapO = -0.02  * k * p * torch.exp(-t)
        lapI = -0.008 * k * p * (1.0 - torch.exp(-0.3 * t))
        lapS = -0.02  * k * p * (1.0 - torch.exp(-0.5 * t))

        # f = ∂_t u_ex - D · Δu_ex - reaction(u_ex)
        fT = (dTdt - pr['D_T'] * lapT
              - pr['rho_T'] * Te * (1 - Te / pr['theta'])
              + pr['delta_T'] * Te + pr['alpha_S'] * Se * Te)

        fB = (dBdt - pr['D_B'] * lapB
              - pr['rho_B'] * Be * (pr['K_H'] / (pr['K_H'] + Oe + 1e-8))
              + pr['delta_B'] * Be + pr['beta_I'] * Ie * Be)

        fO = (dOdt - pr['D_O'] * lapO
              + pr['gamma_T'] * Te * Oe
              - pr['gamma_E'] * (pr['O_ext'] - Oe))

        fI = (dIdt - pr['D_I'] * lapI
              - pr['beta_T'] * Te + pr['delta_I'] * Ie)

        fS = (dSdt - pr['D_S'] * lapS
              - pr['beta_B_signal'] * Be + pr['delta_S'] * Se)

        return fT, fB, fO, fI, fS


# =============================================================================
# NETWORK WITH HARD IC
# =============================================================================

class PINN_HardIC(nn.Module):
    """u(t,x,y) = IC(x,y) + t · NN(x,y,t)  — exact IC by construction."""

    def __init__(self, layers, mms):
        super().__init__()
        self.act = nn.Tanh()
        self.lins = nn.ModuleList()
        for i in range(len(layers) - 1):
            l = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.lins.append(l)
        self.mms = mms

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        for l in self.lins[:-1]:
            inp = self.act(l(inp))
        nn_out = self.lins[-1](inp)
        return self.mms.ic_t(x, y) + t * nn_out


# =============================================================================
# PINN SOLVER
# =============================================================================

class BioPINN:
    # Per-variable scales used to normalize PDE residuals (see note 3 above).
    # Chosen as the typical magnitude of each field over [0,1]×Ω×[0,1].
    VAR_SCALES = {'T': 0.4, 'B': 0.08, 'O': 0.2, 'I': 0.02, 'S': 0.05}

    def __init__(self, layers, params, device, L=1.0, tau=1.0):
        self.device = device
        self.params = params
        self.L = L
        self.tau = tau
        self.mms = BioMMS(L=L)
        self.model = PINN_HardIC(layers, self.mms).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # squared scales as a tensor in variable order [T,B,O,I,S]
        self._scale_sq = torch.tensor(
            [self.VAR_SCALES[v] ** 2 for v in ['T', 'B', 'O', 'I', 'S']],
            device=device)
        print(f"  PINN: {layers}, {self.n_params} params")

    # -------------------------------------------------------------------------
    # DATA GENERATION
    # -------------------------------------------------------------------------
    def generate_data(self, n_pde=15000, n_bc=600):
        """Sample collocation, boundary, and validation points.

        Boundary points are split by face-orientation:
          - x-faces (x=0 and x=L): outward normal is along x → only ∂_x penalized
          - y-faces (y=0 and y=L): outward normal is along y → only ∂_y penalized

        n_bc total boundary points are split equally among the 4 sides.
        """
        L, tau = self.L, self.tau

        # --- Interior collocation points ---
        self.x_pde = (torch.rand(n_pde, 1) * L).to(self.device)
        self.y_pde = (torch.rand(n_pde, 1) * L).to(self.device)
        self.t_pde = (torch.rand(n_pde, 1) * tau).to(self.device)

        # --- Boundary points: x-normal faces (left x=0, right x=L) ---
        n_per_side = n_bc // 4
        x_left  = torch.zeros(n_per_side, 1)
        x_right = torch.full((n_per_side, 1), L)
        y_xface = torch.cat([torch.rand(n_per_side, 1) * L,
                             torch.rand(n_per_side, 1) * L], dim=0)
        self.x_bcx = torch.cat([x_left, x_right], dim=0).to(self.device)
        self.y_bcx = y_xface.to(self.device)
        self.t_bcx = (torch.rand(2 * n_per_side, 1) * tau).to(self.device)

        # --- Boundary points: y-normal faces (bottom y=0, top y=L) ---
        y_bot = torch.zeros(n_per_side, 1)
        y_top = torch.full((n_per_side, 1), L)
        x_yface = torch.cat([torch.rand(n_per_side, 1) * L,
                             torch.rand(n_per_side, 1) * L], dim=0)
        self.x_bcy = x_yface.to(self.device)
        self.y_bcy = torch.cat([y_bot, y_top], dim=0).to(self.device)
        self.t_bcy = (torch.rand(2 * n_per_side, 1) * tau).to(self.device)

        # --- Validation set ---
        nv = 3000
        self.x_val = (torch.rand(nv, 1) * L).to(self.device)
        self.y_val = (torch.rand(nv, 1) * L).to(self.device)
        self.t_val = (torch.rand(nv, 1) * tau).to(self.device)

        self.n_pde = n_pde
        self.n_bcx = self.x_bcx.shape[0]
        self.n_bcy = self.x_bcy.shape[0]

        # Diagnostic: print source-term magnitudes
        with torch.no_grad():
            fT, fB, fO, fI, fS = self.mms.source_terms(
                self.x_pde, self.y_pde, self.t_pde, self.params)
            for name, f in [('fT', fT), ('fB', fB), ('fO', fO),
                            ('fI', fI), ('fS', fS)]:
                print(f"    Source {name}: rms="
                      f"{torch.sqrt(torch.mean(f**2)).item():.4f}")

        print(f"  Data: {n_pde} PDE, "
              f"{self.n_bcx} BC-x (∂_x), {self.n_bcy} BC-y (∂_y), {nv} val")

    def resample_pde(self):
        L, tau = self.L, self.tau
        self.x_pde = (torch.rand(self.n_pde, 1) * L).to(self.device)
        self.y_pde = (torch.rand(self.n_pde, 1) * L).to(self.device)
        self.t_pde = (torch.rand(self.n_pde, 1) * tau).to(self.device)

    # -------------------------------------------------------------------------
    # PDE RESIDUAL & GRADIENTS
    # -------------------------------------------------------------------------
    def _pde_residual(self, x_in, y_in, t_in):
        """Return (residuals, grads) for the 5 variables.

        residuals : list of 5 tensors, each [N,1]
        grads     : list of 5 tensors, each [N,2] = [∂_x u, ∂_y u]
        """
        x = x_in.clone().detach().requires_grad_(True)
        y = y_in.clone().detach().requires_grad_(True)
        t = t_in.clone().detach().requires_grad_(True)

        out = self.model(x, y, t)
        T, B, O, I, S = [out[:, i:i + 1] for i in range(5)]

        def g1(u, v):
            return torch.autograd.grad(u, v, torch.ones_like(u),
                                       create_graph=True)[0]

        def lap(u):
            ux = g1(u, x)
            uy = g1(u, y)
            return g1(ux, x) + g1(uy, y), ux, uy

        Tl, Tx, Ty = lap(T); Bl, Bx, By = lap(B); Ol, Ox, Oy = lap(O)
        Il, Ix, Iy = lap(I); Sl, Sx, Sy = lap(S)
        Tt = g1(T, t); Bt = g1(B, t); Ot = g1(O, t); It = g1(I, t); St = g1(S, t)

        p = self.params

        # Source terms: do not depend on network params, detach from graph
        with torch.no_grad():
            fT, fB, fO, fI, fS = self.mms.source_terms(x, y, t, p)
        fT = fT.detach(); fB = fB.detach(); fO = fO.detach()
        fI = fI.detach(); fS = fS.detach()

        rT = (Tt - p['D_T'] * Tl
              - p['rho_T'] * T * (1 - T / p['theta'])
              + p['delta_T'] * T + p['alpha_S'] * S * T - fT)
        rB = (Bt - p['D_B'] * Bl
              - p['rho_B'] * B * (p['K_H'] / (p['K_H'] + O + 1e-8))
              + p['delta_B'] * B + p['beta_I'] * I * B - fB)
        rO = (Ot - p['D_O'] * Ol
              + p['gamma_T'] * T * O
              - p['gamma_E'] * (p['O_ext'] - O) - fO)
        rI = (It - p['D_I'] * Il
              - p['beta_T'] * T + p['delta_I'] * I - fI)
        rS = (St - p['D_S'] * Sl
              - p['beta_B_signal'] * B + p['delta_S'] * S - fS)

        grads = [torch.cat([Tx, Ty], 1), torch.cat([Bx, By], 1),
                 torch.cat([Ox, Oy], 1), torch.cat([Ix, Iy], 1),
                 torch.cat([Sx, Sy], 1)]
        return [rT, rB, rO, rI, rS], grads

    # -------------------------------------------------------------------------
    # LOSS
    # -------------------------------------------------------------------------
    def _loss(self, xp, yp, tp,
              xbx, ybx, tbx,
              xby, yby, tby):
        """Total loss = scaled PDE residual + Neumann BC (normal-only).

        - PDE: each residual normalized by VAR_SCALES[v]^2 so all five
          variables contribute comparably to the gradient.
        - BC on x-faces: only ∂_x u is penalized (g[:, 0:1]).
        - BC on y-faces: only ∂_y u is penalized (g[:, 1:2]).
        """
        # PDE residuals
        res, _ = self._pde_residual(xp, yp, tp)
        loss_pde = sum(torch.mean(r ** 2) / self._scale_sq[i]
                       for i, r in enumerate(res))

        # Neumann on x-faces: penalize ∂_x of each variable only
        _, grads_x = self._pde_residual(xbx, ybx, tbx)
        loss_bx = sum(torch.mean(g[:, 0:1] ** 2) for g in grads_x)

        # Neumann on y-faces: penalize ∂_y of each variable only
        _, grads_y = self._pde_residual(xby, yby, tby)
        loss_by = sum(torch.mean(g[:, 1:2] ** 2) for g in grads_y)

        loss_bc = loss_bx + loss_by
        loss = loss_pde + loss_bc
        return loss, loss_pde.item(), loss_bc.item()

    # -------------------------------------------------------------------------
    # TRAINING — ADAM
    # -------------------------------------------------------------------------
    def train_adam(self, n_epochs=10000, patience=2000, batch_size=512,
                   print_every=500, val_every=5, resample_every=2000):
        best_val = float('inf'); best_state = None; best_ep = 0; pat = 0
        hist = {'train': [], 'val': [], 'pde': [], 'bc': []}
        print(f"\n  ADAM: {n_epochs} ep, batch={batch_size}")
        t0 = datetime.now()

        for ep in range(n_epochs):
            if ep > 0 and ep % resample_every == 0:
                self.resample_pde()
            self.model.train()

            ip  = torch.randperm(self.n_pde, device=self.device)[:batch_size]
            ibx = torch.randperm(self.n_bcx, device=self.device)[
                  :min(batch_size, self.n_bcx)]
            iby = torch.randperm(self.n_bcy, device=self.device)[
                  :min(batch_size, self.n_bcy)]

            self.optimizer.zero_grad()
            loss, lp, lb = self._loss(
                self.x_pde[ip],  self.y_pde[ip],  self.t_pde[ip],
                self.x_bcx[ibx], self.y_bcx[ibx], self.t_bcx[ibx],
                self.x_bcy[iby], self.y_bcy[iby], self.t_bcy[iby])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            hist['train'].append(loss.item())
            hist['pde'].append(lp); hist['bc'].append(lb)

            if (ep + 1) % val_every == 0:
                vl, _, _ = self._loss(
                    self.x_val,  self.y_val,  self.t_val,
                    self.x_bcx, self.y_bcx, self.t_bcx,
                    self.x_bcy, self.y_bcy, self.t_bcy)
                vv = vl.item()
                self.scheduler.step(vv)
                hist['val'].append(vv)
                if vv < best_val:
                    best_val = vv; best_ep = ep + 1
                    best_state = copy.deepcopy(self.model.state_dict()); pat = 0
                else:
                    pat += 1
            else:
                hist['val'].append(hist['val'][-1] if hist['val'] else loss.item())

            if (ep + 1) % print_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                el = (datetime.now() - t0).total_seconds() / 60
                print(f"    Ep {ep+1:>6}  Train {loss.item():.4e}  "
                      f"Val {hist['val'][-1]:.4e}  "
                      f"PDE {lp:.3e}  BC {lb:.3e}  "
                      f"Best {best_val:.4e}@{best_ep}  "
                      f"LR {lr:.1e}  Pat {pat}  [{el:.1f}m]")

            if pat >= patience:
                print(f"    Early stop @{ep+1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
        el = (datetime.now() - t0).total_seconds()
        print(f"  Adam done. Best={best_val:.6e}@{best_ep} ({el:.0f}s)")
        return hist, el

    # -------------------------------------------------------------------------
    # TRAINING — L-BFGS
    # -------------------------------------------------------------------------
    def train_lbfgs(self, n_iters=1000, print_every=50, n_points=5000):
        L, tau = self.L, self.tau
        xp = (torch.rand(n_points, 1) * L).to(self.device)
        yp = (torch.rand(n_points, 1) * L).to(self.device)
        tp = (torch.rand(n_points, 1) * tau).to(self.device)

        opt = torch.optim.LBFGS(self.model.parameters(), lr=1.0,
                                max_iter=20, history_size=100,
                                line_search_fn='strong_wolfe')
        print(f"\n  L-BFGS: {n_iters} iters, {n_points} pts")
        t0 = datetime.now(); losses = []
        best_l = float('inf'); best_s = None

        for it in range(n_iters):
            def closure():
                opt.zero_grad()
                l, _, _ = self._loss(
                    xp, yp, tp,
                    self.x_bcx, self.y_bcx, self.t_bcx,
                    self.x_bcy, self.y_bcy, self.t_bcy)
                l.backward()
                return l
            loss = opt.step(closure)
            lv = loss.item(); losses.append(lv)
            if lv < best_l:
                best_l = lv
                best_s = copy.deepcopy(self.model.state_dict())
            if (it + 1) % print_every == 0:
                el = (datetime.now() - t0).total_seconds() / 60
                print(f"    Iter {it+1:>5}  Loss {lv:.6e}  "
                      f"Best {best_l:.6e}  [{el:.1f}m]")
            if lv < 1e-10:
                print(f"    Converged!")
                break

        if best_s:
            self.model.load_state_dict(best_s)
        el = (datetime.now() - t0).total_seconds()
        print(f"  L-BFGS done. Best={best_l:.6e} ({el:.0f}s)")
        return losses, el

    # -------------------------------------------------------------------------
    # PREDICT & EVALUATE
    # -------------------------------------------------------------------------
    def predict(self, x, y, t):
        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32,
                                 device=self.device).unsqueeze(1)
            yt = torch.as_tensor(y, dtype=torch.float32,
                                 device=self.device).unsqueeze(1)
            tt = torch.full((len(x), 1), float(t),
                            dtype=torch.float32, device=self.device)
            return self.model(xt, yt, tt).cpu().numpy()

    def compute_errors(self, x, y, times, h):
        """Compute relative AND absolute L2 errors against exact solution."""
        vn = ['T', 'B', 'O', 'I', 'S']
        rel = {v: [] for v in vn}
        abso = {v: [] for v in vn}
        for tv in times:
            ev = self.mms.exact_n(x, y, tv)
            po = self.predict(x, y, tv)
            for j, v in enumerate(vn):
                ue = ev[j]; up = po[:, j]
                ln = np.sqrt(h * h * np.sum(ue ** 2))
                err = np.sqrt(h * h * np.sum((ue - up) ** 2))
                abso[v].append(err)
                rel[v].append(err / ln if ln > 1e-10 else float('nan'))
        return rel, abso

    # -------------------------------------------------------------------------
    # DIAGNOSTIC: confirm Neumann BC behaviour
    # -------------------------------------------------------------------------
    def diagnose_bc(self):
        """Check ∂_y T at (x=0, y=0.5, t=0.3).

        Exact: ∂_y T_ex = 0.2 · cos(πx) · (-π sin(πy)) · e^{-0.3t}
        At (0, 0.5, 0.3): 0.2·1·(-π)·e^{-0.09} ≈ -0.5742
        This is the value the trained network's ∂_y T should approach
        if the Neumann BC is correctly implemented (i.e. NOT forcing
        tangential gradients to zero).
        """
        x = torch.tensor([[0.0]], device=self.device, requires_grad=True)
        y = torch.tensor([[0.5]], device=self.device, requires_grad=True)
        t = torch.tensor([[0.3]], device=self.device, requires_grad=True)
        out = self.model(x, y, t)
        T = out[:, 0:1]
        Ty = torch.autograd.grad(T, y, torch.ones_like(T),
                                 create_graph=False)[0]
        exact = 0.2 * np.cos(0.0) * (-np.pi * np.sin(np.pi * 0.5)) \
                * np.exp(-0.09)
        print(f"\n  [BC diagnostic] ∂_y T(0, 0.5, 0.3):")
        print(f"    Exact value : {exact:+.4f}")
        print(f"    PINN value  : {Ty.item():+.4f}")
        print(f"    Should be close (NOT zero — y is tangential at x=0).")


# =============================================================================
# PARAMETERS (same as Table 1)
# =============================================================================

PARAMS = {
    'D_T': 0.01, 'D_B': 0.1, 'D_O': 1.0, 'D_I': 0.5, 'D_S': 0.3,
    'rho_T': 0.3, 'theta': 1.0, 'delta_T': 0.05, 'alpha_S': 0.2,
    'rho_B': 0.5, 'K_H': 0.1, 'delta_B': 0.1, 'beta_I': 0.3,
    'gamma_T': 0.2, 'gamma_E': 0.5, 'O_ext': 0.2,
    'beta_T': 0.1, 'delta_I': 0.2,
    'beta_B_signal': 0.4, 'delta_S': 0.3,
}


# =============================================================================
# RUN
# =============================================================================

LAYERS = [3, 128, 128, 128, 128, 5]
OUTPUT_DIR = 'pinn_bio_mms'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'#' * 70}")
print(f"  PINN — Biological-Scale MMS  [CORRECTED]")
print(f"  Values in [0, 1] — matching the real problem")
print(f"  Architecture: {LAYERS}")
print(f"{'#' * 70}")

pinn = BioPINN(LAYERS, PARAMS, device, L=1.0, tau=1.0)
pinn.generate_data(n_pde=15000, n_bc=600)

# Phase 1: Adam
h1, t1 = pinn.train_adam(n_epochs=100000, patience=2000, batch_size=1024,
                         print_every=500, resample_every=2000)

# Phase 2: L-BFGS
h2, t2 = pinn.train_lbfgs(n_iters=5000, print_every=50, n_points=5000)

total_time = t1 + t2

# Diagnostic — confirm BC fix took
pinn.diagnose_bc()

# Save model
torch.save({
    'model_state_dict': pinn.model.state_dict(),
    'layers': LAYERS, 'params': PARAMS, 'total_time': total_time,
}, os.path.join(OUTPUT_DIR, 'pinn_bio_mms_model.pt'))


# ── Evaluate on a 61x61 grid (same grid as FEM) ──
nx = 61
xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, nx)
XX, YY = np.meshgrid(xs, ys)
x_mesh = XX.ravel(); y_mesh = YY.ravel()
h = 1.0 / (nx - 1)
times = [round(t, 2) for t in np.arange(0.01, 1.01, 0.01)]

print(f"\n  Evaluating PINN on {nx}x{nx} grid, {len(times)} times...")
pinn_rel, pinn_abs = pinn.compute_errors(x_mesh, y_mesh, times, h)


# ── Print PINN results (split by variable type) ──
vn = ['T', 'B', 'O', 'I', 'S']
print(f"\n  {'=' * 60}")
print(f"  PINN RESULTS")
print(f"  {'=' * 60}")
print(f"  Variables T, O start nonzero — relative L2 valid for all t.")
print(f"  Variables B, I, S start at 0 — relative L2 unreliable for")
print(f"    small t; absolute L2 reported alongside.")
print(f"  {'-' * 60}")
for v in vn:
    rel_arr = np.array(pinn_rel[v])
    abs_arr = np.array(pinn_abs[v])
    if v in ('T', 'O'):
        idx_valid = np.arange(len(times))
    else:
        # For B, I, S: report relative starting from t >= 0.1
        idx_valid = np.array([i for i, t in enumerate(times) if t >= 0.1])
    rel_valid = rel_arr[idx_valid]
    if np.all(np.isnan(rel_valid)):
        rel_max_str = "n/a"
    else:
        i_max = idx_valid[np.nanargmax(rel_valid)]
        rel_max_str = f"{rel_arr[i_max]:.4e}  (at t = {times[i_max]:.2f})"
    abs_max = np.max(abs_arr)
    abs_t = times[np.argmax(abs_arr)]
    print(f"    {v}: max rel L2 = {rel_max_str}")
    print(f"       max abs L2 = {abs_max:.4e}  (at t = {abs_t:.2f})")
print(f"  {'=' * 60}")


# ── Save exact solution for FEM ──
print(f"\n  Saving exact solution for FEM validation...")
mms = BioMMS(L=1.0)
exact_dir = os.path.join(OUTPUT_DIR, 'exact_solution')
for vname, idx in [('Tex', 0), ('Bex', 1), ('Oex', 2), ('Iex', 3), ('Sex', 4)]:
    vdir = os.path.join(exact_dir, vname)
    os.makedirs(vdir, exist_ok=True)
    for tv in times:
        ev = mms.exact_n(x_mesh, y_mesh, tv)
        data = np.column_stack([x_mesh, y_mesh, ev[idx]])
        t_str = str(int(tv)) if tv == int(tv) else str(tv)
        if tv == 1.0:
            t_str = '1'
        np.savetxt(os.path.join(vdir, t_str), data, fmt='%.10f')
print(f"    Saved to {exact_dir}/")


# ── Training history plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogy(h1['train'], lw=0.5, alpha=0.4, label='Train (total)')
axes[0].semilogy(h1['pde'],   lw=0.7, alpha=0.6, label='PDE')
axes[0].semilogy(h1['bc'],    lw=0.7, alpha=0.6, label='BC')
axes[0].semilogy(h1['val'],   lw=1.5, label='Val')
axes[0].set_title('Phase 1: Adam'); axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')

axes[1].semilogy(h2, lw=2, color='r', label='L-BFGS')
axes[1].set_title('Phase 2: L-BFGS'); axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Loss')
plt.suptitle('Training History — Biological-Scale MMS',
             fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'),
            dpi=150, bbox_inches='tight')
plt.close()


# ── Error plot (relative + absolute) ──
fig, axes = plt.subplots(2, 5, figsize=(25, 9))
ta = np.array(times)
vc = ['#d32f2f', '#1565c0', '#2e7d32', '#e65100', '#6a1b9a']
vl = ['$T$', '$B$', '$O$', '$I$', '$S$']
fig.suptitle('PINN $L^2$ Error (Biological-Scale MMS)',
             fontsize=15, weight='bold', y=1.01)
for j, (v, vlb, c) in enumerate(zip(vn, vl, vc)):
    # Relative (top row) — mask early times for B/I/S
    rel_arr = np.array(pinn_rel[v])
    if v in ('B', 'I', 'S'):
        mask = ta >= 0.1
        axes[0, j].semilogy(ta[mask], rel_arr[mask], '-', color=c, lw=2.5)
        axes[0, j].text(0.05, 0.95,
                        '(plotted from $t \\geq 0.1$)',
                        transform=axes[0, j].transAxes,
                        fontsize=9, va='top', color='gray')
    else:
        axes[0, j].semilogy(ta, rel_arr, '-', color=c, lw=2.5)
    axes[0, j].set_xlabel('Time $t$')
    axes[0, j].set_title(f'{vlb}  (relative)', fontsize=12, weight='bold')
    axes[0, j].grid(True, alpha=0.3, which='both')
    axes[0, j].set_ylabel('Relative $L^2$ error')

    # Absolute (bottom row) — full range
    axes[1, j].semilogy(ta, np.array(pinn_abs[v]), '-', color=c, lw=2.5)
    axes[1, j].set_xlabel('Time $t$')
    axes[1, j].set_title(f'{vlb}  (absolute)', fontsize=12, weight='bold')
    axes[1, j].grid(True, alpha=0.3, which='both')
    axes[1, j].set_ylabel('Absolute $L^2$ error')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_errors.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_errors.png'),
            dpi=150, bbox_inches='tight')
plt.close()


# ── Solution snapshots ──
fig, axes = plt.subplots(2, 5, figsize=(25, 8))
fig.suptitle('Exact vs PINN at $t = 0.5$',
             fontsize=15, weight='bold', y=1.02)
cms = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']
ev = mms.exact_n(x_mesh, y_mesh, 0.5)
pv = pinn.predict(x_mesh, y_mesh, 0.5)
X, Y = x_mesh.reshape(nx, nx), y_mesh.reshape(nx, nx)

for j, (v, vlb, cm) in enumerate(zip(vn, vl, cms)):
    im0 = axes[0, j].pcolormesh(X, Y, ev[j].reshape(nx, nx),
                                cmap=cm, shading='auto')
    plt.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)
    axes[0, j].set_title(f'Exact {vlb}', fontsize=11, weight='bold')
    axes[0, j].set_aspect('equal')

    im1 = axes[1, j].pcolormesh(X, Y, pv[:, j].reshape(nx, nx),
                                cmap=cm, shading='auto')
    plt.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)
    axes[1, j].set_title(f'PINN {vlb}', fontsize=11, weight='bold')
    axes[1, j].set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'solution_comparison.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'solution_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Results: {OUTPUT_DIR}/")
print(f"\n  NEXT STEP: Give {exact_dir}/ to David for FEM validation")
print(f"  Then run comparison with both results.")
