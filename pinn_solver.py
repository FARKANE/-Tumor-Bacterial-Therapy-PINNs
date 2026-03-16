import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =============================================================================
# QUADRANGLE GEOMETRY  (replaces MeshGeometry / breast mesh)
# =============================================================================

class QuadGeometry:
    """
    Rectangular (quadrangle) domain  [x_min, x_max] × [y_min, y_max].
    Generates a uniform triangular mesh by splitting each quad cell into
    2 triangles — same interface as MeshGeometry so no other code changes.
    """

    def __init__(self, x_min=0.0, x_max=6.0, y_min=0.0, y_max=6.0,
                 nx=60, ny=60):
        """
        Parameters
        ----------
        x_min, x_max : horizontal extent
        y_min, y_max : vertical extent
        nx, ny       : number of intervals in x and y  (nodes = (nx+1)*(ny+1))
        """
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.nx, self.ny = nx, ny

        self.nodes, self.elements, self.boundary_edges = self._build_mesh()

        self.n_nodes          = len(self.nodes)
        self.n_elements       = len(self.elements)
        self.n_boundary_edges = len(self.boundary_edges)

        self.x = self.nodes[:, 0]
        self.y = self.nodes[:, 1]

        # boundary node indices (0-based)
        self.boundary_node_indices = np.unique(
            self.boundary_edges[:, :2].flatten()
        )

        print(f"\n{'='*70}")
        print(f"📐 QUADRANGLE DOMAIN")
        print(f"{'='*70}")
        print(f"Domain  : x ∈ [{x_min}, {x_max}],  y ∈ [{y_min}, {y_max}]")
        print(f"Grid    : {nx} × {ny}  →  {(nx+1)*(ny+1)} nodes")
        print(f"Elements: {self.n_elements} triangles")
        print(f"Boundary edges: {self.n_boundary_edges}")
        print(f"Boundary nodes: {len(self.boundary_node_indices)}")
        print(f"{'='*70}\n")

    # ── mesh builder ──────────────────────────────────────────────
    def _build_mesh(self):
        nx, ny = self.nx, self.ny
        xs = np.linspace(self.x_min, self.x_max, nx + 1)
        ys = np.linspace(self.y_min, self.y_max, ny + 1)

        # nodes
        XX, YY = np.meshgrid(xs, ys)          # shape (ny+1, nx+1)
        nodes  = np.column_stack([XX.ravel(), YY.ravel()])   # (N,2)

        def nid(i, j):   # node index for grid point (j-row, i-col)
            return j * (nx + 1) + i

        # triangles: each quad split into 2 triangles
        tris = []
        for j in range(ny):
            for i in range(nx):
                a = nid(i,   j)
                b = nid(i+1, j)
                c = nid(i+1, j+1)
                d = nid(i,   j+1)
                tris.append([a, b, c])
                tris.append([a, c, d])
        elements = np.array(tris, dtype=int)   # 0-based

        # boundary edges (4 sides)
        bedges = []
        # bottom  y = y_min
        for i in range(nx):
            bedges.append([nid(i, 0),   nid(i+1, 0)])
        # right  x = x_max
        for j in range(ny):
            bedges.append([nid(nx, j),  nid(nx, j+1)])
        # top  y = y_max
        for i in range(nx-1, -1, -1):
            bedges.append([nid(i+1, ny), nid(i, ny)])
        # left  x = x_min
        for j in range(ny-1, -1, -1):
            bedges.append([nid(0, j+1),  nid(0, j)])
        boundary_edges = np.array(bedges, dtype=int)   # 0-based

        return nodes, elements, boundary_edges

    # ── interface methods (identical signature to MeshGeometry) ───
    def triangulation(self):
        return Triangulation(self.x, self.y, self.elements)

    def sample_points(self, n_points):
        if n_points <= self.n_nodes:
            idx = np.random.choice(self.n_nodes, n_points, replace=False)
            return torch.FloatTensor(self.nodes[idx])
        else:
            all_nodes  = torch.FloatTensor(self.nodes)
            n_extra    = n_points - self.n_nodes
            xr = torch.rand(n_extra)*(self.x_max-self.x_min) + self.x_min
            yr = torch.rand(n_extra)*(self.y_max-self.y_min) + self.y_min
            extra = torch.stack([xr, yr], dim=1)
            return torch.cat([all_nodes, extra], dim=0)

    def sample_boundary(self, n_points):
        bc = self.nodes[self.boundary_node_indices]
        if n_points <= len(bc):
            idx = np.random.choice(len(bc), n_points, replace=False)
        else:
            idx = np.random.choice(len(bc), n_points, replace=True)
        return torch.FloatTensor(bc[idx])

    def plot_mesh(self, ax, show_elements=True, show_boundary=True, **kwargs):
        if show_elements:
            triang = self.triangulation()
            ax.triplot(triang, 'b-', linewidth=0.2, alpha=0.3)
        if show_boundary:
            for edge in self.boundary_edges:
                idx = edge[:2]
                ax.plot(self.x[idx], self.y[idx], 'r-', linewidth=2)

    def draw_boundary(self, ax, color='k', lw=1.5, alpha=0.9):
        """Draw the 4 outer edges of the rectangle."""
        for edge in self.boundary_edges:
            idx = edge[:2]
            ax.plot(self.x[idx], self.y[idx], '-', color=color, lw=lw, alpha=alpha)


# =============================================================================
# DATASET
# =============================================================================

class PINNDataset(Dataset):
    def __init__(self, xy_pde, t_pde, xy_ic, t_ic, xy_bc, t_bc):
        self.xy_pde = xy_pde; self.t_pde = t_pde
        self.xy_ic  = xy_ic;  self.t_ic  = t_ic
        self.xy_bc  = xy_bc;  self.t_bc  = t_bc
        self.n_pde  = xy_pde.shape[0]
        self.n_ic   = xy_ic.shape[0]
        self.n_bc   = xy_bc.shape[0]

    def __len__(self): return self.n_pde

    def __getitem__(self, idx):
        return {
            'xy_pde': self.xy_pde[idx],       't_pde': self.t_pde[idx],
            'xy_ic' : self.xy_ic[idx % self.n_ic], 't_ic' : self.t_ic[idx % self.n_ic],
            'xy_bc' : self.xy_bc[idx % self.n_bc], 't_bc' : self.t_bc[idx % self.n_bc],
        }


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class PINN_2D_MultiVariable(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation    = nn.Tanh()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.linear_layers.append(lin)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        for layer in self.linear_layers[:-1]:
            inp = self.activation(layer(inp))
        return self.linear_layers[-1](inp)


# =============================================================================
# PINN  (unchanged — only geometry object differs)
# =============================================================================

class BreastTumorPINN:
    def __init__(self, layers, params, geometry, device, checkpoint_dir='checkpoints'):
        self.device         = device
        self.geometry       = geometry
        self.model          = PINN_2D_MultiVariable(layers).to(device)
        self.params         = params
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1000)
        self.lambda_pde     = 1.0
        self.lambda_ic      = 50.0
        self.lambda_bc      = 1.0
        self.loss_history   = {k: [] for k in
            ['train_total','train_pde','train_ic','train_bc',
             'val_total',  'val_pde',  'val_ic',  'val_bc']}
        self.best_val_loss  = float('inf')
        self.best_epoch     = 0
        self.patience_counter = 0
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.train_loader = self.val_loader = self.test_loader = None

    # ── data ──────────────────────────────────────────────────────
    def generate_training_data(self, n_pde_total=10000, n_ic_total=600,
                               n_bc_total=400, t_domain=[0, 30],
                               train_ratio=0.6, val_ratio=0.2,
                               batch_size=512, num_workers=0):
        print("\n" + "="*70)
        print("Generating training data on QUADRANGLE DOMAIN...")
        print("="*70)

        n_pde_tr = int(n_pde_total*train_ratio); n_pde_v = int(n_pde_total*val_ratio)
        n_ic_tr  = int(n_ic_total *train_ratio); n_ic_v  = int(n_ic_total *val_ratio)
        n_bc_tr  = int(n_bc_total *train_ratio); n_bc_v  = int(n_bc_total *val_ratio)

        xy_pde = self.geometry.sample_points(n_pde_total)
        t_pde  = torch.rand(n_pde_total,1)*(t_domain[1]-t_domain[0])+t_domain[0]
        xy_ic  = self.geometry.sample_points(n_ic_total)
        t_ic   = torch.zeros(n_ic_total, 1)
        xy_bc  = self.geometry.sample_boundary(n_bc_total)
        t_bc   = torch.rand(n_bc_total,1)*(t_domain[1]-t_domain[0])+t_domain[0]

        def mk(a, b, xp, tp, xi, ti, xb, tb):
            return PINNDataset(xp[a:b], tp[a:b], xi[a:b], ti[a:b], xb[a:b], tb[a:b])

        tr_ds  = PINNDataset(xy_pde[:n_pde_tr], t_pde[:n_pde_tr],
                             xy_ic[:n_ic_tr],   t_ic[:n_ic_tr],
                             xy_bc[:n_bc_tr],   t_bc[:n_bc_tr])
        v_ds   = PINNDataset(xy_pde[n_pde_tr:n_pde_tr+n_pde_v],
                             t_pde[n_pde_tr:n_pde_tr+n_pde_v],
                             xy_ic[n_ic_tr:n_ic_tr+n_ic_v],
                             t_ic[n_ic_tr:n_ic_tr+n_ic_v],
                             xy_bc[n_bc_tr:n_bc_tr+n_bc_v],
                             t_bc[n_bc_tr:n_bc_tr+n_bc_v])
        te_ds  = PINNDataset(xy_pde[n_pde_tr+n_pde_v:], t_pde[n_pde_tr+n_pde_v:],
                             xy_ic[n_ic_tr+n_ic_v:],    t_ic[n_ic_tr+n_ic_v:],
                             xy_bc[n_bc_tr+n_bc_v:],    t_bc[n_bc_tr+n_bc_v:])

        pin = True if device.type == 'cuda' else False
        self.train_loader = DataLoader(tr_ds, batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin)
        self.val_loader   = DataLoader(v_ds,  batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin)
        self.test_loader  = DataLoader(te_ds, batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin)
        print(f"Batches → Train:{len(self.train_loader)}  Val:{len(self.val_loader)}  "
              f"Test:{len(self.test_loader)}")
        print("="*70)

    # ── PDE residuals ─────────────────────────────────────────────
    def pde_residuals(self, x, y, t):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        out = self.model(x, y, t)
        T, B, O, I, S = [out[:, i:i+1] for i in range(5)]

        def grad1(u, v):
            return torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u),
                                       create_graph=True)[0]
        def lapl(u, vx, vy):
            ux = grad1(u, vx); uy = grad1(u, vy)
            return grad1(ux, vx) + grad1(uy, vy), ux, uy

        T_lap, T_x, T_y = lapl(T, x, y)
        B_lap, B_x, B_y = lapl(B, x, y)
        O_lap, O_x, O_y = lapl(O, x, y)
        I_lap, I_x, I_y = lapl(I, x, y)
        S_lap, S_x, S_y = lapl(S, x, y)

        T_t = grad1(T, t); B_t = grad1(B, t)
        O_t = grad1(O, t); I_t = grad1(I, t); S_t = grad1(S, t)

        p = self.params
        f_T = T_t - p['D_T']*T_lap - p['rho_T']*T*(1-T/p['theta']) + p['delta_T']*T + p['alpha_S']*S*T
        f_B = B_t - p['D_B']*B_lap - p['rho_B']*B*(p['K_H']/(p['K_H']+O+1e-8)) + p['delta_B']*B + p['beta_I']*I*B
        f_O = O_t - p['D_O']*O_lap + p['gamma_T']*T*O  - p['gamma_E']*(p['O_ext']-O)
        #f_O = O_t - p['D_O']*O_lap + p['gamma_T']*T*O + p['beta_B']*B*O - p['gamma_E']*(p['O_ext']-O)
        f_I = I_t - p['D_I']*I_lap - p['beta_T']*T + p['delta_I']*I
        f_S = S_t - p['D_S']*S_lap - p['beta_B_signal']*B + p['delta_S']*S

        grad_T = torch.cat([T_x, T_y], dim=1)
        grad_B = torch.cat([B_x, B_y], dim=1)
        grad_O = torch.cat([O_x, O_y], dim=1)
        grad_I = torch.cat([I_x, I_y], dim=1)
        grad_S = torch.cat([S_x, S_y], dim=1)

        return f_T, f_B, f_O, f_I, f_S, grad_T, grad_B, grad_O, grad_I, grad_S

    # ── initial conditions ────────────────────────────────────────
    def initial_conditions(self, x, y):
        g = self.geometry
        x_center   = (g.x_max + g.x_min) / 2
        y_center   = (g.y_max + g.y_min) / 2
        domain_size = max(g.x_max - g.x_min, g.y_max - g.y_min)

        p = self.params
        tx = p.get('tumor_x', x_center)
        ty = p.get('tumor_y', y_center)
        ts = p.get('tumor_size', domain_size / 5)

        r2  = (x - tx)**2 + (y - ty)**2
        T0  = 1.0 * torch.exp(-r2 / (2 * ts**2))

        bx  = p.get('bacteria_x', g.x_min + 0.3*(g.x_max - g.x_min))
        by  = p.get('bacteria_y', g.y_min + 0.6*(g.y_max - g.y_min))
        bs  = p.get('bacteria_spread', domain_size / 8)

        rb2 = (x - bx)**2 + (y - by)**2
        B0  = 0.3 * torch.exp(-rb2 / (2 * bs**2))

        O0  = p['O_ext'] * torch.ones_like(x)
        I0  = 0.01       * torch.ones_like(x)
        S0  = torch.zeros_like(x)

        return T0, B0, O0, I0, S0

    # ── loss ──────────────────────────────────────────────────────
    def loss_function(self, batch):
        xy_pde = batch['xy_pde'].to(self.device, non_blocking=True)
        t_pde  = batch['t_pde'] .to(self.device, non_blocking=True)
        xy_ic  = batch['xy_ic'] .to(self.device, non_blocking=True)
        t_ic   = batch['t_ic']  .to(self.device, non_blocking=True)
        xy_bc  = batch['xy_bc'] .to(self.device, non_blocking=True)
        t_bc   = batch['t_bc']  .to(self.device, non_blocking=True)

        x_pde, y_pde = xy_pde[:, 0:1], xy_pde[:, 1:2]
        f_T,f_B,f_O,f_I,f_S,_,_,_,_,_ = self.pde_residuals(x_pde, y_pde, t_pde)
        loss_pde = sum(torch.mean(f**2) for f in [f_T,f_B,f_O,f_I,f_S])

        x_ic, y_ic = xy_ic[:, 0:1], xy_ic[:, 1:2]
        out_ic = self.model(x_ic, y_ic, t_ic)
        T0,B0,O0,I0,S0 = self.initial_conditions(x_ic, y_ic)
        loss_ic = sum(torch.mean((out_ic[:,i:i+1] - ref)**2)
                      for i, ref in enumerate([T0,B0,O0,I0,S0]))

        x_bc, y_bc = xy_bc[:, 0:1], xy_bc[:, 1:2]
        _,_,_,_,_,gT,gB,gO,gI,gS = self.pde_residuals(x_bc, y_bc, t_bc)
        loss_bc = sum(torch.mean(g**2) for g in [gT,gB,gO,gI,gS])

        loss = self.lambda_pde*loss_pde + self.lambda_ic*loss_ic + self.lambda_bc*loss_bc
        return loss, loss_pde.item(), loss_ic.item(), loss_bc.item()

    # ── train / validate ──────────────────────────────────────────
    def _run_loader(self, loader, train=True):
        self.model.train(train)
        acc = {k: 0. for k in ['total','pde','ic','bc']}
        for batch in loader:
            if train: self.optimizer.zero_grad()
            loss, lp, li, lb = self.loss_function(batch)
            if train: loss.backward(); self.optimizer.step()
            acc['total'] += loss.item(); acc['pde'] += lp
            acc['ic']    += li;          acc['bc']  += lb
        n = len(loader)
        return {k: v/n for k, v in acc.items()}

    def train_epoch(self):  return self._run_loader(self.train_loader, train=True)
    def validate(self):     return self._run_loader(self.val_loader,   train=False)

    # ── checkpoint ────────────────────────────────────────────────
    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss_history': self.loss_history,
                'best_val_loss': self.best_val_loss, 'best_epoch': self.best_epoch}
        torch.save(ckpt, os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, 'best_model.pt'))
            print(f"  ✓ Best saved  (Val: {self.best_val_loss:.4e})")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.loss_history  = ckpt['loss_history']
        self.best_val_loss = ckpt['best_val_loss']
        self.best_epoch    = ckpt['best_epoch']
        print(f"Checkpoint loaded  (epoch {ckpt['epoch']})")

    # ── training loop ─────────────────────────────────────────────
    def train(self, n_epochs, patience=2000, print_every=500):
        print("\n" + "="*70)
        print("Training on QUADRANGLE DOMAIN")
        print("="*70)
        start = datetime.now()

        for epoch in range(n_epochs):
            tl = self.train_epoch()
            vl = self.validate()
            self.scheduler.step(vl['total'])

            for k in ['total','pde','ic','bc']:
                self.loss_history[f'train_{k}'].append(tl[k])
                self.loss_history[f'val_{k}'].append(vl[k])

            is_best = vl['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = vl['total']
                self.best_epoch    = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch+1) % 500 == 0 or is_best:
                self.save_checkpoint(epoch+1, is_best)

            if (epoch+1) % print_every == 0 or is_best:
                lr  = self.optimizer.param_groups[0]['lr']
                ela = (datetime.now()-start).total_seconds()/60
                print(f"Ep {epoch+1}/{n_epochs} | {ela:.1f}min")
                print(f"  Train {tl['total']:.4e}  (pde {tl['pde']:.4e}  "
                      f"ic {tl['ic']:.4e}  bc {tl['bc']:.4e})")
                print(f"  Val   {vl['total']:.4e}  LR {lr:.2e}  "
                      f"Best {self.best_val_loss:.4e}@{self.best_epoch}  "
                      f"Pat {self.patience_counter}/{patience}")
                print('-'*70)

            if self.patience_counter >= patience:
                print(f"\nEarly stop @ epoch {epoch+1}")
                break

        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)
        print(f"\nDone in {(datetime.now()-start).total_seconds()/60:.1f} min")

    # ── evaluate test ─────────────────────────────────────────────
    def evaluate_test_set(self):
        print("\n" + "="*70 + "\nTest Set Evaluation\n" + "="*70)
        tl = self._run_loader(self.test_loader, train=False)
        for k, v in tl.items(): print(f"  {k}: {v:.4e}")
        return tl

    # ── predict ───────────────────────────────────────────────────
    def predict(self, x, y, t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x, y, t)

    # ── training history plot ─────────────────────────────────────
    def plot_training_history(self, save_path='training_history.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        keys = [('train_total','val_total','Total'),
                ('train_pde',  'val_pde',  'PDE'),
                ('train_ic',   'val_ic',   'IC'),
                ('train_bc',   'val_bc',   'BC')]
        for ax, (tk, vk, title) in zip(axes.flatten(), keys):
            ax.semilogy(self.loss_history[tk], label='Train', lw=2)
            ax.semilogy(self.loss_history[vk], label='Val',   lw=2, alpha=0.8)
            ax.axvline(self.best_epoch-1, color='r', ls='--', alpha=0.6,
                       label=f'Best @{self.best_epoch}')
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training history → {save_path}")


# =============================================================================
# VISUALIZATION  (no internal markers)
# =============================================================================

def plot_results(pinn, times=[0, 5, 10, 15, 20, 25, 30],
                 save_path='results_quad.png'):
    print("\nGenerating result plots …")
    geo    = pinn.geometry
    triang = geo.triangulation()
    vars_  = ['Tumor (T)', 'Bacteria (B)', 'Oxygen (O)', 'Cytokines (I)', 'Signal (S)']
    cmaps  = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']
    nt, nv = len(times), 5

    fig, axes = plt.subplots(nt, nv, figsize=(4.5*nv, 3.5*nt))
    fig.suptitle('PDE Solution on Quadrangle Domain', fontsize=15, y=1.002)

    for ti, t_val in enumerate(times):
        xm = torch.FloatTensor(geo.x).unsqueeze(1).to(device)
        ym = torch.FloatTensor(geo.y).unsqueeze(1).to(device)
        tm = torch.full_like(xm, t_val)
        out = pinn.predict(xm, ym, tm).cpu().numpy()

        for vi in range(nv):
            ax  = axes[ti, vi]
            val = out[:, vi]
            tcf = ax.tricontourf(triang, val, levels=50, cmap=cmaps[vi])
            geo.draw_boundary(ax, color='k', lw=1.5)
            ax.set_title(f'{vars_[vi]}  t={t_val}d', fontsize=9)
            ax.set_aspect('equal')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results → {save_path}")


def plot_initial_conditions(pinn, save_path='initial_conditions_quad.png'):
    print("Plotting initial conditions …")
    geo    = pinn.geometry
    triang = geo.triangulation()
    vars_  = ['Tumor (T)', 'Bacteria (B)', 'Oxygen (O)', 'Cytokines (I)', 'Signal (S)']
    cmaps  = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']

    xm = torch.FloatTensor(geo.x).unsqueeze(1).to(device)
    ym = torch.FloatTensor(geo.y).unsqueeze(1).to(device)
    tm = torch.zeros_like(xm)
    out = pinn.predict(xm, ym, tm).cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('Initial Conditions  (t = 0)', fontsize=14, weight='bold')
    for vi in range(5):
        ax  = axes[vi]
        tcf = ax.tricontourf(triang, out[:, vi], levels=50, cmap=cmaps[vi])
        geo.draw_boundary(ax, color='k', lw=1.5)
        ax.set_title(vars_[vi], fontsize=12, weight='bold')
        ax.set_aspect('equal')
        plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"IC plot → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ── 1. Domain (edit dimensions here) ─────────────────────────
    geo = QuadGeometry(
        x_min=0.0, x_max=6.0,
        y_min=0.0, y_max=6.0,
        nx=60, ny=60          # 3721 nodes, 7200 triangles
    )

    # visualize mesh
    fig, ax = plt.subplots(figsize=(6, 6))
    geo.plot_mesh(ax, show_elements=True, show_boundary=True)
    ax.set_title('Quadrangle Mesh', fontsize=13, weight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quad_mesh.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Mesh saved → quad_mesh.png")

    # ── 2. Physical parameters ────────────────────────────────────
    params = {
        'D_T': 0.01, 'D_B': 0.1, 'D_O': 1.0, 'D_I': 0.5, 'D_S': 0.3,
        'rho_T': 0.3, 'theta': 1.0, 'delta_T': 0.05, 'alpha_S': 0.2,
        'rho_B': 0.5, 'K_H': 0.1, 'delta_B': 0.1, 'beta_I': 0.3,
        'gamma_T': 0.2, 'beta_B': 0.1, 'gamma_E': 0.5, 'O_ext': 0.2,
        'beta_T': 0.1, 'delta_I': 0.2,
        'beta_B_signal': 0.4, 'delta_S': 0.3,

        # Tumor IC  (centre of quad domain)
        'tumor_x'   : (geo.x_min + geo.x_max) / 2,
        'tumor_y'   : (geo.y_min + geo.y_max) / 2,
        'tumor_size': (geo.x_max - geo.x_min) / 5,       # large IC

        # Bacteria injection  (30 % x, 60 % y)
        'bacteria_x'     : geo.x_min + 0.3*(geo.x_max - geo.x_min),
        'bacteria_y'     : geo.y_min + 0.6*(geo.y_max - geo.y_min),
        'bacteria_spread': (geo.x_max - geo.x_min) / 8,
    }

    # ── 3. Build PINN ─────────────────────────────────────────────
    layers = [3, 64, 64, 64, 64, 5]
    pinn   = BreastTumorPINN(layers, params, geo, device,
                             checkpoint_dir='checkpoints_quad')

    # ── 4. Training data ──────────────────────────────────────────
    pinn.generate_training_data(
        n_pde_total=10000, n_ic_total=600, n_bc_total=400,
        t_domain=[0, 30], train_ratio=0.6, val_ratio=0.2,
        batch_size=512, num_workers=0
    )

    # ── 5. Train ──────────────────────────────────────────────────
    pinn.train(n_epochs=8000, patience=2000, print_every=10)

    # ── 6. Evaluate ───────────────────────────────────────────────
    pinn.evaluate_test_set()

    # ── 7. Plots ──────────────────────────────────────────────────
    pinn.plot_training_history('training_history_quad.png')
    plot_initial_conditions(pinn, 'initial_conditions_quad.png')
    plot_results(pinn, times=[0, 5, 10, 15, 20, 25, 30],
                 save_path='results_quad.png')

    print("\n" + "="*70)
    print(f"✅  DONE   best epoch {pinn.best_epoch}   "
          f"val loss {pinn.best_val_loss:.4e}")
    print("="*70)