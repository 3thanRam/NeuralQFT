"""
Training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from thimble import LagrangianParams, PhysicsInformedThimble, effective_sample_size


# ─────────────────────────────────────────────────────────────────────────────
# Randomised Lagrangian sampler  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

def random_params(epoch: int, cfg: dict) -> LagrangianParams:
    """
    Draw a random Lagrangian each step covering the full validation range.
    Range always includes g4=0 so the network doesn't forget the free theory.
    """
    N = cfg['n_epochs']
    t = epoch / N

    m2 = float(np.random.uniform(0.5, 2.0))

    kinetic_coeff_max=cfg.get('train_kin', 1.0) * min(1.0, max(0.0, (t - 0.7) / 0.3))
    kinetic_coeff= float(np.random.uniform(0.05, kinetic_coeff_max)) if kinetic_coeff_max > 0.1 else 0.1

    g4_max = cfg.get('train_g4', 0.5) * min(1.0, max(0.0, (t - 0.15) / 0.35))
    g4 = float(np.random.uniform(0.0, g4_max)) if g4_max > 0 else 0.0

    g3_max = cfg.get('train_g3', 0.0) * min(1.0, max(0.0, (t - 0.5) / 0.3))
    g3 = float(np.random.uniform(-g3_max, g3_max)) if g3_max > 0 else 0.0

    g6_max = cfg.get('train_g6', 0.0) * min(1.0, max(0.0, (t - 0.5) / 0.3))
    g6 = float(np.random.uniform(0.0, g6_max)) if g6_max > 0 else 0.0

    mu_max = cfg.get('train_mu', 0.0) * min(1.0, max(0.0, (t - 0.7) / 0.3))
    mu = float(np.random.uniform(0.0, mu_max)) if mu_max > 0 else 0.0

    box_coeff_max=cfg.get('train_box_coeff', 0.0) * min(1.0, max(0.0, (t - 0.7) / 0.3))
    box_coeff= float(np.random.uniform(0.0, box_coeff_max)) if box_coeff_max > 0 else 0.0

    deriv_interact_max=cfg.get('train_deriv_interact', 0.0) * min(1.0, max(0.0, (t - 0.7) / 0.3))
    deriv_interact= float(np.random.uniform(0.0, deriv_interact_max)) if deriv_interact_max > 0 else 0.0
    return LagrangianParams(
        m2=m2, kinetic_coeff=kinetic_coeff,
        g3=g3, g4=g4, g6=g6,
        box_coeff=box_coeff, deriv_interact=deriv_interact, mu=mu,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history: dict, save_path):
    keys   = ['var_Feff', 'var_imS', 'var_logJ', 'ess', 'lr', 'g4_sample']
    labels = ['Var(F_eff) [primary]', 'Var(Im S) [thimble]',
              'Var(log det) [stability]', 'ESS training (fraction)',
              'Learning Rate (warm restarts)', 'g4 (sampled this step)']
    logy   = [True, True, True, False, False, False]
    colors = ['k', 'm', 'teal', 'g', 'b', 'r']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep = np.arange(len(history['var_Feff']))
    for ax, key, lbl, col, do_log in zip(axes.flat, keys, labels, colors, logy):
        if key not in history: continue
        vals = np.array(history[key], dtype=float)
        fn   = ax.semilogy if do_log else ax.plot
        fn(ep, np.clip(vals, 1e-15, None) if do_log else vals, color=col, alpha=0.6)
        if key == 'ess':
            ax.axhline(0.1, color='r', ls='--', lw=1.5, label='10% floor')
            ax.set_ylim(0, 1); ax.legend()
        ax.set_title(lbl, fontsize=12); ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.model = PhysicsInformedThimble(cfg).to(cfg['device'])
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])

        # Warm restarts: re-heat LR every T_0 epochs so optimiser escapes
        # flat regions — crucial when training hasn't converged at T_max.
        T0 = cfg.get('restart_period', 2000)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=T0, T_mult=1, eta_min=cfg['lr'] * 5e-3)

    def step(self, epoch: int):
        self.opt.zero_grad()
        p   = random_params(epoch, self.cfg)
        N   = self.cfg['n_epochs']
        L   = self.cfg['L']
        dev = self.cfg['device']

        z = torch.randn(self.cfg['batch_size'], L, L, L, L, device=dev)
        phi, log_det, scale = self.model(z, p, return_scale=True)

        var_Feff, var_imS, var_logJ, mean_Feff, S = \
            self.model.compute_thimble_loss(phi, p, log_det)

        # lambda_im: ramp 0.01 → 5.0 over first 60% of training
        # Higher ceiling (was 0.5) because Var(Im S) needs more pressure.
        t         = epoch / N
        lambda_im = 0.01 + 4.99 * min(1.0, max(0.0, (t - 0.10) / 0.50))
        lambda_J  = 0.01

        loss = var_Feff + lambda_im * var_imS + lambda_J * var_logJ

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sched.step(epoch)   # warm-restart scheduler takes epoch as arg

        with torch.no_grad():
            log_w = -S.real.detach() + log_det.detach()
            ess   = effective_sample_size(log_w)

        return {
            'var_Feff':  var_Feff.item(),
            'var_imS':   var_imS.item(),
            'var_logJ':  var_logJ.item(),
            'ess':       ess,
            'lr':        self.opt.param_groups[0]['lr'],
            'g4_sample': p.g4,
            'mu_sample': p.mu,
        }

    def train(self, n_epochs: int):
        keys = ('var_Feff','var_imS','var_logJ','ess','lr','g4_sample','mu_sample')
        h    = {k: [] for k in keys}
        T0   = self.cfg.get('restart_period', 2000)

        print(f"Training v6  |  Var(F_eff) + λ_im·Var(Im S) + λ_J·Var(log det)")
        print(f"  Randomised params | CosineWarmRestarts T_0={T0}")
        print(f"  L={self.cfg['L']}  batch={self.cfg['batch_size']}"
              f"  epochs={n_epochs}  lr={self.cfg['lr']:.1e}")

        for e in range(n_epochs):
            m = self.step(e)
            for k in keys: h[k].append(m[k])
            if e % 100 == 0:
                print(f"Ep {e:5d}: Var(F)={m['var_Feff']:.2e}"
                      f"  Var(ImS)={m['var_imS']:.2e}"
                      f"  ESS={m['ess']:.3f}"
                      f"  LR={m['lr']:.2e}"
                      f"  g4={m['g4_sample']:.3f}")
        return h


# ─────────────────────────────────────────────────────────────────────────────

def train(CONFIG: dict):
    t = Trainer(CONFIG)
    h = t.train(CONFIG['n_epochs'])
    torch.save(
        {'model': t.model.state_dict(), 'config': CONFIG, 'history': h},
        CONFIG['data_dir'] / 'universal_thimble_model.pt',
    )
    plot_training(h, CONFIG['data_dir'] / 'conv_training.png')