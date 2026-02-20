"""
Training — unified Euclidean + Minkowski thimble training.

Changes vs v7
-------------
- model.forward() now receives the training fraction `t` so the deformation
  clamp scale can grow with training (clamp_scale_start → clamp_scale_end).
- Two new CONFIG keys:
    clamp_scale_start  float  Initial log_sigma clamp (default 0.3)
    clamp_scale_end    float  Final log_sigma clamp   (default 0.7)
- temporal_kernel_size is forwarded from CONFIG to the model constructor
  (default 3; use 5 for larger-L lattices where mu correlations are longer).

Everything else is identical to v7.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from thimble import LagrangianParams, PhysicsInformedThimble, effective_sample_size


# ─────────────────────────────────────────────────────────────────────────────
# Randomised Lagrangian sampler  (unchanged from v7)
# ─────────────────────────────────────────────────────────────────────────────

def random_params(epoch: int, cfg: dict) -> LagrangianParams:
    N = cfg['n_epochs']
    t = epoch / N

    m2 = float(np.random.uniform(0.5, 2.0))

    kinetic_coeff_max = cfg.get('train_kin', 1.0) * min(1.0, max(0.0, (t - 0.7) / 0.3))
    kinetic_coeff = float(np.random.uniform(0.05, kinetic_coeff_max)) \
                    if kinetic_coeff_max > 0.1 else 0.1

    g4_max = cfg.get('train_g4', 0.5) * min(1.0, max(0.0, (t - 0.15) / 0.35))
    g4 = float(np.random.uniform(0.0, g4_max)) if g4_max > 0 else 0.0

    g3_max = cfg.get('train_g3', 0.0) * min(1.0, max(0.0, (t - 0.5) / 0.3))
    g3 = float(np.random.uniform(-g3_max, g3_max)) if g3_max > 0 else 0.0

    g6_max = cfg.get('train_g6', 0.0) * min(1.0, max(0.0, (t - 0.5) / 0.3))
    g6 = float(np.random.uniform(0.0, g6_max)) if g6_max > 0 else 0.0

    # mu ramp: starts earlier than before (mu_ramp_start, default 0.30)
    # so the network has more time to learn finite-mu geometry before the
    # Minkowski phase turns on at mink_ramp_start.
    mu_ramp_start = cfg.get('mu_ramp_start', 0.30)
    mu_max = cfg.get('train_mu', 0.0) * \
             min(1.0, max(0.0, (t - mu_ramp_start) / 0.40))
    mu = float(np.random.uniform(0.0, mu_max)) if mu_max > 0 else 0.0

    box_coeff_max = cfg.get('train_box_coeff', 0.0) * \
                    min(1.0, max(0.0, (t - 0.7) / 0.3))
    box_coeff = float(np.random.uniform(0.0, box_coeff_max)) \
                if box_coeff_max > 0 else 0.0

    deriv_interact_max = cfg.get('train_deriv_interact', 0.0) * \
                         min(1.0, max(0.0, (t - 0.7) / 0.3))
    deriv_interact = float(np.random.uniform(0.0, deriv_interact_max)) \
                     if deriv_interact_max > 0 else 0.0

    return LagrangianParams(
        m2=m2, kinetic_coeff=kinetic_coeff,
        g3=g3, g4=g4, g6=g6,
        box_coeff=box_coeff, deriv_interact=deriv_interact, mu=mu,
    )


def _mink_lambda(t: float, cfg: dict) -> float:
    start   = cfg.get('mink_ramp_start',  0.70)
    end     = cfg.get('mink_ramp_end',    0.90)
    lam_max = cfg.get('train_lambda_M',   2.0)

    if t < start:
        return 0.0
    if t >= end:
        return lam_max
    return lam_max * (t - start) / (end - start)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history: dict, save_path):
    keys   = ['var_Feff', 'var_imS_E', 'var_imS_M', 'var_logJ', 'ess', 'lambda_M']
    labels = [
        'Var(F_eff) [primary]',
        'Var(Im S_E) [Euclidean thimble]',
        'Var(Im S_M) [Minkowski sign problem]',
        'Var(log det) [stability]',
        'ESS training (fraction)',
        'λ_M (Minkowski weight)',
    ]
    logy   = [True, True, True, True, False, False]
    colors = ['k', 'm', 'darkorange', 'teal', 'g', 'r']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep = np.arange(len(history['var_Feff']))
    for ax, key, lbl, col, do_log in zip(axes.flat, keys, labels, colors, logy):
        if key not in history or not history[key]:
            continue
        vals = np.array(history[key], dtype=float)
        fn   = ax.semilogy if do_log else ax.plot
        fn(ep, np.clip(vals, 1e-15, None) if do_log else vals, color=col, alpha=0.6)

        if 'lambda_M' in history:
            lm = np.array(history['lambda_M'])
            onset = np.argmax(lm > 0)
            if onset > 0:
                ax.axvline(onset, color='darkorange', ls=':', lw=1.2,
                           label='Mink. onset')

        if key == 'ess':
            ax.axhline(0.1, color='r', ls='--', lw=1.5, label='10% floor')
            ax.set_ylim(0, 1)
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Unified Euclidean + Minkowski Thimble Training', fontsize=13)
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

        T0 = cfg.get('restart_period', 2000)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=T0, T_mult=2, eta_min=cfg['lr'] * 5e-3)

    def step(self, epoch: int):
        self.opt.zero_grad()

        p   = random_params(epoch, self.cfg)
        N   = self.cfg['n_epochs']
        L   = self.cfg['L']
        dev = self.cfg['device']
        t   = epoch / N   # training fraction

        z = torch.randn(self.cfg['batch_size'], L, L, L, L, device=dev)

        # ── Forward pass — pass t so clamp scale can grow ─────────────────
        phi, log_det, scale = self.model(z, p, return_scale=True, t=t)

        # ── Loss weights ─────────────────────────────────────────────────────
        lambda_im = 0.01 + 4.99 * min(1.0, max(0.0, (t - 0.10) / 0.50))
        lambda_J = 0.01 + 0.49 * min(1.0, max(0.0, (t - 0.05) / 0.45))
        lambda_M  = _mink_lambda(t, self.cfg)

        # ── Combined loss ─────────────────────────────────────────────────────
        result = self.model.compute_combined_loss(
            phi, p, log_det,
            lambda_im=lambda_im,
            lambda_J=lambda_J,
            lambda_M=lambda_M,
        )

        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sched.step(epoch)

        with torch.no_grad():
            log_w = -result['S_E'].real.detach() + log_det.detach()
            ess   = effective_sample_size(log_w)

        return {
            'var_Feff':  result['var_Feff'].item(),
            'var_imS_E': result['var_imS_E'].item(),
            'var_imS_M': result['var_imS_M'].item(),
            'var_logJ':  result['var_logJ'].item(),
            'ess':       ess,
            'lr':        self.opt.param_groups[0]['lr'],
            'lambda_M':  lambda_M,
            'g4_sample': p.g4,
            'mu_sample': p.mu,
        }

    def train(self, n_epochs: int):
        keys = ('var_Feff', 'var_imS_E', 'var_imS_M', 'var_logJ',
                'ess', 'lr', 'lambda_M', 'g4_sample', 'mu_sample')
        h    = {k: [] for k in keys}
        T0   = self.cfg.get('restart_period', 2000)

        mink_start_ep = int(self.cfg.get('mink_ramp_start', 0.70) * n_epochs)
        mink_end_ep   = int(self.cfg.get('mink_ramp_end',   0.90) * n_epochs)
        mu_ramp_start = self.cfg.get('mu_ramp_start', 0.30)

        print(f"Unified training  |  Euclidean + Minkowski thimble  (dual-path v8)")
        print(f"  Loss     : Var(F_eff) + λ_im·Var(Im S_E) + λ_J·Var(log det)"
              f" + λ_M·Var(Im S_M)")
        print(f"  λ_M ramp : epochs {mink_start_ep}–{mink_end_ep}"
              f"  (0 → {self.cfg.get('train_lambda_M', 2.0):.1f})")
        print(f"  μ ramp   : starts at t={mu_ramp_start:.2f}"
              f"  (earlier than Mink. ramp to warm-start temporal path)")
        print(f"  Clamp    : {self.cfg.get('clamp_scale_start', 0.3):.2f}"
              f" → {self.cfg.get('clamp_scale_end', 0.7):.2f}  (log_sigma range)")
        print(f"  TempKern : {self.cfg.get('temporal_kernel_size', 3)}")
        print(f"  LR       : CosineWarmRestarts T_0={T0}")
        print(f"  L={self.cfg['L']}  batch={self.cfg['batch_size']}"
              f"  epochs={n_epochs}  lr={self.cfg['lr']:.1e}\n")

        for e in range(n_epochs):
            m = self.step(e)
            for k in keys:
                h[k].append(m[k])

            if e % 100 == 0:
                mink_active = "🔥 Mink" if m['lambda_M'] > 0 else "   Eucl"
                print(f"Ep {e:5d} {mink_active} | "
                      f"Var(F)={m['var_Feff']:.2e}  "
                      f"Im(E)={m['var_imS_E']:.2e}  "
                      f"Im(M)={m['var_imS_M']:.2e}  "
                      f"ESS={m['ess']:.3f}  "
                      f"λ_M={m['lambda_M']:.3f}  "
                      f"LR={m['lr']:.2e}  "
                      f"g4={m['g4_sample']:.3f}  μ={m['mu_sample']:.3f}")
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