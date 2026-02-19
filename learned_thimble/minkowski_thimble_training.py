# minkowski_thimble_training.py
# Add this to learned_thimble/ alongside the existing training code.
#
# The key change: the thimble's training objective must include
# Im(S_M) minimization, not just Euclidean action optimization.
#
# Euclidean thimble training objective:
#   L = Var(Im(S_E)) + KL(weights, uniform)
#   Im(S_E) ~ 0 by construction since S_E is real on real axis
#   → thimble had nothing to solve, so it never learned Minkowski contours
#
# Minkowski thimble training objective:
#   L = Var(Im(S_M)) + KL(weights, uniform)
#   Im(S_M) is large and parameter-dependent → thimble must work

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from thimble import (PhysicsInformedThimble, LagrangianParams,
                     GeneralAction, effective_sample_size, PARAM_DIM)
from run_thimble import CONFIG


def minkowski_action(phi: torch.Tensor, lp: LagrangianParams,
                     dx: float = 1.0) -> torch.Tensor:
    """
    Minkowski action S_M with correct sign conventions.
    phi: [B, L, L, L, L] complex
    Returns S_M: [B] complex

    S_M = sum_x [ ½(d_t phi)^2 - ½m² phi^2 - g4/4! phi^4 - ½ mu_hop ]

    The thimble is trained to make Im(S_M) ~ 0 on the deformed contour.
    This is fundamentally different from the Euclidean case where
    Im(S_E) = 0 trivially on the real axis.
    """
    # Kinetic: forward difference on time axis (dim=1)
    dphi    = torch.roll(phi, -1, dims=1) - phi
    kinetic = 0.5 * (dphi ** 2).sum(dim=(1, 2, 3, 4))      # [B]

    m2  = lp.m2
    g4  = lp.g4
    g6  = lp.g6
    mu  = lp.mu

    # Minkowski sign conventions: V appears with minus sign
    mass  = -0.5        * m2 * (phi ** 2).sum(dim=(1, 2, 3, 4))
    phi4  = -(g4 / 24.0)     * (phi ** 4).sum(dim=(1, 2, 3, 4))
    phi6  = -(g6 / 720.0)    * (phi ** 6).sum(dim=(1, 2, 3, 4))

    # Chemical potential hop
    hop   = (torch.roll(phi, -1, dims=1) * phi +
             torch.roll(phi,  1, dims=1) * phi).sum(dim=(1, 2, 3, 4))
    chem  = -0.5 * mu * hop

    return kinetic + mass + phi4 + phi6 + chem              # [B] complex


def train_minkowski_thimble(
        n_epochs: int = 50,
        n_samples: int = 256,
        lr: float = 1e-4,
        save_path: Path = None,
        resume_euclidean: bool = True,
):
    """
    Train the thimble to solve the Minkowski sign problem.

    resume_euclidean: if True, initialise from the pre-trained Euclidean
    thimble. This gives a warm start — the network already knows how to
    deform contours, it just needs to learn the Minkowski geometry.
    Starting from scratch would take much longer.
    """
    device     = CONFIG['device']
    L          = CONFIG['L']
    dx         = CONFIG['dx']
    save_path  = save_path or (CONFIG['data_dir'] / 'minkowski_thimble.pt')

    thimble = PhysicsInformedThimble(CONFIG).to(device)

    if resume_euclidean:
        eucl_path = CONFIG['data_dir'] / 'universal_thimble_model.pt'
        if eucl_path.exists():
            ckpt = torch.load(eucl_path, map_location=device,
                              weights_only=False)
            thimble.load_state_dict(ckpt['model'])
            print(f"✅ Warm start from Euclidean thimble: {eucl_path}")
        else:
            print("⚠️  Euclidean thimble not found, training from scratch")

    optimizer = torch.optim.Adam(thimble.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * 20, eta_min=lr * 0.1)

    # Training distribution — same as Euclidean but with mu included
    param_ranges = {
        'm2':  (0.3, 1.7),
        'g4':  (0.0, 0.6),
        'g6':  (0.0, 0.3),
        'mu':  (0.0, 0.28),  # ← Euclidean training used mu but not S_M
    }

    print(f"\nTraining Minkowski thimble for {n_epochs} epochs")
    print(f"Target: |Im(S_M)| < 1.0, ESS > 0.1\n")

    best_ess = 0.0

    for epoch in range(n_epochs):
        epoch_im_s  = []
        epoch_ess   = []
        epoch_loss  = []

        # Sample random Lagrangian params each step
        for step in range(20):
            rng = np.random
            lp  = LagrangianParams(
                m2            = rng.uniform(*param_ranges['m2']),
                kinetic_coeff = 1.0,
                g3            = 0.0,
                g4            = rng.uniform(*param_ranges['g4']),
                g6            = rng.uniform(*param_ranges['g6']),
                box_coeff     = 0.0,
                deriv_interact= 0.0,
                mu            = rng.uniform(*param_ranges['mu']),
            )

            z        = torch.randn(n_samples, L, L, L, L, device=device)
            phi, log_det = thimble(z, lp)

            # ── Minkowski action on deformed contour ─────────────────
            S_M = minkowski_action(phi, lp, dx)                # [B] complex

            # ── Training objective ────────────────────────────────────
            # 1. Minimise Im(S_M): the thimble should find the contour
            #    where the Minkowski phase is stationary
            im_loss = S_M.imag.pow(2).mean()

            # 2. Importance weight variance (same as Euclidean training)
            #    Maximise ESS by minimising variance of log weights
            log_w   = -S_M.real + log_det
            log_w_c = log_w - log_w.mean()
            var_loss = log_w_c.pow(2).mean()

            # 3. Prevent the contour from collapsing to trivial solution
            #    (phi = 0 everywhere minimises Im(S_M) trivially)
            #    Penalise if field variance is too small
            field_var    = phi.real.var()
            collapse_pen = torch.clamp(0.1 - field_var, min=0.0) * 10.0

            loss = im_loss + 0.1 * var_loss + collapse_pen

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(thimble.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                ess = effective_sample_size(log_w.detach())
                epoch_im_s.append(S_M.imag.abs().mean().item())
                epoch_ess.append(ess)
                epoch_loss.append(loss.item())

        mean_im_s = np.mean(epoch_im_s)
        mean_ess  = np.mean(epoch_ess)
        mean_loss = np.mean(epoch_loss)

        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Loss={mean_loss:.4f} | "
              f"|Im(S_M)|={mean_im_s:.4f} | "
              f"ESS={mean_ess:.4f}"
              + (" ← best" if mean_ess > best_ess else ""))

        if mean_ess > best_ess:
            best_ess = mean_ess
            torch.save({
                'epoch':  epoch,
                'model':  thimble.state_dict(),
                'ess':    best_ess,
                'im_s':   mean_im_s,
                'config': CONFIG,
            }, save_path)

        # Early stopping: if we've solved the sign problem well enough
        if mean_im_s < 0.5 and mean_ess > 0.15:
            print(f"\n✅ Sign problem solved at epoch {epoch+1}")
            print(f"   |Im(S_M)|={mean_im_s:.4f}  ESS={mean_ess:.4f}")
            break

    print(f"\nBest ESS: {best_ess:.4f}")
    print(f"Saved to: {save_path}")
    print(f"\nNext step: update THIMBLE_PATH in train_minkowski.py to:")
    print(f"  {save_path}")

    return save_path


if __name__ == '__main__':
    train_minkowski_thimble(
        n_epochs        = 100,
        n_samples       = 256,
        lr              = 1e-4,
        resume_euclidean= True,
    )