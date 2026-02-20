"""
run — unified Euclidean + Minkowski thimble training entry point.

New CONFIG keys added in v8 (dual-path temporal conditioner):

  temporal_kernel_size  int    Kernel size for the 1D temporal Conv blocks.
                               Must be odd.  3 = each site sees its two
                               immediate temporal neighbours (the sites
                               directly coupled by e^{±μ} hops).  Use 5
                               for larger lattices (L≥16) where mu
                               induces longer-range temporal correlations.
                               Default: 3.

  clamp_scale_start     float  Initial soft-clamp on log_sigma.
                               At epoch 0 the deformation is near-identity.
                               Default: 0.3  (same as v7 throughout).

  clamp_scale_end       float  Final soft-clamp on log_sigma.
                               Reached at epoch n_epochs.  Larger values
                               allow more aggressive contour deformation
                               at the cost of higher Jacobian variance —
                               watch var_logJ in the training plots.
                               Default: 0.7.

  mu_ramp_start         float  Training fraction at which μ begins ramping
                               into the randomised Lagrangian sampler.
                               Set earlier than mink_ramp_start so the
                               temporal path has time to learn finite-μ
                               geometry before the Minkowski phase turns on.
                               Default: 0.30.

Unchanged keys from v7:

  train_lambda_M        float  Max weight for λ_M · Var(Im S_M).
  mink_ramp_start       float  Training fraction where λ_M begins.
  mink_ramp_end         float  Training fraction where λ_M reaches max.
  train_m2, train_g4, train_g6, train_mu, train_kin  — param ranges.
"""

import torch
from pathlib import Path
import os
from train import train
from validation import run_all_validations

CONFIG = {
    # ── Lattice ───────────────────────────────────────────────────────────────
    'L':                     8,
    'dx':                    1.0,
    'batch_size':            32,
    'thimble_hidden_dim':    64,

    # ── Temporal path (NEW in v8) ─────────────────────────────────────────────
    'temporal_kernel_size':  3,    # 3 = nearest-neighbour temporal conv
                                   # matches the e^{±μ} hop structure exactly
                                   # increase to 5 for L >= 16

    # ── Deformation clamp schedule (NEW in v8) ────────────────────────────────
    'clamp_scale_start':     0.3,  # initial log_sigma clamp (near-identity)
    'clamp_scale_end':       0.4,  # final clamp (more expressive late in training)

    # ── Training schedule ─────────────────────────────────────────────────────
    'n_epochs':              12000,
    'lr':                    2e-4,
    'restart_period':        3000,

    # ── Euclidean parameter ranges ────────────────────────────────────────────
    'train_m2':              5.0,
    'train_kin':             5.0,
    'train_g3':              3.0,
    'train_g4':              4.0,
    'train_g6':              3.0,
    'train_mu':              3.0,

    # ── μ ramp schedule (NEW in v8) ───────────────────────────────────────────
    # Starts earlier (0.30) so the temporal path warms up before the
    # Minkowski phase turns on at mink_ramp_start (0.70).
    'mu_ramp_start':         0.30,

    # ── Minkowski training phase ──────────────────────────────────────────────
    'train_lambda_M':        2.0,
    'mink_ramp_start':       0.70,
    'mink_ramp_end':         0.90,

    # ── I/O ───────────────────────────────────────────────────────────────────
    'device':                'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir':              Path(__file__).parent.resolve() / 'data',
}

os.makedirs(CONFIG['data_dir'], exist_ok=True)


def main():
    print("=" * 70)
    print("PHYSICS-INFORMED THIMBLE  v8  (Dual-Path Temporal Conditioner)")
    print("  Architecture  : SpatialPath (3D conv) + TemporalPath (1D conv) fused")
    print(f"  Temporal kern : {CONFIG['temporal_kernel_size']}")
    print(f"  Clamp schedule: {CONFIG['clamp_scale_start']} → {CONFIG['clamp_scale_end']}")
    print(f"  μ ramp start  : t={CONFIG['mu_ramp_start']:.2f}"
          f"  (before Mink. phase at t={CONFIG['mink_ramp_start']:.2f})")
    print("  Loss (early)  : Var(F_eff) + λ_im·Var(Im S_E) + λ_J·Var(log det)")
    print("  Loss (late)   : + λ_M·Var(Im S_M)  [Minkowski sign problem]")
    print(f"  Mink. ramp    : {CONFIG['mink_ramp_start']*100:.0f}%"
          f"–{CONFIG['mink_ramp_end']*100:.0f}% of training "
          f"(λ_M → {CONFIG['train_lambda_M']})")
    print("  Training      : randomised Lagrangian params per step")
    print("  LR            : CosineAnnealingWarmRestarts")
    print("=" * 70 + "\n")
    train(CONFIG)
    print("\n" + "=" * 70)
    run_all_validations(CONFIG)


if __name__ == "__main__":
    main()