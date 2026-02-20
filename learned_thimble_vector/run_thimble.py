"""
run — unified Euclidean + Minkowski thimble training entry point.

New CONFIG keys for the Minkowski training phase:

  train_lambda_M   float  Maximum weight for the Minkowski loss term
                          λ_M · Var(Im S_M).  Suggested range: 1.0–5.0.
                          Higher values push harder on the sign problem but
                          can destabilise training if set too large before
                          the Euclidean phase has converged.  Default: 2.0.

  mink_ramp_start  float  Training fraction at which λ_M begins ramping up.
                          Before this point λ_M = 0 and no Minkowski gradient
                          is computed.  Default: 0.70 (after 70% of epochs).

  mink_ramp_end    float  Training fraction at which λ_M reaches its maximum
                          and holds constant for the rest of training.
                          Default: 0.90.

The Euclidean loss keys (train_m2, train_g4, etc.) are unchanged.
"""

import torch
from pathlib import Path
import os
from train import train
from validation import run_all_validations

CONFIG = {
    # ── Lattice ───────────────────────────────────────────────────────────────
    'L':                  10,
    'dx':                 1.0,
    'batch_size':         32,
    'thimble_hidden_dim': 96,

    # ── Training schedule ─────────────────────────────────────────────────────
    'n_epochs':           12000,
    'lr':                 2e-4,
    'restart_period':     2000,   # CosineAnnealingWarmRestarts T_0

    # ── Euclidean parameter ranges ────────────────────────────────────────────
    'train_m2':           5.0,    # centre of m2 range (actual range [0.5, 2.0])
    'train_kin':          5.0,
    'train_g3':           3.0,    # set nonzero for Z2-breaking theories
    'train_g4':           4.0,    # maximum g4 in training distribution
    'train_g6':           3.0,    # set nonzero for UV-stable theories
    'train_mu':           3.0,    # set 0.3 to train on sign problem

    # ── Minkowski training phase ──────────────────────────────────────────────
    # λ_M ramps from 0 → train_lambda_M over epochs
    # [mink_ramp_start * n_epochs,  mink_ramp_end * n_epochs]
    # then holds at train_lambda_M for the final 10% of training.
    'train_lambda_M':     2.0,    # max weight for λ_M · Var(Im S_M)
    'mink_ramp_start':    0.70,   # start ramp at 70% of training
    'mink_ramp_end':      0.90,   # reach max λ_M at 90% of training

    # ── I/O ───────────────────────────────────────────────────────────────────
    'device':             'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir':           Path(__file__).parent.resolve() / 'data',
}

os.makedirs(CONFIG['data_dir'], exist_ok=True)


def main():
    print("=" * 65)
    print("PHYSICS-INFORMED THIMBLE  v7  (Unified Euclidean + Minkowski)")
    print("  Loss (early)  : Var(F_eff) + λ_im·Var(Im S_E) + λ_J·Var(log det)")
    print("  Loss (late)   : + λ_M·Var(Im S_M)  [Minkowski sign problem]")
    print(f"  Mink. ramp    : {CONFIG['mink_ramp_start']*100:.0f}%"
          f"–{CONFIG['mink_ramp_end']*100:.0f}% of training "
          f"(λ_M → {CONFIG['train_lambda_M']})")
    print("  Training      : randomised Lagrangian params per step")
    print("  LR            : CosineAnnealingWarmRestarts")
    print("  Network       : position-space ResNet + GroupNorm + exact log-det")
    print("=" * 65 + "\n")
    train(CONFIG)
    print("\n" + "=" * 65)
    run_all_validations(CONFIG)


if __name__ == "__main__":
    main()