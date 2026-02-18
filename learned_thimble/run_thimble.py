"""
run
"""

import torch
from pathlib import Path
import os
from train import train
from validation import run_all_validations

CONFIG = {
    'L':                  8,
    'dx':                 1.0,
    'batch_size':         32,
    'thimble_hidden_dim': 32,
    'n_epochs':           12000,
    'lr':                 3e-4,
    'restart_period':     2000,   # CosineAnnealingWarmRestarts T_0
    'device':             'cuda' if torch.cuda.is_available() else 'cpu',
    # Parameter ranges for randomised training
    'train_m2':           2.0,    # centre of m2 range (actual range is [0.5, 2.0])
    'train_kin':          2.0,
    'train_g3':           0.6,    # set nonzero for Z2-breaking theories
    'train_g4':           1.0,    # maximum g4 in training distribution
    'train_g6':           0.6,    # set nonzero for UV-stable theories
    'train_mu':           0.6,    # set 0.3 to train on sign problem
    # I/O
    'data_dir':           Path(__file__).parent.resolve() / 'data',
}

os.makedirs(CONFIG['data_dir'], exist_ok=True)


def main():
    print("=" * 65)
    print("PHYSICS-INFORMED THIMBLE  v6")
    print("  Loss     : Var(F_eff) + λ_im·Var(Im S) + λ_J·Var(log det)")
    print("  Training : randomised Lagrangian params per step")
    print("  LR       : CosineAnnealingWarmRestarts")
    print("  Network  : position-space ResNet + GroupNorm + exact log-det")
    print("=" * 65 + "\n")
    train(CONFIG)
    print("\n" + "=" * 65)
    run_all_validations(CONFIG)


if __name__ == "__main__":
    main()