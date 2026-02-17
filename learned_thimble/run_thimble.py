"""
Ultimate Thimble Training and Validation

This script trains the model with aggressive scale regularization
and validates with OOM fixes built into the code.
"""

import torch
from pathlib import Path
import os

from train import train
from validation import run_all_validations


CONFIG = {
    'L': 8,
    'dx': 1.0,
    'batch_size': 32,
    'thimble_hidden_dim': 32,
    'n_epochs': 10000,
    'lr': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': Path(__file__).parent.resolve()/'data',
    'Naverage': 100
}

os.makedirs(CONFIG['data_dir'], exist_ok=True)


def main():
    print("="*70)
    print("ULTIMATE THIMBLE TRAINING")
    print("="*70 + "\n")
    
    # TRAIN
    print("STARTING TRAINING...")
    train(CONFIG)
    
    # VALIDATE
    print("\n" + "="*70)
    print("STARTING VALIDATION")
    print("="*70)
    
    run_all_validations(CONFIG)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()