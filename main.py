import torch

from pathlib import Path
import os

from experiment import QFTExperiment
# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'L': 12,
    'dim': 4,
    'dx': 0.5,
    'M': 1.0,
    'g': 0.01,
    'batch_size': 64,    # Smaller batch for stability
    'n_layers': 8,       
    'hidden_dim': 48,    
    'pretrain_epochs': 500, 
    'fine_tune_epochs': 1000, 
    'lr': 1e-3,              
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': Path(__file__).parent.resolve()/'data',
}

os.makedirs(CONFIG['data_dir'], exist_ok=True)


if __name__ == "__main__":
    torch.manual_seed(42)
    QFTExperiment(CONFIG).run()