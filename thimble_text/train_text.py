import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from pathlib import Path
import sys

# Ensure physics_lm is importable
try:
    from physics_lm import LagrangianLanguageModel
except ImportError:
    print("❌ Could not import 'physics_lm.py'. Make sure it is in the same directory.")
    sys.exit(1)

# ============================================================================
# 1. Data Pipeline (Wikitext-2 Character Level)
# ============================================================================

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        
        # Create character vocabulary from the text
        # We limit to unique chars found in dataset
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        # Convert full text to integers
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        print(f"✅ Data loaded. Length: {len(text)} chars.")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Sample vocab: {''.join(chars[30:50])} ...")

    def __len__(self):
        # We start at 0 and go up to len - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of text
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # Input: tokens 0..N-1
        # Target: tokens 1..N
        x = chunk[:-1]
        y = chunk[-1] # predicting the NEXT token only (Many-to-One)
        return x, y

    def decode(self, indices):
        return ''.join([self.itos[int(i)] for i in indices])

def get_wikitext_data(block_size=32, split="train"):
    print(f"Downloading Wikitext-2 ({split})...")
    # wikitext-2-raw-v1 keeps punctuation and case, which is better for char-level
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Concatenate all lines into one massive string
    text = "\n".join(dataset['text'])
    return CharDataset(text, block_size)

# ============================================================================
# 2. Generation Tool
# ============================================================================

def generate_text(model, dataset, device, start_str="The ", length=50):
    model.eval()
    
    # Encode start string
    context_idx = [dataset.stoi.get(c, 0) for c in start_str]
    context = torch.tensor(context_idx, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = list(context_idx)
    last_params = None

    for _ in range(length):
        # Crop context to block_size if needed
        input_seq = context[:, -dataset.block_size:]
        
        with torch.no_grad():
            logits, params = model(input_seq)
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=1)
            # Use multinomial sampling for variety
            next_token = torch.multinomial(probs, num_samples=1)
            
            context = torch.cat((context, next_token), dim=1)
            generated.append(next_token.item())
            last_params = params[0]

    text = dataset.decode(generated)
    return text, last_params

# ============================================================================
# 3. Main Training Loop
# ============================================================================
import sys

sys.path.insert(0,'/home/ethan/Documents/Code/github/NeuralQFT')
sys.path.insert(0,'/home/ethan/Documents/Code/github/NeuralQFT/learned_thimble')

# Import your existing modules
from learned_thimble.thimble import PhysicsInformedThimble, GeneralAction, PARAM_DIM
from learned_thimble.run_thimble import CONFIG 


def train():
    # --- Configuration ---
    BLOCK_SIZE = 32         # Context length
    BATCH_SIZE = 64         # Small batch size due to heavy Path Integral
    EMBED_DIM  = 64
    HIDDEN_DIM = 128
    LR         = 1e-3
    EPOCHS     = 3          # Wikitext is larger, 3 epochs is plenty for a demo
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Path to your pre-trained physics model
    # (Update this path if your folder structure is different)
    THIMBLE_PATH = CONFIG['data_dir'] /"universal_thimble_model.pt"
    

    if not THIMBLE_PATH.exists():
        print(f"❌ Error: {THIMBLE_PATH} not found.")
        print("   Please run 'train.py' first to learn the QFT manifold.")
        return

    # --- Setup Data ---
    # We use a subset/stride to make epochs faster for demonstration
    # Wikitext is ~2MB of text.
    full_dataset = get_wikitext_data(BLOCK_SIZE, split="train")
    
    # Create DataLoader
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Setup Model ---
    print("\nInitializing Physics-Informed LM...")
    model = LagrangianLanguageModel(
        vocab_size=full_dataset.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        thimble_model_path=THIMBLE_PATH,
        config=CONFIG
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {DEVICE}...")
    print(f"   Architecture: Text -> LSTM -> Lagrangian -> Thimble(phi^4) -> Observables -> Next Char")

    # --- Loop ---
    model.train()
    step = 0
    
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: 
            # 1. LSTM predicts Lagrangian parameters
            # 2. Thimble solves Path Integral for those parameters
            # 3. Observables predict next char
            logits, phys_params = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Clip gradients: The exponential in the path integral can cause spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            step += 1
            
            # Logging
            if batch_idx % 50 == 0:
                p = phys_params[0].detach().cpu()
                # Monitor the physics parameters. 
                # If they saturate (e.g. g4 stays at 0.0 or 1.0), the model is struggling.
                print(f"Step {step} | Loss: {loss.item():.4f}")
                print(f"   Lagrangian: m²={p[0]:.2f}  g4={p[3]:.2f}  μ={p[7]:.2f}")

            # Every 500 steps, generate a sample
            if batch_idx > 0 and batch_idx % 500 == 0:
                print("\n--- Generating Wikitext via QFT ---")
                sample, p = generate_text(model, full_dataset, DEVICE, start_str="The ", length=100)
                print(f"Sample: {sample}")
                print(f"Physics state at end: m²={p[0]:.2f}, Interaction={p[3]:.2f}")
                print("-----------------------------------\n")
                model.train()

    # Save
    save_path = "data/wikitext_physics_lm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train()