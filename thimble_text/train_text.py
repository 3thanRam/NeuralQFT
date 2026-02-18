import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from pathlib import Path
import sys
import sys
# Ensure physics_lm is importable
try:
    from physics_lm import LagrangianLanguageModel
except ImportError:
    print("❌ Could not import 'physics_lm.py'. Make sure it is in the same directory.")
    sys.exit(1)

project_dir=Path(__file__).parent.parent.resolve()
sys.path.insert(0,str(project_dir))
sys.path.insert(0,str(project_dir/'learned_thimble'))


from learned_thimble.thimble import PhysicsInformedThimble, GeneralAction, PARAM_DIM
from learned_thimble.run_thimble import CONFIG 


from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset
import os

def get_or_train_tokenizer(vocab_size=512, save_path="physics_bpe.json"):
    if os.path.exists(save_path):
        print(f"✅ Loading existing tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    print(f"🔨 Training new BPE tokenizer (vocab_size={vocab_size})...")
    # Load raw text from wikitext
    raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Initialize BPE
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    )
    
    # We pass the generator of strings directly to the trainer
    tokenizer.train_from_iterator(raw_data['text'], trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


class BPEDataset(Dataset):
    def __init__(self, text_list, block_size, tokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        
        # Join list of strings and encode
        full_text = "\n".join(text_list)
        encoded = self.tokenizer.encode(full_text)
        self.data = torch.tensor(encoded.ids, dtype=torch.long)
        
        print(f"✅ BPE Data ready. Total tokens: {len(self.data)}")
        print(f"   Vocab size: {self.vocab_size}")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[-1]
        return x, y

    def decode(self, indices):
        if torch.is_tensor(indices):
            indices = indices.tolist()
        return self.tokenizer.decode(indices)

def get_wikitext_bpe_data(tokenizer, block_size=32, split="train"):
    print(f"Fetching Wikitext-2 ({split}) for BPE...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    # Filter out empty strings to keep it clean
    text_list = [t for t in dataset['text'] if len(t.strip()) > 0]
    return BPEDataset(text_list, block_size, tokenizer)


# ============================================================================
# 2. Generation Tool
# ============================================================================

def generate_text(model, dataset, device, start_str="The ", length=30):
    model.eval()
    
    # Encode start string using BPE
    encoding = dataset.tokenizer.encode(start_str)
    context_idx = encoding.ids
    context = torch.tensor(context_idx, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = list(context_idx)
    last_params = None

    for _ in range(length):
        input_seq = context[:, -dataset.block_size:]
        
        with torch.no_grad():
            logits, params = model(input_seq)
            probs = torch.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            context = torch.cat((context, next_token), dim=1)
            generated.append(next_token.item())
            last_params = params[0]

    # Use the BPE decoder to turn IDs back into text
    text = dataset.decode(generated)
    return text, last_params

# ============================================================================
# 3. Main Training Loop
# ============================================================================




def train():
    # --- Configuration ---
    BLOCK_SIZE = 32         # Context length
    BATCH_SIZE = 64         # Small batch size due to heavy Path Integral
    EMBED_DIM  = 256
    HIDDEN_DIM = 512
    LR         = 1e-4
    EPOCHS     = 3          # Wikitext is larger, 3 epochs is plenty for a demo
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    VOCAB_SIZE = 4096 

    # Path to your pre-trained physics model
    # (Update this path if your folder structure is different)
    THIMBLE_PATH = CONFIG['data_dir'] /"universal_thimble_model.pt"
    

    if not THIMBLE_PATH.exists():
        print(f"❌ Error: {THIMBLE_PATH} not found.")
        print("   Please run 'train.py' first to learn the QFT manifold.")
        return

    tokenizer = get_or_train_tokenizer(vocab_size=VOCAB_SIZE)
    
    # 2. Setup Data with BPE
    full_dataset = get_wikitext_bpe_data(tokenizer, BLOCK_SIZE, split="train")
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Setup Model (Now with 512 output classes)
    model = LagrangianLanguageModel(
        vocab_size=full_dataset.vocab_size, # This is now 512
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