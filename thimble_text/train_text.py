"""
train_minkowski.py
==================
Training loop for MinkowskiFieldLM.

Loss terms
----------
1. CE loss      : standard next-token prediction (primary)
2. commit_loss  : VQ codebook commitment (keeps embeddings quantized)
3. sign_loss    : penalises large Im(S_M) — the sign problem severity
                  This term directly rewards the thimble for doing its job:
                  if Im(S_M) is large the thimble hasn't solved the sign
                  problem, so we penalise it.
4. ess_loss     : penalises low effective sample size
                  ESS < 0.05 means importance weights have collapsed —
                  the thimble contour is far from the true saddle point.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import numpy as np
from pathlib import Path
import sys

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent / 'learned_thimble'))

from run_thimble import CONFIG as THIMBLE_CONFIG
from physics_lm import MinkowskiFieldLM

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_CONFIG = {
    'vocab_size':    50257,
    'embed_dim':     256,
    'n_quanta':      512,
    'n_layers':      4,
    'block_size':    64,
    'batch_size':    16,    # small: thimble adds memory overhead
    'lr':            1e-4,
    'epochs':        3,
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
    'thimble_path':  THIMBLE_CONFIG['data_dir'] / 'minkowski_thimble.pt',
    'save_path':     THIMBLE_CONFIG['data_dir'] / 'minkowski_lm.pt',
    # Loss weights
    'w_commit':      0.1,
    'w_sign':        0.05,   # sign problem penalty weight
    'w_ess':         0.1,    # ESS penalty weight
    # ESS floor: below this we consider the thimble to have failed
    'ess_floor':     0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


class WikiDataset(Dataset):
    def __init__(self, tokenizer, block_size: int, split: str = 'train'):
        raw = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

        cleaned = []
        for t in raw['text']:
            t = t.strip()
            if not t:                               continue
            if t.startswith('='):                   continue
            if t.startswith('<'):                   continue
            if '@' in t:                            continue   # @-@ artifacts
            if len(t.split()) < 8:                  continue
            alpha = sum(c.isalpha() for c in t) / (len(t) + 1e-8)
            if alpha < 0.4:                         continue
            cleaned.append(t)

        full   = '\n'.join(cleaned)
        enc    = tokenizer(full, return_tensors='pt', truncation=False)
        self.data       = enc['input_ids'].squeeze(0)
        self.block_size = block_size
        print(f"✅ Tokens: {len(self.data):,}")

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        return self.data[idx: idx + self.block_size + 1]

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    cfg    = TRAIN_CONFIG
    device = cfg['device']

    if not cfg['thimble_path'].exists():
        print(f"❌  Thimble model not found at {cfg['thimble_path']}")
        print("    Run learned_thimble/run_thimble.py first.")
        return

    tokenizer = get_tokenizer()
    dataset   = WikiDataset(tokenizer, cfg['block_size'], split='train')
    loader    = DataLoader(dataset, batch_size=cfg['batch_size'],
                           shuffle=True, num_workers=2, pin_memory=True)

    model = MinkowskiFieldLM(
        vocab_size        = cfg['vocab_size'],
        embed_dim         = cfg['embed_dim'],
        n_quanta          = cfg['n_quanta'],
        n_layers          = cfg['n_layers'],
        thimble_model_path= cfg['thimble_path'],
        thimble_config    = THIMBLE_CONFIG,
    ).to(device)

    # Separate LR groups: action params need slower learning
    # so the thimble's pre-trained contour stays valid
    # Replace the optimizer block in train() with:

    # Collect embedding params (shared between field_embed and measure)
    embed_params = list(model.field_embed.parameters())

    # All other params excluding embedding and action
    other_params = (
        list(model.action_proj.parameters()) +
        list(model.propagator.parameters())
    )

    # Action params (slower LR)
    action_params_list = list(model.action.parameters())

    # Verify no overlap
    embed_ids  = {id(p) for p in embed_params}
    other_ids  = {id(p) for p in other_params}
    action_ids = {id(p) for p in action_params_list}

    assert not (embed_ids & other_ids),  "embed/other overlap"
    assert not (embed_ids & action_ids), "embed/action overlap"
    assert not (other_ids & action_ids), "other/action overlap"

    optimizer = torch.optim.AdamW([
        {'params': embed_params,        'lr': cfg['lr']},
        {'params': other_params,        'lr': cfg['lr']},
        {'params': action_params_list,  'lr': cfg['lr'] * 0.1,
         'weight_decay': 0.0},
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'],
        steps_per_epoch=len(loader), epochs=cfg['epochs'],
        pct_start=0.05,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*65}")
    print(f"MinkowskiFieldLM training")
    print(f"  embed_dim={cfg['embed_dim']}  n_quanta={cfg['n_quanta']}"
          f"  n_layers={cfg['n_layers']}")
    print(f"  batch={cfg['batch_size']}  block={cfg['block_size']}"
          f"  device={device}")
    print(f"{'='*65}\n")

    for epoch in range(cfg['epochs']):
        model.train()
        for step, chunk in enumerate(loader):
            x_in  = chunk[:, :-1].to(device)
            y_seq = chunk[:, 1:].to(device)

            optimizer.zero_grad()

            (logits, commit_loss, perplexity,
             quanta_ids, S_density, ess,
             action_params,param_reg) = model(x_in)

            # ── 1. Primary: next-token prediction ────────────────────
            ce_loss = criterion(
                logits.reshape(-1, cfg['vocab_size']),
                y_seq.reshape(-1))

            # ── 2. VQ commitment ─────────────────────────────────────
            vq_loss = commit_loss

            # ── 3. Sign problem penalty ───────────────────────────────
            # Penalise large variance in S_density across the sequence.
            # When the thimble is working well, Im(S) ~ 0 and S_density
            # is approximately constant (flat action landscape on thimble).
            # Large variance means the contour deformation is insufficient.
            sign_loss = S_density.var(dim=-1).mean()

            # ── 4. ESS penalty ────────────────────────────────────────
            # If ESS drops below floor, importance weights have collapsed
            ess_tensor = torch.tensor(ess, device=device, dtype=torch.float32)
            ess_loss   = torch.clamp(cfg['ess_floor'] - ess_tensor, min=0.0)

            loss = (ce_loss
                    + cfg['w_commit'] * vq_loss
                    + cfg['w_sign']   * sign_loss
                    + cfg['w_ess']    * ess_loss
                    + param_reg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                n_active = quanta_ids.unique().numel()
                p        = action_params
                print(
                    f"Ep {epoch+1} | Step {step:5d} | "
                    f"CE={ce_loss.item():.4f} | "
                    f"VQ={vq_loss.item():.4f} | "
                    f"SignVar={sign_loss.item():.4f} | "
                    f"ESS={ess:.3f} | "
                    f"Perp={perplexity.item():.0f}/{cfg['n_quanta']}\n"
                    f"   m²={p['m2']:.4f} g4={p['g4']:.4f} "
                    f"g6={p['g6']:.5f} μ={p['mu']:.4f} | "
                    f"Active quanta: {n_active}/{cfg['n_quanta']}"
                )

            if step > 0 and step % 500 == 0:
                _generate_sample(model, tokenizer, dataset, device, step)
                model.train()

        # Checkpoint after each epoch
        torch.save({
            'epoch':        epoch,
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'train_config': cfg,
        }, cfg['save_path'])
        print(f"\n✅ Checkpoint saved → {cfg['save_path']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Generation (called during training)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sample(model, tokenizer, dataset, device, step,
                     length=60, temperature=0.85, top_k=50):
    model.eval()
    print(f"\n--- Generation at step {step} ---")

    for start in ["The ", "In the ", "Scientists "]:
        text, stats = generate_text(
            model, tokenizer, device,
            start_str=start, length=length,
            block_size=TRAIN_CONFIG['block_size'],
            temperature=temperature, top_k=top_k)
        print(f"  [{start.strip()}] {text}")
        print(f"    ESS={stats['ess_mean']:.3f}  "
              f"SignVar={stats['sign_var_mean']:.3f}  "
              f"Switches={stats['quantum_switches']}/{length}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# generate_text  (importable for use outside training)
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, device,
                  start_str: str = "The ",
                  length: int = 80,
                  block_size: int = 64,
                  temperature: float = 0.85,
                  top_k: int = 50):
    """
    Autoregressive generation with per-step physics diagnostics.

    Returns
    -------
    text  : str   generated text
    stats : dict  physics diagnostics collected during generation
    """
    model.eval()
    enc      = tokenizer(start_str, return_tensors='pt')
    context  = enc['input_ids'].to(device)
    generated = context[0].tolist()

    ess_history        = []
    sign_var_history   = []
    quanta_history     = []
    action_log         = []

    with torch.no_grad():
        for i in range(length):
            x_in = context[:, -block_size:]

            (logits, _, _, quanta_ids,
             S_density, ess, action_params,param_reg) = model(x_in)

            # Physics diagnostics at the last (prediction) position
            ess_history.append(ess)
            sign_var_history.append(S_density[0, -1].item())
            quanta_history.append(quanta_ids[0, -1].item())

            if i % 20 == 0:
                action_log.append(action_params.copy())

            # Sample next token
            next_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                top_v, top_i = torch.topk(next_logits, top_k)
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(0, top_i, top_v)
                next_logits = mask

            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context    = torch.cat([context, next_token.unsqueeze(0)], dim=1)
            generated.append(next_token.item())

    text = tokenizer.decode(generated, skip_special_tokens=True)

    # ── Physics report ────────────────────────────────────────────────
    quanta_arr  = np.array(quanta_history)
    switches    = int(np.sum(np.diff(quanta_arr) != 0))
    unique_q    = np.unique(quanta_arr)
    ess_arr     = np.array(ess_history)

    stats = {
        'ess_mean':           float(np.mean(ess_arr)),
        'ess_min':            float(np.min(ess_arr)),
        'sign_var_mean':      float(np.mean(np.abs(sign_var_history))),
        'quantum_switches':   switches,
        'unique_quanta':      len(unique_q),
        'quanta_ids':         quanta_arr,
        'action_log':         action_log,
    }
    return text, stats


def generate_text_verbose(model, tokenizer, device,
                          start_str: str = "The ",
                          length: int = 80,
                          block_size: int = 64,
                          temperature: float = 0.85,
                          top_k: int = 50):
    """
    Like generate_text but prints a full physics report.
    """
    text, stats = generate_text(
        model, tokenizer, device, start_str, length,
        block_size, temperature, top_k)

    p = model.get_action_params()

    print(f"\nGenerated:\n  {text}\n")
    print("=== Minkowski Field State ===")
    print(f"  Action params:  m²={p['m2']:.4f}  g4={p['g4']:.4f}  "
          f"g6={p['g6']:.5f}  μ={p['mu']:.4f}")
    print(f"  ESS:            mean={stats['ess_mean']:.3f}  "
          f"min={stats['ess_min']:.3f}")
    print(f"  Sign var:       {stats['sign_var_mean']:.4f}  "
          f"(lower = thimble working better)")
    print(f"  Quantum sectors used: {stats['unique_quanta']}/512")
    print(f"  Quantum switches: {stats['quantum_switches']}/{length-1} steps "
          f"({'high' if stats['quantum_switches'] > length*0.3 else 'low'}"
          f" mobility)")

    # Highlight instanton-like events (large action spikes)
    s_arr   = np.array(stats['quanta_ids'])   # reuse as proxy
    ess_arr = np.array([stats['ess_mean']] * length)  # placeholder

    if stats['action_log']:
        print(f"\n  Action param evolution (every 20 steps):")
        for i, ap in enumerate(stats['action_log']):
            print(f"    step {i*20:3d}: m²={ap['m2']:.4f}  "
                  f"g4={ap['g4']:.4f}  μ={ap['mu']:.4f}")

    return text, stats


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train()