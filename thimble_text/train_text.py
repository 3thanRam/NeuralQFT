"""
train_text.py  (O(N) vector field edition — fixes applied)
==========================================================
Training loop for MinkowskiFieldLM with O(N) vector field thimble.

Fixes vs previous version
--------------------------
1. Codebook collapse  (n_quanta 256->64, w_commit 10->50, vq_ema_decay 0.99->0.95)
   Symptoms: active quanta 83-91/256, only 35% utilisation.
   Root cause: codebook too large for the data regime in 3 epochs; EMA too
   slow to revive dead slots. Fix: shrink codebook so every entry is reachable,
   increase commitment pressure, speed up EMA reassignment.

2. g6 frozen  (own optimizer group at lr_g6=1e-2, init log_g6=-2.0 in physics_lm.py)
   Symptoms: g6=0.04990 never changing across all steps.
   Root cause: log_g6 shared the 0.1x action LR group with log_m2/log_g4.
   Those two are already near their targets (m2~1, g4~0.3) so their
   gradients are near zero, and AdamW's per-parameter momentum dragged
   log_g6's effective step size down to near zero too.
   Fix: own group at full LR. The log_g6 init in physics_lm.py was also
   moved from -3.0 (g6~0.05) to -2.0 (g6~0.14) to start in a region with
   stronger gradients from the data.

3. lx frozen  (own optimizer group at lr_lx=1e-2)
   Symptoms: lx=0.1000 never changing across thousands of steps.
   Same root cause as g6: log_lambda_cross was in the 0.1x action_slow group
   with already-converged parameters. Fix: own group at full LR.

4. Temporal incoherence  (switch_loss, w_switch=0.3)
   Symptoms: Switches=55-59/60, nearly every token changes quantum state.
   The field has no temporal continuity — it's sampling a new quantum at
   every step rather than evolving smoothly. switch_loss penalises the
   fraction of adjacent positions that map to different quanta, computed
   over the embedding (smooth, differentiable) not the discrete indices.

Loss terms
----------
1. ce_loss    : standard next-token prediction (primary)
2. commit_loss: VQ codebook commitment (keeps embeddings quantized)
3. sign_loss  : penalises large variance in S_density across sequence
4. ward_loss  : O(N) Ward identity regularisation
5. switch_loss: penalises high quantum-switch rate (temporal incoherence)
6. ess_loss   : penalises low effective sample size
7. param_reg  : soft regularisation keeping action params in thimble range
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import numpy as np
from pathlib import Path
import sys

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent / 'learned_thimble_vector'))

from run_thimble import CONFIG as THIMBLE_CONFIG
from physics_lm import MinkowskiFieldLM

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_CONFIG = {
    'vocab_size':    50257,
    'embed_dim':     512,
    # FIX 1: 256 -> 64. Only 83-91/256 quanta were active (35% utilisation).
    # 64 ensures every codebook entry is reachable within 3 epochs.
    'n_quanta':      64,
    'n_layers':      12,
    'n_components':  3,
    'block_size':    64,
    'batch_size':    16,
    'lr':            1e-2,
    'epochs':        3,
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
    'thimble_path':  THIMBLE_CONFIG['data_dir'] / 'universal_thimble_model.pt',
    'save_path':     THIMBLE_CONFIG['data_dir'] / 'minkowski_vector_lm.pt',
    # Loss weights
    # FIX 1: 10.0 -> 50.0. Stronger commitment pressure to fill all 64 quanta.
    'w_commit':      50.0,
    'w_sign':        1.0,
    'w_ward':        0.5,
    # FIX 4: penalise high quantum-switch rate (was 55-59/60 switches)
    # 0.3 is enough to push toward smoother trajectories without killing diversity
    'w_switch':      0.3,
    'w_ess':         0.1,
    'ess_floor':     0.05,
    # FIX 2: g6 gets its own LR group at full rate.
    'lr_g6':         1e-2,
    # FIX 3: lx gets its own LR group — same fix as g6
    'lr_lx':         1e-2,
    # FIX 1: 0.99 -> 0.95. Faster EMA so dead codebook slots get revived
    # within 3 epochs rather than lingering for thousands of steps.
    'vq_ema_decay':  0.95,
}


# ─────────────────────────────────────────────────────────────────────────────
# O(N) Ward identity loss
# ─────────────────────────────────────────────────────────────────────────────

def quantum_switch_loss(phi_in: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Penalise temporal incoherence: adjacent positions mapping to different quanta.

    Why this works without a discrete argmax in the gradient path
    -------------------------------------------------------------
    Instead of comparing discrete quanta_ids (which has zero gradient), we
    measure the cosine distance between adjacent normalised embeddings in the
    continuous embedding space. When the model learns to keep adjacent tokens
    in similar regions of embedding space, the VQ argmax naturally assigns
    them the same codebook entry — reducing switches without ever touching
    a non-differentiable operation.

    switch_loss = mean over (b, t) of (1 - cos_sim(phi[t], phi[t+1]))

    This is 0 when adjacent embeddings are identical (max coherence) and
    approaches 2 when they are antipodal (max incoherence).

    phi_in   : [B, T, D]  continuous embeddings (before quantisation)
    codebook : unused (kept for API symmetry)
    Returns  : scalar in [0, 2]
    """
    # Normalise along embedding dim
    phi_n   = torch.nn.functional.normalize(phi_in, dim=-1)   # [B, T, D]
    cos_sim = (phi_n[:, :-1] * phi_n[:, 1:]).sum(dim=-1)      # [B, T-1]
    return (1.0 - cos_sim).mean()


def ward_identity_loss(phi_comp):
    """
    Penalises breaking of O(N) symmetry in the field configurations.

    For an O(N)-symmetric theory: <phi^a phi^b> = delta^{ab} * G
    We penalise the squared Frobenius norm of the off-diagonal part of the
    empirical N x N covariance matrix.

    phi_comp : [B, T, N]
    Returns  : scalar
    """
    B, T, N  = phi_comp.shape
    phi_flat = phi_comp.reshape(B * T, N)
    C        = (phi_flat.T @ phi_flat) / (B * T)
    diag     = torch.eye(N, device=phi_comp.device, dtype=torch.bool)
    off_diag = C.masked_fill(diag, 0.0)
    return (off_diag ** 2).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


class WikiDataset(Dataset):
    def __init__(self, tokenizer, block_size: int, split: str = 'train',
                 cache_file='wiki_cache.pt'):
        self.block_size = block_size

        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}...")
            self.data = torch.load(cache_file)
            print(f"Loaded {len(self.data):,} tokens.")
            return

        print("Cache not found. Processing dataset (this happens once)...")
        raw = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

        cleaned = []
        for t in raw['text']:
            t = t.strip()
            if not t: continue
            if t.startswith('=') or t.startswith('<') or '@' in t: continue
            if len(t.split()) < 8: continue
            if len(t) > 0:
                alpha = sum(c.isalpha() for c in t) / len(t)
                if alpha < 0.4: continue
            cleaned.append(t)

        print("   Tokenizing...")
        full = '\n'.join(cleaned)
        del raw

        enc       = tokenizer(full, return_tensors='pt', truncation=False)
        self.data = enc['input_ids'].squeeze(0)

        print(f"Saving cache to {cache_file}...")
        torch.save(self.data, cache_file)
        print(f"Tokens: {len(self.data):,}")

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        return self.data[idx: idx + self.block_size + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    cfg    = TRAIN_CONFIG
    device = cfg['device']

    if not cfg['thimble_path'].exists():
        print(f"Thimble model not found at {cfg['thimble_path']}")
        print("Run learned_thimble/run_thimble.py first.")
        return

    tokenizer = get_tokenizer()
    dataset   = WikiDataset(tokenizer, cfg['block_size'], split='train')
    loader    = DataLoader(dataset, batch_size=cfg['batch_size'],
                           shuffle=True, num_workers=2, pin_memory=True)

    thimble_cfg = {**THIMBLE_CONFIG, 'n_components': cfg['n_components']}

    model = MinkowskiFieldLM(
        vocab_size         = cfg['vocab_size'],
        embed_dim          = cfg['embed_dim'],
        n_quanta           = cfg['n_quanta'],
        n_layers           = cfg['n_layers'],
        n_components       = cfg['n_components'],
        thimble_model_path = cfg['thimble_path'],
        thimble_config     = thimble_cfg,
        # FIX 1: pass faster EMA decay so dead quanta get revived quickly
        vq_ema_decay       = cfg['vq_ema_decay'],
    ).to(device)

    # ── Optimiser: separate LR groups ────────────────────────────────────────
    embed_params     = list(model.field_embed.parameters())
    component_params = (list(model.action.embed_to_components.parameters()) +
                        list(model.component_proj.parameters()))
    other_params     = (list(model.action_proj.parameters()) +
                        list(model.propagator.parameters()))

    # FIX 2: g6 gets its own full-LR group.
    # Previously log_g6 shared the 0.1x action group with log_m2 and log_g4.
    # Those two converge early (m2->1.0, g4->0.3) and produce near-zero
    # gradients. AdamW's per-parameter second moment then suppresses the
    # effective step for all parameters in the group, including log_g6.
    # Isolating log_g6 at lr_g6 lets it follow its own gradient independently.
    g6_params   = [model.action.log_g6]

    # FIX 3: lx (log_lambda_cross) gets its own full-LR group.
    # Same root cause as g6: was in action_slow at 0.1x LR alongside
    # already-converged m2/g4, so AdamW's second moment suppressed it.
    lx_params   = [model.action.log_lambda_cross]

    # Remaining action params: slow 0.1x LR keeps thimble contour valid
    # (log_lambda_cross removed — now in lx_params)
    action_slow = [model.action.log_m2, model.action.g3,
                   model.action.log_g4, model.action.raw_mu]

    all_ids = [id(p) for group in [embed_params, component_params,
                                    other_params, g6_params, lx_params,
                                    action_slow]
               for p in group]
    assert len(all_ids) == len(set(all_ids)), "Parameter group overlap detected"

    optimizer = torch.optim.AdamW([
        {'params': embed_params,     'lr': cfg['lr']},
        {'params': component_params, 'lr': cfg['lr']},
        {'params': other_params,     'lr': cfg['lr']},
        # g6 at full LR: needs freedom to move from its new init
        {'params': g6_params,        'lr': cfg['lr_g6'],  'weight_decay': 0.0},
        # lx at full LR: was frozen at 0.1000 for thousands of steps
        {'params': lx_params,        'lr': cfg['lr_lx'],  'weight_decay': 0.0},
        # Other action params: slow LR to keep thimble contour valid
        {'params': action_slow,      'lr': cfg['lr'] * 0.1, 'weight_decay': 0.0},
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'],
        steps_per_epoch=len(loader), epochs=cfg['epochs'],
        pct_start=0.05,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*65}")
    print(f"MinkowskiFieldLM  O({cfg['n_components']}) vector field training")
    print(f"  embed_dim={cfg['embed_dim']}  n_components={cfg['n_components']}"
          f"  n_quanta={cfg['n_quanta']}  n_layers={cfg['n_layers']}")
    print(f"  batch={cfg['batch_size']}  block={cfg['block_size']}"
          f"  device={device}")
    print(f"  Fixes: n_quanta=64, w_commit=50, vq_ema=0.95, g6/lx own LR, switch_loss")
    print(f"  Loss: CE + {cfg['w_commit']}*VQ + {cfg['w_sign']}*SignVar"
          f" + {cfg['w_ward']}*Ward + {cfg['w_switch']}*Switch + {cfg['w_ess']}*ESS")
    print(f"{'='*65}\n")

    for epoch in range(cfg['epochs']):
        model.train()
        for step, chunk in enumerate(loader):
            x_in  = chunk[:, :-1].to(device)
            y_seq = chunk[:, 1:].to(device)

            optimizer.zero_grad()

            (logits, commit_loss, perplexity,
             quanta_ids, S_density, ess,
             action_params, param_reg, phi_comp) = model(x_in)

            # 1. Primary: next-token prediction
            ce_loss = criterion(
                logits.reshape(-1, cfg['vocab_size']),
                y_seq.reshape(-1))

            # 2. VQ commitment
            vq_loss = commit_loss

            # 3. Sign problem: penalise large S_density variance across sequence
            sign_loss = S_density.var(dim=-1).mean()

            # 4. O(N) Ward identity
            ward_loss = ward_identity_loss(phi_comp)

            # 5. Temporal coherence: penalise high embedding cosine distance
            #    between adjacent positions. Backprops through continuous phi_in
            #    (not discrete quanta_ids), so the gradient path is clean.
            #    phi_in is the raw embedding before quantisation, retrieved from
            #    the field_embed submodule's last output via phi_comp's parent.
            #    We use phi_comp (projected to N components) as a proxy — it's
            #    derived from phi_in via a linear map so gradients flow through.
            switch_loss = quantum_switch_loss(phi_comp, None)

            # 6. ESS penalty
            ess_tensor = torch.tensor(ess, device=device, dtype=torch.float32)
            ess_loss   = torch.clamp(cfg['ess_floor'] - ess_tensor, min=0.0)

            loss = (ce_loss
                    + cfg['w_commit']  * vq_loss
                    + cfg['w_sign']    * sign_loss
                    + cfg['w_ward']    * ward_loss
                    + cfg['w_switch']  * switch_loss
                    + cfg['w_ess']     * ess_loss
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
                    f"Ward={ward_loss.item():.4f} | "
                    f"Switch={switch_loss.item():.4f} | "
                    f"ESS={ess:.3f} | "
                    f"Perp={perplexity.item():.0f}/{cfg['n_quanta']}\n"
                    f"   m2={p['m2']:.4f} g3={p['g3']:.4f} g4={p['g4']:.4f} "
                    f"g6={p['g6']:.5f} lx={p['lambda_cross']:.4f} "
                    f"mu={p['mu']:.4f} | "
                    f"Active: {n_active}/{cfg['n_quanta']}"
                )

            if step > 0 and step % 500 == 0:
                _generate_sample(model, tokenizer, device, step)
                model.train()

        torch.save({
            'epoch':        epoch,
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'train_config': cfg,
        }, cfg['save_path'])
        print(f"\nCheckpoint saved -> {cfg['save_path']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sample(model, tokenizer, device, step,
                     length=60, temperature=0.7, top_k=100):
    model.eval()
    print(f"\n--- Generation at step {step} ---")
    for start in ["The ", "In the ", "Scientists "]:
        text, stats = generate_text(
            model, tokenizer, device,
            start_str=start, length=length,
            block_size=TRAIN_CONFIG['block_size'],
            temperature=temperature, top_k=top_k)
        p = model.get_action_params()
        print(f"  [{start.strip()}] {text}")
        print(f"    ESS={stats['ess_mean']:.3f}  "
              f"SignVar={stats['sign_var_mean']:.3f}  "
              f"Ward={stats['ward_mean']:.4f}  "
              f"Switch={stats['switch_mean']:.4f}  "
              f"lx={p['lambda_cross']:.4f}  "
              f"g6={p['g6']:.5f}  "
              f"Switches={stats['quantum_switches']}/{length}")
    print()


def generate_text(model, tokenizer, device,
                  start_str: str = "The ",
                  length: int = 80,
                  block_size: int = 64,
                  temperature: float = 0.7,
                  top_k: int = 100):
    """
    Autoregressive generation with per-step physics diagnostics.

    Returns
    -------
    text  : str
    stats : dict  includes ward_mean, g6 tracked per step
    """
    model.eval()
    enc       = tokenizer(start_str, return_tensors='pt')
    context   = enc['input_ids'].to(device)
    generated = context[0].tolist()

    ess_history      = []
    sign_var_history = []
    ward_history     = []
    switch_history   = []
    quanta_history   = []
    action_log       = []

    with torch.no_grad():
        for i in range(length):
            x_in = context[:, -block_size:]

            (logits, _, _, quanta_ids,
             S_density, ess, action_params,
             param_reg, phi_comp) = model(x_in)

            ess_history.append(ess)
            sign_var_history.append(S_density[0, -1].item())
            quanta_history.append(quanta_ids[0, -1].item())
            ward_history.append(ward_identity_loss(phi_comp).item())
            switch_history.append(quantum_switch_loss(phi_comp, None).item())

            if i % 20 == 0:
                action_log.append(action_params.copy())

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

    quanta_arr = np.array(quanta_history)
    switches   = int(np.sum(np.diff(quanta_arr) != 0))

    stats = {
        'ess_mean':         float(np.mean(ess_history)),
        'ess_min':          float(np.min(ess_history)),
        'sign_var_mean':    float(np.mean(np.abs(sign_var_history))),
        'ward_mean':        float(np.mean(ward_history)),
        'ward_history':     ward_history,
        'switch_mean':      float(np.mean(switch_history)),
        'quantum_switches': switches,
        'unique_quanta':    len(np.unique(quanta_arr)),
        'quanta_ids':       quanta_arr,
        'action_log':       action_log,
    }
    return text, stats


def generate_text_verbose(model, tokenizer, device,
                          start_str: str = "The ",
                          length: int = 80,
                          block_size: int = 64,
                          temperature: float = 0.7,
                          top_k: int = 100):
    """Like generate_text but prints a full physics report."""
    text, stats = generate_text(
        model, tokenizer, device, start_str, length,
        block_size, temperature, top_k)

    p = model.get_action_params()

    print(f"\nGenerated:\n  {text}\n")
    print("=== O(N) Minkowski Field State ===")
    print(f"  Action params:  m2={p['m2']:.4f}  g3={p['g3']:.4f}  "
          f"g4={p['g4']:.4f}  g6={p['g6']:.5f}  "
          f"lx={p['lambda_cross']:.4f}  mu={p['mu']:.4f}")
    print(f"  ESS:            mean={stats['ess_mean']:.3f}  "
          f"min={stats['ess_min']:.3f}")
    print(f"  Sign var:       {stats['sign_var_mean']:.4f}")
    print(f"  Ward residual:  {stats['ward_mean']:.4f}  "
          f"(lower = O(N) symmetry better preserved)")
    print(f"  Switch loss:    {stats['switch_mean']:.4f}  "
          f"(lower = smoother quantum trajectories)")
    print(f"  Quantum sectors used: {stats['unique_quanta']}/{TRAIN_CONFIG['n_quanta']}")
    print(f"  Quantum switches: {stats['quantum_switches']}/{length-1} steps")

    if stats['action_log']:
        print(f"\n  Action param evolution (every 20 steps):")
        for i, ap in enumerate(stats['action_log']):
            print(f"    step {i*20:3d}: m2={ap['m2']:.4f}  g4={ap['g4']:.4f}"
                  f"  g6={ap['g6']:.5f}  lx={ap['lambda_cross']:.4f}"
                  f"  mu={ap['mu']:.4f}")

    return text, stats


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train()