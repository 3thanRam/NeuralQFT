import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0,'/home/ethan/Documents/Code/github/NeuralQFT')
sys.path.insert(0,'/home/ethan/Documents/Code/github/NeuralQFT/learned_thimble')

# Import your existing modules
from learned_thimble.thimble import PhysicsInformedThimble, GeneralAction, PARAM_DIM
from learned_thimble.run_thimble import CONFIG 

class DifferentiableAction:
    """
    A tensor-only version of GeneralAction.
    """
    def __init__(self, dx=1.0):
        self.dx = dx

    def compute(self, phi: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Unpack parameters [Batch, 8] -> [Batch, 8, 1, 1, 1, 1]
        p = params.reshape(-1, 8, 1, 1, 1, 1)
        m2             = p[:, 0]
        kinetic_coeff  = p[:, 1]
        g3, g4, g6     = p[:, 2], p[:, 3], p[:, 4]
        box_coeff      = p[:, 5]
        deriv_interact = p[:, 6]
        mu             = p[:, 7]

        dx = self.dx
        vol = dx ** 4

        # Gradients on the lattice
        dt  = torch.roll(phi, -1, dims=-4) - phi
        ddx = torch.roll(phi, -1, dims=-3) - phi
        dy  = torch.roll(phi, -1, dims=-2) - phi
        dz  = torch.roll(phi, -1, dims=-1) - phi
        grad2 = (dt**2 + ddx**2 + dy**2 + dz**2) / dx**2

        # Base Action
        S = 0.5 * kinetic_coeff * grad2 + 0.5 * m2 * phi**2

        # Interactions
        S = S + (g3 / 6.0) * phi**3
        S = S + (g4 / 24.0) * phi**4
        S = S + (g6 / 720.0) * phi**6

        # Higher derivative terms
        if (box_coeff != 0).any():
            box = (
                (torch.roll(phi,-1,dims=-4) - 2*phi + torch.roll(phi,1,dims=-4)) +
                (torch.roll(phi,-1,dims=-3) - 2*phi + torch.roll(phi,1,dims=-3)) +
                (torch.roll(phi,-1,dims=-2) - 2*phi + torch.roll(phi,1,dims=-2)) +
                (torch.roll(phi,-1,dims=-1) - 2*phi + torch.roll(phi,1,dims=-1))
            ) / dx**2
            S = S + box_coeff * box**2
        
        if (deriv_interact != 0).any():
            S = S + deriv_interact * phi**2 * grad2

        # Chemical Potential
        emu, emum = torch.exp(mu), torch.exp(-mu)
        hop_fwd = torch.roll(phi, -1, dims=-4) * phi
        hop_bwd = torch.roll(phi,  1, dims=-4) * phi
        S = S - 0.5 * (emu * hop_fwd + emum * hop_bwd)

        return torch.sum(S, dim=(-4,-3,-2,-1)) * vol


class ThimbleLayer(nn.Module):
    def __init__(self, model_path, config, n_mc_samples=16):
        super().__init__()
        self.device = config['device']
        self.L = config['L']
        self.dx = config['dx']
        self.n_mc = n_mc_samples
        
        # Load Thimble (Same as before)
        self.thimble = PhysicsInformedThimble(config).to(self.device)
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            self.thimble.load_state_dict(ckpt['model'])
            print(f"✅ Loaded Thimble model from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load: {e}"); sys.exit(1)
        
        for param in self.thimble.parameters():
            param.requires_grad = False
        self.thimble.eval()

        self.diff_action = DifferentiableAction(config['dx'])

    def forward(self, lagrangian_params):
        BatchSize = lagrangian_params.shape[0]
        
        # 1. Generate Noise
        z = torch.randn(BatchSize * self.n_mc, self.L, self.L, self.L, self.L, 
                        device=self.device)
        params_expanded = lagrangian_params.repeat_interleave(self.n_mc, dim=0)

        # 2. Run Flow
        with torch.no_grad():
            phi_free, kernel = self.thimble.generate_free_field_tensor_params(z, params_expanded)
            p_emb = self.thimble.param_embedder(params_expanded)
            phi_im, log_det_im = self.thimble.imaginary_layer(phi_free, p_emb)
            base_log_det = torch.sum(-0.5 * torch.log(kernel), dim=(-4,-3,-2,-1))
            log_det = base_log_det + log_det_im
            phi = torch.complex(phi_free, phi_im)

        # 3. Action & Weights
        S = self.diff_action.compute(phi, params_expanded)
        log_w = -S.real + log_det
        log_w = log_w.reshape(BatchSize, self.n_mc)
        max_log_w, _ = torch.max(log_w, dim=1, keepdim=True)
        w = torch.softmax(log_w - max_log_w, dim=1)

        # 4. Compute Observables (Standard)
        phi_flat = phi.reshape(BatchSize, self.n_mc, -1)
        phi_re = torch.clamp(phi_flat.real, -20.0, 20.0)
        phi_im = torch.clamp(phi_flat.imag, -20.0, 20.0)
        phi_safe = torch.complex(phi_re, phi_im)
        
        obs_phi_mean = phi_safe.mean(dim=2)
        exp_phi = (obs_phi_mean * w).sum(dim=1) 

        obs_phi2 = (phi_safe**2).mean(dim=2)
        exp_phi2 = (obs_phi2 * w).sum(dim=1)
        
        obs_phi4 = (phi_safe**4).mean(dim=2)
        exp_phi4 = (obs_phi4 * w).sum(dim=1)

        obs_phase = torch.exp(1j * S.imag).reshape(BatchSize, self.n_mc)
        exp_phase = (obs_phase * w).sum(dim=1)

        # 5. NEW: Advanced Observables (Kinetic & Correlation)
        
        # A. Kinetic Energy (Gradients)
        # How "rough" is the field?
        # We compute sum((phi - neighbor)^2) roughly via rolls
        # We do this on the unflattened phi [B*N, L, L, L, L]
        dx_phi = torch.roll(phi, -1, dims=-1) - phi
        dy_phi = torch.roll(phi, -1, dims=-2) - phi
        grad_sq = (dx_phi**2 + dy_phi**2) # Simplified kinetic term
        
        # Flatten and Average over lattice
        grad_sq_flat = grad_sq.reshape(BatchSize, self.n_mc, -1).mean(dim=2)
        
        # Clamp complex parts
        g_re = torch.clamp(grad_sq_flat.real, -50.0, 50.0)
        g_im = torch.clamp(grad_sq_flat.imag, -50.0, 50.0)
        grad_safe = torch.complex(g_re, g_im)
        
        exp_grad = (grad_safe * w).sum(dim=1)

        # B. Nearest Neighbor Correlation
        # <phi(x) * phi(x+1)>
        # Distinguishes "Heavy" vs "Light" fields better than just phi^2
        corr = phi * torch.roll(phi, 1, dims=-1)
        corr_flat = corr.reshape(BatchSize, self.n_mc, -1).mean(dim=2)
        
        c_re = torch.clamp(corr_flat.real, -20.0, 20.0)
        c_im = torch.clamp(corr_flat.imag, -20.0, 20.0)
        corr_safe = torch.complex(c_re, c_im)
        
        exp_corr = (corr_safe * w).sum(dim=1)

        # 6. Stack All Features (8 + 4 = 12 dims)
        features = torch.stack([
            exp_phi.real,  exp_phi.imag,
            exp_phi2.real, exp_phi2.imag,
            exp_phi4.real, exp_phi4.imag,
            exp_phase.real, exp_phase.imag,
            exp_grad.real, exp_grad.imag,    # NEW
            exp_corr.real, exp_corr.imag     # NEW
        ], dim=1) 
        
        return features

# Monkey-patch PhysicsInformedThimble to accept tensor params
def generate_free_field_tensor_params(self, z, p_tensor):
    # p_tensor: [B, 8] -> m2 is idx 0, kin is idx 1
    # FIXED: Use reshape here as well for safety
    m2 = p_tensor[:, 0].reshape(-1, 1, 1, 1, 1)
    kin = p_tensor[:, 1].reshape(-1, 1, 1, 1, 1)
    
    eig = self.laplacian_eig.unsqueeze(0)
    
    kernel = torch.clamp(kin * self.dx**2 * eig + m2 * self.dx**4, min=1e-8)
    
    phi_free = torch.fft.ifftn(
        torch.fft.fftn(z.to(torch.complex64), dim=(-4,-3,-2,-1)) / torch.sqrt(kernel),
        dim=(-4,-3,-2,-1)
    ).real
    return phi_free, kernel

PhysicsInformedThimble.generate_free_field_tensor_params = generate_free_field_tensor_params


class LagrangianLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, thimble_model_path, config):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.to_params = nn.Linear(hidden_dim, 8)
        
        self.thimble_layer = ThimbleLayer(thimble_model_path, config, n_mc_samples=16)
        
        # Increased dims (12 now)
        self.norm = nn.LayerNorm(12)
        
        # NEW: A small "Physics Interpreter" MLP
        # Instead of a single Linear layer, give the model a chance to 
        # mix the physical signals (e.g. "High Gradient" AND "Low Mass")
        self.interpreter = nn.Sequential(
            nn.Linear(12, 32),
            nn.GELU(),
            nn.Linear(32, vocab_size)
        )
        
        self.config = config

    # ... (_scale_params remains same as previous step) ...
    def _scale_params(self, raw):
        s = torch.sigmoid(raw)
        p = torch.zeros_like(s)
        p[:, 0] = 2.5 * s[:, 0] - 0.5   # m2
        p[:, 1] = 0.5 + 2.5 * s[:, 1]   # kin
        p[:, 2] = 2.0 * s[:, 2] - 1.0   # g3
        p[:, 3] = 2.5 * s[:, 3] - 0.5   # g4
        p[:, 4] = 1.0 * s[:, 4] + 0.05  # g6
        p[:, 5] = 0.0
        p[:, 6] = 0.0 
        p[:, 7] = 1.5 * s[:, 7]         # mu
        return p

    def forward(self, x):
        embed = self.embedding(x)
        output, _ = self.rnn(embed)
        last_hidden = output[:, -1, :]
        
        raw_params = self.to_params(last_hidden)
        phys_params = self._scale_params(raw_params)
        
        # Get 12 physical features
        observables = self.thimble_layer(phys_params)
        observables = self.norm(observables)
        
        # Interpret and Predict
        logits = self.interpreter(observables)
        
        return logits, phys_params

