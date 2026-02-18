"""
Physics-Informed Thimble Network
"""

import torch
import torch.nn as nn
import numpy as np
import torch.fft as fft
from dataclasses import dataclass


# ============================================================================
# Physics
# ============================================================================

@dataclass
class LagrangianParams:
    m2:             float = 1.0   # phi^2 coefficient (× 0.5)
    kinetic_coeff:  float = 1.0   # (∂_μ φ)^2 coefficient (× 0.5)
    g3:             float = 0.0   # phi^3 / 3!
    g4:             float = 0.0   # phi^4 / 4!
    g6:             float = 0.0   # phi^6 / 6!
    box_coeff:      float = 0.0   # (□φ)^2 coefficient
    deriv_interact: float = 0.0   # phi^2 * (∂_μ phi)^2
    mu:             float = 0.0   # chemical potential (finite density)

    def __repr__(self):
        parts = [f"{self.m2:.2f}φ²", f"{self.kinetic_coeff:.2f}(∂φ)²"]
        if self.g3:  parts.append(f"{self.g3:.3f}φ³/3!")
        if self.g4:  parts.append(f"{self.g4:.3f}φ⁴/4!")
        if self.g6:  parts.append(f"{self.g6:.3f}φ⁶/6!")
        if self.mu:  parts.append(f"μ={self.mu:.3f}")
        return "S=[" + " + ".join(parts) + "]"

    def to_tensor(self, device='cpu') -> torch.Tensor:
        return torch.tensor(
            [self.m2, self.kinetic_coeff, self.g3, self.g4,
             self.g6, self.box_coeff, self.deriv_interact, self.mu],
            dtype=torch.float32, device=device,
        )


PARAM_DIM = 8


class GeneralAction:
    """Full extended Lagrangian. phi may be complex. Returns [B] complex."""

    def __init__(self, dx: float = 1.0):
        self.dx = dx

    def compute(self, phi: torch.Tensor, p: LagrangianParams) -> torch.Tensor:
        dx = self.dx
        vol = dx ** 4

        dt  = torch.roll(phi, -1, dims=-4) - phi
        ddx = torch.roll(phi, -1, dims=-3) - phi
        dy  = torch.roll(phi, -1, dims=-2) - phi
        dz  = torch.roll(phi, -1, dims=-1) - phi
        grad2 = (dt**2 + ddx**2 + dy**2 + dz**2) / dx**2

        S = 0.5 * p.kinetic_coeff * grad2 + 0.5 * p.m2 * phi**2

        if p.g3 != 0.0:
            S = S + (p.g3  / 6.0)   * phi**3
        if p.g4 != 0.0:
            S = S + (p.g4  / 24.0)  * phi**4
        if p.g6 != 0.0:
            S = S + (p.g6  / 720.0) * phi**6
        if p.box_coeff != 0.0:
            box = (
                (torch.roll(phi,-1,dims=-4) - 2*phi + torch.roll(phi,1,dims=-4)) +
                (torch.roll(phi,-1,dims=-3) - 2*phi + torch.roll(phi,1,dims=-3)) +
                (torch.roll(phi,-1,dims=-2) - 2*phi + torch.roll(phi,1,dims=-2)) +
                (torch.roll(phi,-1,dims=-1) - 2*phi + torch.roll(phi,1,dims=-1))
            ) / dx**2
            S = S + p.box_coeff * box**2
        if p.deriv_interact != 0.0:
            S = S + p.deriv_interact * phi**2 * grad2
        if p.mu != 0.0:
            emu, emum = np.exp(p.mu), np.exp(-p.mu)
            hop_fwd = torch.roll(phi, -1, dims=-4) * phi
            hop_bwd = torch.roll(phi,  1, dims=-4) * phi
            S = S - 0.5 * (emu * hop_fwd + emum * hop_bwd)

        return torch.sum(S, dim=(-4,-3,-2,-1)) * vol


# ============================================================================
# Conditioner network  (position-space ResNet)
# ============================================================================

class ResNetConditioner(nn.Module):
    """
    phi_free [B, L, L, L, L]  →  (log_sigma, tau)  each [B, L, L, L, L]

    phi_im_x = exp(log_sigma_x) * phi_free_x + tau_x
    log|det J|_im = sum_x log_sigma_x   (exact)

    Two genuine residual blocks for gradient stability.
    Soft-clamp on log_sigma to keep deformation moderate.
    """

    def __init__(self, L: int, hidden_dim: int, param_dim: int = PARAM_DIM):
        super().__init__()
        # Physics parameter → per-channel bias
        self.param_proj = nn.Linear(param_dim, L)

        self.in_conv = nn.Conv3d(L, hidden_dim, 3, padding=1, padding_mode='circular')

        self.res1 = nn.Sequential(
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
        )
        self.res2 = nn.Sequential(
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
        )

        # Output: log_sigma and tau
        self.out_log_sigma = nn.Conv3d(hidden_dim, L, 1)
        self.out_tau       = nn.Conv3d(hidden_dim, L, 1)

        # Critical: zero-init so phi_im = 0 at start (identity map)
        nn.init.zeros_(self.out_log_sigma.weight); nn.init.zeros_(self.out_log_sigma.bias)
        nn.init.zeros_(self.out_tau.weight);       nn.init.zeros_(self.out_tau.bias)

    def forward(self, phi_free: torch.Tensor, p_vec: torch.Tensor):
        # [B, L, X, Y, Z]  with T treated as channels
        p_bias = self.param_proj(p_vec).view(-1, phi_free.shape[1], 1, 1, 1)
        x = self.in_conv(phi_free + p_bias)
        x = x + self.res1(x)
        x = x + self.res2(x)

        log_sigma = self.out_log_sigma(x)
        tau       = self.out_tau(x)

        # Soft-clamp: log_sigma in (-0.3, 0.3)  →  sigma in (0.74, 1.35)
        # Small range keeps importance weights healthy
        log_sigma = torch.tanh(log_sigma) * 0.3

        return log_sigma, tau


# ============================================================================
# Imaginary deformation layer
# ============================================================================

class ImaginaryLayer(nn.Module):
    """
    phi_im_x = exp(log_sigma_x) * phi_free_x + tau_x
    log|det J|_im = sum_x log_sigma_x    (exact, no Hutchinson)
    """

    def __init__(self, L: int, hidden_dim: int, param_dim: int = PARAM_DIM):
        super().__init__()
        self.net = ResNetConditioner(L, hidden_dim, param_dim)

    def forward(self, phi_free: torch.Tensor, p_emb: torch.Tensor):
        log_sigma, tau = self.net(phi_free, p_emb)
        phi_im  = torch.exp(log_sigma) * phi_free + tau
        log_det = torch.sum(log_sigma, dim=(-4,-3,-2,-1))  # [B], exact
        return phi_im, log_det


# ============================================================================
# Parameter embedder
# ============================================================================

class ParamEmbedder(nn.Module):
    def __init__(self, param_dim: int = PARAM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, 32), nn.SiLU(),
            nn.Linear(32, 32),        nn.SiLU(),
            nn.Linear(32, param_dim),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p) + p   # residual


# ============================================================================
# Main model
# ============================================================================

class PhysicsInformedThimble(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.L      = config['L']
        self.device = config['device']
        self.dx     = config['dx']
        self.action_computer = GeneralAction(self.dx)

        k = 2 * np.pi * torch.arange(self.L, device=self.device) / self.L
        kt, kx, ky, kz = torch.meshgrid(k, k, k, k, indexing='ij')
        eig = (2*(1-torch.cos(kt)) + 2*(1-torch.cos(kx)) +
               2*(1-torch.cos(ky)) + 2*(1-torch.cos(kz)))
        self.register_buffer('laplacian_eig', eig)

        hidden_dim = config.get('thimble_hidden_dim', 32)
        self.param_embedder  = ParamEmbedder(param_dim=PARAM_DIM)
        self.imaginary_layer = ImaginaryLayer(self.L, hidden_dim, param_dim=PARAM_DIM)

    def _kernel(self, p: LagrangianParams) -> torch.Tensor:
        return torch.clamp(
            p.kinetic_coeff * self.dx**2 * self.laplacian_eig +
            p.m2            * self.dx**4,
            min=1e-8
        )

    def generate_free_field(self, z: torch.Tensor, p: LagrangianParams):
        kernel = self._kernel(p)
        phi_free = fft.ifftn(
            fft.fftn(z.to(torch.complex64), dim=(-4,-3,-2,-1)) / torch.sqrt(kernel),
            dim=(-4,-3,-2,-1)
        ).real
        return phi_free, kernel

    def get_base_log_det(self, p: LagrangianParams) -> torch.Tensor:
        return torch.sum(-0.5 * torch.log(self._kernel(p)))   # scalar

    def forward(self, z: torch.Tensor, p: LagrangianParams,
                return_scale: bool = False):
        phi_free, kernel = self.generate_free_field(z, p)
        p_emb = self.param_embedder(p.to_tensor(self.device))

        phi_im, log_det_im = self.imaginary_layer(phi_free, p_emb)

        log_det = self.get_base_log_det(p) + log_det_im   # [B]
        phi     = torch.complex(phi_free, phi_im)

        if return_scale:
            return phi, log_det, torch.mean(1.0/torch.sqrt(kernel)).item()
        return phi, log_det

    def compute_thimble_loss(self, phi: torch.Tensor, p: LagrangianParams,
                             log_det: torch.Tensor, lambda_im: float = 0.1):
        """
        PRIMARY loss: Var(F_eff)  where F_eff = Re(S) - log_det
        This is the correct variational objective — when Var(F_eff) = 0,
        importance weights are uniform → ESS = 1.

        SECONDARY loss: Var(Im S) * lambda_im
        Regularises the imaginary part but is NOT the primary signal.

        TERTIARY: penalty on Var(log_det) to prevent log_det from
        having large per-sample variance that tanks ESS independently.
        """
        S = self.action_computer.compute(phi, p)

        F_eff    = S.real - log_det          # [B]  effective action
        var_Feff = torch.var(F_eff)          # PRIMARY: minimise this
        var_imS  = torch.var(S.imag)         # SECONDARY: thimble constraint
        var_logJ = torch.var(log_det)        # TERTIARY: log-det stability

        # Mean effective action (normalised by volume)
        mean_Feff = torch.mean(F_eff) / self.L**4

        return var_Feff, var_imS, var_logJ, mean_Feff, S

    def compute_log_jacobian(self, z: torch.Tensor, p: LagrangianParams) -> torch.Tensor:
        _, log_det = self.forward(z, p)
        return log_det


# ============================================================================
# Statistical utilities
# ============================================================================

def effective_sample_size(log_weights: torch.Tensor) -> float:
    """ESS = (Σw)²/Σw² as fraction of N."""
    lw = log_weights - torch.max(log_weights)
    w  = torch.exp(lw)
    return (w.sum()**2 / (w**2).sum() / len(w)).item()


def bootstrap_error(values: np.ndarray, weights: np.ndarray,
                    n_boot: int = 500) -> float:
    """Bootstrap standard error of weighted mean (ESS-corrected)."""
    n = len(values)
    w = weights / weights.sum()
    boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        wb  = w[idx]; wb /= wb.sum()
        boots.append(np.dot(wb, values[idx]))
    return float(np.std(boots))


def integrated_autocorrelation(x: np.ndarray, max_lag: int = 50) -> float:
    """Integrated autocorrelation time via Sokal window method."""
    n   = len(x)
    xm  = x - x.mean()
    c0  = np.dot(xm, xm) / n
    if c0 == 0: return 1.0
    tau = 0.5
    for t in range(1, min(max_lag, n // 2)):
        ct  = np.dot(xm[:-t], xm[t:]) / (n - t)
        tau += ct / c0
        if t > 5 * tau: break
    return max(tau, 0.5)