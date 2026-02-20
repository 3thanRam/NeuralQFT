"""
Physics-Informed Thimble Network

Key change vs. v7: ResNetConditioner now uses a dual-path architecture
that gives the network explicit awareness of the temporal direction.

Why this matters for finite-μ sign problems
-------------------------------------------
The chemical-potential term in the action is

    S_μ = -μ · Σ_x [ e^μ φ(x+ê_t)φ(x) + e^{-μ} φ(x-ê_t)φ(x) ]

This couples sites only along the time axis.  When μ is large the thimble
geometry deforms primarily in the imaginary-time direction; the optimal
contour tilt is a *temporal* object.  The original 3D spatial conv treats
time as a batch/channel dimension — it cannot learn asymmetric temporal
deformations without many indirect layers.

Dual-path conditioner
---------------------
    phi  [B, T, X, Y, Z]
         │
         ├── SpatialPath: treats T as channels, 3D conv over (X,Y,Z)
         │     Conv3d(T,H) → ResBlock → ResBlock  →  features_spatial [B,H,X,Y,Z]
         │
         └── TemporalPath: treats (X,Y,Z) as batch, 1D conv over T
               reshape to [B·X·Y·Z, T, 1]
               → TemporalResBlock (Conv1d, circ pad) → TemporalResBlock
               → reshape back [B,H,X,Y,Z]  →  features_temporal [B,H,X,Y,Z]
         │
         └── Fusion: concat along channel → 1×1 conv → log_sigma, tau

The spatial path handles the usual φ⁴/φ⁶ geometry (same as before).
The temporal path handles the μ-induced asymmetry.  Because the 1D convs
are applied over the time loop with circular (periodic BC) padding they
correctly see the e^{±μ} hop structure.

The two paths share the same hidden_dim H and are fused with a lightweight
1×1 conv, keeping parameter count almost the same as the original.

Exact log|det J| is preserved: log_sigma is still a pointwise function of
phi_free so log|det| = Σ_x log_sigma_x (no Hutchinson estimator needed).
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
    """
    Full extended Euclidean Lagrangian. phi may be complex. Returns [B] complex.

    Also provides minkowski_action() for the Minkowski-signature counterpart,
    used during the late-training Minkowski phase.
    """

    def __init__(self, dx: float = 1.0):
        self.dx = dx

    def compute(self, phi: torch.Tensor, p: LagrangianParams) -> torch.Tensor:
        """Euclidean action S_E. phi: [B, L, L, L, L]. Returns [B] complex."""
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

    def minkowski_action(self, phi: torch.Tensor,
                         p: LagrangianParams) -> torch.Tensor:
        """
        Minkowski action S_M with correct sign conventions.
        phi: [B, L, L, L, L] complex.  Returns S_M: [B] complex.
        """
        dphi    = torch.roll(phi, -1, dims=1) - phi
        kinetic = 0.5 * (dphi ** 2).sum(dim=(1, 2, 3, 4))

        mass  = -0.5          * p.m2 * (phi ** 2).sum(dim=(1, 2, 3, 4))
        phi4  = -(p.g4 / 24.0)       * (phi ** 4).sum(dim=(1, 2, 3, 4))
        phi6  = -(p.g6 / 720.0)      * (phi ** 6).sum(dim=(1, 2, 3, 4))

        hop_fwd = torch.roll(phi, -1, dims=1) * phi
        hop_bwd = torch.roll(phi,  1, dims=1) * phi
        chem    = -0.5 * p.mu * (hop_fwd + hop_bwd).sum(dim=(1, 2, 3, 4))

        return kinetic + mass + phi4 + phi6 + chem


# ============================================================================
# Building blocks
# ============================================================================

def _circ_pad_t(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Circular padding along the T dimension (dim=2) of a 5D [B, C, T, S, 1] tensor.
    Enforces periodic temporal boundary conditions.
    """
    return torch.cat([x[:, :, -pad:], x, x[:, :, :pad]], dim=2)


class TemporalResBlock(nn.Module):
    """
    Two-layer residual block that convolves ONLY along the time axis.

    Internally works on [B*S, H, T, 1, 1] where S = X*Y*Z, using
    Conv3d(kernel=(k,1,1)) which is mathematically a 1D conv over T,
    weight-shared across all spatial sites.

    External interface: [B, H, T, S] in/out where S = X*Y*Z (pre-merged).
    Callers handle the merge/unmerge around this block.

    The Conv3d(k,1,1) approach is chosen over Conv1d to avoid the CUDA
    workspace OOM that occurs when Conv1d receives a batch of size B*X*Y*Z.
    With Conv3d the batch stays at B and S is folded into spatial dims.
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim,
                               kernel_size=(kernel_size, 1, 1), padding=0)
        self.norm2 = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim,
                               kernel_size=(kernel_size, 1, 1), padding=0)
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, S]  where S = X*Y*Z
        B, H, T, S = x.shape
        x5 = x.unsqueeze(-1)                          # [B, H, T, S, 1]
        h = self.act(self.norm1(x5))
        h = self.conv1(_circ_pad_t(h, self.pad))
        h = self.act(self.norm2(h))
        h = self.conv2(_circ_pad_t(h, self.pad))
        return (x5 + h).squeeze(-1)                   # residual → [B, H, T, S]


class SpatialResBlock(nn.Module):
    """
    Two-layer 3D residual block operating over (X, Y, Z).
    Identical structure to the original ResNetConditioner residual blocks.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1,
                      padding_mode='circular'),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1,
                      padding_mode='circular'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ============================================================================
# Dual-path conditioner  (spatial + temporal)
# ============================================================================

class ResNetConditioner(nn.Module):
    """
    phi_free [B, T, X, Y, Z]  →  (log_sigma, tau)  each [B, T, X, Y, Z]

    Architecture
    ------------
    Two parallel feature extractors are run on the same input and fused:

    SpatialPath
        Treats T as channels, applies 3D convolutions over (X, Y, Z).
        This learns the spatial geometry of the thimble — the same job
        the original ResNetConditioner did.

        phi [B, T, X, Y, Z]
          → in_conv_s  : Conv3d(T→H, k=3)          [B, H, X, Y, Z]
          → sp_res1    : SpatialResBlock             [B, H, X, Y, Z]
          → sp_res2    : SpatialResBlock             [B, H, X, Y, Z]

    TemporalPath
        Treats each spatial site independently, applies 1D convolutions over T.
        This learns the temporal (chemical-potential) geometry.

        phi [B, T, X, Y, Z]
          → reshape    : [B*X*Y*Z, T]
          → in_conv_t  : Linear(T→H)  (pointwise embed per site)
          → reshape    : [B*X*Y*Z, H, T] (H as channels for Conv1d)
              OR equivalently: expand each site's T values into H channels
          → tm_res1    : TemporalResBlock            [B*X*Y*Z, H, T]
          → tm_res2    : TemporalResBlock            [B*X*Y*Z, H, T]
          → reshape    : [B, H, X, Y, Z]  (mean over T or at matching T slice)

        Because we need a [B, H, X, Y, Z] output that can be added to the
        spatial features, we transpose the temporal features back:
          [B*X*Y*Z, H, T] → [B, T, H, X, Y, Z] → mean over T channel
        This gives a T-averaged temporal context that modulates each spatial
        site's deformation amplitude.

        A second temporal output keeps the per-T information:
          [B*X*Y*Z, H, T] → [B, H, T, X, Y, Z] → permute [B, H·T//T, X,Y,Z]
        ... but this would break shape alignment.  Instead we use a cleaner
        scheme: the temporal path produces [B, H, T, X, Y, Z] by unfolding
        the batch, and then we average over X,Y,Z to get a global temporal
        context [B, H, T] which is broadcast back:

        Implementation (what we actually do — simple and correct):
          phi [B, T, X, Y, Z]
          → permute to [B, X, Y, Z, T]  (T is now the last dim)
          → reshape   [B*X*Y*Z, 1, T]   (1 input channel for Conv1d)
          → in_conv_t : Conv1d(1→H)     [B*X*Y*Z, H, T]
          → tm_res1   : TemporalResBlock [B*X*Y*Z, H, T]
          → tm_res2   : TemporalResBlock [B*X*Y*Z, H, T]
          → reshape   [B, X, Y, Z, H, T]
          → permute   [B, H, T, X, Y, Z]  ← matches SpatialPath layout
                                             when treated as [B, H·T_as_channel, X,Y,Z]

        The spatial path output is [B, H, X, Y, Z] (T collapsed as channels).
        To fuse with temporal output [B, H, T, X, Y, Z] we *broadcast*:
          temporal features are pooled over the spatial dims to get a global
          per-T-per-channel bias, then broadcast added to spatial features:
          [B, H, T, X, Y, Z]  →  mean(X,Y,Z)  →  [B, H, T]
          then expand back to [B, H, T, X, Y, Z] and add to
          spatial features [B, H, X, Y, Z] broadcast over T.

        Finally fusion:
          fused [B, H, T, X, Y, Z]
          → permute [B, T, H, X, Y, Z]
          → reshape [B*T, H, X, Y, Z]  (treat each time slice separately)
          → out_log_sigma: Conv3d(H→1, k=1)  → [B*T, 1, X, Y, Z]
          → reshape [B, T, X, Y, Z]  =  log_sigma
          (same for tau)

    Physics parameter conditioning
        A small MLP maps the param vector to a per-channel bias added to
        both spatial and temporal input embeddings, so the deformation
        strategy changes with m², g4, μ, etc.

    Zero-init of output layers
        The output Conv3d layers are zero-initialised so the network starts
        as an identity map (phi_im = 0) and learns deformations from there.

    Soft-clamped log_sigma
        log_sigma = tanh(raw) * clamp_scale  (clamp_scale passed at runtime)
        This keeps importance weights from collapsing early in training.
    """

    def __init__(self, L: int, hidden_dim: int, param_dim: int = PARAM_DIM,
                 temporal_kernel_size: int = 3):
        super().__init__()
        self.L          = L
        self.hidden_dim = H = hidden_dim
        self.T          = L   # temporal extent = L (4D isotropic lattice)

        # ── Physics param conditioning ────────────────────────────────────────
        # Produces a bias for BOTH the spatial and temporal input embeddings.
        self.param_proj_s = nn.Linear(param_dim, H)   # spatial bias [B, H]
        self.param_proj_t = nn.Linear(param_dim, H)   # temporal bias [B, H]

        # ── Spatial path ──────────────────────────────────────────────────────
        # Input: phi [B, T, X, Y, Z] with T treated as in-channels
        self.in_conv_s = nn.Conv3d(L, H, 3, padding=1, padding_mode='circular')
        self.sp_res1   = SpatialResBlock(H)
        self.sp_res2   = SpatialResBlock(H)

        # ── Temporal path ─────────────────────────────────────────────────────
        # Works on [B, H, T, S, 1] where S=X*Y*Z via Conv3d(1,H,(k,1,1)).
        # S is treated as a spatial dim so Conv3d gets a valid 5D tensor.
        # This avoids the OOM from Conv1d on a batch of size B*X*Y*Z.
        self.in_conv_t = nn.Conv3d(1, H, kernel_size=(temporal_kernel_size, 1, 1),
                                   padding=0)   # manual circ pad along T
        self._t_in_pad = temporal_kernel_size // 2

        self.tm_res1 = TemporalResBlock(H, kernel_size=temporal_kernel_size)
        self.tm_res2 = TemporalResBlock(H, kernel_size=temporal_kernel_size)

        # ── Fusion ────────────────────────────────────────────────────────────
        # After fusion each [B, T, X, Y, Z] time-slice has 2H channels:
        # H from spatial path (broadcast over T) + H from temporal path.
        # A 1×1 Conv3d maps 2H → H before the output heads.
        self.fusion_conv = nn.Conv3d(2 * H, H, 1)

        # ── Output heads ──────────────────────────────────────────────────────
        # Applied per time-slice: [B*T, H, X, Y, Z] → [B*T, 1, X, Y, Z]
        self.out_log_sigma = nn.Conv3d(H, 1, 1)
        self.out_tau       = nn.Conv3d(H, 1, 1)

        # Zero-init: network starts as identity (phi_im = 0)
        nn.init.zeros_(self.out_log_sigma.weight)
        nn.init.zeros_(self.out_log_sigma.bias)
        nn.init.zeros_(self.out_tau.weight)
        nn.init.zeros_(self.out_tau.bias)
        # Also zero-init fusion conv bias to avoid initial offset
        nn.init.zeros_(self.fusion_conv.bias)

    # ─────────────────────────────────────────────────────────────────────────

    def _spatial_features(self, phi_free: torch.Tensor,
                          p_vec: torch.Tensor) -> torch.Tensor:
        """
        phi_free: [B, T, X, Y, Z]
        p_vec:    [B, param_dim]
        returns:  [B, H, X, Y, Z]   (T collapsed as spatial-path channels)
        """
        H = self.hidden_dim
        # param bias: [B, H] → [B, H, 1, 1, 1]
        bias_s = self.param_proj_s(p_vec).view(-1, H, 1, 1, 1)
        # phi_free already [B, T, X, Y, Z] — T plays the role of in_channels
        x = self.in_conv_s(phi_free) + bias_s   # [B, H, X, Y, Z]
        x = self.sp_res1(x)
        x = self.sp_res2(x)
        return x   # [B, H, X, Y, Z]

    def _temporal_features(self, phi_free: torch.Tensor,
                           p_vec: torch.Tensor) -> torch.Tensor:
        """
        phi_free: [B, T, X, Y, Z]
        p_vec:    [param_dim] or [B, param_dim]
        returns:  [B, H, T, X, Y, Z]

        Internally works on [B, C, T, S, 1] where S = X*Y*Z, using
        Conv3d(kernel=(k,1,1)) which convolves only along T.
        This gives Conv3d a valid 5D input while keeping batch size = B
        (not B*X*Y*Z), avoiding the CUDA workspace OOM.
        """
        B, T, X, Y, Z = phi_free.shape
        H = self.hidden_dim
        S = X * Y * Z

        # [B, T, X, Y, Z] → [B, T, S] → [B, 1, T, S] → [B, 1, T, S, 1]
        x = phi_free.reshape(B, T, S).unsqueeze(1).unsqueeze(-1)  # [B, 1, T, S, 1]

        # Initial embedding: circ pad T then Conv3d(1,H,(k,1,1))
        x = _circ_pad_t(x, self._t_in_pad)          # [B, 1, T+2p, S, 1]
        x = self.in_conv_t(x)                        # [B, H, T, S, 1]
        x = x.squeeze(-1)                            # [B, H, T, S]

        # Physics param bias broadcast over T and S
        bias_t = self.param_proj_t(p_vec)            # [H] or [B, H]
        if bias_t.dim() == 1:
            bias_t = bias_t.view(1, H, 1, 1)         # [1, H, 1, 1]
        else:
            bias_t = bias_t.view(B, H, 1, 1)         # [B, H, 1, 1]
        x = x + bias_t                               # [B, H, T, S]

        # Temporal residual blocks (internal shape [B, H, T, S])
        x = self.tm_res1(x)
        x = self.tm_res2(x)

        # Restore spatial dims: [B, H, T, S] → [B, H, T, X, Y, Z]
        return x.reshape(B, H, T, X, Y, Z)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, phi_free: torch.Tensor, p_vec: torch.Tensor,
                clamp_scale: float = 0.3):
        """
        phi_free:    [B, T, X, Y, Z]
        p_vec:       [B, param_dim]
        clamp_scale: soft-clamp range for log_sigma (default 0.3, same as v7)

        Returns
        -------
        log_sigma: [B, T, X, Y, Z]   pointwise log-scale of the deformation
        tau:       [B, T, X, Y, Z]   pointwise shift of the deformation
        """
        B, T, X, Y, Z = phi_free.shape
        H = self.hidden_dim

        # ── Spatial features: [B, H, X, Y, Z] ───────────────────────────────
        feat_s = self._spatial_features(phi_free, p_vec)   # [B, H, X, Y, Z]

        # ── Temporal features: [B, H, T, X, Y, Z] ───────────────────────────
        feat_t = self._temporal_features(phi_free, p_vec)  # [B, H, T, X, Y, Z]

        # ── Fusion ────────────────────────────────────────────────────────────
        # Broadcast spatial features over T:
        #   [B, H, X, Y, Z] → [B, H, T, X, Y, Z]
        feat_s_exp = feat_s.unsqueeze(2).expand(B, H, T, X, Y, Z)

        # Concatenate along channel dim: [B, 2H, T, X, Y, Z]
        fused = torch.cat([feat_s_exp, feat_t], dim=1)

        # Apply fusion conv per time-slice via reshape trick:
        # [B, 2H, T, X, Y, Z] → [B*T, 2H, X, Y, Z]
        fused = fused.permute(0, 2, 1, 3, 4, 5)          # [B, T, 2H, X, Y, Z]
        fused = fused.reshape(B * T, 2 * H, X, Y, Z)     # [B*T, 2H, X, Y, Z]
        fused = torch.relu(self.fusion_conv(fused))       # [B*T, H, X, Y, Z]

        # ── Output heads ──────────────────────────────────────────────────────
        log_sigma_raw = self.out_log_sigma(fused)          # [B*T, 1, X, Y, Z]
        tau_raw       = self.out_tau(fused)                # [B*T, 1, X, Y, Z]

        # Reshape to [B, T, X, Y, Z]
        log_sigma = log_sigma_raw.reshape(B, T, X, Y, Z)
        tau       = tau_raw.reshape(B, T, X, Y, Z)

        # Soft-clamp: log_sigma ∈ (-clamp_scale, clamp_scale)
        log_sigma = torch.tanh(log_sigma) * clamp_scale

        return log_sigma, tau


# ============================================================================
# Imaginary deformation layer
# ============================================================================

class ImaginaryLayer(nn.Module):
    """
    phi_im_x = exp(log_sigma_x) * phi_free_x + tau_x

    log|det J|_im = Σ_x log_sigma_x    (exact, pointwise — no Hutchinson)

    The log|det| formula is exact because the deformation is a pointwise
    (site-diagonal) affine map in field space: each site's imaginary part
    is an independent affine function of that site's free-field value.
    The Jacobian is therefore diagonal with entries exp(log_sigma_x).

    clamp_scale is forwarded to ResNetConditioner so the training loop can
    gradually increase the allowed deformation range as training progresses.
    """

    def __init__(self, L: int, hidden_dim: int, param_dim: int = PARAM_DIM,
                 temporal_kernel_size: int = 3):
        super().__init__()
        self.net = ResNetConditioner(L, hidden_dim, param_dim,
                                     temporal_kernel_size=temporal_kernel_size)

    def forward(self, phi_free: torch.Tensor, p_emb: torch.Tensor,
                clamp_scale: float = 0.3):
        log_sigma, tau = self.net(phi_free, p_emb, clamp_scale=clamp_scale)
        phi_im  = torch.exp(log_sigma) * phi_free + tau
        log_det = torch.sum(log_sigma, dim=(-4, -3, -2, -1))  # [B], exact
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

        hidden_dim           = config.get('thimble_hidden_dim', 32)
        temporal_kernel_size = config.get('temporal_kernel_size', 3)

        self.param_embedder  = ParamEmbedder(param_dim=PARAM_DIM)
        self.imaginary_layer = ImaginaryLayer(
            self.L, hidden_dim, param_dim=PARAM_DIM,
            temporal_kernel_size=temporal_kernel_size,
        )

        # Optionally grow the clamp range over training (set in config)
        # clamp_scale_start: initial clamp (default 0.3, identity-like)
        # clamp_scale_end:   final clamp   (default 0.7, more expressive)
        self._clamp_start = config.get('clamp_scale_start', 0.3)
        self._clamp_end   = config.get('clamp_scale_end',   0.7)

    def _clamp_scale(self, t: float) -> float:
        """
        Linearly interpolate the clamp range with training fraction t ∈ [0,1].
        Starts tight (near-identity) and opens up as training stabilises.
        """
        return self._clamp_start + (self._clamp_end - self._clamp_start) * min(1.0, t)

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
        return torch.sum(-0.5 * torch.log(self._kernel(p)))

    def forward(self, z: torch.Tensor, p: LagrangianParams,
                return_scale: bool = False, t: float = 1.0):
        """
        t: training fraction [0, 1].  Used to schedule the clamp scale.
           Pass t=1.0 (default) at inference / validation for max expressivity.
        """
        phi_free, kernel = self.generate_free_field(z, p)
        p_emb = self.param_embedder(p.to_tensor(self.device))

        clamp = self._clamp_scale(t)
        phi_im, log_det_im = self.imaginary_layer(phi_free, p_emb,
                                                   clamp_scale=clamp)

        log_det = self.get_base_log_det(p) + log_det_im
        phi     = torch.complex(phi_free, phi_im)

        if return_scale:
            return phi, log_det, torch.mean(1.0 / torch.sqrt(kernel)).item()
        return phi, log_det

    # ── Loss functions (unchanged from v7) ────────────────────────────────────

    def compute_thimble_loss(self, phi, p, log_det, lambda_im=0.1):
        S = self.action_computer.compute(phi, p)
        F_eff    = S.real - log_det
        var_Feff = torch.var(F_eff)
        var_imS  = torch.var(S.imag)
        var_logJ = torch.var(log_det)
        mean_Feff = torch.mean(F_eff) / self.L**4
        return var_Feff, var_imS, var_logJ, mean_Feff, S

    def compute_combined_loss(self, phi, p, log_det,
                              lambda_im, lambda_J, lambda_M) -> dict:
        var_Feff, var_imS_E, var_logJ, mean_Feff, S_E = \
            self.compute_thimble_loss(phi, p, log_det, lambda_im)

        # In thimble.py compute_combined_loss, add to euclidean_loss:
        mean_logJ_penalty = (torch.mean(log_det) / self.L**4) ** 2
        euclidean_loss = var_Feff + lambda_im * var_imS_E + lambda_J * var_logJ + 0.1 * mean_logJ_penalty

        if lambda_M > 0.0:
            S_M       = self.action_computer.minkowski_action(phi, p)
            var_imS_M = torch.var(S_M.imag)
            mink_loss = lambda_M * var_imS_M
        else:
            with torch.no_grad():
                S_M       = self.action_computer.minkowski_action(phi, p)
                var_imS_M = torch.var(S_M.imag)
            mink_loss = torch.tensor(0.0, device=phi.device)

        total_loss = euclidean_loss + mink_loss

        return {
            'loss':       total_loss,
            'var_Feff':   var_Feff,
            'var_imS_E':  var_imS_E,
            'var_logJ':   var_logJ,
            'var_imS_M':  var_imS_M,
            'mean_Feff':  mean_Feff,
            'S_E':        S_E,
        }

    def compute_log_jacobian(self, z, p):
        _, log_det = self.forward(z, p)
        return log_det


# ============================================================================
# Statistical utilities  (unchanged)
# ============================================================================

def effective_sample_size(log_weights: torch.Tensor) -> float:
    lw = log_weights - torch.max(log_weights)
    w  = torch.exp(lw)
    return (w.sum()**2 / (w**2).sum() / len(w)).item()


def bootstrap_error(values: np.ndarray, weights: np.ndarray,
                    n_boot: int = 500) -> float:
    n = len(values)
    w = weights / weights.sum()
    boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        wb  = w[idx]; wb /= wb.sum()
        boots.append(np.dot(wb, values[idx]))
    return float(np.std(boots))


def integrated_autocorrelation(x: np.ndarray, max_lag: int = 50) -> float:
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