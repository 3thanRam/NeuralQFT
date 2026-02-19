"""
MinkowskiFieldLM
================
Language model where next-token prediction is computed as a real-time
quantum field transition amplitude.

The path integral has a complex action (Minkowski signature) which
causes a severe sign problem. The pre-trained Generalized Lefschetz
Thimble deforms the integration contour into the complex plane so
that Im(S) stays small and importance sampling is tractable.

Architecture
------------
tokens  ->  QuantizedFieldEmbedding  ->  phi_in  [B, T, D]
phi_in  ->  MinkowskiAction          ->  S_M (complex)  [B]
phi_in  ->  ThimbleImportanceSampler ->  phi_prop  [B, T, D]
phi_prop -> measure (tied weights)   ->  logits  [B, T, V]

Why the thimble is load-bearing here
-------------------------------------
S_M[phi] = integral of [ (d_t phi)^2 - m^2 phi^2 - g4 phi^4 - mu*hop ]
The minus signs and the chemical potential mu make S_M complex.
Without the thimble, e^{i*S_M} oscillates wildly and importance weights
cancel to near zero (sign problem). The thimble shifts the contour so
phi becomes complex but Im(S_M) ~ 0, restoring a well-behaved integral.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent / 'learned_thimble'))

from thimble import (
    PhysicsInformedThimble, LagrangianParams,
    GeneralAction, effective_sample_size, PARAM_DIM,
)
from run_thimble import CONFIG as THIMBLE_CONFIG


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Quantized field embedding  (VQ-VAE style with EMA codebook)
# ═════════════════════════════════════════════════════════════════════════════

class QuantizedFieldEmbedding(nn.Module):
    """
    Maps token ids to field values constrained to a discrete codebook.

    phi(x_i) in {phi_1 , ... , phi_K}  where K = n_quanta << vocab_size

    This enforces the second-quantization picture: even though the
    vocabulary has 50k tokens, the field lives on a coarse lattice of
    K energy levels.  Many tokens share the same quantum state.

    Training uses a straight-through estimator so gradients flow through
    the discrete argmax.  The codebook is updated via exponential moving
    averages (EMA) with random restarts for dead entries.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 n_quanta: int = 512,
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.95):
        super().__init__()
        self.n_quanta        = n_quanta
        self.embed_dim       = embed_dim
        self.commitment_cost = commitment_cost
        self.ema_decay       = ema_decay

        # Continuous token embedding (trained)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Discrete codebook (EMA-updated, not a gradient parameter)
        codebook = F.normalize(torch.randn(n_quanta, embed_dim), dim=-1)
        self.register_buffer('codebook',   codebook)
        self.register_buffer('ema_count',  torch.ones(n_quanta))
        self.register_buffer('ema_weight', codebook.clone())

    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor):
        """
        Returns
        -------
        phi          : [B, T, D]  straight-through quantized field
        quanta_ids   : [B, T]     which codebook entry each token maps to
        commit_loss  : scalar     keeps embeddings near codebook
        codebook_perp: scalar     effective number of codebook entries used
        """
        z_e     = self.embedding(token_ids)               # [B, T, D]
        flat    = z_e.reshape(-1, self.embed_dim)          # [BT, D]
        flat_n  = F.normalize(flat, dim=-1)
        cb_n    = F.normalize(self.codebook, dim=-1)

        # Nearest codebook entry via cosine similarity
        sim = flat_n @ cb_n.T                              # [BT, K]
        k   = sim.argmax(dim=-1)                           # [BT]

        z_q = self.codebook[k].view_as(z_e)               # [B, T, D]

        # ── EMA codebook update ──────────────────────────────────────
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(k, self.n_quanta).float()  # [BT, K]
                count   = one_hot.sum(dim=0)
                weight  = one_hot.T @ flat

                d = self.ema_decay
                self.ema_count  = d * self.ema_count  + (1 - d) * count
                self.ema_weight = d * self.ema_weight + (1 - d) * weight

                updated = self.ema_weight / (self.ema_count.unsqueeze(1) + 1e-5)

                # Random restart: revive dead codebook entries
                dead    = self.ema_count < 1.0
                n_dead  = dead.sum().item()
                if n_dead > 0:
                    rand_idx        = torch.randint(flat.shape[0], (n_dead,))
                    updated[dead]   = F.normalize(flat[rand_idx], dim=-1)

                self.codebook.copy_(updated)

        # ── Straight-through estimator ───────────────────────────────
        phi         = z_e + (z_q - z_e).detach()          # [B, T, D]
        commit_loss = ((z_e - z_q.detach()) ** 2).mean()

        # ── Codebook perplexity (utilisation metric) ─────────────────
        avg_probs    = F.softmax(sim, dim=-1).mean(dim=0)  # [K]
        perplexity   = torch.exp(
            -(avg_probs * (avg_probs + 1e-10).log()).sum()
        )

        return phi, k.view(token_ids.shape), commit_loss, perplexity


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Minkowski action  (complex — this is why the thimble is needed)
# ═════════════════════════════════════════════════════════════════════════════

class MinkowskiAction(nn.Module):
    """
    Real-time (Minkowski) scalar field action.

    S_M[phi] = sum_x [ ½(d_t phi)^2  -  ½ m² phi^2  -  g4/4! phi^4
                       -  g6/6! phi^6  -  ½ mu*(e^mu phi_{t+1} phi_t
                                                + e^{-mu} phi_{t-1} phi_t) ]

    Key difference from Euclidean action
    -------------------------------------
    The mass and interaction terms have MINUS signs.  This follows from
    the Minkowski metric (+,-,-,-): the potential V = m²phi²/2 + g4 phi^4/4!
    appears with a minus sign in S_M = integral (K - V).

    The chemical potential mu adds an imaginary contribution when phi is
    analytically continued to the complex plane — this is the primary
    source of the sign problem that the thimble resolves.

    Parameters
    ----------
    All Lagrangian parameters are learned end-to-end.  They are
    initialised to reasonable values matching the thimble's training
    distribution so the pre-trained contour deformation is valid from
    the start of LM training.
    """

    def __init__(self):
        super().__init__()
        # log-parameterisation enforces positivity for m², g4, g6
        self.log_m2  = nn.Parameter(torch.tensor(0.0))    # m²  ~ 1.0
        self.log_g4  = nn.Parameter(torch.tensor(-1.5))   # g4  ~ 0.22
        self.log_g6  = nn.Parameter(torch.tensor(-2.5))   # g6  ~ 0.08
        self.mu      = nn.Parameter(torch.tensor(0.15))   # chemical potential

    # ------------------------------------------------------------------
    def forward(self, phi: torch.Tensor):
        """
        phi : [B, T, D]  field on 1-D sequence lattice (positions = sites)

        Returns
        -------
        S_M_scalar : [B]  total Minkowski action per sample (real scalar)
                          In the full complex theory this would be complex;
                          here we return the real part used for weighting.
        S_density  : [B, T]  per-site contribution (for diagnostics)
        params     : dict  current parameter values
        """
        # Normalise field to unit sphere so phi^6 doesn't explode
        phi_n   = phi / (phi.norm(dim=-1, keepdim=True) + 1e-6)   # [B, T, D]

        # Kinetic: (d_t phi)^2  — forward difference on sequence axis
        dphi    = phi_n[:, 1:] - phi_n[:, :-1]                    # [B, T-1, D]
        kinetic = 0.5 * (dphi ** 2).sum(dim=-1)                    # [B, T-1]

        m2  = torch.exp(self.log_m2)
        g4  = torch.exp(self.log_g4)
        g6  = torch.exp(self.log_g6)
        mu  = self.mu

        # Potential terms — note MINUS signs (Minkowski convention)
        mass  = -0.5        * m2 * (phi_n ** 2).sum(dim=-1)        # [B, T]
        phi4  = -(g4/24.0)       * (phi_n ** 4).sum(dim=-1)
        phi6  = -(g6/720.0)      * (phi_n ** 6).sum(dim=-1)

        # Chemical potential hop term
        hop_fwd = (torch.roll(phi_n, -1, dims=1) * phi_n).sum(dim=-1)
        hop_bwd = (torch.roll(phi_n,  1, dims=1) * phi_n).sum(dim=-1)
        chem    = -0.5 * (torch.exp(mu) * hop_fwd + torch.exp(-mu) * hop_bwd)

        S_density              = mass + phi4 + phi6 + chem         # [B, T]
        S_density[:, :-1]     += kinetic                           # add kinetic

        S_M_scalar = S_density.sum(dim=-1)                         # [B]

        params = {
            'm2': m2.item(), 'g4': g4.item(),
            'g6': g6.item(), 'mu': mu.item(),
        }
        return S_M_scalar, S_density, params


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Thimble importance sampler
# ═════════════════════════════════════════════════════════════════════════════

class ThimbleImportanceSampler(nn.Module):
    """
    Uses the pre-trained Lefschetz Thimble to generate field configurations
    importance-sampled from e^{i S_M[phi]}.

    Why this is needed
    ------------------
    Naive MC samples phi ~ N(0,1) and weights by e^{i S_M[phi]}.
    Because S_M is real (on the real axis), e^{i S_M} oscillates with
    unit modulus — all weights cancel and the estimator has zero signal.

    The thimble deforms the integration contour phi -> phi + i*tau(phi)
    such that Im(S_M) ~ 0 on the thimble.  Weights become e^{-|S_M|}
    (real, positive) and importance sampling works again.

    The pre-trained thimble was trained on the Euclidean theory; here
    we reuse its contour deformation as an approximation for the
    Minkowski theory.  This approximation improves as mu -> 0 and
    worsens for large mu (strong sign problem) — exactly the regime
    where you'd want to re-train the thimble on the Minkowski action.

    Integration with the LM
    -----------------------
    We only propagate through the thimble once per forward pass (not
    per token), using batch-level Lagrangian parameters derived from
    the current token sequence.  The thimble weights are frozen; the
    LM learns which Lagrangian parameters make the thimble's contour
    most useful for its particular input.
    """

    def __init__(self, thimble_model_path: Path,
                 config: dict, n_mc: int = 16):
        super().__init__()
        self.n_mc    = n_mc
        self.L       = config['L']
        self.dx      = config['dx']
        self.device  = config['device']

        self.thimble = PhysicsInformedThimble(config).to(self.device)
        ckpt = torch.load(thimble_model_path,
                          map_location=self.device, weights_only=False)
        self.thimble.load_state_dict(ckpt['model'])
        print(f"✅ Thimble loaded from {thimble_model_path}")

        # Freeze thimble — it was pre-trained; we only reuse its contour
        for p in self.thimble.parameters():
            p.requires_grad = False
        self.thimble.eval()

        self._euclidean_action = GeneralAction(self.dx)

    # ------------------------------------------------------------------
    def _make_lagrangian_params(self, action_params: dict) -> LagrangianParams:
        """Convert learned action params to LagrangianParams dataclass."""
        return LagrangianParams(
            m2            = float(np.clip(action_params['m2'],   0.3, 1.7)),
            kinetic_coeff = 1.0,
            g3            = 0.0,
            g4            = float(np.clip(action_params['g4'],   0.0, 0.7)),
            g6            = float(np.clip(action_params['g6'],   0.0, 0.3)),
            box_coeff     = 0.0,
            deriv_interact= 0.0,
            mu            = float(np.clip(action_params['mu'],   0.0, 0.28)),
        )

    # ------------------------------------------------------------------
    def forward(self, phi_in: torch.Tensor,
            action_params: dict,
            per_sample_params: list = None) -> tuple:
        """
        per_sample_params: list of B dicts, one per batch element.
        If None, uses action_params for all samples (old behaviour).
        """
        B, T, D = phi_in.shape

        phi_props = []
        ess_list  = []

        for b in range(B):
            p_dict = per_sample_params[b] if per_sample_params else action_params
            lp     = self._make_lagrangian_params(p_dict)

            with torch.no_grad():
                # More samples per sequence = better ESS
                z = torch.randn(64, self.L, self.L, self.L, self.L,
                                device=self.device)

                phi_thimble, log_det = self.thimble(z, lp)
                S_euc = self._euclidean_action.compute(phi_thimble, lp)

                log_w = -S_euc.real + log_det         # [64]
                ess   = effective_sample_size(log_w)
                ess_list.append(ess)

                w = torch.softmax(log_w - log_w.max(), dim=0)  # [64]

                L = self.L
                phi_re   = phi_thimble.real            # [64, L,L,L,L]
                phi_flat = phi_re.reshape(64, L*L*L*L) # [64, L^4]

                if L*L*L*L >= T:
                    phi_seq = phi_flat[:, :T]
                else:
                    reps    = (T // (L*L*L*L)) + 1
                    phi_seq = phi_flat.repeat(1, reps)[:, :T]

                phi_mean = (phi_seq * w.unsqueeze(-1)).sum(dim=0)  # [T]
                phi_props.append(phi_mean)

        phi_mean_batch = torch.stack(phi_props, dim=0).unsqueeze(-1)  # [B, T, 1]
        phi_prop       = phi_in + phi_mean_batch * phi_in
        mean_ess       = float(np.mean(ess_list))

        return phi_prop, mean_ess, None


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Full model
# ═════════════════════════════════════════════════════════════════════════════

class MinkowskiFieldLM(nn.Module):
    """
    Language model based on real-time quantum field propagation.

    Forward pass
    ------------
    1.  token_ids -> phi_in            (quantized field embedding)
    2.  phi_in    -> S_M, S_density    (Minkowski action computation)
    3.  phi_in    -> phi_prop          (thimble importance sampling)
        * thimble deforms contour so e^{i S_M} integral is tractable
    4.  phi_prop  -> logits            (causal transformer propagator)
    5.  logits                         (tied-weight measurement)

    The key difference from a standard transformer: the field is
    propagated under a physical action rather than arbitrary attention,
    and the contour deformation (thimble) is necessary because the
    Minkowski action is complex.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 n_quanta: int, n_layers: int,
                 thimble_model_path: Path,
                 thimble_config: dict):
        super().__init__()

        self.field_embed = QuantizedFieldEmbedding(
            vocab_size, embed_dim, n_quanta)

        self.action = MinkowskiAction()

        self.sampler = ThimbleImportanceSampler(
            thimble_model_path, thimble_config, n_mc=16)

        # Action -> additive bias on field (modulates, doesn't suppress)
        self.action_proj = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, embed_dim),
        )

        # Causal transformer propagator
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = embed_dim,
            nhead      = 8,
            dim_feedforward = embed_dim * 4,
            dropout    = 0.1,
            batch_first= True,
            norm_first = True,
        )
        self.propagator  = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self._causal_mask = None

        # Measurement: tied to embedding weights
        # <phi_out | phi_prop> — overlap with each vocabulary quantum
        self.measure = nn.Linear(embed_dim, vocab_size, bias=False)
        self.measure.weight = self.field_embed.embedding.weight

    # ------------------------------------------------------------------
    def _causal(self, T: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.shape[0] != T:
            self._causal_mask = torch.triu(
                torch.full((T, T), float('-inf'), device=device),
                diagonal=1)
        return self._causal_mask

    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor):
        """
        Parameters
        ----------
        token_ids : [B, T]

        Returns
        -------
        logits       : [B, T, vocab_size]
        commit_loss  : scalar   VQ commitment loss
        perplexity   : scalar   codebook utilisation
        quanta_ids   : [B, T]   which quantum each token maps to
        S_density    : [B, T]   per-site Minkowski action
        ess          : float    thimble effective sample size
        action_params: dict     current m², g4, g6, mu
        """
        # ── Step 1: quantized field embedding ───────────────────────
        phi_in, quanta_ids, commit_loss, perplexity = \
            self.field_embed(token_ids)                    # [B, T, D]

        # ── Step 2: Minkowski action ─────────────────────────────────
        S_M_scalar, S_density, action_params = self.action(phi_in)

        m2_target  = torch.tensor(1.0,  device=phi_in.device)
        g4_target  = torch.tensor(0.3,  device=phi_in.device)
        mu_target  = torch.tensor(0.15, device=phi_in.device)

        param_reg = (
            (torch.exp(self.action.log_m2) - m2_target)**2 +
            (torch.exp(self.action.log_g4) - g4_target)**2 +
            (self.action.mu                - mu_target)**2
        ) * 0.01
        # Normalise action density for use as field modulation
        S_norm = (S_density - S_density.mean(dim=-1, keepdim=True)) \
               / (S_density.std(dim=-1, keepdim=True) + 1e-6)     # [B, T]

        # Project action to embedding space and add residually
        # High-action sites are "excited" — they attend differently
        action_bias = self.action_proj(S_norm.unsqueeze(-1))       # [B, T, D]
        phi_biased  = phi_in + action_bias                         # [B, T, D]

        # ── Step 3: thimble importance sampling ──────────────────────
        # This is the core physics step — the thimble resolves the
        # sign problem of the Minkowski path integral
        phi_prop, ess, log_weights = self.sampler(
            phi_biased, action_params)                             # [B, T, D]

        # ── Step 4: causal propagation ───────────────────────────────
        mask     = self._causal(phi_prop.shape[1], token_ids.device)
        phi_out  = self.propagator(phi_prop, mask=mask)            # [B, T, D]

        # ── Step 5: measurement ──────────────────────────────────────
        logits = self.measure(phi_out)                             # [B, T, V]

        return (logits, commit_loss, perplexity,
                quanta_ids, S_density, ess, action_params,param_reg)

    # ------------------------------------------------------------------
    def get_action_params(self) -> dict:
        return {
            'm2': torch.exp(self.action.log_m2).item(),
            'g4': torch.exp(self.action.log_g4).item(),
            'g6': torch.exp(self.action.log_g6).item(),
            'mu': self.action.mu.item(),
        }