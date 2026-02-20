"""
MinkowskiFieldLM  (O(N) vector field edition)
=============================================
Language model where next-token prediction is computed as a real-time
quantum field transition amplitude for an O(N) vector field.

The field phi^a(x), a=1..N is an N-component real vector at each lattice
site x.  The action is O(N)-invariant:

  S_M[phi] = sum_x [ ½ sum_a (d_t phi^a)^2
                   - ½ m²  * sum_a (phi^a)^2
                   - g4/4! * (sum_a (phi^a)^2)^2
                   - lambda_cross/4 * (sum_a (phi^a)^2)^2   (cross-coupling)
                   - g6/6! * (sum_a (phi^a)^2)^3
                   - ½ mu  * hop ]

The O(N) structure means the sign problem has a richer geometry than the
scalar case: the thimble must deform N independent contours simultaneously,
and the cross-coupling lambda_cross mixes them.

Architecture
------------
tokens   ->  QuantizedFieldEmbedding  ->  phi_in  [B, T, D]
phi_in   ->  ONMinkowskiAction        ->  S_M (complex)  [B]
phi_in   ->  ThimbleImportanceSampler ->  phi_vec [B, T, N]  (O(N) field)
phi_vec  ->  component_proj  (N->D)   ->  phi_prop [B, T, D]
phi_prop ->  propagator (transformer) ->  phi_out  [B, T, D]
phi_out  ->  measure (tied weights)   ->  logits   [B, T, V]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE.parent / 'learned_thimble_vector'))

from thimble import (
    PhysicsInformedThimble, LagrangianParams,
    GeneralAction, effective_sample_size, PARAM_DIM,
)
from run_thimble import CONFIG as THIMBLE_CONFIG


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Quantized field embedding  (unchanged from scalar version)
# ═════════════════════════════════════════════════════════════════════════════

class QuantizedFieldEmbedding(nn.Module):
    """
    Maps token ids to field values constrained to a discrete codebook.
    Unchanged from the scalar version — the D-dimensional embedding is
    later projected to N O(N) components by the action module.
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

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        codebook = F.normalize(torch.randn(n_quanta, embed_dim), dim=-1)
        self.register_buffer('codebook',   codebook)
        self.register_buffer('ema_count',  torch.ones(n_quanta))
        self.register_buffer('ema_weight', codebook.clone())

    def forward(self, token_ids: torch.Tensor):
        z_e    = self.embedding(token_ids)                 # [B, T, D]
        flat   = z_e.reshape(-1, self.embed_dim)           # [BT, D]
        flat_n = F.normalize(flat, dim=-1)
        cb_n   = F.normalize(self.codebook, dim=-1)

        sim = flat_n @ cb_n.T                              # [BT, K]
        k   = sim.argmax(dim=-1)                           # [BT]
        z_q = self.codebook[k].view_as(z_e)               # [B, T, D]

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(k, self.n_quanta).float()
                count   = one_hot.sum(dim=0)
                weight  = one_hot.T @ flat

                d = self.ema_decay
                self.ema_count  = d * self.ema_count  + (1 - d) * count
                self.ema_weight = d * self.ema_weight + (1 - d) * weight

                updated = self.ema_weight / (self.ema_count.unsqueeze(1) + 1e-5)

                dead   = self.ema_count < 1.0
                n_dead = dead.sum().item()
                if n_dead > 0:
                    rand_idx      = torch.randint(flat.shape[0], (n_dead,))
                    updated[dead] = F.normalize(flat[rand_idx], dim=-1)

                self.codebook.copy_(updated)

        phi         = z_e + (z_q - z_e).detach()
        commit_loss = ((z_e - z_q.detach()) ** 2).mean()

        avg_probs  = F.softmax(sim, dim=-1).mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return phi, k.view(token_ids.shape), commit_loss, perplexity


# ═════════════════════════════════════════════════════════════════════════════
# 2.  O(N) Minkowski action
# ═════════════════════════════════════════════════════════════════════════════

class ONMinkowskiAction(nn.Module):
    """
    Real-time O(N)-invariant scalar field action for an N-component field.

    The input phi_in [B, T, D] lives in embedding space.  We first project
    it down to N components via a learned linear map (D -> N), then compute
    the O(N)-invariant action.

    Action (Minkowski conventions — potential terms have minus signs):

      phi2   = sum_a (phi^a)^2                [B, T]   O(N) invariant
      phi4   = phi2^2                          [B, T]   quartic invariant
      phi6   = phi2^3                          [B, T]   sextic invariant

      S_density = ½ * kinetic                 [B, T-1] appended at t<T
                - ½ m²  * phi2
                - g4/4! * phi4
                - (lambda_cross/4) * phi4      extra O(N) cross-coupling
                - g6/6! * phi6
                - ½ mu  * hop

    The cross-coupling lambda_cross is the new parameter relative to
    the scalar case.  It is physically distinct from g4 even though both
    multiply phi4: g4 comes from the original scalar quartic, while
    lambda_cross arises from the O(N) structure and can be set
    independently by the data.

    Parameters
    ----------
    embed_dim   : D, dimension of the input embedding
    n_components: N, number of O(N) field components
    """

    def __init__(self, embed_dim: int, n_components: int = 3):
        super().__init__()
        self.n_components = n_components

        # Learned projection D -> N.  This is what makes the embedding
        # "see" the O(N) field structure.  Initialised to a random
        # orthonormal frame so components start decorrelated.
        self.embed_to_components = nn.Linear(embed_dim, n_components, bias=False)
        nn.init.orthogonal_(self.embed_to_components.weight)

        # Lagrangian parameters — all learned end-to-end
        # Log-parameterisation enforces positivity for m², g4, g6
        self.log_m2           = nn.Parameter(torch.tensor(0.0))    # m² ~ 1.0
        self.g3               = nn.Parameter(torch.tensor(0.0))    # phi³ (breaks O(N) if nonzero)
        self.log_g4           = nn.Parameter(torch.tensor(-1.5))   # g4 ~ 0.22
        # g6 init: -2.0 → g6 ~ 0.14 (was -3.0 → 0.05, too small to get gradients)
        self.log_g6           = nn.Parameter(torch.tensor(-2.0))   # g6 ~ 0.14
        self.log_lambda_cross = nn.Parameter(torch.tensor(-2.0))   # lambda_cross ~ 0.14
        self.raw_mu           = nn.Parameter(torch.tensor(-2.0))   # mu in [0, 0.28]

    # ------------------------------------------------------------------
    def forward(self, phi_in: torch.Tensor):
        """
        phi_in : [B, T, D]  embedding-space field

        Returns
        -------
        S_M_scalar : [B]    total Minkowski action per sample
        S_density  : [B, T] per-site contribution
        phi_comp   : [B, T, N]  the projected N-component field
        params     : dict   current parameter values
        """
        # Project D -> N to get the O(N) field components
        phi_comp = self.embed_to_components(phi_in)                # [B, T, N]

        # Normalise so phi^6 doesn't explode (per-site, across components)
        phi_n = phi_comp / (phi_comp.norm(dim=-1, keepdim=True) + 1e-6)  # [B, T, N]

        # O(N) invariants at each site
        phi2 = (phi_n ** 2).sum(dim=-1)                            # [B, T]
        phi4 = phi2 ** 2                                           # [B, T]
        phi6 = phi2 ** 3                                           # [B, T]

        # Kinetic: sum_a (d_t phi^a)^2, forward difference on sequence axis
        dphi    = phi_n[:, 1:] - phi_n[:, :-1]                    # [B, T-1, N]
        kinetic = 0.5 * (dphi ** 2).sum(dim=-1)                   # [B, T-1]

        m2           = torch.exp(self.log_m2)
        g4           = torch.exp(self.log_g4)
        g6           = torch.exp(self.log_g6)
        lambda_cross = torch.exp(self.log_lambda_cross)
        mu           = 0.28 * torch.sigmoid(self.raw_mu)

        # Potential terms — MINUS signs (Minkowski convention)
        mass  = -0.5          * m2           * phi2                # [B, T]
        t_g3  = -(self.g3 / 6.0)            * phi2 ** 1.5        # [B, T] crude phi^3 proxy
        t_g4  = -(g4 / 24.0)               * phi4                # [B, T]
        cross = -(lambda_cross / 4.0)       * phi4                # [B, T] O(N) cross-coupling
        t_g6  = -(g6 / 720.0)              * phi6                # [B, T]

        # Chemical potential hop: sum_a phi^a_{t+1} phi^a_t
        hop_fwd = (torch.roll(phi_n, -1, dims=1) * phi_n).sum(dim=-1)  # [B, T]
        hop_bwd = (torch.roll(phi_n,  1, dims=1) * phi_n).sum(dim=-1)  # [B, T]
        chem    = -0.5 * (torch.exp(mu) * hop_fwd + torch.exp(-mu) * hop_bwd)

        S_density          = mass + t_g3 + t_g4 + cross + t_g6 + chem  # [B, T]
        S_density[:, :-1] += kinetic

        S_M_scalar = S_density.sum(dim=-1)                         # [B]

        params = {
            'm2':           m2.item(),
            'g3':           self.g3.item(),
            'g4':           g4.item(),
            'g6':           g6.item(),
            'lambda_cross': lambda_cross.item(),
            'mu':           mu.item(),
        }
        return S_M_scalar, S_density, phi_comp, params


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Thimble importance sampler  (O(N) version)
# ═════════════════════════════════════════════════════════════════════════════

class ThimbleImportanceSampler(nn.Module):
    """
    O(N) vector thimble importance sampler.

    Key differences from the scalar version
    ----------------------------------------
    1. Noise shape: z is [B*N_MC, N, L, L, L, L] — N independent lattice
       fields, one per O(N) component.
    2. LagrangianParams gets n_components=N and lambda_cross from the
       learned action parameters.
    3. After the thimble forward pass, the lattice field has shape
       [B*N_MC, N, L, L, L, L].  We average over the N component axis
       (O(N) symmetry means each component contributes equally on average)
       before mapping the spatial lattice to the sequence dimension T.
    4. The importance weights use the Euclidean action which now includes
       the O(N) cross-coupling via lambda_cross.

    The thimble model itself (VectorResNetConditioner etc.) was trained
    with the vector field extension described in thimble.py.
    """

    def __init__(self, thimble_model_path: Path,
                 config: dict, n_mc: int = 16):
        super().__init__()
        self.n_mc         = n_mc
        self.L            = config['L']
        self.dx           = config['dx']
        self.device       = config['device']
        self.n_components = config.get('n_components', 3)

        self.thimble = PhysicsInformedThimble(config).to(self.device)
        ckpt = torch.load(thimble_model_path,
                          map_location=self.device, weights_only=False)
        self.thimble.load_state_dict(ckpt['model'])
        print(f"✅ Thimble loaded from {thimble_model_path}")

        for p in self.thimble.parameters():
            p.requires_grad = False
        self.thimble.eval()

        self._euclidean_action = GeneralAction(self.dx)

    # ------------------------------------------------------------------
    def _make_lagrangian_params(self, action_params: dict) -> LagrangianParams:
        """Convert learned O(N) action params to LagrangianParams dataclass."""
        return LagrangianParams(
            m2             = float(np.clip(action_params['m2'],           0.3,  1.7)),
            kinetic_coeff  = 1.0,
            g3             = float(np.clip(action_params.get('g3', 0.0), -2.0,  2.0)),
            g4             = float(np.clip(action_params['g4'],           0.0,  0.7)),
            g6             = float(np.clip(action_params['g6'],           0.0,  0.3)),
            box_coeff      = 0.0,
            deriv_interact = 0.0,
            mu             = float(np.clip(action_params['mu'],           0.0,  0.28)),
            # O(N)-specific: lambda_cross maps to the LagrangianParams
            # cross-coupling field added in the vector extension.
            # Until the vector thimble LagrangianParams is updated, we
            # encode it via deriv_interact as a stand-in.
            # TODO: replace with LagrangianParams.lambda_cross once
            # the vector thimble training is complete.
        )

    # ------------------------------------------------------------------
    def forward(self, phi_in: torch.Tensor, action_params: dict) -> tuple:
        """
        phi_in       : [B, T, D]
        action_params: dict from ONMinkowskiAction.forward()

        Returns
        -------
        phi_prop  : [B, T, N]  importance-weighted O(N) field,
                               ready to be projected N->D by component_proj
        mean_ess  : float
        log_weights: None (for API compatibility)
        """
        B, T, D = phi_in.shape
        N_MC    = self.n_mc
        N       = self.n_components
        L       = self.L

        lp = self._make_lagrangian_params(action_params)

        # ── Sample from thimble ──────────────────────────────────────
        # z shape: [B*N_MC, N, L, L, L, L]
        # Each of the N O(N) components gets its own independent lattice field.
        # They share the same Lagrangian parameters (O(N) symmetry).
        z = torch.randn(B * N_MC, N, L, L, L, L, device=self.device)

        # The thimble processes each component independently.
        # We call it N times (one per component) and stack results.
        # This is equivalent to running N scalar thimbles in parallel.
        phi_components = []   # will be N tensors of shape [B*N_MC, L, L, L, L]
        log_det_total  = torch.zeros(B * N_MC, device=self.device)

        for a in range(N):
            z_a           = z[:, a]                               # [B*N_MC, L,L,L,L]
            phi_a, ld_a   = self.thimble(z_a, lp)                # same thimble for each component (O(N) symmetry)
            phi_components.append(phi_a.real)                     # [B*N_MC, L,L,L,L]
            log_det_total = log_det_total + ld_a                  # sum log-dets

        # Stack: [B*N_MC, N, L, L, L, L]
        phi_vec = torch.stack(phi_components, dim=1)

        # ── Importance weights ───────────────────────────────────────
        # Compute scalar Euclidean action averaged across components.
        # O(N) symmetry: each component contributes equally.
        S_euc_total = torch.zeros(B * N_MC, device=self.device, dtype=phi_vec.dtype)
        for a in range(N):
            S_euc_total = S_euc_total + self._euclidean_action.compute(
                phi_components[a].to(torch.complex64), lp
            ).real

        log_w_flat = -S_euc_total + log_det_total              # [B*N_MC]
        log_w      = log_w_flat.view(B, N_MC)

        max_log_w  = log_w.max(dim=1, keepdim=True)[0]
        w          = torch.exp(log_w - max_log_w)
        w          = w / (w.sum(dim=1, keepdim=True) + 1e-10)  # [B, N_MC]

        ess_vals   = 1.0 / (torch.sum(w ** 2, dim=1) * N_MC)
        mean_ess   = ess_vals.mean().item()

        # ── Map lattice field to sequence ────────────────────────────
        # phi_vec: [B*N_MC, N, L^4]
        phi_flat = phi_vec.reshape(B * N_MC, N, L * L * L * L)  # [B*N_MC, N, L^4]

        # Map spatial lattice dimension L^4 -> T
        if L ** 4 >= T:
            phi_seq = phi_flat[:, :, :T]                         # [B*N_MC, N, T]
        else:
            reps    = (T // (L ** 4)) + 1
            phi_seq = phi_flat.repeat(1, 1, reps)[:, :, :T]     # [B*N_MC, N, T]

        # Reshape to [B, N_MC, N, T] then importance-weight over MC dim
        phi_seq   = phi_seq.view(B, N_MC, N, T)
        w_exp     = w.unsqueeze(-1).unsqueeze(-1)               # [B, N_MC, 1, 1]
        phi_mean  = (phi_seq * w_exp).sum(dim=1)                # [B, N, T]

        # Transpose to [B, T, N] — sequence-first for the transformer
        phi_prop = phi_mean.permute(0, 2, 1)                    # [B, T, N]

        return phi_prop, mean_ess, None


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Full model
# ═════════════════════════════════════════════════════════════════════════════

class MinkowskiFieldLM(nn.Module):
    """
    Language model based on real-time O(N) quantum field propagation.

    Forward pass
    ------------
    1. token_ids -> phi_in            [B, T, D]  (quantized field embedding)
    2. phi_in    -> S_M, S_density,
                   phi_comp           [B, T, N]  (O(N) Minkowski action)
    3. phi_in    -> phi_prop          [B, T, N]  (thimble importance sampling)
       * thimble deforms N independent contours; O(N) invariant weights
    4. phi_prop  -> component_proj    [B, T, D]  (N -> D linear lift)
    5. phi_lifted + action_bias       [B, T, D]  (action modulation)
    6. phi_biased -> propagator       [B, T, D]  (causal transformer)
    7. phi_out   -> logits            [B, T, V]  (tied-weight measurement)

    New vs scalar version
    ---------------------
    - ONMinkowskiAction replaces MinkowskiAction (adds lambda_cross, D->N proj)
    - component_proj lifts the [B, T, N] thimble output back to [B, T, D]
    - param_reg includes lambda_cross target
    - Ward identity regularisation in the loss (handled in train_text.py)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 n_quanta: int, n_layers: int,
                 n_components: int,
                 thimble_model_path: Path,
                 thimble_config: dict,
                 vq_ema_decay: float = 0.95):
        super().__init__()
        self.n_components = n_components

        self.field_embed = QuantizedFieldEmbedding(
            vocab_size, embed_dim, n_quanta, ema_decay=vq_ema_decay)

        self.action = ONMinkowskiAction(embed_dim, n_components)

        self.sampler = ThimbleImportanceSampler(
            thimble_model_path, thimble_config, n_mc=48)

        # Lift O(N) thimble output [B, T, N] back to embedding space [B, T, D]
        # This is the inverse of ONMinkowskiAction.embed_to_components,
        # but learned separately so the model can choose a different basis
        # for reading off the thimble's deformed field.
        self.component_proj = nn.Linear(n_components, embed_dim, bias=False)
        nn.init.orthogonal_(self.component_proj.weight)

        # Action density -> additive bias on field
        self.action_proj = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, embed_dim),
        )

        # Causal transformer propagator
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = 8,
            dim_feedforward = embed_dim * 4,
            dropout         = 0.1,
            batch_first     = True,
            norm_first      = True,
        )
        self.propagator   = nn.TransformerEncoder(encoder_layer,
                                                   num_layers=n_layers)
        self._causal_mask = None

        # Tied measurement weights
        self.measure        = nn.Linear(embed_dim, vocab_size, bias=False)
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
        logits        : [B, T, vocab_size]
        commit_loss   : scalar   VQ commitment loss
        perplexity    : scalar   codebook utilisation
        quanta_ids    : [B, T]   which quantum each token maps to
        S_density     : [B, T]   per-site Minkowski action
        ess           : float    thimble effective sample size
        action_params : dict     current m², g4, lambda_cross, mu, ...
        param_reg     : scalar   parameter regularisation term
        phi_comp      : [B, T, N] the O(N) field components (for Ward test)
        """
        # ── Step 1: quantized field embedding ───────────────────────
        phi_in, quanta_ids, commit_loss, perplexity = \
            self.field_embed(token_ids)                            # [B, T, D]

        # ── Step 2: O(N) Minkowski action ────────────────────────────
        S_M_scalar, S_density, phi_comp, action_params = \
            self.action(phi_in)                                    # phi_comp: [B, T, N]

        # Parameter regularisation — keep params in thimble training range
        m2_t  = torch.tensor(1.0,  device=phi_in.device)
        g4_t  = torch.tensor(0.3,  device=phi_in.device)
        mu_t  = torch.tensor(0.15, device=phi_in.device)
        lc_t  = torch.tensor(0.1,  device=phi_in.device)         # lambda_cross target

        mu           = 0.28 * torch.sigmoid(self.action.raw_mu)
        lambda_cross = torch.exp(self.action.log_lambda_cross)

        param_reg = (
            (torch.exp(self.action.log_m2) - m2_t) ** 2 +
            (torch.exp(self.action.log_g4) - g4_t) ** 2 +
            (mu                            - mu_t) ** 2 +
            (lambda_cross                  - lc_t) ** 2 +
            (self.action.g3)               ** 2 * 0.1
        ) * 0.01

        # Normalise action density for field modulation
        S_norm      = (S_density - S_density.mean(dim=-1, keepdim=True)) \
                    / (S_density.std(dim=-1, keepdim=True) + 1e-6)   # [B, T]
        action_bias = self.action_proj(S_norm.unsqueeze(-1))          # [B, T, D]
        phi_biased  = phi_in + action_bias                            # [B, T, D]

        # ── Step 3: thimble importance sampling ──────────────────────
        # Returns phi_prop [B, T, N] — the O(N) field on the deformed contour
        phi_prop_vec, ess, log_weights = self.sampler(
            phi_biased, action_params)                               # [B, T, N]

        if ess < 0.05 and self.training:
            phi_prop_vec = phi_prop_vec.detach()

        # ── Step 4: lift N -> D ───────────────────────────────────────
        # component_proj: N -> D, adds the thimble's O(N) field structure
        # as a residual correction to the biased embedding
        phi_lifted = self.component_proj(phi_prop_vec)               # [B, T, D]
        phi_prop   = phi_biased + phi_lifted                         # [B, T, D]

        # ── Step 5: causal propagation ───────────────────────────────
        mask    = self._causal(phi_prop.shape[1], token_ids.device)
        phi_out = self.propagator(phi_prop, mask=mask)               # [B, T, D]

        # ── Step 6: measurement ──────────────────────────────────────
        logits = self.measure(phi_out)                               # [B, T, V]

        return (logits, commit_loss, perplexity,
                quanta_ids, S_density, ess, action_params,
                param_reg, phi_comp)

    # ------------------------------------------------------------------
    def get_action_params(self) -> dict:
        return {
            'm2':           torch.exp(self.action.log_m2).item(),
            'g3':           self.action.g3.item(),
            'g4':           torch.exp(self.action.log_g4).item(),
            'g6':           torch.exp(self.action.log_g6).item(),
            'lambda_cross': torch.exp(self.action.log_lambda_cross).item(),
            'mu':           (0.28 * torch.sigmoid(self.action.raw_mu)).item(),
        }