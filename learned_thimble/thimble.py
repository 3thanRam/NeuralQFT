import torch
import torch.nn as nn
import numpy as np
import torch.fft as fft
from dataclasses import dataclass

@dataclass
class LagrangianParams:
    m2: float
    kinetic_coeff: float
    g3: float
    g4: float
    g6: float
    box_coeff: float
    deriv_interact: float
    
    def __repr__(self):
        return f"L={self.kinetic_coeff:.1f}(dphi)^2 + {self.m2:.1f}phi^2 + {self.g4:.2f}phi^4"

class GeneralAction:
    def __init__(self, dx: float):
        self.dx = dx
    
    def compute(self, phi: torch.Tensor, params: LagrangianParams) -> torch.Tensor:
        """Vectorized computation of the Action S[phi] over a batch."""
        dx = self.dx
        vol_element = dx**4

        # Use negative indexing to target the last 4 lattice dimensions [T, X, Y, Z]
        dt = torch.roll(phi, shifts=-1, dims=-4) - phi
        dx_ = torch.roll(phi, shifts=-1, dims=-3) - phi
        dy = torch.roll(phi, shifts=-1, dims=-2) - phi
        dz = torch.roll(phi, shifts=-1, dims=-1) - phi

        # Kinetic Density: 0.5 * (d_mu phi)^2
        kinetic_density = 0.5 * params.kinetic_coeff * (dt**2 + dx_**2 + dy**2 + dz**2) / (dx**2)

        # Potential Densities
        phi2 = phi**2
        mass_density = 0.5 * params.m2 * phi2
        interaction_density = (params.g3/3.0)*(phi**3) + (params.g4/4.0)*(phi**4)

        S_density = kinetic_density + mass_density + interaction_density

        # Sum over lattice dimensions, preserving Batch dimension [Batch]
        S = torch.sum(S_density, dim=(-4, -3, -2, -1)) * vol_element
        return S

class ConvResidualNet(nn.Module):
    """
    3D Convolutional Network acting on Lattice.
    Treats Time (dim 1) as Channels for 3D Conv processing.
    """
    def __init__(self, L, hidden_dim=32):
        super().__init__()
        self.L = L
        
        # Maps physics params to a learnable bias for the channels
        self.param_map = nn.Linear(7, L)
        
        self.net = nn.Sequential(
            nn.Conv3d(L, hidden_dim, kernel_size=3, padding=1, padding_mode='circular'),
            nn.SiLU(), # Smooth activation for better Jacobian gradients
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='circular'),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, 2*L, kernel_size=3, padding=1, padding_mode='circular')
        )
        
        # Initialize final layer to small values to break symmetry without exploding phase
        nn.init.normal_(self.net[-1].weight, std=1e-6)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phi_free, params):
        # Embed physics parameters as a channel-wise bias
        emb = self.param_map(params) 
        p_bias = emb.view(-1, self.L, 1, 1, 1) # Matches batch size
        
        x = phi_free + p_bias
        out = self.net(x) 
        
        # Split into Real and Imaginary components of the complex deformation
        delta_re, delta_im = torch.split(out, self.L, dim=1)
        return delta_re, delta_im

class PhysicsInformedThimble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.L = config['L']
        self.device = config['device']
        self.dx = config['dx']
        self.action_computer = GeneralAction(self.dx)
        
        # Precompute Momentum Grid and Laplacian Eigenvalues
        k = 2 * np.pi * torch.arange(self.L, device=self.device) / self.L
        kt, kx, ky, kz = torch.meshgrid(k, k, k, k, indexing='ij')
        eig = 2*(1-torch.cos(kt)) + 2*(1-torch.cos(kx)) + 2*(1-torch.cos(ky)) + 2*(1-torch.cos(kz))
        self.register_buffer('laplacian_eig', eig)
        
        self.residual_net = ConvResidualNet(self.L, hidden_dim=config.get('thimble_hidden_dim', 32))

    def encode_lagrangian(self, p: LagrangianParams):
        return torch.tensor([p.m2, p.kinetic_coeff, p.g3, p.g4, p.g6, 
                           p.box_coeff, p.deriv_interact], 
                           dtype=torch.float32, device=self.device)

    def generate_free_field(self, z, params):
        """Maps Gaussian noise to Free Field configurations using FFT."""
        k_val = params.kinetic_coeff * (self.dx**2) * self.laplacian_eig
        m_val = params.m2 * (self.dx**4)
        kernel = torch.clamp(k_val + m_val, min=1e-8)
        
        scale = 1.0 / torch.sqrt(kernel)
        z_fft = fft.fftn(z.to(torch.complex64), dim=(-4,-3,-2,-1))
        
        # Broadcast scale over batch dimension
        phi_free = fft.ifftn(z_fft * scale.unsqueeze(0), dim=(-4,-3,-2,-1)).real
        return phi_free, kernel

    def get_base_log_det(self, params):
        """Analytical log-determinant of the noise-to-free-field transformation."""
        k_val = params.kinetic_coeff * (self.dx**2) * self.laplacian_eig
        m_val = params.m2 * (self.dx**4)
        kernel = torch.clamp(k_val + m_val, min=1e-8)
        return torch.sum(-0.5 * torch.log(kernel))

    def forward(self, z, params, return_scale=False):
        phi_free, kernel = self.generate_free_field(z, params)
        p_vec = self.encode_lagrangian(params)
        
        delta_re, delta_im = self.residual_net(phi_free, p_vec)
        
        phi = torch.complex(phi_free + delta_re, delta_im)
        
        if return_scale:
            avg_scale = torch.mean(1.0/torch.sqrt(kernel))
            return phi, avg_scale
        return phi

    def compute_log_jacobian(self, z, params, n_vectors=4):
        """Hutchinson Trace Estimator for the Jacobian log-determinant."""
        log_det_base = self.get_base_log_det(params)
        
        with torch.enable_grad():
            phi_free, _ = self.generate_free_field(z, params)
            phi_free = phi_free.detach().requires_grad_(True)
            
            p_vec = self.encode_lagrangian(params)
            delta_re, _ = self.residual_net(phi_free, p_vec)
            
            traces = []
            for _ in range(n_vectors):
                v = torch.randn_like(delta_re)
                # Compute gradient of the residual w.r.t the input field
                grad_v = torch.autograd.grad(delta_re, phi_free, grad_outputs=v, 
                                           create_graph=True, retain_graph=True)[0]
                # Stochastic trace estimate (vectorized sum)
                traces.append(torch.sum(grad_v * v, dim=(-4, -3, -2, -1)))
            
            trace_estimate = torch.stack(traces).mean(dim=0)
            return log_det_base + trace_estimate

    def compute_thimble_loss(self, phi, params, log_jac):
        """Calculates Free Energy density and the Imaginary-Variance penalty."""
        S = self.action_computer.compute(phi, params)
        
        # Minimize variance of Im(S) to solve the sign problem
        im_var = torch.var(S.imag)
        
        # Effective Action (Re(S) - LogDet)
        free_E = torch.mean(S.real - log_jac)
        
        return im_var, free_E, S