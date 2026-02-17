import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# NEURAL NETWORK
# ==============================================================================
class MultiScaleWickRotation(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.max_freq = int(np.ceil(np.sqrt(4 * (L//2)**2)))
        self.theta_per_freq = nn.Parameter(torch.ones(self.max_freq + 2) * 0.1)
        
        k = torch.arange(L).float()
        k_centered = torch.where(k <= L//2, k, k - L)
        KT, KX, KY, KZ = torch.meshgrid(k_centered, k_centered, k_centered, k_centered, indexing='ij')
        freq_mag = torch.sqrt(KT**2 + KX**2 + KY**2 + KZ**2)
        self.register_buffer('freq_mag_idx', freq_mag.long().clamp(max=self.max_freq))

    def forward(self, z, force_theta=None):
        if force_theta is not None:
            theta = torch.full_like(self.freq_mag_idx, force_theta, dtype=torch.float32)
        else:
            theta = self.theta_per_freq[self.freq_mag_idx]
            
        z_k = torch.fft.fftn(z, dim=(1,2,3,4), norm="ortho")
        rotator = torch.exp(1j * theta)
        phi = torch.fft.ifftn(z_k * rotator, dim=(1,2,3,4), norm="ortho")
        
        # LogDet = Sum(i * theta)
        log_det = 1j * torch.sum(theta).unsqueeze(0).repeat(z.shape[0])
        return phi, log_det, theta

class FourierScaleFlow(nn.Module):
    def __init__(self, L, dx, M):
        super().__init__()
        n = torch.arange(L, dtype=torch.float32)
        k = torch.where(n <= L//2, n, n - L) * (2.0 * np.pi / L)
        eigen = (2.0 / dx**2) * (1.0 - torch.cos(k))
        Lap = (eigen.view(L,1,1,1) + eigen.view(1,L,1,1) + 
               eigen.view(1,1,L,1) + eigen.view(1,1,1,L))
        
        G_k = 1.0 / (Lap + M**2)
        target_log_scales = 0.5 * torch.log(G_k) - 2.0 * np.log(dx)
        self.log_scales_4d = nn.Parameter(target_log_scales)
    
    def forward(self, z):
        scales = torch.exp(self.log_scales_4d)
        z_k = torch.fft.fftn(z, dim=(1,2,3,4), norm="ortho")
        phi_k = z_k * scales
        phi = torch.fft.ifftn(phi_k, dim=(1,2,3,4), norm="ortho").real
        log_det = torch.sum(self.log_scales_4d).unsqueeze(0).repeat(z.shape[0])
        return phi, log_det

class AffineCoupling(nn.Module):
    def __init__(self, channels, hidden_dim, mask_parity):
        super().__init__()
        self.mask_parity = mask_parity
        self.net = nn.Sequential(
            nn.Conv3d(channels + 1, hidden_dim, 3, padding=1, padding_mode='circular'),  # +1 for theta
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, padding_mode='circular'),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, 2 * channels, 3, padding=1, padding_mode='circular')
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, theta_value):
        B, T, X, Y, Z = x.shape
        theta_channel = torch.full((B, 1, X, Y, Z), theta_value, device=x.device)
        grid = torch.meshgrid(torch.arange(X), torch.arange(Y), torch.arange(Z), indexing='ij')
        mask_3d = ((grid[0] + grid[1] + grid[2]) % 2 == self.mask_parity).float().to(x.device)
        mask = mask_3d.view(1, 1, X, Y, Z).expand(B, T, X, Y, Z)
        
        x_masked = x * mask
        x_with_theta = torch.cat([x_masked, theta_channel], dim=1)
        s, t = torch.chunk(self.net(x_with_theta), 2, dim=1)
        s = torch.tanh(s)
        x_out = x_masked + (1 - mask) * (x * torch.exp(s) + t)
        log_det = torch.sum(s * (1 - mask), dim=(1,2,3,4))
        return x_out, log_det

class ExactQFTFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config['L']
        self.fourier_flow = FourierScaleFlow(config['L'], config['dx'], config['M'])
        self.coupling_layers = nn.ModuleList([
            AffineCoupling(self.L, config['hidden_dim'], (i%2)) 
            for i in range(config['n_layers'])
        ])
        self.wick = MultiScaleWickRotation(self.L)
        
    def forward(self, z, force_theta=None):
        x, log_det_fourier = self.fourier_flow(z)
        if force_theta is not None:
            theta_val = force_theta
        else:
            theta_val = self.wick.theta_per_freq.mean()

        log_det_coupling = 0
        for layer in self.coupling_layers:
            x, ld = layer(x, theta_val)  
            log_det_coupling = log_det_coupling + ld
        
        phi, log_det_wick, theta = self.wick(x, force_theta=force_theta)
        total_log_det = log_det_fourier + log_det_coupling + log_det_wick
        return phi, total_log_det, theta