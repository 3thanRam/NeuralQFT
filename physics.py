import torch
import numpy as np
# ==============================================================================
# PHYSICS ENGINES
# ==============================================================================
def get_action_density(phi, dx, M, g):
    # Forward difference for global action calculation
    dt = (torch.roll(phi, -1, 1) - phi) / dx
    dx_ = (torch.roll(phi, -1, 2) - phi) / dx
    dy = (torch.roll(phi, -1, 3) - phi) / dx
    dz = (torch.roll(phi, -1, 4) - phi) / dx
    
    grad_sq = dt**2 + dx_**2 + dy**2 + dz**2 
    kinetic_mink = dt**2 - (dx_**2 + dy**2 + dz**2) 
    
    mass_term = M**2 * phi**2
    interaction = (g / 4.0) * phi**4
    
    return grad_sq, kinetic_mink, mass_term, interaction

def get_euclidean_action(phi, dx, M, g):
    grad_sq, _, mass, inter = get_action_density(phi, dx, M, g)
    L_density = 0.5 * grad_sq + 0.5 * mass + inter
    return torch.sum(L_density, dim=(1,2,3,4)) * (dx**4)

def get_minkowski_action(phi, dx, M, g):
    _, kin_mink, mass, inter = get_action_density(phi, dx, M, g)
    L_density = 0.5 * (kin_mink - mass) - inter
    return torch.sum(L_density, dim=(1,2,3,4)) * (dx**4)

def get_theory_propagator(L, dx, M, theta):
    """Unified Analytic Continuation for Propagator."""
    n = torch.arange(L, dtype=torch.float64)
    k = 2.0 * np.pi * n / L
    eigen = (4.0 / dx**2) * torch.sin(k / 2.0)**2
    
    KT = eigen.view(L,1,1,1)
    KX = eigen.view(1,L,1,1)
    KY = eigen.view(1,1,L,1)
    KZ = eigen.view(1,1,1,L)
    
    # Metric signature rotation
    # At theta=0: (+,+,+,+) Euclidean
    # At theta=pi/2: (-,+,+,+) Minkowski  
    K_time = KT * torch.cos(torch.tensor(2*theta)) - 1j * KT * torch.sin(torch.tensor(2*theta))
    K_spatial = KX + KY + KZ
    
    epsilon = 1e-6
    denominator = K_time + K_spatial + M**2 + 1j*epsilon
    G_k = 1.0 / denominator
    
    G_x = torch.fft.ifftn(G_k)
    return G_x.numpy()* (1.0 / dx**4)
