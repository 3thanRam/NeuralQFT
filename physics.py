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
    """
    Unified Analytic Continuation for Propagator.
    
    FIXED: Correct Wick rotation formula
    """
    n = torch.arange(L, dtype=torch.float64)  
    k = 2.0 * np.pi * n / L
    k_centered = torch.where(k.real <= np.pi, k, k - 2.0 * np.pi)
    
    # Lattice derivative eigenvalues
    eigen = (2.0 / dx**2) * (1.0 - torch.cos(k_centered * dx))
    
    KT = eigen.view(L,1,1,1)
    KX = eigen.view(1,L,1,1)
    KY = eigen.view(1,1,L,1)
    KZ = eigen.view(1,1,1,L)
    
    rotation_factor = torch.exp(torch.tensor(2j * theta, dtype=torch.complex128))
    K_time = KT * rotation_factor
    K_spatial = KX + KY + KZ
    
    # Propagator
    epsilon = 1e-8
    denominator = K_time + K_spatial + M**2 + 1j*epsilon
    G_k = 1.0 / denominator
    
    # Transform to position space
    G_x = torch.fft.ifftn(G_k, dim=(0,1,2,3))
    
    # Normalization: keep your original for now
    return G_x.cpu().numpy() * (1.0 / dx**4)