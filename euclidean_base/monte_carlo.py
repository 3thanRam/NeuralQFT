import torch


# ==============================================================================
# ROBUST MCMC (VECTORIZED LOCAL CHECKERBOARD)
# ==============================================================================
class MCMC_Checkerboard:
    def __init__(self, cfg):
        self.cfg = cfg
        self.L = cfg['L']
        self.dx = cfg['dx']
        self.device = cfg['device']
        
        # Initialize Hot
        self.state = torch.randn(1, self.L, self.L, self.L, self.L, device=self.device)
        
        # Masks
        grid = torch.meshgrid(
            torch.arange(self.L), torch.arange(self.L), 
            torch.arange(self.L), torch.arange(self.L), 
            indexing='ij'
        )
        sum_indices = sum(grid).to(self.device)
        self.mask_even = (sum_indices % 2 == 0).float()
        self.mask_odd  = (sum_indices % 2 == 1).float()
        
    def compute_local_action_change(self, phi, noise, mask):
        """
        Compute dS locally for updating sites specified by 'mask'.
        dS = S(phi+noise) - S(phi)
        This depends on:
          1. Mass/Interaction change at site x
          2. Kinetic change at site x (forward links)
          3. Kinetic change at neighbors (backward links entering x)
        """
        phi_new = phi + noise
        
        # 1. Potential Term V(phi)
        # dV = V(phi') - V(phi)
        M2 = self.cfg['M']**2
        g = self.cfg['g']
        dV = (0.5 * M2 * (phi_new**2 - phi**2) + 
              0.25 * g * (phi_new**4 - phi**4))
        
        # 2. Kinetic Term
        # For site x, we need neighbors x+mu and x-mu
        # Since we use checkerboard, neighbors are fixed (masked out in noise)
        
        # Sum of neighbor values
        neighbor_sum = torch.zeros_like(phi)
        for dim in [1, 2, 3, 4]:
            neighbor_sum += torch.roll(phi, 1, dim) + torch.roll(phi, -1, dim)
            
        # dK = (1/dx^2) * [ (phi' - phi_nb)^2 - (phi - phi_nb)^2 ]
        #    = (1/dx^2) * [ phi'^2 - phi^2 - 2*phi'*phi_nb + 2*phi*phi_nb ]
        #    = (1/dx^2) * [ (phi'^2 - phi^2)*2*dim - 2*(phi' - phi)*sum(phi_nb) ]
        
        dK = (1.0 / self.dx**2) * (
            (phi_new**2 - phi**2) * 8.0 -  # 8 neighbors in 4D
            2.0 * (phi_new - phi) * neighbor_sum
        )
        
        # Total local dS (accounting for dx^4 measure)
        # The factor 0.5 in kinetic term is canceled by double counting links?
        # Standard lattice action: S = sum [0.5(dphi)^2 + V]
        # Our dK formula above effectively sums the changes on all 8 links connected to x.
        # Each link contributes 0.5 * (phi_x - phi_y)^2.
        # So the full change is exactly 0.5 * dK_above.
        
        dS_density = 0.5 * dK + dV
        return dS_density * (self.dx**4)

    def step_half(self, mask):
        # Propose
        noise = 0.4 * torch.randn_like(self.state) * mask
        
        # Compute exact local dS
        dS = self.compute_local_action_change(self.state, noise, mask)
        
        # Local Metropolis Step
        # Accept if rand < exp(-dS)
        # Note: Sites are decoupled, so we can use random numbers per site
        rand_vals = torch.rand_like(self.state)
        accept_mask = (rand_vals < torch.exp(-dS)) * mask
        
        # Update
        self.state = self.state + noise * accept_mask
        
        return accept_mask.sum() / mask.sum()

    def step(self):
        acc1 = self.step_half(self.mask_even)
        acc2 = self.step_half(self.mask_odd)
        return (acc1 + acc2) / 2.0

    def run_simulation(self, n_samples=200):
        print(f"  MCMC: Thermalizing...")
        for _ in range(1000): self.step()
        
        print(f"  MCMC: Sampling {n_samples}...")
        p2_sum, p4_sum = 0, 0
        for _ in range(n_samples):
            for _ in range(5): self.step() # Decorrelate
            p2_sum += torch.mean(self.state**2).item()
            p4_sum += torch.mean(self.state**4).item()
            
        return p2_sum/n_samples, p4_sum/n_samples