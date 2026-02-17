import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from thimble import LagrangianParams, PhysicsInformedThimble

def plot_training(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ep = np.arange(len(history['loss']))
    
    axes[0,0].plot(ep, history['loss'], 'k-', alpha=0.6)
    axes[0,0].set_title('Free Energy Density')
    
    axes[0,1].semilogy(ep, history['im_var'], 'm-', alpha=0.6)
    axes[0,1].set_title('Thimble Constraint Im(S)')
    
    axes[1,0].plot(ep, history['scale'], 'c-')
    axes[1,0].set_title('Free Field Scale')
    
    plt.savefig(save_path)
    plt.close()

def compute_topological_charge(phi):
    """
    Calculates a proxy for the topological winding of the complex field.
    phi: [Batch, T, X, Y, Z]
    """
    # Look at the real part of the field winding
    # We use the curvature (Laplacian) as a proxy for the charge density
    dt = torch.roll(phi.real, -1, dims=-4) - phi.real
    dx = torch.roll(phi.real, -1, dims=-3) - phi.real
    dy = torch.roll(phi.real, -1, dims=-2) - phi.real
    dz = torch.roll(phi.real, -1, dims=-1) - phi.real
    
    # Quadratic density Q ~ \int (d phi)^2
    q_density = (dt**2 + dx**2 + dy**2 + dz**2)
    
    # Sum over lattice to get total charge per batch element
    Q = torch.sum(q_density, dim=(-4, -3, -2, -1))
    return Q
class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = PhysicsInformedThimble(config).to(config['device'])
        # Lower LR slightly for ConvNet stability
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        
    def train_step(self, epoch):
        self.opt.zero_grad()
        
        # 1. Physics Curriculum (m2 and g4)
        m2 = 1.0 # Keep mass stable
        g4_target = 0.5
        # Ramp g4 over the first 50% of epochs
        g4 = g4_target * min(1.0, epoch / (self.config['n_epochs'] * 0.5))
        
        params = LagrangianParams(m2=m2, kinetic_coeff=1.0, g3=0.0, g4=g4, 
                                 g6=0.0, box_coeff=0.0, deriv_interact=0.0)
        
        z = torch.randn(self.config['batch_size'], self.config['L'], 
                        self.config['L'], self.config['L'], self.config['L'], 
                        device=self.config['device'])
        
        # 2. Forward Pass
        log_jac = self.model.compute_log_jacobian(z, params)
        phi, scale = self.model(z, params, return_scale=True)
        
        # 3. Diagnostics: Topological Charge
        Q = compute_topological_charge(phi)
        topo_suscept = torch.var(Q) # High variance = Good exploration
        
        # 4. Loss calculation
        im_var, free_E, S = self.model.compute_thimble_loss(phi, params, log_jac)
        
        # Adaptive Thimble Weighting (Ramp up lambda)
        thimble_lambda = min(100.0, 1.0 + epoch * 0.02)
        loss = (free_E / self.config['L']**4) + thimble_lambda * im_var
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.opt.step()
        
        return {
            'loss': (free_E / self.config['L']**4).item(),
            'im_var': im_var.item(),
            'topo_suscept': topo_suscept.item(),
            'scale': scale.item()
        }

    def train(self, epochs):
        h = {'loss':[], 'im_var':[], 'scale':[],'topo_suscept':[]}
        print(f"Training {epochs} epochs (Conv3d Thimble)...")
        for e in range(epochs):
            m = self.train_step(e)
            for k,v in m.items(): h[k].append(v)
            if e%100==0:
                print(f"Ep {e}: Loss={m['loss']:.4f} ImVar={m['im_var']:.2e} topo_suscept={m['topo_suscept']:.2e}")
        return h

def train(CONFIG):
    t = Trainer(CONFIG)
    h = t.train(CONFIG['n_epochs'])
    torch.save({'model': t.model.state_dict(), 'config': CONFIG, 'history': h}, 
               CONFIG['data_dir']/'universal_thimble_model.pt')
    plot_training(h, CONFIG['data_dir']/'conv_training.png')

