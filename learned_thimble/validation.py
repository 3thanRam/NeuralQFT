"""
Complete Validation Suite for Physics-Informed Thimble Network.

Tests included:
1. Free Field Propagator G(t) vs Analytical (FFT)
2. <phi^2> Expectation vs Exact Lattice Sum
3. Perturbative <phi^2> Correction vs 1-Loop Tadpole Diagram
4. Consistency Checks (Symmetry, Positivity, Kurtosis)

Includes OOM fixes and batched sampling.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Import the model and physics definitions
from thimble import PhysicsInformedThimble, LagrangianParams, GeneralAction

class AnalyticalSolutions:
    """Exact solutions for Lattice Scalar Field Theory."""
    
    @staticmethod
    def get_lattice_eigenvalues(L, dx=1.0):
        """Compute eigenvalues of the discrete Laplacian 2(1-cos(k))."""
        k_idx = torch.arange(L)
        k = 2 * np.pi * k_idx / L
        # 4D Grid of momenta
        kt, kx, ky, kz = torch.meshgrid(k, k, k, k, indexing='ij')
        
        # Eigenvalues: 2(1-cos(kx)) + ...
        eig = 2.0*(1.0 - torch.cos(kt)) + \
              2.0*(1.0 - torch.cos(kx)) + \
              2.0*(1.0 - torch.cos(ky)) + \
              2.0*(1.0 - torch.cos(kz))
        return eig

    @staticmethod
    def get_action_kernel(L, m, kinetic_coeff, dx):
        """
        Returns the quadratic kernel D(k) in momentum space.
        S = 0.5 * sum_k D(k) |phi_k|^2
        """
        eig = AnalyticalSolutions.get_lattice_eigenvalues(L, dx)
        
        # Match GeneralAction: 
        # Kinetic: 0.5 * k_coeff * dx^2 * (dphi)^2  -> Fourier: eig
        # Mass:    0.5 * m^2 * dx^4 * phi^2
        
        # D(k) = kinetic_coeff * dx^2 * eig + m^2 * dx^4
        kernel = kinetic_coeff * (dx**2) * eig + (m**2) * (dx**4)
        return torch.clamp(kernel, min=1e-9)

    @staticmethod
    def expected_phi2_free(L, m, kinetic_coeff=1.0, dx=1.0):
        """Exact <phi^2> for free discrete lattice field."""
        kernel = AnalyticalSolutions.get_action_kernel(L, m, kinetic_coeff, dx)
        
        # <phi^2> = (1/Vol) * Sum(1/D(k))
        # Note: 1/D(k) is variance of phi_k. Parseval's thm handles the rest.
        total_variance = torch.sum(1.0 / kernel)
        return (total_variance / (L**4)).item()

    @staticmethod
    def expected_phi2_perturbative(L, m, g, kinetic_coeff=1.0, dx=1.0):
        """
        1-Loop correction to <phi^2> for lambda*phi^4/4 theory.
        Formula: <phi^2> = <phi^2>_0 - 3 * g * <phi^2>_0 * (1/Vol)*Sum(1/D(k)^2)
        (Tadpole diagram correction to mass)
        """
        kernel = AnalyticalSolutions.get_action_kernel(L, m, kinetic_coeff, dx)
        
        # Zero-order expectation
        phi2_0 = torch.sum(1.0 / kernel) / (L**4)
        
        # "Bubble" sum for the correction term (integral of propagator squared)
        bubble = torch.sum(1.0 / (kernel**2)) / (L**4)
        
        # Correction: - delta_m^2 * bubble
        # delta_m^2 = 3 * g * phi2_0
        # This assumes interaction g*phi^4/4. Our code uses params.g4 * phi^4 / 4.
        correction = -3.0 * g * phi2_0 * bubble
        
        return (phi2_0 + correction).item()

    @staticmethod
    def get_propagator_free(L, m, kinetic_coeff=1.0, dx=1.0):
        kernel = AnalyticalSolutions.get_action_kernel(L, m, kinetic_coeff, dx)
        G_k = 1.0 / kernel  # [L, L, L, L]

        # G(t, Δx=0) = (1/L⁴) Σ_{all k} (1/D(k)) * e^{i k_t t}
        # Sum over spatial k first, keeping the k_t dependence
        G_kt = torch.sum(G_k, dim=(1, 2, 3))  # [L], sum over kx,ky,kz — NO divide by L³

        # ifft: result[t] = (1/L) Σ_{kt} G_kt[kt] * e^{i kt t}
        # Combined: (1/L) * Σ_{kt} [Σ_{kperp} 1/D(k)] * e^{i kt t}
        # = (1/L) * L³ * (1/L⁴) * Σ_k e^{i kt t}/D(k)  — need factor 1/L³ back
        G_t = torch.fft.ifft(G_kt).real / (L**3)  # divide by L³ here instead
        return G_t.numpy()

class ValidationSuite:
    def __init__(self, config, model_path):
        self.config = config
        self.device = config['device']
        self.L = config['L']
        self.dx = config['dx']
        
        # Load Model
        self.model = PhysicsInformedThimble(config).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            print(f"✅ Loaded model from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
            
        self.model.eval()
        self.action_tool = GeneralAction(self.dx)
        self.results = {}

    def compute_observables(self, params, n_samples=1000, batch_size=50):
        """
        Compute observables using Importance Sampling with OOM protection.
        """
        accum_phi = 0.0
        accum_phi2 = 0.0
        accum_phi4 = 0.0
        accum_prop = np.zeros(self.L)
        total_weight = 0.0
        
        n_batches = max(1, n_samples // batch_size)
        print(f"   Sampling {n_samples} configs ({n_batches} batches)...", end='', flush=True)
        
        with torch.no_grad():
            for _ in range(n_batches):
                z = torch.randn(batch_size, self.L, self.L, self.L, self.L, device=self.device)
                
                # 1. Forward Pass
                phi, _ = self.model(z, params, return_scale=True)
                
                # 2. Compute Importance Weights
                # w = exp(-S + logJ)
                log_jac = self.model.compute_log_jacobian(z, params)#, create_graph=False
                
                # Compute Action (Batched manually if needed, or via loop)
                S_list = []
                for i in range(batch_size):
                    # Compute EXACT action for reweighting
                    s_val = self.action_tool.compute(phi[i], params)
                    S_list.append(s_val.real)
                S = torch.stack(S_list)
                
                log_w = -S + log_jac
                
                # Stable Softmax
                max_log_w = torch.max(log_w)
                w = torch.exp(log_w - max_log_w)
                
                # 3. Accumulate Observables
                batch_weight = torch.sum(w).item()
                total_weight += batch_weight
                
                # Scalar Obs (averaged over volume)
                phi_mean = torch.mean(phi.real, dim=(1,2,3,4))
                phi2_mean = torch.mean(phi.real**2, dim=(1,2,3,4))
                phi4_mean = torch.mean(phi.real**4, dim=(1,2,3,4))
                
                accum_phi += torch.sum(w * phi_mean).item()
                accum_phi2 += torch.sum(w * phi2_mean).item()
                accum_phi4 += torch.sum(w * phi4_mean).item()
                
                # Propagator G(t) = <phi(0) phi(t, 0, 0, 0)>
                # Slice at x=y=z=0
                phi_real = phi.real  # [B, T, X, Y, Z]
                # Correlate each time slice t with t=0, averaged over all spatial sites
                phi_t0 = phi_real[:, 0:1, :, :, :]   # [B, 1, X, Y, Z]  — the t=0 slice
                corr = torch.mean(phi_real * phi_t0, dim=(2, 3, 4))  # [B, T], averaged over X,Y,Z

                phi_mean_spatial = torch.mean(phi_real, dim=(2, 3, 4))   # [B, T]
                phi_mean_t0      = phi_mean_spatial[:, 0:1]              # [B, 1]
                corr_connected   = corr - phi_mean_spatial * phi_mean_t0 # [B, T]
                
                for t in range(self.L):
                    accum_prop[t] += torch.sum(w * corr_connected[:, t]).item()
        
        print(" Done.")
        
        # Normalize
        if total_weight == 0: total_weight = 1e-12
        
        return {
            'phi': accum_phi / total_weight,
            'phi2': accum_phi2 / total_weight,
            'phi4': accum_phi4 / total_weight,
            'propagator': accum_prop / total_weight
        }

    def test_propagator(self):
        print("\n=== Test 1: Free Field Propagator G(t) ===")
        m = 1.0
        params = LagrangianParams(m2=m**2, kinetic_coeff=1.0, g3=0.0, g4=0.0, g6=0.0, box_coeff=0.0, deriv_interact=0.0)
        
        obs = self.compute_observables(params, n_samples=10000)
        G_thimble = obs['propagator']
        G_exact = AnalyticalSolutions.get_propagator_free(self.L, m, dx=self.dx)
        
        # Avoid div by zero
        denom = np.abs(G_exact) + 1e-10
        rel_err = np.abs(G_thimble - G_exact) / denom
        avg_err = np.mean(rel_err)
        
        print(f"   Max G(0): {G_exact[0]:.4f}")
        print(f"   Avg Relative Error: {avg_err*100:.2f}%")
        
        self.results['propagator'] = {'thimble': G_thimble, 'exact': G_exact, 'error': avg_err}

    def test_phi2_expectation(self):
        print("\n=== Test 2: <phi^2> Expectation (Vacuum Variance) ===")
        m = 1.0
        params = LagrangianParams(m2=m**2, kinetic_coeff=1.0, g3=0.0, g4=0.0, g6=0.0, box_coeff=0.0, deriv_interact=0.0)
        
        obs = self.compute_observables(params, n_samples=1000)
        val_thimble = obs['phi2']
        val_exact = AnalyticalSolutions.expected_phi2_free(self.L, m, dx=self.dx)
        
        err = abs(val_thimble - val_exact) / abs(val_exact)
        print(f"   Thimble: {val_thimble:.6f}")
        print(f"   Exact:   {val_exact:.6f}")
        print(f"   Error:   {err*100:.2f}%")
        
        self.results['phi2'] = {'thimble': val_thimble, 'exact': val_exact, 'error': err}

    def test_perturbative(self):
        print("\n=== Test 3: Perturbative Correction (phi^4 theory) ===")
        m = 1.0
        g = 0.1 # Weak coupling
        params = LagrangianParams(m2=m**2, kinetic_coeff=1.0, g3=0.0, g4=g, g6=0.0, box_coeff=0.0, deriv_interact=0.0)
        
        # Use more samples for interacting theory
        obs = self.compute_observables(params, n_samples=2000)
        val_thimble = obs['phi2']
        val_pert = AnalyticalSolutions.expected_phi2_perturbative(self.L, m, g, dx=self.dx)
        
        err = abs(val_thimble - val_pert) / abs(val_pert)
        print(f"   Thimble:      {val_thimble:.6f}")
        print(f"   Perturbative: {val_pert:.6f} (1-Loop Tadpole)")
        print(f"   Error:        {err*100:.2f}%")
        
        self.results['perturbative'] = {'thimble': val_thimble, 'exact': val_pert, 'error': err}

    def test_consistency(self):
        print("\n=== Test 4: Consistency Checks ===")
        m = 1.0
        g = 0.2
        params = LagrangianParams(m2=m**2, kinetic_coeff=1.0, g3=0.0, g4=g, g6=0.0, box_coeff=0.0, deriv_interact=0.0)
        
        obs = self.compute_observables(params, n_samples=500)
        
        phi_mean = obs['phi']
        phi2 = obs['phi2']
        phi4 = obs['phi4']
        
        # 1. Symmetry <phi> ~ 0
        check_sym = abs(phi_mean) < 0.05
        # 2. Positivity <phi^4> > 0
        check_pos = phi4 > 0
        # 3. Kurtosis/Inequality <phi^4> >= <phi^2>^2
        # For Gaussian it is 3 * <phi^2>^2. For interacting it deviates but strictly >= 1.
        check_ineq = phi4 >= (phi2**2)
        
        print(f"   <phi> = {phi_mean:.4f} (~0) ....... {'✅ PASS' if check_sym else '❌ FAIL'}")
        print(f"   <phi^4> = {phi4:.4f} (>0) ......... {'✅ PASS' if check_pos else '❌ FAIL'}")
        print(f"   <phi^4>/<phi^2>^2 = {phi4/phi2**2:.2f} (>=1) .. {'✅ PASS' if check_ineq else '❌ FAIL'}")
        
        self.results['consistency'] = {'sym': check_sym, 'pos': check_pos, 'ineq': check_ineq}

    def plot_summary(self):
        save_path = self.config['data_dir'] / 'validation_summary.png'
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Propagator
        if 'propagator' in self.results:
            ax = fig.add_subplot(gs[0, :])
            res = self.results['propagator']
            t = np.arange(len(res['thimble']))
            ax.semilogy(t, res['exact'], 'o-', label='Exact (Free)', linewidth=2, alpha=0.7)
            ax.semilogy(t, res['thimble'], 's--', label='Thimble', linewidth=2, alpha=0.7)
            ax.set_title(f"Propagator G(t) (Avg Error: {res['error']*100:.1f}%)", fontsize=14)
            ax.set_xlabel("Time Slice")
            ax.set_ylabel("G(t)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 2. Phi2
        if 'phi2' in self.results:
            ax = fig.add_subplot(gs[1, 0])
            res = self.results['phi2']
            labels = ['Exact (Free)', 'Thimble']
            vals = [res['exact'], res['thimble']]
            ax.bar(labels, vals, color=['blue', 'orange'], alpha=0.7, edgecolor='black')
            ax.set_title(f"<phi^2> Free Theory (Error: {res['error']*100:.1f}%)", fontsize=14)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add text
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        
        # 3. Perturbative
        if 'perturbative' in self.results:
            ax = fig.add_subplot(gs[1, 1])
            res = self.results['perturbative']
            labels = ['Perturbative (1-Loop)', 'Thimble']
            vals = [res['exact'], res['thimble']]
            ax.bar(labels, vals, color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax.set_title(f"<phi^2> Interacting g=0.1 (Error: {res['error']*100:.1f}%)", fontsize=14)
            ax.grid(True, axis='y', alpha=0.3)
            
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\n✅ Validation plots saved to {save_path}")

def run_all_validations(config):
    print("="*60)
    print("STARTING FULL VALIDATION SUITE")
    print("="*60)
    
    suite = ValidationSuite(config, config['data_dir'] / 'universal_thimble_model.pt')
    
    suite.test_propagator()
    suite.test_phi2_expectation()
    suite.test_perturbative()
    suite.test_consistency()
    
    suite.plot_summary()
    print("\nValidation Complete.")
