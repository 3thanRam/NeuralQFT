import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from physics import get_euclidean_action,get_minkowski_action,get_theory_propagator
from monte_carlo import MCMC_Checkerboard
from model import ExactQFTFlow
# ==============================================================================
# EXPERIMENT
# ==============================================================================

class QFTExperiment:
    def __init__(self, config):
        self.cfg = config
        self.model = ExactQFTFlow(config).to(config['device'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        print(f"Initialized QFT Flow on {config['device']}")

    def pretrain_step(self):
        self.optimizer.zero_grad()
        z = torch.randn(self.cfg['batch_size'], self.cfg['L'], self.cfg['L'], 
                       self.cfg['L'], self.cfg['L'], device=self.cfg['device'])
        
        phi_complex, log_det, _ = self.model(z, force_theta=0.0)
        phi_real = phi_complex.real
        
        S_E = get_euclidean_action(phi_real, self.cfg['dx'], self.cfg['M'], self.cfg['g'])
        log_p_z = -0.5 * torch.sum(z**2, dim=(1,2,3,4))
        loss = torch.mean(S_E + log_p_z - log_det.real)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), S_E.mean().item()

    def finetune_step(self,epoch):
        self.optimizer.zero_grad()

        max_theta = min(1.57, 0.3 + epoch * 0.003)  
    
        if np.random.rand() < 0.3:
            theta_target = np.random.uniform(max(0, max_theta - 0.5), max_theta)
        else:
            theta_target = np.random.uniform(0, max_theta)

        z = torch.randn(self.cfg['batch_size'], self.cfg['L'], self.cfg['L'],
                       self.cfg['L'], self.cfg['L'], device=self.cfg['device'])

        phi, log_det, _ = self.model(z, force_theta=theta_target)

        S_M = get_minkowski_action(phi, self.cfg['dx'], self.cfg['M'], self.cfg['g'])
        log_p_z = -0.5 * torch.sum(z**2, dim=(1,2,3,4))
        log_w = 1j * S_M + log_det - log_p_z

        phase = log_w.imag
        phase_centered = phase - phase.detach().mean()
        sign_loss = -torch.log(torch.abs(torch.mean(torch.exp(1j * phase_centered))) + 1e-8)

        vol = self.cfg['L']**4
        eff_loss = torch.mean(torch.abs(log_w.real - log_w.real.detach().mean())) / vol

        loss = sign_loss + 1.0 * eff_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return sign_loss.item(), eff_loss.item(), theta_target, max_theta

    def check_physics_mcmc(self):
        print("\n=== PHYSICS CHECK: FLOW vs MCMC (EUCLIDEAN) ===")
        self.model.eval()
        
        z = torch.randn(1000, self.cfg['L'], self.cfg['L'], 
                       self.cfg['L'], self.cfg['L'], device=self.cfg['device'])
        with torch.no_grad():
            phi, log_det, _ = self.model(z, force_theta=0.0)
            phi = phi.real
        
        S = get_euclidean_action(phi, self.cfg['dx'], self.cfg['M'], self.cfg['g'])
        log_q = -0.5 * torch.sum(z**2, dim=(1,2,3,4)) - log_det.real
        log_w = -S - log_q
        
        log_w_shifted = log_w - log_w.max()
        w = torch.exp(log_w_shifted)
        w = w / w.sum()
        
        phi2 = torch.mean(phi**2, dim=(1,2,3,4))
        phi4 = torch.mean(phi**4, dim=(1,2,3,4))
        
        phi2_flow = torch.sum(phi2 * w).item()
        phi4_flow = torch.sum(phi4 * w).item()
        U_L_flow = 1.0 - phi4_flow / (3.0 * phi2_flow**2)
        
        print(f"[FLOW] <φ²>={phi2_flow:.4f}, U_L={U_L_flow:.4f}")
        
        mcmc = MCMC_Checkerboard(self.cfg)
        phi2_mc, phi4_mc = mcmc.run_simulation(n_samples=200)
        U_L_mc = 1.0 - phi4_mc / (3.0 * phi2_mc**2)
        
        print(f"[MCMC] <φ²>={phi2_mc:.4f}, U_L={U_L_mc:.4f}")
        
        err = abs(phi2_flow - phi2_mc) / phi2_mc
        if err < 0.10: 
            print(f"✅ PASSED (Error: {err*100:.1f}%)")
        else:
            print(f"❌ FAILED (Error: {err*100:.1f}%)")

    def validate_analytic_continuation(self, theta_target):
        """Compute propagator and error metrics at given theta."""
        self.model.eval()
        G_sum = torch.zeros(self.cfg['L'], self.cfg['L'], self.cfg['L'], self.cfg['L'], 
                           dtype=torch.complex128)
        with torch.no_grad():
            for _ in range(20):
                z = torch.randn(50, self.cfg['L'], self.cfg['L'], self.cfg['L'], 
                              self.cfg['L'], device=self.cfg['device'])
                phi, _, _ = self.model(z, force_theta=theta_target)
                phi_cpu = phi.cpu().to(torch.complex128)

                phi_0 = phi_cpu[:, 0:1, 0:1, 0:1, 0:1]
                G_sum += torch.sum(phi_cpu * phi_0, dim=0)
        with torch.no_grad():
            z_test = torch.randn(500, self.cfg['L'], self.cfg['L'], self.cfg['L'], 
                                self.cfg['L'], device=self.cfg['device'])
            phi_test, log_det_test, _ = self.model(z_test, force_theta=theta_target)

            S_M = get_minkowski_action(phi_test, self.cfg['dx'], self.cfg['M'], self.cfg['g'])
            log_w = -1j * S_M + log_det_test - (-0.5 * torch.sum(z_test**2, dim=(1,2,3,4)))

            log_w_real = log_w.real
            w = torch.exp(log_w_real - log_w_real.max())
            w_norm = w / w.sum()

            ess = 1.0 / torch.sum(w_norm**2)
            ess_ratio = ess / len(z_test)

        G_learned = (G_sum / 1000.0).numpy()
        G_theory = get_theory_propagator(self.cfg['L'], self.cfg['dx'], 
                                         self.cfg['M'], theta_target)
        print(f"  θ={theta_target:.3f}: "
          f"|G_learned|: min={np.min(np.abs(G_learned)):.2e}, "
          f"max={np.max(np.abs(G_learned)):.2e}, "
          f"mean={np.mean(np.abs(G_learned)):.2e}")
        print(f"  θ={theta_target:.3f}: "
          f"|G_theory|: min={np.min(np.abs(G_theory)):.2e}, "
          f"max={np.max(np.abs(G_theory)):.2e}, "
          f"mean={np.mean(np.abs(G_theory)):.2e}")
        # Compute errors
        diff = G_learned - G_theory

        # Real and imaginary errors (RMS over all points)
        error_real = np.sqrt(np.mean(np.abs(diff.real)**2))
        error_imag = np.sqrt(np.mean(np.abs(diff.imag)**2))

        # Relative errors (normalized by theory magnitude)
        theory_mag = np.sqrt(np.mean(np.abs(G_theory)**2))
        rel_error_real = error_real / theory_mag
        rel_error_imag = error_imag / theory_mag

        # Total complex error
        error_total = np.sqrt(np.mean(np.abs(diff)**2))
        rel_error_total = error_total / theory_mag

        return {
            'theta': theta_target,
            'error_real': error_real,
            'error_imag': error_imag,
            'error_total': error_total,
            'rel_error_real': rel_error_real,
            'rel_error_imag': rel_error_imag,
            'rel_error_total': rel_error_total,
            'G_learned': G_learned,
            'G_theory': G_theory,
            'ess_ratio': ess_ratio.item()
        }

    def run(self):
        print("=== PHASE 1: EUCLIDEAN PRETRAINING ===")
        for i in range(self.cfg['pretrain_epochs']):
            loss, se = self.pretrain_step()
            if i % 200 == 0: 
                print(f"Step {i:4d}: Loss={loss:.1f}, <S_E>={se:.1f}")

        self.check_physics_mcmc()

        print("\n=== PHASE 2: FINE-TUNING (Multi-Theta) ===")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 5e-5

        for i in range(self.cfg['fine_tune_epochs']):
            s_loss, e_loss, theta, max_theta = self.finetune_step(i)
            if i % 100 == 0:
                print(f"Step {i:4d}: SignLoss={s_loss:.2f}, EffLoss={e_loss:.4f}, "f"Theta={theta:.3f}, MaxTheta={max_theta:.3f}")

        print("\n=== ANALYTIC CONTINUATION ERROR ANALYSIS ===")

        # Fine sweep of theta values
        theta_values = np.linspace(0, np.pi/2, 20)
        results = []

        for theta in theta_values:
            result = self.validate_analytic_continuation(theta)
            results.append(result)
            print(f"θ={theta:.3f}: Error={result['rel_error_total']:.4f}, ESS={result['ess_ratio']:.3f}")

        # Create error plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        thetas = [r['theta'] for r in results]

        # Plot 1: Absolute errors
        ax = axes[0, 0]
        ax.plot(thetas, [r['error_real'] for r in results], 'o-', label='Real', linewidth=2)
        ax.plot(thetas, [r['error_imag'] for r in results], 's-', label='Imag', linewidth=2)
        ax.plot(thetas, [r['error_total'] for r in results], '^-', label='Total', linewidth=2, color='black')
        ax.set_xlabel('θ (radians)', fontsize=12)
        ax.set_ylabel('RMS Error', fontsize=12)
        ax.set_title('Absolute Propagator Error vs Rotation Angle', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=np.pi/2, color='red', linestyle='--', alpha=0.5, label='Minkowski')

        # Plot 2: Relative errors (log scale)
        ax = axes[0, 1]
        ax.semilogy(thetas, [r['rel_error_real'] for r in results], 'o-', label='Real', linewidth=2)
        ax.semilogy(thetas, [r['rel_error_imag'] for r in results], 's-', label='Imag', linewidth=2)
        ax.semilogy(thetas, [r['rel_error_total'] for r in results], '^-', label='Total', linewidth=2, color='black')
        ax.set_xlabel('θ (radians)', fontsize=12)
        ax.set_ylabel('Relative Error (log scale)', fontsize=12)
        ax.set_title('Relative Error Growth (Normalized)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.axvline(x=np.pi/2, color='red', linestyle='--', alpha=0.5)

        # Plot 3: Sample propagators at key angles
        ax = axes[1, 0]
        key_indices = [0, len(results)//4, len(results)//2, -1]
        for idx in key_indices:
            r = results[idx]
            G_t = r['G_learned'][:, 0, 0, 0]
            ax.plot(np.abs(G_t), 'o-', label=f'θ={r["theta"]:.2f}', alpha=0.7)
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('|G(t)|', fontsize=12)
        ax.set_title('Learned Propagator Magnitude', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Error decomposition
        ax = axes[1, 1]
        error_real_vals = [r['error_real'] for r in results]
        error_imag_vals = [r['error_imag'] for r in results]
        ax.fill_between(thetas, 0, error_real_vals, alpha=0.5, label='Real Error', color='C0')
        ax.fill_between(thetas, error_real_vals, 
                        [r+i for r,i in zip(error_real_vals, error_imag_vals)], 
                        alpha=0.5, label='Imag Error', color='C1')
        ax.plot(thetas, [r['error_total'] for r in results], 'k-', linewidth=2, label='Total')
        ax.set_xlabel('θ (radians)', fontsize=12)
        ax.set_ylabel('Error Contribution', fontsize=12)
        ax.set_title('Error Decomposition', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=np.pi/2, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.cfg['data_dir']/'error_vs_theta.png', dpi=150)
        print(f"\nSaved error analysis to {self.cfg['data_dir']/'error_vs_theta.png'}")


        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

        # Plot propagator magnitudes
        ax = axes2[0, 0]
        for idx in [0, len(results)//4, len(results)//2, -1]:
            r = results[idx]
            mag = np.abs(r['G_theory'][:, 0, 0, 0])
            # Filter out zeros for log plot
            mag = np.maximum(mag, 1e-10)  # Clamp to small positive value
            ax.plot(mag, '--', label=f'Theory θ={r["theta"]:.2f}', alpha=0.7)
        ax.set_title('Theory Propagator Magnitude')
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylim(1e-10, None)  # Set lower limit
        ax.grid(True, alpha=0.3)

        ax = axes2[0, 1]
        for idx in [0, len(results)//4, len(results)//2, -1]:
            r = results[idx]
            mag = np.abs(r['G_learned'][:, 0, 0, 0])
            mag = np.maximum(mag, 1e-10)
            ax.plot(mag, '-', label=f'Learned θ={r["theta"]:.2f}', alpha=0.7)
        ax.set_title('Learned Propagator Magnitude')
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylim(1e-10, None)
        ax.grid(True, alpha=0.3)

        # Plot Real and Imaginary parts separately (linear scale)
        ax = axes2[1, 0]
        for idx in [0, len(results)//4, len(results)//2, -1]:
            r = results[idx]
            ax.plot(r['G_theory'][:, 0, 0, 0].real, '--', label=f'Theory θ={r["theta"]:.2f}', alpha=0.7)
        ax.set_title('Theory Propagator Real Part')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Index')

        ax = axes2[1, 1]
        for idx in [0, len(results)//4, len(results)//2, -1]:
            r = results[idx]
            ax.plot(r['G_learned'][:, 0, 0, 0].real, '-', label=f'Learned θ={r["theta"]:.2f}', alpha=0.7)
        ax.set_title('Learned Propagator Real Part')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Index')

        plt.tight_layout()
        plt.savefig(self.cfg['data_dir']/'propagator_comparison.png', dpi=150)
        print(f"Saved propagator comparison to {self.cfg['data_dir']/'propagator_comparison.png'}")

        # Also save numerical results
        import json
        with open(self.cfg['data_dir']/'error_data.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for r in results:
                json_results.append({
                    'theta': float(r['theta']),
                    'error_real': float(r['error_real']),
                    'error_imag': float(r['error_imag']),
                    'error_total': float(r['error_total']),
                    'rel_error_real': float(r['rel_error_real']),
                    'rel_error_imag': float(r['rel_error_imag']),
                    'rel_error_total': float(r['rel_error_total'])
                })
            json.dump(json_results, f, indent=2)
        print(f"Saved numerical data to {self.cfg['data_dir']/'error_data.json'}")