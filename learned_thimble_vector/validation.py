"""
Validation

Changes vs original:
- Added _forward_batched() helper that runs model inference in small chunks.
  This prevents OOM when n_samples is large (e.g. 2000) because the new
  dual-path temporal conditioner has peak memory proportional to batch size.
  Training uses batch=32 so it's fine; the original validation code used
  single batches of 2000 which caused CUDA OOM on 8GB GPUs.
- test_chemical_potential: replaced single randn(2000,...) with batched loop.
- test_minkowski_sign_problem: same fix.
- All other tests already used self.sample() which batches correctly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

from thimble import (
    PhysicsInformedThimble, LagrangianParams, GeneralAction,
    effective_sample_size, bootstrap_error, integrated_autocorrelation,
    PARAM_DIM,
)


class AnalyticalSolutions:

    @staticmethod
    def kernel(L, m, kinetic_coeff=1.0, dx=1.0) -> torch.Tensor:
        k = 2 * np.pi * torch.arange(L, dtype=torch.float64) / L
        kt, kx, ky, kz = torch.meshgrid(k, k, k, k, indexing='ij')
        eig = (2*(1-torch.cos(kt)) + 2*(1-torch.cos(kx)) +
               2*(1-torch.cos(ky)) + 2*(1-torch.cos(kz)))
        return torch.clamp(kinetic_coeff * dx**2 * eig + m**2 * dx**4, min=1e-9)

    @staticmethod
    def phi2_free(L, m, kinetic_coeff=1.0, dx=1.0) -> float:
        K = AnalyticalSolutions.kernel(L, m, kinetic_coeff, dx)
        return (torch.sum(1.0 / K) / L**4).item()

    @staticmethod
    def phi2_perturbative(L, m, g4, kinetic_coeff=1.0, dx=1.0) -> float:
        """1-loop tadpole for g4/4! phi^4.  delta_m^2 = g4/2 * G(0)."""
        K      = AnalyticalSolutions.kernel(L, m, kinetic_coeff, dx)
        G0     = torch.sum(1.0 / K) / L**4
        bubble = torch.sum(1.0 / K**2) / L**4
        return (G0 - (g4 / 2.0) * G0 * bubble).item()

    @staticmethod
    def propagator_free(L, m, kinetic_coeff=1.0, dx=1.0) -> np.ndarray:
        K    = AnalyticalSolutions.kernel(L, m, kinetic_coeff, dx)
        G_kt = torch.sum(1.0 / K, dim=(1,2,3))
        G_t  = torch.fft.ifft(G_kt).real / L**3
        return G_t.numpy()


class ValidationSuite:

    def __init__(self, config: dict, model_path):
        self.config = config
        self.device = config['device']
        self.L      = config['L']
        self.dx     = config['dx']
        # Safe batch size for validation forward passes.
        # Kept small so the dual-path temporal conditioner doesn't OOM.
        self.val_batch = config.get('val_batch_size', 50)

        self.model = PhysicsInformedThimble(config).to(self.device)
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            print(f"✅ Loaded model from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load: {e}"); sys.exit(1)

        self.model.eval()
        self.action = GeneralAction(self.dx)
        self.results = {}

    # ── Memory-safe batched forward pass ──────────────────────────────────────

    def _forward_batched(self, n: int, params: LagrangianParams):
        """
        Run n samples through the model in chunks of self.val_batch.

        Returns
        -------
        phi     : [n, L, L, L, L] complex,  concatenated across batches
        log_det : [n] real
        S_E     : [n] complex  Euclidean action
        S_M     : [n] complex  Minkowski action

        All tensors are on CPU to avoid accumulating GPU memory.
        """
        all_phi, all_log_det, all_SE, all_SM = [], [], [], []
        n_done = 0
        with torch.no_grad():
            while n_done < n:
                bs = min(self.val_batch, n - n_done)
                z  = torch.randn(bs, self.L, self.L, self.L, self.L,
                                 device=self.device)
                phi, log_det = self.model(z, params)
                S_E = self.action.compute(phi, params)
                S_M = self.action.minkowski_action(phi, params)
                all_phi.append(phi.cpu())
                all_log_det.append(log_det.cpu())
                all_SE.append(S_E.cpu())
                all_SM.append(S_M.cpu())
                n_done += bs

        return (torch.cat(all_phi,     dim=0),
                torch.cat(all_log_det, dim=0),
                torch.cat(all_SE,      dim=0),
                torch.cat(all_SM,      dim=0))

    # ── Importance-sampling estimator ─────────────────────────────────────────

    def sample(self, params: LagrangianParams,
               n_samples: int = 1000, batch_size: int = 50) -> dict:
        """Importance-sampling estimator with ESS-corrected bootstrap errors."""
        n_batches = max(1, n_samples // batch_size)
        print(f"   Sampling {n_samples} ({n_batches} batches) for {params}")

        all_phi_re, all_log_w = [], []

        with torch.no_grad():
            for _ in range(n_batches):
                z = torch.randn(batch_size, self.L, self.L, self.L, self.L,
                                device=self.device)
                phi, log_det = self.model(z, params)
                S   = self.action.compute(phi, params)
                lw  = -S.real + log_det
                all_phi_re.append(phi.real.cpu())
                all_log_w.append(lw.cpu())

        phi_re = torch.cat(all_phi_re, dim=0)
        log_w  = torch.cat(all_log_w,  dim=0)

        ess   = effective_sample_size(log_w)
        n_eff = max(1, int(ess * n_samples))
        flag  = "✅" if ess > 0.1 else ("⚠️ LOW" if n_eff < 10 else "⚠️ (n_eff OK)")
        print(f"   ESS = {ess:.4f}  (N_eff ≈ {n_eff}/{n_samples})  {flag}")

        lw_s = log_w - log_w.max()
        w    = torch.exp(lw_s).numpy(); w /= w.sum()

        phi_np = phi_re.numpy()
        N      = phi_np.shape[0]

        phi_mean = phi_np.reshape(N, -1).mean(axis=1)
        phi2     = (phi_np**2).reshape(N, -1).mean(axis=1)
        phi4     = (phi_np**4).reshape(N, -1).mean(axis=1)

        prop   = np.zeros(self.L)
        phi_t0 = phi_np[:, 0, :, :, :]
        for t in range(self.L):
            corr_t  = (phi_np[:, t, :, :, :] * phi_t0).mean(axis=(1, 2, 3))
            prop[t] = np.dot(w, corr_t)

        return {
            'phi':        float(np.dot(w, phi_mean)),
            'phi2':       float(np.dot(w, phi2)),
            'phi4':       float(np.dot(w, phi4)),
            'err_phi2':   bootstrap_error(phi2, w),
            'err_phi4':   bootstrap_error(phi4, w),
            'propagator': prop,
            'ess':        ess,
            'n_eff':      n_eff,
            'tau_phi2':   integrated_autocorrelation(phi2),
            'w':          w,
            'phi2_arr':   phi2,
        }

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_propagator(self):
        print("\n=== Test 1: Free Field Propagator G(t) ===")
        p    = LagrangianParams(m2=1.0, kinetic_coeff=1.0)
        obs  = self.sample(p, n_samples=10000, batch_size=self.val_batch)
        G_th = obs['propagator']
        G_ex = AnalyticalSolutions.propagator_free(self.L, 1.0, dx=self.dx)

        abs_err = np.abs(G_th - G_ex)
        rel_err = abs_err / (np.abs(G_ex) + 1e-10)
        mae     = np.mean(abs_err)
        mre     = np.mean(rel_err)

        print(f"   G(0): exact={G_ex[0]:.5f}  thimble={G_th[0]:.5f}")
        print(f"   Mean abs error: {mae:.5f}   Mean rel error: {mre*100:.1f}%")
        print(f"   N_eff={obs['n_eff']}  τ={obs['tau_phi2']:.1f}")
        self.results['propagator'] = {
            'thimble': G_th, 'exact': G_ex,
            'error': mre, 'mae': mae,
            'ess': obs['ess'], 'n_eff': obs['n_eff'],
        }

    def test_phi2_free(self):
        print("\n=== Test 2: <phi²> Free Theory ===")
        p   = LagrangianParams(m2=1.0, kinetic_coeff=1.0)
        obs = self.sample(p, n_samples=5000, batch_size=self.val_batch)
        vt, et = obs['phi2'], obs['err_phi2']
        ve  = AnalyticalSolutions.phi2_free(self.L, 1.0, dx=self.dx)
        err = abs(vt - ve) / abs(ve)
        print(f"   Thimble: {vt:.5f} ± {et:.5f}  |  Exact: {ve:.5f}  |  "
              f"Error: {err*100:.2f}%  N_eff={obs['n_eff']}")
        self.results['phi2'] = {'thimble': vt, 'err': et, 'exact': ve,
                                 'error': err, 'ess': obs['ess']}

    def test_phi2_phi4(self):
        print("\n=== Test 3: <phi²> with phi^4 (g4=0.1) ===")
        g4  = 0.1
        p   = LagrangianParams(m2=1.0, kinetic_coeff=1.0, g4=g4)
        obs = self.sample(p, n_samples=5000, batch_size=self.val_batch)
        vt, et = obs['phi2'], obs['err_phi2']
        ve  = AnalyticalSolutions.phi2_perturbative(self.L, 1.0, g4, dx=self.dx)
        err = abs(vt - ve) / abs(ve)
        print(f"   Thimble: {vt:.5f} ± {et:.5f}  |  1-loop: {ve:.5f}  |  "
              f"Error: {err*100:.2f}%  N_eff={obs['n_eff']}")
        self.results['perturbative'] = {'thimble': vt, 'err': et, 'exact': ve,
                                         'error': err, 'ess': obs['ess']}

    def test_phi2_phi6(self):
        print("\n=== Test 4: <phi²> with phi^6 (g6=0.05) ===")
        p   = LagrangianParams(m2=1.0, kinetic_coeff=1.0, g6=0.05)
        obs = self.sample(p, n_samples=3000, batch_size=self.val_batch)
        vt, et = obs['phi2'], obs['err_phi2']
        ve  = AnalyticalSolutions.phi2_free(self.L, 1.0, dx=self.dx)
        direction_ok = vt < ve
        print(f"   Thimble: {vt:.5f} ± {et:.5f}  |  Free (ref): {ve:.5f}")
        print(f"   g6>0 suppresses <φ²>: {'✅' if direction_ok else '❌'}  N_eff={obs['n_eff']}")
        self.results['phi6'] = {'thimble': vt, 'err': et, 'exact': ve,
                                 'direction_ok': direction_ok, 'ess': obs['ess']}

    def test_chemical_potential(self):
        print("\n=== Test 5: Chemical Potential μ=0.3 (Euclidean sign diagnostics) ===")
        mu = 0.3
        p  = LagrangianParams(m2=1.0, kinetic_coeff=1.0, mu=mu)
        obs = self.sample(p, n_samples=3000, batch_size=self.val_batch)

        # Batched forward pass for sign diagnostics (avoids OOM)
        n_diag = 2000
        phi, _, S_E, _ = self._forward_batched(n_diag, p)
        avg_phase = torch.mean(torch.exp(1j * S_E.imag)).abs().item()
        std_imS   = S_E.imag.std().item()

        sign_problem  = avg_phase < 0.95
        thimble_helps = std_imS < 1.0
        print(f"   |<e^iθ>| = {avg_phase:.4f}  std(Im S_E) = {std_imS:.4f}")
        print(f"   ESS = {obs['ess']:.4f}  N_eff = {obs['n_eff']}")
        self.results['mu'] = {
            'mu': mu, 'avg_phase': avg_phase, 'std_imS': std_imS,
            'ess': obs['ess'], 'sign_problem': sign_problem,
            'thimble_helps': thimble_helps,
        }

    def test_minkowski_sign_problem(self):
        """
        Test 7: Minkowski sign-problem suppression.
        Uses batched forward pass to avoid OOM on GPUs with limited VRAM.
        """
        print("\n=== Test 7: Minkowski Sign-Problem Suppression ===")

        mu_values   = [0.0, 0.15, 0.28]
        std_thresh  = 1.5
        n_diag      = 2000

        rows = []
        all_std_imSM = []

        for mu in mu_values:
            p = LagrangianParams(m2=1.0, kinetic_coeff=1.0, g4=0.1, mu=mu)

            # Batched forward pass — safe for any GPU
            phi, log_det, _, S_M = self._forward_batched(n_diag, p)

            std_imSM  = S_M.imag.std().item()
            avg_phase = torch.mean(torch.exp(1j * S_M.imag)).abs().item()
            log_w_M   = -S_M.real + log_det
            ess_M     = effective_sample_size(log_w_M)

            passed = std_imSM < std_thresh
            status = "✅" if passed else "⚠️"
            print(f"   μ={mu:.2f} | std(Im S_M)={std_imSM:.4f}  "
                  f"|<e^{{iθ_M}}>|={avg_phase:.4f}  ESS_M={ess_M:.4f}  {status}")

            rows.append({
                'mu':          mu,
                'std_imSM':    std_imSM,
                'avg_phase_M': avg_phase,
                'ess_M':       ess_M,
                'passed':      passed,
            })
            all_std_imSM.append(std_imSM)

        overall_pass = all(r['passed'] for r in rows)
        print(f"\n   Overall: {'✅ PASS' if overall_pass else '⚠️  Some μ above threshold'}"
              f"  (threshold: std < {std_thresh})")

        self.results['minkowski'] = {
            'rows':         rows,
            'std_thresh':   std_thresh,
            'overall_pass': overall_pass,
            'all_std_imSM': all_std_imSM,
            'ess':          rows[-1]['ess_M'],
            'n_eff':        max(1, int(rows[-1]['ess_M'] * n_diag)),
        }

    def test_consistency(self):
        print("\n=== Test 6: Consistency Checks (phi^4, g4=0.2) ===")
        p   = LagrangianParams(m2=1.0, kinetic_coeff=1.0, g4=0.2)
        obs = self.sample(p, n_samples=3000, batch_size=self.val_batch)
        phi, phi2, phi4 = obs['phi'], obs['phi2'], obs['phi4']
        kurt = phi4 / phi2**2 if phi2 > 0 else 0.0

        print(f"   <φ>        = {phi:.4f}  (~0)   {'✅' if abs(phi)<0.05 else '❌'}")
        print(f"   <φ⁴>       = {phi4:.4f}  (>0)   {'✅' if phi4>0 else '❌'}")
        print(f"   <φ⁴>/<φ²>² = {kurt:.2f}    (≥1)   {'✅' if phi4>=phi2**2 else '❌'}")
        print(f"   ESS = {obs['ess']:.4f}  N_eff = {obs['n_eff']}"
              f"  τ = {obs['tau_phi2']:.1f}")

        self.results['consistency'] = {
            'sym': abs(phi) < 0.05, 'pos': phi4 > 0, 'ineq': phi4 >= phi2**2,
            'phi': phi, 'phi2': phi2, 'phi4': phi4,
            'kurt': kurt, 'ess': obs['ess'], 'tau': obs['tau_phi2'],
            'n_eff': obs['n_eff'],
        }

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot_summary(self):
        save_path = self.config['data_dir'] / 'validation_summary.png'

        has_mink = 'minkowski' in self.results
        nrows    = 4 if has_mink else 3
        fig      = plt.figure(figsize=(20, 5 * nrows))
        gs       = fig.add_gridspec(nrows, 3, hspace=0.55, wspace=0.35)

        if 'propagator' in self.results:
            ax  = fig.add_subplot(gs[0, :2])
            res = self.results['propagator']
            t   = np.arange(len(res['thimble']))
            G_ex = res['exact']
            G_th = res['thimble']

            ax.semilogy(t, G_ex,            'o-',  lw=2, label='Exact (Free)', alpha=0.9)
            ax.semilogy(t, np.abs(G_th),    's--', lw=2, label='Thimble',      alpha=0.9)
            ax.set_title(
                f"Propagator G(t)  —  MAE={res['mae']:.5f}  "
                f"RelErr={res['error']*100:.1f}%  "
                f"N_eff={res['n_eff']}",
                fontsize=11
            )
            ax.set_xlabel("t"); ax.set_ylabel("G(t)")
            ax.grid(True, alpha=0.3); ax.legend()

            ax_in = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
            ax_in.bar(t, np.abs(G_th - G_ex), color='coral', alpha=0.8)
            ax_in.set_title("|residual|", fontsize=9)
            ax_in.set_xlabel("t", fontsize=8)
            ax_in.tick_params(labelsize=8)
            ax_in.grid(True, alpha=0.3)

        ax_ess = fig.add_subplot(gs[0, 2])
        ekeys  = ['propagator', 'phi2', 'perturbative',
                  'phi6', 'mu', 'consistency', 'minkowski']
        ev     = {k: self.results[k]['ess']          for k in ekeys if k in self.results}
        nv     = {k: self.results[k].get('n_eff', 0) for k in ekeys if k in self.results}
        cols   = plt.cm.Set2(np.linspace(0, 1, len(ev)))
        bars   = ax_ess.bar(list(ev.keys()), list(ev.values()),
                            color=cols, alpha=0.85, edgecolor='k')
        ax_ess.axhline(0.1, color='r', ls='--', lw=1.5, label='10% floor')
        ax_ess.set_ylim(0, max(0.15, max(ev.values()) * 1.2))
        ax_ess.set_title("ESS per Test", fontsize=12)
        ax_ess.set_ylabel("ESS fraction")
        ax_ess.tick_params(axis='x', rotation=30)
        ax_ess.legend(); ax_ess.grid(True, axis='y', alpha=0.3)
        for bar, (k, n) in zip(bars, nv.items()):
            ax_ess.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"N={n}", ha='center', va='bottom', fontsize=7)

        for col_i, (key, lbl, clrs) in enumerate([
            ('phi2',        '<φ²> Free',          ['royalblue', 'darkorange']),
            ('perturbative','<φ²> phi^4 g4=0.1',  ['forestgreen', 'crimson']),
            ('phi6',        '<φ²> phi^6 g6=0.05', ['steelblue', 'mediumpurple']),
        ]):
            if key not in self.results: continue
            ax  = fig.add_subplot(gs[1, col_i])
            res = self.results[key]
            ax.bar(['Exact/Ref', 'Thimble'], [res['exact'], res['thimble']],
                   color=clrs, alpha=0.85, edgecolor='k')
            ax.errorbar([1], [res['thimble']], yerr=[res.get('err', 0)],
                        fmt='none', color='k', capsize=5, lw=2)
            err_s = f"Error:{res['error']*100:.1f}%" if 'error' in res else ""
            dir_s = (f" {'✅' if res.get('direction_ok', True) else '❌'}"
                     if 'direction_ok' in res else "")
            ax.set_title(f"{lbl}  {err_s}{dir_s}", fontsize=11)
            for i, v in enumerate([res['exact'], res['thimble']]):
                ax.text(i, v, f"{v:.4f}", ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)

        if 'mu' in self.results:
            ax  = fig.add_subplot(gs[2, :2])
            res = self.results['mu']
            sp  = "✅" if res['sign_problem'] else "—"
            th  = "✅" if res['thimble_helps'] else "—"
            metrics = ['|<e^{iθ_E}>|', 'std(Im S_E) [capped@2]']
            vals    = [res['avg_phase'], min(res['std_imS'], 2.0)]
            x = np.arange(len(metrics))
            ax.bar(x, vals, color=['darkorange', 'crimson'], alpha=0.85, edgecolor='k')
            ax.axhline(1.0, color='gray', ls='--', lw=1)
            ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
            ax.set_title(
                f"μ={res['mu']:.2f}  Euclidean sign diagnostics  "
                f"Sign problem:{sp}  Im(S_E) reduced:{th}",
                fontsize=12)
            ax.grid(True, axis='y', alpha=0.3)

        if 'consistency' in self.results:
            ax  = fig.add_subplot(gs[2, 2])
            ax.axis('off')
            res = self.results['consistency']
            rows = [
                ['<φ>',        f"{res['phi']:.4f}",  '~0',   '✅' if res['sym']  else '❌'],
                ['<φ⁴>',       f"{res['phi4']:.4f}",  '>0',   '✅' if res['pos']  else '❌'],
                ['<φ⁴>/<φ²>²', f"{res['kurt']:.2f}",  '≥1',   '✅' if res['ineq'] else '❌'],
                ['ESS',        f"{res['ess']:.4f}",  '>0.1', '✅' if res['ess']>0.1 else '⚠️'],
                ['N_eff',      f"{res['n_eff']}",    '>10',  '✅' if res['n_eff']>10 else '⚠️'],
                ['τ_int(φ²)',  f"{res['tau']:.1f}",  '—',    '—'],
            ]
            tbl = ax.table(cellText=rows,
                           colLabels=['Observable', 'Value', 'Expected', 'Status'],
                           loc='center', cellLoc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
            ax.set_title("Consistency Summary", fontsize=12)

        if has_mink:
            res   = self.results['minkowski']
            rows  = res['rows']
            mu_v  = [r['mu']          for r in rows]
            std_v = [r['std_imSM']    for r in rows]
            ph_v  = [r['avg_phase_M'] for r in rows]
            ess_v = [r['ess_M']       for r in rows]

            ax_std = fig.add_subplot(gs[3, 0])
            bar_cols = ['forestgreen' if r['passed'] else 'crimson' for r in rows]
            ax_std.bar([str(m) for m in mu_v], std_v,
                       color=bar_cols, alpha=0.85, edgecolor='k')
            ax_std.axhline(res['std_thresh'], color='r', ls='--', lw=1.5,
                           label=f"threshold ({res['std_thresh']})")
            ax_std.set_title("std(Im S_M) vs μ\n(green = pass)", fontsize=11)
            ax_std.set_xlabel("μ"); ax_std.set_ylabel("std(Im S_M)")
            ax_std.legend(fontsize=8); ax_std.grid(True, axis='y', alpha=0.3)
            for i, v in enumerate(std_v):
                ax_std.text(i, v, f"{v:.3f}", ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

            ax_ph = fig.add_subplot(gs[3, 1])
            ax_ph.plot([str(m) for m in mu_v], ph_v,
                       'o-', color='darkorange', lw=2, ms=8)
            ax_ph.axhline(1.0, color='gray',   ls='--', lw=1, label='ideal (1.0)')
            ax_ph.axhline(0.1, color='red',    ls=':',  lw=1, label='severe (<0.1)')
            ax_ph.set_ylim(0, 1.1)
            ax_ph.set_title("|<e^{iθ_M}>| vs μ\n(1 = no sign problem)", fontsize=11)
            ax_ph.set_xlabel("μ"); ax_ph.set_ylabel("|<e^{iθ_M}>|")
            ax_ph.legend(fontsize=8); ax_ph.grid(True, alpha=0.3)
            for i, v in enumerate(ph_v):
                ax_ph.text(i, v + 0.02, f"{v:.3f}", ha='center',
                           fontsize=9, fontweight='bold')

            ax_em = fig.add_subplot(gs[3, 2])
            ax_em.bar([str(m) for m in mu_v], ess_v,
                      color='steelblue', alpha=0.85, edgecolor='k')
            ax_em.axhline(0.1, color='r', ls='--', lw=1.5, label='10% floor')
            ax_em.set_ylim(0, max(0.15, max(ess_v) * 1.3))
            ax_em.set_title("Minkowski ESS vs μ", fontsize=11)
            ax_em.set_xlabel("μ"); ax_em.set_ylabel("ESS fraction")
            ax_em.legend(fontsize=8); ax_em.grid(True, axis='y', alpha=0.3)
            for i, v in enumerate(ess_v):
                ax_em.text(i, v, f"{v:.3f}", ha='center', va='bottom',
                           fontsize=9, fontweight='bold')

            overall = "✅ PASS" if res['overall_pass'] else "⚠️ PARTIAL"
            fig.text(0.5, 0.01,
                     f"Minkowski Sign-Problem Test: {overall}  "
                     f"(threshold: std(Im S_M) < {res['std_thresh']})",
                     ha='center', fontsize=12,
                     color='green' if res['overall_pass'] else 'darkorange')

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────

def run_all_validations(config: dict):
    print("=" * 65)
    print("VALIDATION SUITE  —  v8")
    print("=" * 65)
    suite = ValidationSuite(config, config['data_dir'] / 'universal_thimble_model.pt')
    suite.test_propagator()
    suite.test_phi2_free()
    suite.test_phi2_phi4()
    suite.test_phi2_phi6()
    suite.test_chemical_potential()
    suite.test_consistency()
    suite.test_minkowski_sign_problem()
    suite.plot_summary()
    print("\nValidation complete.")