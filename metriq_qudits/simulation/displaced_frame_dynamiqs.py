"""dynamiqs/JAX port of DisplacedFrameSimulator (see displaced_frame.py)
for GPU execution. It uses the same Hamiltonian, frame, and collapse operators and the same
constructor and simulate()/to_physical_frame() interface. Time-dependent
coefficients are linearly interpolated (the QuTiP version uses cubic splines).

to_physical_frame returns a qutip.Qobj so subsequent ptrace/metrics code is
backend-agnostic. Only the time evolution runs under JAX.
"""

import sys, os

if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import qutip as qt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import dynamiqs as dq
from scipy.interpolate import CubicSpline

from metriq_qudits.pulses.drive_envelopes import CavityMode
from metriq_qudits.pulses.conditional_dynamics import DispersiveRates, evolve_pair

K_Q_DEFAULT_MHZ = -200.0  # transmon anharmonicity K_q/2π [MHz]

_MESOLVE_OPTIONS = dq.Options(progress_meter=False)
_MESOLVE_METHOD  = dq.method.Tsit5(rtol=1e-6, atol=1e-8, max_steps=10_000_000)


def _embed(op: np.ndarray, idx: int, dims: list[int]) -> jnp.ndarray:
    full = np.eye(1, dtype=complex)
    for i, d in enumerate(dims):
        part = op if i == idx else np.eye(d, dtype=complex)
        full = np.kron(full, part)
    return jnp.asarray(full)


class DisplacedFrameSimulatorDQ:

    def __init__(
        self,
        cavity_dim: int,
        mode: CavityMode,
        qubit_dim: int = 3,
        K_q_MHz: float = K_Q_DEFAULT_MHZ,
    ):
        self.cavity_dim = cavity_dim
        self.qubit_dim = qubit_dim
        self.mode = mode
        self.dims = [cavity_dim, qubit_dim]

        destroy = lambda d: np.diag(np.sqrt(np.arange(1, d)), k=1).astype(complex)
        self.a    = _embed(destroy(cavity_dim), 0, self.dims)
        self.q_op = _embed(destroy(qubit_dim), 1, self.dims)
        self.n_q  = _embed(np.diag(np.arange(qubit_dim)).astype(complex), 1, self.dims)
        self.sigx = self.q_op + self.q_op.conj().T
        self.sigy = 1j * (self.q_op.conj().T - self.q_op)

        self.I_op = jnp.eye(int(np.prod(self.dims)), dtype=complex)
        self.K_q  = 2.0 * np.pi * K_q_MHz * 1e-3  # MHz -> rad/ns

    def _solve_alpha(self, epsilon: np.ndarray) -> np.ndarray:
        chi, _, self_kerr, kappa = self.mode.angular_rates()
        rates = DispersiveRates(delta=chi / 2, kerr=self_kerr, kappa=kappa)
        alpha, _ = evolve_pair(np.asarray(epsilon, dtype=complex), rates)
        return alpha

    def _static_hamiltonian(self) -> jnp.ndarray:
        a = self.a
        adag = a.conj().T
        n = adag @ a
        quartic = adag @ adag @ a @ a
        # static residual coupling in the g/e-midpoint frame:
        # (chi/2) n - chi n*nq - (chi'/2) a†²a²*nq - (K/2) a†²a²
        chi, chi_prime, self_kerr, _ = self.mode.angular_rates()
        c_n, c_nnq, c_qnq, c_q = chi / 2.0, -chi, -chi_prime / 2.0, -self_kerr / 2.0
        H = c_n * n + c_nnq * n @ self.n_q
        H = H + c_qnq * quartic @ self.n_q + c_q * quartic
        H = H + self.K_q / 2.0 * self.n_q @ (self.n_q - self.I_op)
        return H

    @staticmethod
    def _modulated(tlist, coeff, op):
        # Same clamped cubic spline as the QuTiP backend, evaluated in JAX.
        cs = CubicSpline(np.asarray(tlist), np.real(coeff), bc_type="clamped")
        x = jnp.asarray(cs.x)
        c = jnp.asarray(cs.c)  # (4, n-1) polynomial coefficients

        def f(t):
            i = jnp.clip(jnp.searchsorted(x, t, side="right") - 1, 0, c.shape[1] - 1)
            dt = t - x[i]
            return ((c[0, i] * dt + c[1, i]) * dt + c[2, i]) * dt + c[3, i]

        return dq.modulated(f, dq.asqarray(op))

    def _diag_hamiltonian(self, alpha: np.ndarray, tlist) -> list:
        a = self.a
        n = a.conj().T @ a
        # frequency shifts induced by the classical displacement alpha(t)
        chi, chi_prime, self_kerr, _ = self.mode.angular_rates()
        abs2 = np.abs(alpha) ** 2
        c_n   = -2.0 * self_kerr * abs2
        c_nnq = -2.0 * chi_prime * abs2
        c_nq  = -chi * abs2 - (chi_prime / 2.0) * abs2 ** 2
        return [
            self._modulated(tlist, c_n, n),
            self._modulated(tlist, c_nnq, n @ self.n_q),
            self._modulated(tlist, c_nq, self.n_q),
        ]

    def _offdiag_hamiltonian(self, alpha: np.ndarray, omega: np.ndarray, tlist) -> list:
        a = self.a
        adag = a.conj().T
        # qubit-conditioned cavity displacement (the ECD term): -chi * alpha
        chi = self.mode.angular_rates()[0]
        c_x, c_y = -chi * np.real(alpha), -chi * np.imag(alpha)
        drive_i, drive_q = np.real(omega), np.imag(omega)
        return [
            self._modulated(tlist, c_x, self.n_q @ (a + adag)),
            self._modulated(tlist, c_y, self.n_q @ (1j * (adag - a))),
            self._modulated(tlist, drive_i, self.sigx),
            self._modulated(tlist, drive_q, self.sigy),
        ]

    def _make_jump_ops(self, T1_us=None, T2_us=None) -> list:
        # Eq. B6: γ1 D[q̂] + 2γφ D[n̂_q] + κ D[â]
        jump_ops = []
        _, _, _, kappa = self.mode.angular_rates()
        if kappa > 0:
            jump_ops.append(dq.asqarray(np.sqrt(kappa) * self.a))
        if T1_us is not None:
            gamma1 = 1.0 / (T1_us * 1e3)
            jump_ops.append(dq.asqarray(np.sqrt(gamma1) * self.q_op))
        if T1_us is not None and T2_us is not None:
            gamma_phi = max(0.0, 1.0 / (T2_us * 1e3) - 0.5 / (T1_us * 1e3))
            if gamma_phi > 0:
                jump_ops.append(dq.asqarray(np.sqrt(2.0 * gamma_phi) * self.n_q))
        return jump_ops

    def to_physical_frame(self, state, alpha: np.ndarray,
                          cavity_phase: float = None) -> qt.Qobj:
        """Undo the residual frame displacement α(T) and return a qutip.Qobj so
        downstream ptrace/metrics code is shared with the QuTiP backend.

        cavity_phase: spurious rotation angle φ tracked by the pulse builder,
        undone as a virtual rotation e^{−iφ n̂} (see the QuTiP backend docstring)."""
        rho_np = np.asarray(state.to_jax() if hasattr(state, "to_jax") else state)
        if rho_np.shape[-1] == 1:
            ket = qt.Qobj(rho_np, dims=[self.dims, [1] * len(self.dims)])
            rho = qt.ket2dm(ket)
        else:
            rho = qt.Qobj(rho_np, dims=[self.dims, self.dims])
        D = qt.tensor(qt.displace(self.cavity_dim, alpha[-1]), qt.qeye(self.qubit_dim))
        rho = D * rho * D.dag()
        if cavity_phase is not None and cavity_phase != 0.0:
            R = qt.tensor((-1j * cavity_phase * qt.num(self.cavity_dim)).expm(),
                          qt.qeye(self.qubit_dim))
            rho = R * rho * R.dag()
        return rho

    def simulate(
        self,
        epsilon: np.ndarray,
        omega: np.ndarray,
        psi0=None,
        T1_us: float = None,
        T2_us: float = None,
        save_trajectory: bool = False,
    ) -> tuple:
        alpha = self._solve_alpha(epsilon)
        tlist = jnp.arange(len(epsilon), dtype=float)

        H = dq.asqarray(self._static_hamiltonian())
        for term in self._diag_hamiltonian(alpha, tlist):
            H = H + term
        for term in self._offdiag_hamiltonian(alpha, omega, tlist):
            H = H + term

        if psi0 is None:
            dim = int(np.prod(self.dims))
            vac = np.zeros(dim, dtype=complex)
            vac[0] = 1.0  # vacuum ⊗ |g⟩ (index 0 in row-major cavity⊗qubit order)
            psi0 = dq.asqarray(vac[:, None])

        t_final = float(len(epsilon) - 1)
        tsave = (np.arange(len(epsilon), dtype=float)
                 if save_trajectory else np.array([0.0, t_final]))

        jump_ops = self._make_jump_ops(T1_us, T2_us)
        if not jump_ops and psi0.shape[-1] == 1:
            # Preserve a ket for closed-system evolution: this is both faster
            # and O(dim) in state storage instead of O(dim²).
            result = dq.sesolve(H, psi0, tsave, method=_MESOLVE_METHOD,
                                options=_MESOLVE_OPTIONS)
        else:
            result = dq.mesolve(H, jump_ops, psi0, tsave,
                                method=_MESOLVE_METHOD, options=_MESOLVE_OPTIONS)
        return result, alpha
