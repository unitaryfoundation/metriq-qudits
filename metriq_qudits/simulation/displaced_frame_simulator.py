"""Displaced-frame Lindblad simulator for a cavity dispersively coupled to a
transmon ancilla — You et al., arXiv:2403.00275 (Eqs. B2, B4, B5, B6).

Cavity drive frame at the g/e midpoint: α(t) solved with detuning χ/2, and the
residual static coupling is +χ/2·n̂ − χ·n̂·n̂_q. All transmon coupling is written
with n̂_q = q̂†q̂ so one class covers 2- and 3-level ancillas (qubit_dim).
"""

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline

from metriq_qudits.pulses.pulse_primitives import Storage
from metriq_qudits.physics.alpha_dynamics import alpha_from_epsilon_finite_difference
from metriq_qudits.physics.displaced_frame_model import (
    ancilla_drive_coefficients, mode_conditional_coefficients,
    mode_diagonal_coefficients, mode_static_coefficients, storage_parameters,
)
from metriq_qudits.physics.units import angular_frequency_from_mhz

K_Q_DEFAULT_MHZ = -200.0  # transmon anharmonicity K_q/2π [MHz]


def embed(op: qt.Qobj, idx: int, dims: list[int]) -> qt.Qobj:
    parts = [qt.qeye(d) for d in dims]
    parts[idx] = op
    return qt.tensor(parts)


class DisplacedFrameSimulator:

    def __init__(
        self,
        cavity_dim: int,
        storage: Storage,
        qubit_dim: int = 3,
        K_q_MHz: float = K_Q_DEFAULT_MHZ,
    ):
        self.cavity_dim = cavity_dim
        self.qubit_dim = qubit_dim
        self.storage = storage
        self.dims = [cavity_dim, qubit_dim]

        self.a      = embed(qt.destroy(cavity_dim), 0, self.dims)
        self.q_op   = embed(qt.destroy(qubit_dim), 1, self.dims)
        self.n_q    = embed(qt.num(qubit_dim),     1, self.dims)
        self.sigx   = self.q_op + self.q_op.dag()
        self.sigy   = 1j * (self.q_op.dag() - self.q_op)

        self.I_op = qt.tensor([qt.qeye(d) for d in self.dims])
        self.K_q  = angular_frequency_from_mhz(K_q_MHz)

    def _solve_alpha(self, epsilon: np.ndarray) -> np.ndarray:
        chi, _, self_kerr, kappa = storage_parameters(self.storage)
        alpha, _ = alpha_from_epsilon_finite_difference(
            np.asarray(epsilon, dtype=complex), delta=chi / 2,
            self_kerr=self_kerr, kappa=kappa)
        return alpha

    def _static_hamiltonian(self) -> qt.Qobj:
        a = self.a
        n = a.dag() * a
        quartic = a.dag() * a.dag() * a * a
        c_n, c_nnq, c_qnq, c_q = mode_static_coefficients(self.storage)
        H = c_n * n + c_nnq * n * self.n_q
        H += c_qnq * quartic * self.n_q + c_q * quartic
        H += self.K_q / 2.0 * self.n_q * (self.n_q - self.I_op)
        return H

    @staticmethod
    def _spline(tlist, arr):
        cs = CubicSpline(tlist, arr, bc_type="clamped")
        return lambda t, **kwargs: float(cs(t))

    def _diag_hamiltonian(self, alpha: np.ndarray) -> list:
        tlist = np.arange(len(alpha), dtype=float)
        sp = lambda arr: self._spline(tlist, arr)

        a = self.a
        n = a.dag() * a
        c_n, c_nnq, c_nq = mode_diagonal_coefficients(alpha, self.storage)
        return [
            [n,            sp(c_n)],
            [n * self.n_q, sp(c_nnq)],
            [self.n_q,     sp(c_nq)],
        ]

    def _offdiag_hamiltonian(self, alpha: np.ndarray, omega: np.ndarray) -> list:
        tlist = np.arange(len(alpha), dtype=float)
        sp = lambda arr: self._spline(tlist, arr)

        a = self.a
        c_x, c_y = mode_conditional_coefficients(alpha, self.storage)
        drive_i, drive_q = ancilla_drive_coefficients(omega)
        return [
            [self.n_q * (a + a.dag()),        sp(c_x)],
            [self.n_q * (1j * (a.dag() - a)), sp(c_y)],
            [self.sigx, sp(drive_i)],
            [self.sigy, sp(drive_q)],
        ]

    def _make_c_ops(self, T1_us=None, T2_us=None) -> list:
        # Eq. B6: γ1 D[q̂] + 2γφ D[n̂_q] + κ D[â]
        c_ops = []
        _, _, _, kappa = storage_parameters(self.storage)
        if kappa > 0:
            c_ops.append(np.sqrt(kappa) * self.a)
        if T1_us is not None:
            gamma1 = 1.0 / (T1_us * 1e3)
            c_ops.append(np.sqrt(gamma1) * self.q_op)
        if T1_us is not None and T2_us is not None:
            gamma_phi = max(0.0, 1.0 / (T2_us * 1e3) - 0.5 / (T1_us * 1e3))
            if gamma_phi > 0:
                c_ops.append(np.sqrt(2.0 * gamma_phi) * self.n_q)
        return c_ops

    def to_physical_frame(self, state: qt.Qobj, alpha: np.ndarray,
                          cavity_phase: float = None) -> qt.Qobj:
        """Undo the residual frame displacement α(T) so Fock populations are
        lab frame. The frame does not return to zero after echoed gates.

        cavity_phase: spurious rotation angle φ tracked by the pulse builder
        (final state ≈ e^{+iφ n̂}·target). It is undone here as a virtual
        rotation e^{−iφ n̂}, the cavity analog of the qubit virtual-Z."""
        rho = qt.ket2dm(state) if state.type == 'ket' else state
        D = embed(qt.displace(self.cavity_dim, alpha[-1]), 0, self.dims)
        rho = D * rho * D.dag()
        if cavity_phase is not None and cavity_phase != 0.0:
            R = embed((-1j * cavity_phase * qt.num(self.cavity_dim)).expm(),
                      0, self.dims)
            rho = R * rho * R.dag()
        return rho

    def simulate(
        self,
        epsilon: np.ndarray,
        omega: np.ndarray,
        psi0: qt.Qobj = None,
        T1_us: float = None,
        T2_us: float = None,
        save_trajectory: bool = False,
    ) -> tuple:
        alpha = self._solve_alpha(epsilon)
        Ht = ([self._static_hamiltonian()]
              + self._diag_hamiltonian(alpha)
              + self._offdiag_hamiltonian(alpha, omega))

        if psi0 is None:
            psi0 = qt.tensor(qt.basis(self.cavity_dim, 0),
                             qt.basis(self.qubit_dim, 0))  # vacuum ⊗ |g⟩

        t_final = float(len(epsilon) - 1)
        tlist = (np.arange(len(epsilon), dtype=float)
                 if save_trajectory else np.array([0.0, t_final]))
        e_ops = [self.a.dag() * self.a, self.n_q]

        # max_step: without a cap, zvode grows its step during idle/slow segments
        # and can integrate straight across a ~40 ns qubit pulse. A 5 ns step resolves
        # the shortest waveform feature (qubit sigma = 10 ns).
        result = qt.mesolve(Ht, psi0, tlist,
                            c_ops=self._make_c_ops(T1_us, T2_us),
                            e_ops=e_ops,
                            options={"store_states": True, "nsteps": 1_000_000,
                                     "max_step": 5.0})
        return result, alpha
