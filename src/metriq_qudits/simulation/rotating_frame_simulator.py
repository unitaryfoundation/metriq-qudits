"""Undisplaced rotating-frame Lindblad simulator — independent verification backend.

Simulates the *same* ε(t)/ω(t) waveforms as ``DisplacedFrameSimulator`` but in the
full, undisplaced cavity frame. Because the cavity is driven directly (no α(t)),
it explores large coherent states, so ``cavity_dim`` must be chosen generously —
avoiding exactly that is the point of the displaced frame. This backend is meant
for cross-checking the displaced-frame simulator on small cases, not for
production sweeps.

Built self-contained: it shares no Hamiltonian code with the displaced frame, so
agreement between the two is a real cross-check. The constructor and
``simulate()``/``to_physical_frame()`` interface mirror ``DisplacedFrameSimulator``
so it can be swapped in wherever that backend is used.
"""

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline

from metriq_qudits.pulses.pulse_primitives import Storage
from metriq_qudits.physics.rotating_frame_model import (
    ancilla_drive_coefficients,
    cavity_drive_coefficients,
    static_coefficients,
    storage_parameters,
)

K_Q_DEFAULT_MHZ = -200.0  # transmon anharmonicity K_q/2π [MHz]


def embed(op: qt.Qobj, idx: int, dims: list[int]) -> qt.Qobj:
    parts = [qt.qeye(d) for d in dims]
    parts[idx] = op
    return qt.tensor(parts)


def _angular_frequency_from_mhz(value_mhz: float) -> float:
    return 2.0 * np.pi * value_mhz * 1e-3


class RotatingFrameSimulator:

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

        a = embed(qt.destroy(cavity_dim), 0, self.dims)
        q = embed(qt.destroy(qubit_dim), 1, self.dims)
        self.a = a
        self.q_op = q
        self.n = a.dag() * a
        self.quartic = a.dag() * a.dag() * a * a
        self.n_q = embed(qt.num(qubit_dim), 1, self.dims)
        self.sigx = q + q.dag()
        self.sigy = 1j * (q.dag() - q)
        self.x_c = a + a.dag()
        self.y_c = 1j * (a.dag() - a)

        self.I_op = qt.tensor([qt.qeye(d) for d in self.dims])
        self.K_q = _angular_frequency_from_mhz(K_q_MHz)

    def _static_hamiltonian(self) -> qt.Qobj:
        c_n, c_nnq, c_qnq, c_q = static_coefficients(self.storage)
        H = c_n * self.n + c_nnq * self.n * self.n_q
        H += c_qnq * self.quartic * self.n_q + c_q * self.quartic
        H += self.K_q / 2.0 * self.n_q * (self.n_q - self.I_op)
        return H

    @staticmethod
    def _spline(tlist, arr):
        cs = CubicSpline(tlist, arr, bc_type="clamped")
        return lambda t, **kwargs: float(cs(t))

    def _drive_hamiltonian(self, epsilon: np.ndarray, omega: np.ndarray) -> list:
        tlist = np.arange(len(omega), dtype=float)
        sp = lambda arr: self._spline(tlist, arr)

        eps_i, eps_q = cavity_drive_coefficients(epsilon)
        drive_i, drive_q = ancilla_drive_coefficients(omega)
        return [
            [self.x_c,  sp(eps_i)],    # Re(ε)·(a+a†)
            [self.y_c,  sp(eps_q)],    # Im(ε)·i(a†−a)
            [self.sigx, sp(drive_i)],
            [self.sigy, sp(drive_q)],
        ]

    def _make_c_ops(self, T1_us=None, T2_us=None) -> list:
        # γ1 D[q̂] + 2γφ D[n̂_q] + κ D[â]  (same collapse structure as the
        # displaced backend, rebuilt here independently)
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

    def to_physical_frame(self, state: qt.Qobj, alpha=None,
                          cavity_phase: float = None) -> qt.Qobj:
        """The rotating frame is already physical — there is no displacement to
        undo. Only the spurious cavity phase φ tracked by the pulse builder is
        removed, as the virtual rotation e^{−iφ n̂}, exactly as the displaced
        backend does, so the resulting states are directly comparable.

        ``alpha`` is accepted and ignored to keep a drop-in interface."""
        rho = qt.ket2dm(state) if state.type == 'ket' else state
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
        Ht = [self._static_hamiltonian()] + self._drive_hamiltonian(epsilon, omega)

        if psi0 is None:
            psi0 = qt.tensor(qt.basis(self.cavity_dim, 0),
                             qt.basis(self.qubit_dim, 0))  # vacuum ⊗ |g⟩

        t_final = float(len(epsilon) - 1)
        tlist = (np.arange(len(epsilon), dtype=float)
                 if save_trajectory else np.array([0.0, t_final]))
        e_ops = [self.n, self.n_q]

        result = qt.mesolve(Ht, psi0, tlist,
                            c_ops=self._make_c_ops(T1_us, T2_us),
                            e_ops=e_ops,
                            options={"store_states": True, "nsteps": 1_000_000,
                                     "max_step": 5.0})
        # Second return value mirrors DisplacedFrameSimulator.simulate (which
        # returns the trajectory α); the rotating frame has none.
        return result, None
