from dataclasses import dataclass

import numpy as np


def truncated_gaussian(sigma, span=4.0, dt=1.0):
    n_samples = int(round(span * sigma / dt))
    if n_samples < 3:
        raise ValueError(
            f"span * sigma / dt = {span * sigma / dt:.2f} gives too few samples"
        )
    t = (np.arange(n_samples) - 0.5 * (n_samples - 1)) * dt
    g = np.exp(-0.5 * (t / sigma) ** 2)
    edge = g[0]
    return (g - edge) / (1.0 - edge)


def unit_area(envelope, dt=1.0):
    return envelope / (float(np.sum(envelope)) * dt)


@dataclass
class TransmonAncilla:
    sigma_ns: float = 10.0
    span: float = 4.0
    detune_rad_ns: float = 0.0

    def envelope(self, dt=1.0):
        return truncated_gaussian(self.sigma_ns, self.span, dt)

    def rotation(self, angle=np.pi, axis_phase=0.0, dt=1.0):
        g = unit_area(self.envelope(dt), dt)
        return (0.5 * angle) * np.exp(-1j * axis_phase) * g.astype(complex)


@dataclass
class CavityMode:
    chi_kHz: float = 32.8
    chi_prime_Hz: float = 3.0
    kerr_Hz: float = 1.0
    kappa_kHz: float = 0.0
    sigma_ns: float = 15.0
    span: float = 4.0

    _KHZ_TO_RAD_NS = 2.0e-6 * np.pi
    _HZ_TO_RAD_NS = 2.0e-9 * np.pi

    def envelope(self, dt=1.0):
        return truncated_gaussian(self.sigma_ns, self.span, dt)

    def displacement(self, amount=1.0, dt=1.0):
        g = unit_area(self.envelope(dt), dt)
        return 1j * complex(amount) * g.astype(complex)

    def angular_rates(self):
        chi = self.chi_kHz * self._KHZ_TO_RAD_NS
        chi_prime = self.chi_prime_Hz * self._HZ_TO_RAD_NS
        kerr = self.kerr_Hz * self._HZ_TO_RAD_NS
        kappa = self.kappa_kHz * self._KHZ_TO_RAD_NS
        return chi, chi_prime, kerr, kappa
