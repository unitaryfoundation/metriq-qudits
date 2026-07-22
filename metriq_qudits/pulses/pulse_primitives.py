import numpy as np


def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0] # Gaussian value at the trancation edge
    return (P - ofs) / (1 - ofs) # Return normalized gaussian with 0 value at edges


class Pulse:
    def __init__(self, sigma=11, chop=4, detune=0):
        self.sigma = sigma
        self.chop = chop
        self.detune = detune

    def make_wave(self):
        wave = gaussian_wave(sigma=self.sigma, chop=self.chop)
        return np.real(wave), np.imag(wave)

    def rotate(self, theta=np.pi, phi=0, dt=1):
        # Gaussian ancilla rotation with the paper's complex-envelope
        # convention (You et al. 2024, Sec. III.A.1): ancilla_drive = eps_I +
        # i eps_Q, H_drive = Re(drive) X + Im(drive) Y. The exp(-i phi)
        # convention agrees with the compiler, whose transmon blocks are
        # ordered (|e>,|g>) versus the simulator's (|g>,|e>).
        wave = gaussian_wave(sigma=self.sigma, chop=self.chop)
        wave = (1 + 0j) * wave / np.trapezoid(wave, dx=dt)
        return (theta / 2.0) * np.exp(-1j * phi) * wave

    def disp_gaussian(self, alpha=1, dt=1):
        wave = gaussian_wave(sigma=self.sigma, chop=self.chop)
        wave = (1 + 0j) * wave / np.trapezoid(wave, dx=dt)
        return np.abs(alpha) * np.exp(1j * (np.pi / 2.0 + np.angle(alpha))) * wave


class Storage:
    """Cavity parameters quoted as positive cyclic-frequency magnitudes.

    ``Ks_Hz`` is the full Kerr ``K = 2*Kc`` and ``chi_prime_Hz`` is You et
    al.'s ``chi_prime = 2*chi0``. Hamiltonian signs and factors of one half are
    applied centrally in :mod:`displaced_frame_model`.

    The dispersive defaults are the published Eickbusch et al. 2022
    (arXiv:2111.06414) Table S1 values: chi/2pi = 32.8 kHz, chi_prime = 2*chi0 =
    3 Hz (from the quoted chi0/2pi = 1.5 Hz), and K = 1 Hz.
    """

    def __init__(
            self,
            chi_kHz=32.8,
            chi_prime_Hz=3.0,
            Ks_Hz=1.0,
            sigma_ns=15,
            kappa_kHz=0,
            chop=4,
    ):
        self.chi_kHz = chi_kHz
        self.chi_prime_Hz = chi_prime_Hz
        self.Ks_Hz = Ks_Hz
        self.kappa_kHz = kappa_kHz
        self.sigma_ns = sigma_ns
        self.chop = chop
        self.pulse = Pulse(sigma=self.sigma_ns, chop=self.chop)


class Qubit:
    def __init__(self, sigma_ns=10, chop=4, detune=0):
        self.sigma_ns = sigma_ns
        self.chop = chop
        self.detune = detune
        self.pulse = Pulse(sigma=sigma_ns, chop=chop, detune=detune)
