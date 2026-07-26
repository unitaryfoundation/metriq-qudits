"""A numerical solver for the displaced-frame cavity trajectories.

A driven cavity dispersively coupled to a transmon ancilla follows, to leading
order, a *classical* equation of motion for its coherent-state amplitude α(t)
(You et al., arXiv:2403.00275, Eq. B3). Because the cavity frequency depends on
the ancilla's state, α(t) splits into two branches: α_g(t) if the ancilla is in
its ground state |g>, and α_e(t) if it is excited |e>. That state-dependent
separation is what enables an ECD gate to imprint a qubit-conditioned
displacement on the cavity.

The trajectories are solved once, classically, and then reused by both the pulse
builder and the quantum simulators.

Symbols (angular frequencies are in rad/ns; rates are in 1/ns):
    α_g, α_e     complex  cavity coherent-state amplitude; |α|² = mean photon number
    ε (epsilon)  complex  cavity drive envelope [rad/ns]
    δ (delta)    float    frame detuning from the cavity [rad/ns]
    χ (chi)      float    dispersive shift: the ancilla-state-dependent cavity
                          frequency [rad/ns].
    χ' (chi_prime) float  second-order, photon-number-dependent part of χ [rad/ns]
    K (self_kerr)  float  cavity self-Kerr: photon-number-dependent frequency [rad/ns]
    κ (kappa)    float    cavity photon-loss rate [1/ns] -- a rate, so no 2π factor
"""

import numpy as np


def conditional_cavity_rates(
    alpha_g,
    alpha_e,
    drive,
    *,
    delta=0,
    chi=0,
    chi_prime=0,
    self_kerr=0,
    kappa=0,
):
    """Instantaneous dα/dt for the ground and excited ancilla branches.

    Evaluates the mean-field equation of motion (You et al. 2024, Eq. B3)

        dα/dt = -i·δ·α  +  i·K·|α|²·α  -  (κ/2)·α  -  i·ε      (ground branch)

    where the four terms are, in order: rotation from the frame detuning δ, the
    self-Kerr frequency pull (grows with photon number |α|²), photon loss at rate
    κ, and driving by ε. The excited branch adds the dispersive pull
    +i·(χ + χ'·|α|²)·α, so the two branches rotate at different rates and pull
    apart -- the physical origin of the conditional displacement.

    Parameters
    ----------
    alpha_g, alpha_e : complex
        Current cavity amplitude on the ground / excited branch.
    drive : complex
        Cavity drive ε at this instant [rad/ns].
    delta : float, optional
        Frame detuning δ [rad/ns]. The pulse builder solves in a frame at the
        g/e midpoint, i.e. ``delta = chi / 2``.
    chi : float, optional
        Dispersive shift χ [rad/ns], typically > 0. Affects only the excited branch.
    chi_prime : float, optional
        Second-order (photon-number-dependent) dispersive shift χ' [rad/ns].
    self_kerr : float, optional
        Cavity self-Kerr K [rad/ns].
    kappa : float, optional
        Cavity photon-loss rate κ [1/ns] (a rate, so no factor of 2π).

    Returns
    -------
    (complex, complex)
        The rates ``dα_g/dt`` and ``dα_e/dt``.
    """
    abs_g_squared = np.abs(alpha_g) ** 2
    abs_e_squared = np.abs(alpha_e) ** 2
    common_g = (
        -1j * delta * alpha_g
        + 1j * self_kerr * abs_g_squared * alpha_g
        - (kappa / 2.0) * alpha_g
        - 1j * drive
    )
    common_e = (
        -1j * delta * alpha_e
        + 1j * self_kerr * abs_e_squared * alpha_e
        - (kappa / 2.0) * alpha_e
        - 1j * drive
    )
    return common_g, common_e + 1j * (
        chi + chi_prime * abs_e_squared
    ) * alpha_e


def alpha_from_epsilon_finite_difference(
        epsilon_array,
        delta=0,
        chi=0,
        chi_prime=0,
        self_kerr=0,
        kappa=0,
        alpha_g_init=0 + 0j,
        alpha_e_init=0 + 0j,
):
    """Integrate the conditional cavity trajectories α_g(t), α_e(t) from a drive.

    Solves the mean-field equation of motion in :func:`conditional_cavity_rates`
    on a fixed 1-ns grid with a leapfrog (second-order central-difference) step,
    given the whole cavity drive waveform ε(t).

    Parameters
    ----------
    epsilon_array : array_like of complex
        Cavity drive envelope ε(t) sampled every 1 ns [rad/ns]. Needs ≥ 2 samples.
    delta, chi, chi_prime, self_kerr, kappa : float, optional
        Hamiltonian parameters; see :func:`conditional_cavity_rates`. Angular
        frequencies are in rad/ns; ``kappa`` is a rate in 1/ns.
    alpha_g_init, alpha_e_init : complex, optional
        Initial amplitudes at t = 0 for the two branches (default: vacuum, α = 0).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Complex arrays α_g(t) and α_e(t), each the same length as ``epsilon_array``.
    """
    epsilon_array = np.asarray(epsilon_array, dtype=complex)
    if len(epsilon_array) < 2:
        raise ValueError("a trajectory requires at least two time samples")
    dt = 1
    alpha_g = np.zeros(len(epsilon_array), dtype=complex)
    alpha_e = np.zeros(len(epsilon_array), dtype=complex)
    alpha_g[0], alpha_g[1] = alpha_g_init, alpha_g_init
    alpha_e[0], alpha_e[1] = alpha_e_init, alpha_e_init
    for j in range(1, len(epsilon_array) - 1):
        rate_g, rate_e = conditional_cavity_rates(
            alpha_g[j],
            alpha_e[j],
            epsilon_array[j],
            delta=delta,
            chi=chi,
            chi_prime=chi_prime,
            self_kerr=self_kerr,
            kappa=kappa,
        )
        alpha_g[j + 1] = 2 * dt * rate_g + alpha_g[j - 1]
        alpha_e[j + 1] = 2 * dt * rate_e + alpha_e[j - 1]
    return alpha_g, alpha_e


