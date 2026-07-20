"""Classical displaced-frame trajectories shared by builders and simulators."""

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
    """Return the ground/excited classical rates in the code convention.
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
    """Integrate conditional trajectories with a 1-ns leapfrog step."""
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


