import numpy as np
from scipy.optimize import fmin
from scipy.signal import find_peaks

from ecd_qv.pulses.pulse_primitives import Storage, Qubit
from ecd_qv.physics.alpha_dynamics import alpha_from_epsilon_finite_difference
from ecd_qv.physics.displaced_frame_model import storage_parameters
from ecd_qv.pulses.pulse_models import CircuitPulse, ECDPulse


def get_flip_idxs(omega):
    return find_peaks(omega, height=np.max(omega) * 0.975)[0]


def single_flip_idx(omega):
    """Echo π-pulse location within one ECD gate. GRAPE envelopes are not
    unimodal, so an ambiguous detection must fail loudly."""
    idxs = get_flip_idxs(np.abs(omega))
    if len(idxs) != 1:
        raise ValueError(f"expected one echo pulse in |omega|, found {len(idxs)} peaks")
    return int(idxs[0])


class ECDPulseBuilder:

    def __init__(self, storages, qubit, buffer_time=0):
        """
        Args:
            storages: list[Storage] — one Storage per cavity mode.
            qubit: Qubit — shared transmon (single qubit drive ω(t)).
            buffer_time: extra idle ns inserted around ECD pulses (default 0).
        """
        self.storages = storages
        self.storage  = storages[0]  # Active storage, updated per ECD gate in ecd_circuit.
        self.qubit    = qubit
        self.buffer_time = buffer_time

    # Pulse constructor
    def _construct_cd(self, alpha=20, beta=1, tw=100, r=1, r2=1, r3=1, r4=1): # alpha, r = overall scale, {r2,r3,r4} = relative scale  # buffer_time = extra storage wait time during qubit pi-pulse
        storage_unit_displacement_complex = self.storage.pulse.disp_gaussian()
        qubit_pi_rotation_complex = self.qubit.pulse.rotate()
        beta_phase = np.angle(beta) + np.pi/2

        storage_first_pulse = alpha * storage_unit_displacement_complex * np.exp(1j * beta_phase)
        storage_first_wait = np.zeros(tw)
        storage_second_pulse = - r2 * alpha * storage_unit_displacement_complex * np.exp(1j * beta_phase)
        storage_second_wait = np.zeros(len(qubit_pi_rotation_complex) + 2 * self.buffer_time)
        storage_third_pulse =  - r3 * alpha * storage_unit_displacement_complex * np.exp(1j * beta_phase)
        storage_third_wait = np.zeros(tw)
        storage_fourth_pulse = + r4 * alpha * storage_unit_displacement_complex * np.exp(1j * beta_phase)

        epsilon = r * np.concatenate(
            [
                storage_first_pulse,
                storage_first_wait,
                storage_second_pulse,
                storage_second_wait,
                storage_third_pulse,
                storage_third_wait,
                storage_fourth_pulse,
            ]
        )
        omega = np.concatenate(
            [np.zeros(tw + 2 * len(storage_unit_displacement_complex) + self.buffer_time), qubit_pi_rotation_complex, np.zeros(tw + 2 * len(storage_unit_displacement_complex) + self.buffer_time)]
        )
        return epsilon, omega

    def _ecd_trajectory(self, epsilon, omega):
        flip_idxs = get_flip_idxs(np.abs(omega))
        chi, chi_prime, self_kerr, kappa = storage_parameters(self.storage)
        delta = chi / 2

        f = lambda epsilon, alpha_g_init, alpha_e_init: alpha_from_epsilon_finite_difference(
            epsilon,
            delta=delta,
            chi=chi,
            chi_prime=chi_prime,
            self_kerr=self_kerr,
            kappa=kappa,
            alpha_g_init=alpha_g_init,
            alpha_e_init=alpha_e_init,
        )
        epsilons = np.split(epsilon, flip_idxs)
        alpha_g = []  # alpha_g defined as the trajectory that starts in g
        alpha_e = []
        g_state = 0  # this bit will track if alpha_g is in g (0) or e (1)
        alpha_g_init = 0 + 0j
        alpha_e_init = 0 + 0j
        for epsilon in epsilons:
            alpha_g_current, alpha_e_current = f(epsilon, alpha_g_init, alpha_e_init)
            if g_state == 0:
                alpha_g.append(alpha_g_current)
                alpha_e.append(alpha_e_current)
            else:
                alpha_g.append(alpha_e_current)
                alpha_e.append(alpha_g_current)
            # because we will flip the qubit, the initial state for the next g trajectory will be the final from e of the current trajectory.
            alpha_g_init = alpha_e_current[-1]
            alpha_e_init = alpha_g_current[-1]
            g_state = 1 - g_state  # flip the bit
        alpha_g = np.concatenate(alpha_g)
        alpha_e = np.concatenate(alpha_e)

        return alpha_g, alpha_e

    """
     Pulse ratios optimizer
    """
    def _initial_guess(self, alpha, beta):
        chi, chi_prime, _, _ = storage_parameters(self.storage)

        n = np.abs(alpha) ** 2
        chi_effective = chi + chi_prime * n
        tw =  int(np.abs(np.arcsin(np.abs(beta) / (2 * alpha)) / chi_effective))
        r = 1.0
        r2 = np.cos((chi_effective / 2.0) * tw)
        r3 = r2
        r4 = np.cos(chi_effective * tw)

        return r, r2, r3, r4, tw

    def _cost_function(self, alpha=20, beta=2, tw=100, r=1, r2=1, r3=1, r4=1, output=False):

        epsilon, omega = self._construct_cd(alpha, beta, tw, r, r2, r3, r4)

        flip_idx = int(len(epsilon) / 2)

        alpha_g, alpha_e = self._ecd_trajectory(epsilon=epsilon, omega=omega)
        mid_disp = np.abs(alpha_g[flip_idx] + alpha_e[flip_idx])
        final_disp = np.abs(alpha_g[-1] + alpha_e[-1])
        first_radius = np.abs(
            (alpha_g[int(flip_idx / 2)] + alpha_e[int(flip_idx / 2)]) / 2.0
        )
        second_radius = np.abs(
            (alpha_g[int(3 * flip_idx / 2)] + alpha_e[int(3 * flip_idx / 2)]) / 2.0
        )

        total_cost = np.abs(mid_disp) + np.abs(final_disp) + np.abs(first_radius - np.abs(alpha)) + np.abs(second_radius - np.abs(alpha))

        if output:
            print(
                "\r  mid_disp: %.4f" % mid_disp
                + " final_disp: %.4f" % final_disp
                + " first_radius: %.4f" % first_radius
                + " second_radius: %.4f" % second_radius
                + " total_cost: %.4f" % total_cost
            )
        return total_cost

    def _cost_optimizer(self, alpha, beta, output=True):

        # guess ratios:
        initial_ratios = self._initial_guess(alpha=alpha, beta=beta)
        r, r2, r3, r4 = initial_ratios[0], initial_ratios[1], initial_ratios[2], initial_ratios[3]

        def cost(x):
            return self._cost_function(alpha=alpha, beta=beta, tw=initial_ratios[4], r=x[0], r2=x[1], r3=x[2], r4=x[3], output=output)

        result = fmin(cost, x0=[r, r2, r3, r4], ftol=1e-3, xtol=1e-3, disp=False)
        r = result[0]
        r2 = result[1]
        r3 = result[2]
        r4 = result[3]
        tw = initial_ratios[4]
        return r, r2, r3, r4, tw

    def _integrated_beta_and_displacement(self, epsilon, omega, output=False):
        # note that the trajectories are first solved without kerr.
        flip_idx = single_flip_idx(omega)
        alpha_g, alpha_e = self._ecd_trajectory(epsilon=epsilon, omega=omega)
        mid_disp = np.abs(alpha_g[flip_idx] + alpha_e[flip_idx])
        final_disp = np.abs(alpha_g[-1] + alpha_e[-1])
        first_radius = np.abs(
            (alpha_g[int(flip_idx / 2)] + alpha_e[int(flip_idx / 2)]) / 2.0
        )
        second_radius = np.abs(
            (alpha_g[int(3 * flip_idx / 2)] + alpha_e[int(3 * flip_idx / 2)]) / 2.0
        )
        if output:
            print(
                "\r  mid_disp: %.4f" % mid_disp
                + " final_disp: %.4f" % final_disp
                + " first_radius: %.4f" % first_radius
                + " second_radius: %.4f" % second_radius,
                # end="",
            )

        obtained_beta = np.abs(alpha_g[-1] - alpha_e[-1])
        obtained_displacement = np.abs(alpha_g[-1] + alpha_e[-1])

        return obtained_beta, obtained_displacement

    def _alpha_tw_optimizer(self, epsilon, omega, alpha, beta, tw, optimization_threshold=1e-3, output=False):
        current_beta, current_disp = self._integrated_beta_and_displacement(epsilon, omega, output)
        diff = np.abs(current_beta) - np.abs(beta)
        ratio = np.abs(current_beta) / np.abs(beta)
        if output:
            print(
                "tw: "
                + str(tw)
                + "alpha: "
                + str(alpha)
                + "beta: "
                + str(current_beta)
                + "diff: "
                + str(diff)
            )
        #  could look at real/imag part...
        # for now, will only consider absolute value
        # first step: lower tw
        #
        if diff < 0:
            tw = int(tw * 1.5)
            ratio = 1.01
        tw_flag = True
        while np.abs(diff) / np.abs(beta) > optimization_threshold:
            if ratio > 1.0 and tw > 0 and tw_flag:
                tw = int(tw / ratio)
            else:
                tw_flag = False
                # if ratio > 1.02:
                #    ratio = 1.02
                # if ratio < 0.98:
                #    ratio = 0.98
                alpha = alpha / ratio

            # update the ratios for the new tw and alpha given chi_prime
            ratios = self._cost_optimizer(alpha, beta, output=output)
            r = ratios[0]
            r2 = ratios[1]
            r3 = ratios[2]
            r4 = ratios[3]

            epsilon, omega = self._construct_cd(alpha, beta, tw, r, r2, r3, r4)

            current_beta, current_disp = self._integrated_beta_and_displacement(epsilon, omega, output)
            diff = np.abs(current_beta) - np.abs(beta)
            ratio = np.abs(current_beta) / np.abs(beta)

            if output:
                print(
                    "tw: "
                    + str(tw)
                    + "alpha: "
                    + str(alpha)
                    + "beta: "
                    + str(current_beta)
                    + "diff: "
                    + str(diff)
                )

        return epsilon, omega, r, r2, r3, r4, alpha, tw, current_beta, current_disp

    # Analytically computes beta, lambda, and the spurious qubit phase theta' for virtual-Z correction (exact when chi'=K=0).
    def _analytic_ecd_component(self, epsilon, omega):
        chi, _, _, _ = storage_parameters(self.storage)
        flip_idxs = [single_flip_idx(omega)]
        pm = +1
        z = []
        for i in range(len(flip_idxs) + 1):
            l_idx = 0 if i == 0 else flip_idxs[i - 1]
            r_idx = len(omega) if i == len(flip_idxs) else flip_idxs[i]
            z.append(pm * np.ones(r_idx - l_idx))
            pm = -1 * pm
        z = np.concatenate(z)
        phi = -(chi / 2.0) * np.cumsum(z)
        gamma = np.zeros_like(phi, dtype=np.complex64)
        delta = np.zeros_like(gamma)
        for i in range(len(phi)):
            delta[i] = -1 * np.sum(np.sin(phi[: i + 1] - phi[i]) * epsilon[: i + 1])
            gamma[i] = -1j * np.sum(np.cos(phi[: i + 1] - phi[i]) * epsilon[: i + 1])
        theta = -2 * np.cumsum(np.real(np.conj(epsilon) * delta))
        correction = 2 * np.imag(gamma[-1] * delta[-1])
        theta_prime = theta[-1] + correction
        lam = gamma[-1]
        beta = 2 * delta[-1]
        return {
            "z": z,
            "theta": theta,
            "gamma": gamma,
            "delta": delta,
            "phi": phi,

            "correction": correction,
            "theta_prime": theta_prime,
            "lambda": lam,
            "beta": beta
        }

    def ecd_gate(
            self,
            beta=1,
            alpha=10,
            output=False
    ):
        beta = float(beta) if isinstance(beta, int) else beta
        alpha = float(alpha) if isinstance(alpha, int) else alpha
        alpha = np.abs(alpha)
        # initial ratios of the displacements
        initial_ratios = self._cost_optimizer(alpha=alpha, beta=beta, output=output)
        r, r2, r3, r4, tw = initial_ratios[0], initial_ratios[1], initial_ratios[2], initial_ratios[3], initial_ratios[4]

        initial_epsilon, initial_omega = self._construct_cd(alpha=alpha, beta=beta, tw=tw, r=r, r2=r2, r3=r3, r4=r4)

        (final_epsilon, final_omega, _, _, _, _, final_alpha, final_tw,
         final_beta, final_displacement) = self._alpha_tw_optimizer(
            initial_epsilon, initial_omega, alpha, beta, tw, output=output,
        )

        return ECDPulse(
            cavity_drive=final_epsilon,
            ancilla_drive=final_omega,
            peak_displacement=final_alpha,
            wait_time_ns=final_tw,
            achieved_beta=final_beta,
            residual_displacement=final_displacement,
        )

    def _spurious_cavity_phase(self, j, alpha_g, alpha_e):
        """Integrate spurious phase accumulated on cavity mode j during one ECD gate.

        Sources (You et al. 2024, Sec. II):
          - Self-Kerr:             phi_SK = 2 K_j ∫ |α_j|² dt           (Eq. 3)
          - 2nd-order dispersive:  phi_2D = χ'_j ∫ |α_j|² dt            (Eq. 5)
        """
        alpha_sq = 0.5 * (np.abs(alpha_g) ** 2 + np.abs(alpha_e) ** 2)
        integral = float(np.sum(alpha_sq))  # dt = 1 ns

        _, chi_prime, self_kerr, _ = storage_parameters(self.storages[j])
        return 2.0 * self_kerr * integral + chi_prime * integral

    def ecd_circuit(
        self,
        betas,
        rotations,
        alpha_CD=10,
        output=False,
        correct_cavity_phases=False,
    ):
        """Build cavity and ancilla-drive waveforms for an ECD+R circuit.

        betas     : (k, num_modes) array of ECD displacement amplitudes
        rotations : (k*num_modes + 1, 2) array of [theta, phi] per rotation segment
        correct_cavity_phases : if True, apply spurious cavity phase corrections
                                after each ECD gate (You et al. 2024, Sec. II).
                                Has no effect when False (default).
        """
        betas     = np.asarray(betas)
        rotations = np.asarray(rotations)
        k, num_modes = betas.shape
        assert num_modes == len(self.storages), (
            f"betas has {num_modes} modes but builder has {len(self.storages)} storages"
        )
        assert rotations.shape == (k * num_modes + 1, 2), (
            f"Expected rotations shape ({k * num_modes + 1}, 2), got {rotations.shape}"
        )

        epsilons               = [[] for _ in range(num_modes)]
        omega_segs             = []
        alphas                 = []
        tws                    = []
        cd_qubit_phases        = []
        achieved_betas         = []
        cumulative_qubit_phase = 0.0
        last_beta_per_mode     = [None] * num_modes
        last_ecd_per_mode      = [None] * num_modes  # (ECDPulse, optional (alpha_g, alpha_e))
        rot_idx                = 0

        # Cavity phase correction state (only used when correct_cavity_phases=True)
        cumulative_cavity_phase  = np.zeros(num_modes)
        for layer in range(k):
            for j in range(num_modes):
                self.storage = self.storages[j]

                theta = float(rotations[rot_idx, 0])
                phi   = float(rotations[rot_idx, 1])
                rot_idx += 1
                o_r = self.qubit.pulse.rotate(theta=theta, phi=phi)
                omega_segs.append(np.exp(1j * cumulative_qubit_phase) * o_r)
                for m in range(num_modes):
                    epsilons[m].append(np.zeros(len(o_r)))

                beta = betas[layer, j]
                if np.abs(beta) > 1e-3:
                    if correct_cavity_phases:
                        phase_corr = cumulative_cavity_phase[j]
                        beta = beta * np.exp(1j * phase_corr)

                    if beta == last_beta_per_mode[j]:
                        cached = last_ecd_per_mode[j]
                        gate = cached[0]
                        if correct_cavity_phases:
                            trajectory = cached[1]
                    else:
                        gate = self.ecd_gate(
                            beta=beta, alpha=alpha_CD, output=output,
                        )
                        if correct_cavity_phases:
                            alpha_g, alpha_e = self._ecd_trajectory(
                                gate.cavity_drive, gate.ancilla_drive,
                            )
                            trajectory = (alpha_g, alpha_e)
                            last_ecd_per_mode[j] = (gate, trajectory)
                        else:
                            last_ecd_per_mode[j] = (gate,)
                        last_beta_per_mode[j] = beta

                    e_cd = gate.cavity_drive
                    o_cd = gate.ancilla_drive
                    alphas.append(gate.peak_displacement)
                    tws.append(gate.wait_time_ns)
                    analytic_dict = self._analytic_ecd_component(e_cd, o_cd)

                    if correct_cavity_phases:
                        cumulative_cavity_phase[j] += self._spurious_cavity_phase(
                            j, trajectory[0], trajectory[1],
                        )
                else:
                    e_cd, o_cd = np.array([]), np.array([])
                    analytic_dict = {"theta_prime": 0.0, "beta": 0.0}

                cd_qubit_phases.append(analytic_dict["theta_prime"])
                achieved_betas.append(analytic_dict["beta"])

                omega_segs.append(np.exp(1j * cumulative_qubit_phase) * o_cd)
                for m in range(num_modes):
                    epsilons[m].append(e_cd if m == j else np.zeros(len(e_cd)))

                cumulative_qubit_phase += cd_qubit_phases[-1]

        theta = float(rotations[rot_idx, 0])
        phi   = float(rotations[rot_idx, 1])
        o_r   = self.qubit.pulse.rotate(theta=theta, phi=phi)
        omega_segs.append(np.exp(1j * cumulative_qubit_phase) * o_r)
        for m in range(num_modes):
            epsilons[m].append(np.zeros(len(o_r)))

        def _concat(segs):
            nonempty = [s for s in segs if len(s) > 0]
            return np.concatenate(nonempty) if nonempty else np.array([])

        epsilons = [_concat(eps_segs) for eps_segs in epsilons]
        omega    = _concat(omega_segs)

        return CircuitPulse(
            cavity_drives=tuple(epsilons),
            ancilla_drive=omega,
            peak_displacements=tuple(alphas),
            wait_times_ns=tuple(tws),
            ancilla_phases=tuple(cd_qubit_phases),
            achieved_betas=tuple(achieved_betas),
            # Spurious rotation e^{+iφn}. The simulator applies e^{-iφn}.
            final_cavity_phases=cumulative_cavity_phase.copy(),
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ecd_qv.plot_utils import plot_pulse, plot_trajectory_complex

    storage = Storage(
        chi_kHz=30.0,
        chi_prime_Hz=1.0,
        Ks_Hz=2.0,
        T1_us=340.0,
        kappa_kHz=0.0,
        sigma_ns=15,
        chop=4,
    )
    qubit = Qubit(sigma_ns=10, chop=4)
    builder = ECDPulseBuilder([storage], qubit)

    betas     = np.array([[1.0], [1.5]])   
    rotations = np.array([
        [np.pi / 2, 0.0],
        [np.pi / 2, np.pi / 2],
        [0.0,       0.0],
    ])

    result = builder.ecd_circuit(
        betas=betas,
        rotations=rotations,
        alpha_CD=20,
        output=True,
    )

    epsilon = result.cavity_drives[0]
    omega   = result.ancilla_drive

    print(f"Pulse length: {len(epsilon)} ns")
    print(f"Spurious phases (rad): {result.ancilla_phases}")
    print(f"Achieved betas: {result.achieved_betas}")

    plot_pulse(epsilon, omega)
    plt.suptitle("ECD circuit pulses")
    plt.tight_layout()
    plt.show()

    alpha_g, alpha_e = builder._ecd_trajectory(epsilon, omega)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_trajectory_complex(alpha_g, alpha_e, ax=ax)
    plt.tight_layout()
    plt.show()
