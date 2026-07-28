from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize

from metriq_qudits.pulses.conditional_dynamics import DispersiveRates, evolve_with_echoes
from metriq_qudits.pulses.drive_envelopes import CavityMode, TransmonAncilla


LOBE_SIGNS = (1.0, -1.0, -1.0, 1.0)


def detect_echo_centers(ancilla_drive, level=0.95):
    magnitude = np.abs(np.asarray(ancilla_drive))
    if magnitude.size == 0 or magnitude.max() == 0.0:
        return []
    hot = np.flatnonzero(magnitude >= level * magnitude.max())
    runs = np.split(hot, np.flatnonzero(np.diff(hot) > 1) + 1)
    return [int(run[np.argmax(magnitude[run])]) for run in runs]


@dataclass(frozen=True)
class GateRecipe:
    peak: float
    wait_ns: int
    weights: tuple
    direction: complex


@dataclass
class GateWaveforms:
    cavity_drive: np.ndarray
    ancilla_drive: np.ndarray
    echo_sample: int
    peak_displacement: float
    wait_time_ns: int
    achieved_beta: float
    residual_displacement: float


@dataclass
class CircuitWaveforms:
    cavity_drives: tuple
    ancilla_drive: np.ndarray
    peak_displacements: tuple = ()
    wait_times_ns: tuple = ()
    ancilla_phases: tuple = ()
    achieved_betas: tuple = ()
    final_cavity_phases: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        lengths = {len(self.ancilla_drive), *(len(x) for x in self.cavity_drives)}
        if len(lengths) != 1:
            raise ValueError("all circuit drive arrays must have the same length")
        if self.final_cavity_phases.size not in {0, len(self.cavity_drives)}:
            raise ValueError("one final cavity phase is required per mode")

    @property
    def num_modes(self):
        return len(self.cavity_drives)

    @property
    def peak_displacement(self):
        return max(self.peak_displacements, default=0.0)


@dataclass
class WeakDispersiveSummary:
    qubit_sign: np.ndarray
    conditional_phase: np.ndarray
    running_displacement: np.ndarray
    running_conditional: np.ndarray
    qubit_phase: np.ndarray
    theta_prime: float
    residual_displacement: complex
    beta: complex


class ECDPulseBuilder:
    def __init__(
        self,
        modes,
        ancilla,
        pad_ns=0,
        rate_extractor=None,
        beta_tolerance=1e-3,
        max_calibration_rounds=40,
    ):
        self.modes = list(modes) if isinstance(modes, (list, tuple)) else [modes]
        self.ancilla = ancilla
        self.pad_ns = int(pad_ns)
        self._extract = rate_extractor or (lambda mode: mode.angular_rates())
        self.beta_tolerance = float(beta_tolerance)
        self.max_calibration_rounds = int(max_calibration_rounds)


    def frame(self, mode):
        chi, chi_prime, kerr, kappa = self._extract(mode)
        return DispersiveRates(
            chi=chi, chi_prime=chi_prime, kerr=kerr, kappa=kappa, delta=0.5 * chi
        )

    def conditional_trajectories(
        self, cavity_drive, ancilla_drive=None, echo_samples=None, mode_index=0
    ):
        if echo_samples is None:
            if ancilla_drive is None:
                raise ValueError("need either echo_samples or ancilla_drive")
            echo_samples = detect_echo_centers(ancilla_drive)
        return evolve_with_echoes(
            cavity_drive, echo_samples, self.frame(self.modes[mode_index])
        )


    def _assemble(self, mode, recipe):
        lobe = mode.displacement(1.0)
        echo = self.ancilla.rotation()
        amplitudes = [
            sign * weight * recipe.peak * recipe.direction
            for sign, weight in zip(LOBE_SIGNS, recipe.weights)
        ]

        cavity_blocks, ancilla_blocks = [], []

        def emit(cavity=None, ancilla=None, idle=None):
            block = cavity if cavity is not None else ancilla
            n = idle if idle is not None else len(block)
            cavity_blocks.append(
                cavity if cavity is not None else np.zeros(n, dtype=complex)
            )
            ancilla_blocks.append(
                ancilla if ancilla is not None else np.zeros(n, dtype=complex)
            )

        emit(cavity=amplitudes[0] * lobe)
        emit(idle=recipe.wait_ns)
        emit(cavity=amplitudes[1] * lobe)
        emit(idle=self.pad_ns)
        echo_sample = sum(len(b) for b in cavity_blocks) + len(echo) // 2
        emit(ancilla=echo)
        emit(idle=self.pad_ns)
        emit(cavity=amplitudes[2] * lobe)
        emit(idle=recipe.wait_ns)
        emit(cavity=amplitudes[3] * lobe)

        return (
            np.concatenate(cavity_blocks),
            np.concatenate(ancilla_blocks),
            echo_sample,
        )


    def _seed_recipe(self, mode, beta, peak):
        chi, chi_prime, _, _ = self._extract(mode)
        chi_eff = chi + chi_prime * peak**2
        if chi_eff == 0.0:
            raise ValueError("cannot seed an ECD gate with zero dispersive shift")
        fraction = abs(beta) / (2.0 * peak)
        if fraction >= 1.0:
            raise ValueError(
                f"peak displacement {peak} is too small for |beta| = {abs(beta)}"
            )
        wait = int(abs(np.arcsin(fraction) / chi_eff))
        half, full = np.cos(0.5 * chi_eff * wait), np.cos(chi_eff * wait)
        return GateRecipe(
            peak=float(peak),
            wait_ns=wait,
            weights=(1.0, float(half), float(half), float(full)),
            direction=np.exp(1j * (np.angle(beta) + 0.5 * np.pi)),
        )

    def _closure_report(self, mode, cavity_drive, ancilla_drive, echo_sample):
        traj_g, traj_e = evolve_with_echoes(
            cavity_drive, [echo_sample], self.frame(mode)
        )
        common = traj_g + traj_e
        quarter = echo_sample // 2
        three_quarter = echo_sample + (len(cavity_drive) - echo_sample) // 2
        return {
            "mid": float(np.abs(common[echo_sample])),
            "end": float(np.abs(common[-1])),
            "radius_out": float(0.5 * np.abs(common[quarter])),
            "radius_back": float(0.5 * np.abs(common[three_quarter])),
            "beta": float(np.abs(traj_g[-1] - traj_e[-1])),
        }

    def _fit_weights(self, mode, recipe):
        def objective(weights):
            candidate = replace(recipe, weights=tuple(float(w) for w in weights))
            drives = self._assemble(mode, candidate)
            report = self._closure_report(mode, *drives)
            residuals = (
                report["mid"],
                report["end"],
                report["radius_out"] - recipe.peak,
                report["radius_back"] - recipe.peak,
            )
            return float(sum(r * r for r in residuals))

        result = minimize(
            objective,
            x0=np.asarray(recipe.weights, dtype=float),
            method="Powell",
            options={"xtol": 1e-3, "ftol": 1e-3, "maxfev": 4000},
        )
        return replace(recipe, weights=tuple(float(w) for w in result.x))

    def compile_gate(self, beta, peak_amplitude=10.0, mode_index=0, verbose=False):
        mode = self.modes[mode_index]
        beta = complex(beta)
        recipe = self._seed_recipe(mode, beta, abs(float(peak_amplitude)))
        target = abs(beta)

        tuning_wait = True
        previous_wait = None
        for step in range(self.max_calibration_rounds):
            recipe = self._fit_weights(mode, recipe)
            cavity, ancilla, echo_sample = self._assemble(mode, recipe)
            report = self._closure_report(mode, cavity, ancilla, echo_sample)
            achieved = report["beta"]
            if verbose:
                print(
                    f"round {step}: wait={recipe.wait_ns} peak={recipe.peak:.4f} "
                    f"|beta|={achieved:.5f} (target {target:.5f}) "
                    f"closure end={report['end']:.4f}"
                )
            if abs(achieved - target) <= self.beta_tolerance * target:
                return GateWaveforms(
                    cavity_drive=cavity,
                    ancilla_drive=ancilla,
                    echo_sample=echo_sample,
                    peak_displacement=recipe.peak,
                    wait_time_ns=recipe.wait_ns,
                    achieved_beta=achieved,
                    residual_displacement=report["end"],
                )

            rescale = target / achieved
            if tuning_wait and recipe.wait_ns > 0:
                proposal = max(0, int(np.rint(recipe.wait_ns * rescale)))
                if proposal not in (recipe.wait_ns, previous_wait):
                    previous_wait = recipe.wait_ns
                    recipe = replace(recipe, wait_ns=proposal)
                    continue
                tuning_wait = False
            recipe = replace(recipe, peak=recipe.peak * rescale)

        raise RuntimeError(
            f"ECD calibration did not converge to |beta| = {target} within "
            f"{self.max_calibration_rounds} rounds"
        )


    def weak_dispersive_summary(
        self, cavity_drive, ancilla_drive=None, echo_samples=None, mode_index=0
    ):
        if echo_samples is None:
            if ancilla_drive is None:
                raise ValueError("need either echo_samples or ancilla_drive")
            echo_samples = detect_echo_centers(ancilla_drive)
        chi = self._extract(self.modes[mode_index])[0]

        eps = np.asarray(cavity_drive, dtype=complex)
        sign = np.ones(len(eps))
        for idx in sorted(int(i) for i in echo_samples):
            sign[idx:] *= -1.0
        phi = -0.5 * chi * np.cumsum(sign)

        rotor = np.exp(1j * phi)
        forward = np.cumsum(rotor * eps)
        backward = np.cumsum(eps / rotor)
        delta = 0.5j * (forward / rotor - backward * rotor)
        gamma = -0.5j * (forward / rotor + backward * rotor)

        theta = -2.0 * np.cumsum(np.real(np.conj(eps) * delta))
        cross_term = 2.0 * float(np.imag(gamma[-1] * delta[-1]))
        return WeakDispersiveSummary(
            qubit_sign=sign,
            conditional_phase=phi,
            running_displacement=gamma,
            running_conditional=delta,
            qubit_phase=theta,
            theta_prime=float(theta[-1]) + cross_term,
            residual_displacement=complex(gamma[-1]),
            beta=complex(2.0 * delta[-1]),
        )

    def _spurious_cavity_phase(self, mode, traj_g, traj_e):
        _, chi_prime, kerr, _ = self._extract(mode)
        photons = 0.5 * (np.abs(traj_g) ** 2 + np.abs(traj_e) ** 2)
        return float((2.0 * kerr + chi_prime) * np.sum(photons))


    def compile_circuit(
        self,
        betas,
        rotations,
        peak_amplitude=10.0,
        verbose=False,
        correct_cavity_phases=False,
    ):
        betas = np.atleast_2d(np.asarray(betas))
        rotations = np.asarray(rotations, dtype=float)
        layers, num_modes = betas.shape
        if num_modes != len(self.modes):
            raise ValueError(
                f"betas addresses {num_modes} modes but the compiler has "
                f"{len(self.modes)}"
            )
        expected = (layers * num_modes + 1, 2)
        if rotations.shape != expected:
            raise ValueError(
                f"expected rotations of shape {expected}, got {rotations.shape}"
            )

        cavity_tracks = [[] for _ in range(num_modes)]
        ancilla_track = []

        def emit(ancilla_block, active_mode=None, cavity_block=None):
            n = len(ancilla_block)
            ancilla_track.append(np.asarray(ancilla_block, dtype=complex))
            for m in range(num_modes):
                if active_mode is not None and m == active_mode:
                    cavity_tracks[m].append(np.asarray(cavity_block, dtype=complex))
                else:
                    cavity_tracks[m].append(np.zeros(n, dtype=complex))

        peaks, waits, gate_phases, achieved = [], [], [], []
        transmon_phase = 0.0
        cavity_phase = np.zeros(num_modes)
        cache = [None] * num_modes

        rotation_rows = iter(rotations)
        for _layer in range(layers):
            for j in range(num_modes):
                theta, phi = next(rotation_rows)
                segment = self.ancilla.rotation(angle=theta, axis_phase=phi)
                emit(np.exp(1j * transmon_phase) * segment)

                beta = complex(betas[_layer, j])
                if abs(beta) <= 1e-3:
                    gate_phases.append(0.0)
                    achieved.append(0.0)
                    continue
                if correct_cavity_phases:
                    beta = beta * np.exp(1j * cavity_phase[j])

                cached = cache[j]
                if cached is not None and cached[0] == beta:
                    _, gate, trajectories = cached
                else:
                    gate = self.compile_gate(
                        beta,
                        peak_amplitude=peak_amplitude,
                        mode_index=j,
                        verbose=verbose,
                    )
                    trajectories = None
                    if correct_cavity_phases:
                        trajectories = self.conditional_trajectories(
                            gate.cavity_drive,
                            echo_samples=[gate.echo_sample],
                            mode_index=j,
                        )
                    cache[j] = (beta, gate, trajectories)

                peaks.append(gate.peak_displacement)
                waits.append(gate.wait_time_ns)
                summary = self.weak_dispersive_summary(
                    gate.cavity_drive,
                    echo_samples=[gate.echo_sample],
                    mode_index=j,
                )
                gate_phases.append(summary.theta_prime)
                achieved.append(summary.beta)

                emit(
                    np.exp(1j * transmon_phase) * gate.ancilla_drive,
                    active_mode=j,
                    cavity_block=gate.cavity_drive,
                )
                transmon_phase += summary.theta_prime
                if correct_cavity_phases:
                    cavity_phase[j] += self._spurious_cavity_phase(
                        self.modes[j], *trajectories
                    )

        theta, phi = next(rotation_rows)
        segment = self.ancilla.rotation(angle=theta, axis_phase=phi)
        emit(np.exp(1j * transmon_phase) * segment)

        return CircuitWaveforms(
            cavity_drives=tuple(np.concatenate(track) for track in cavity_tracks),
            ancilla_drive=np.concatenate(ancilla_track),
            peak_displacements=tuple(peaks),
            wait_times_ns=tuple(waits),
            ancilla_phases=tuple(gate_phases),
            achieved_betas=tuple(achieved),
            final_cavity_phases=cavity_phase.copy(),
        )


if __name__ == "__main__":
    compiler = ECDPulseBuilder(
        modes=[CavityMode()], ancilla=TransmonAncilla(sigma_ns=10)
    )
    circuit = compiler.compile_circuit(
        betas=np.array([[1.0], [1.5]]),
        rotations=np.array(
            [[np.pi / 2, 0.0], [np.pi / 2, np.pi / 2], [0.0, 0.0]]
        ),
        peak_amplitude=20.0,
        verbose=True,
    )
    print(f"pulse length: {len(circuit.cavity_drives[0])} ns")
    print(f"virtual-Z phases (rad): {circuit.ancilla_phases}")
    print(f"achieved betas: {circuit.achieved_betas}")
