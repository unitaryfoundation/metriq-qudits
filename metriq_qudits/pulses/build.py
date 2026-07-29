"""Turn optimized ECD parameters into synchronized physical waveforms."""

from __future__ import annotations

import os
import time

import numpy as np

from metriq_qudits.compilation.circuit_io import load_circuits
from metriq_qudits.parallel import parallel_map
from metriq_qudits.paths import data_dir
from metriq_qudits.pulses.drive_envelopes import CavityMode, TransmonAncilla
from metriq_qudits.pulses.ecd_pulse_builder import CircuitWaveforms, ECDPulseBuilder
from metriq_qudits.pulses.pulse_io import load_pulses, save_pulses
from metriq_qudits.system_config import SystemConfig

# Eickbusch et al. 2022, Table S1. Quoted frequencies are cycles/s. Numerical
# kernels convert them once at their unit boundaries.
CHI_KHZ = 32.8
# Eickbusch Table S1 quotes chi0/2pi = 1.5 Hz in the Hamiltonian
# ``-chi0*a†2*a2*nq``. You et al. write the same term as ``chi'/2``, so the
# magnitude stored in this code is chi' = 2*chi0 = 3 Hz.
CHI_PRIME_HZ = 3.0
SELF_KERR_HZ = 1.0

CHI_RAD_S = -2 * np.pi * CHI_KHZ * 1e3
CHI_PRIME_RAD_S = 2 * np.pi * CHI_PRIME_HZ
SELF_KERR_RAD_S = 2 * np.pi * SELF_KERR_HZ

PEAK_TRANSIENT_DISPLACEMENT = 10
QUBIT_SIGMA_NS = 10
QUBIT_SPAN = 4
DEFAULT_N_JOBS = int(os.environ.get("N_JOBS", "1"))


def variant_suffix(correct_phases: bool) -> str:
    return "" if correct_phases else "_nopc"


def physics_metadata(correct_phases: bool) -> dict:
    return {
        "chi": CHI_RAD_S,
        "K": SELF_KERR_RAD_S,
        "chi_prime": CHI_PRIME_RAD_S,
        "phase_corrected": correct_phases,
    }


def physics_metadata_matches(actual, correct_phases: bool) -> bool:
    """Return whether a cache was generated with the current physical model."""
    for key, expected in physics_metadata(correct_phases).items():
        if key not in actual:
            return False
        observed = actual[key]
        if isinstance(expected, bool):
            if bool(observed) != expected:
                return False
        elif not np.isclose(float(observed), expected, rtol=1e-12, atol=0.0):
            return False
    return True


def make_modes(num_modes: int) -> list[CavityMode]:
    return [
        CavityMode(chi_kHz=CHI_KHZ, chi_prime_Hz=CHI_PRIME_HZ, kerr_Hz=SELF_KERR_HZ)
        for _ in range(num_modes)
    ]


def make_ancilla() -> TransmonAncilla:
    return TransmonAncilla(sigma_ns=QUBIT_SIGMA_NS, span=QUBIT_SPAN)


def pulse_path(config: SystemConfig, compiled_metadata: dict, correct_phases: bool) -> str:
    suffix = variant_suffix(correct_phases)
    filename = (
        f"{config.key}_k{int(compiled_metadata['k'])}"
        f"_nu{int(compiled_metadata['N_unitaries'])}"
        f"_seed{int(compiled_metadata['seed'])}{suffix}.npz"
    )
    return str(data_dir("pulses") / filename)


def _build_one_pulse(args) -> CircuitWaveforms:
    betas, rotations, correct_phases = args
    compiler = ECDPulseBuilder(make_modes(1), make_ancilla())
    return compiler.compile_circuit(
        betas=betas,
        rotations=rotations,
        peak_amplitude=PEAK_TRANSIENT_DISPLACEMENT,
        correct_cavity_phases=correct_phases,
    )


def build_circuit_pulses(
    compiled_path: str,
    *,
    correct_phases: bool = True,
    overwrite: bool = False,
    n_jobs: int = DEFAULT_N_JOBS,
) -> str:
    """Build and cache physical waveforms for every compiled circuit."""
    compiled, metadata = load_circuits(compiled_path)
    if int(metadata["num_modes"]) != 1:
        raise ValueError("expected a single-qudit pulse cache")
    config = SystemConfig(d=int(metadata["d"]))
    output_path = pulse_path(config, metadata, correct_phases)
    data_dir("pulses").mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_path) and not overwrite:
        _, cached_metadata = load_pulses(output_path)
        settings_match = (
            physics_metadata_matches(cached_metadata, correct_phases)
            and float(cached_metadata.get("alpha_CD", np.nan))
            == PEAK_TRANSIENT_DISPLACEMENT
            and float(cached_metadata.get("qubit_sigma_ns", np.nan))
            == QUBIT_SIGMA_NS
            and int(cached_metadata.get("qubit_span", -1)) == QUBIT_SPAN
        )
        if settings_match:
            print(f"  [pulses] cached -> {os.path.basename(output_path)}")
            return output_path
        print("  [pulses] cached physics changed - rebuilding")

    depths = np.array([circuit.depth for circuit in compiled], dtype=np.int32)
    depth_summary = (
        f"k={depths[0]}"
        if np.all(depths == depths[0])
        else f"k={depths.min()}-{depths.max()}"
    )
    print(
        f"\n  [pulses] {len(compiled)} circuits  {depth_summary}  N_JOBS={n_jobs}"
    )

    jobs = [
        (circuit.betas, circuit.rotations, correct_phases) for circuit in compiled
    ]
    pulses = []
    start = time.perf_counter()
    for index, pulse in enumerate(parallel_map(_build_one_pulse, jobs, n_jobs)):
        pulses.append(pulse)
        print(
            f"    circuit {index + 1:3d}/{len(compiled)}  "
            f"k={depths[index]}  peak_alpha={pulse.peak_displacement:.2f}  "
            f"({time.perf_counter() - start:.1f}s)",
            flush=True,
        )

    save_pulses(
        output_path,
        pulses,
        {
            "d": config.d,
            "k": metadata["k"],
            "num_modes": 1,
            "ansatz": "haar",
            "N_cav": metadata["N_cav"],
            "seed": metadata["seed"],
            "k_per_circuit": depths,
            "dt": 1.0,
            "alpha_CD": PEAK_TRANSIENT_DISPLACEMENT,
            "qubit_sigma_ns": QUBIT_SIGMA_NS,
            "qubit_span": QUBIT_SPAN,
            **physics_metadata(correct_phases),
        },
    )
    print(f"  Saved -> {output_path}")
    return output_path


def save_gate_trajectory_diagnostics(compiled_path: str, output_dir: str) -> None:
    """Plot the first nonzero ECD gate's g/e trajectory for each circuit."""
    from metriq_qudits.plotting.utils import save_conditional_trajectory_diag

    compiled, _ = load_circuits(compiled_path)
    compiler = ECDPulseBuilder(make_modes(1), make_ancilla())
    for circuit_index, circuit in enumerate(compiled):
        betas = np.atleast_1d(circuit.betas[:, 0])
        nonzero = np.where(np.abs(betas) > 1e-3)[0]
        if len(nonzero) == 0:
            continue
        beta = complex(betas[nonzero[0]])
        gate = compiler.compile_gate(beta=beta, peak_amplitude=PEAK_TRANSIENT_DISPLACEMENT)
        alpha_g, alpha_e = compiler.conditional_trajectories(
            gate.cavity_drive, echo_samples=[gate.echo_sample],
        )
        save_conditional_trajectory_diag(
            alpha_g,
            alpha_e,
            os.path.join(output_dir, "ecd_gates"),
            f"c{circuit_index:02d}.png",
            f"ECD gate g/e trajectory  (circuit {circuit_index}, "
            f"|beta|={abs(beta):.2f})",
        )
