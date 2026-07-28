"""Verify the displaced-frame simulator against the rotating-frame one.

Runs the same ECD+R pulse through both simulators and compares the final cavity
state. Exit code is 0 on agreement, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import qutip as qt

from metriq_qudits.pulses.echo_gate_compiler import EchoedDisplacementCompiler
from metriq_qudits.pulses.pulse_stage import make_ancilla, make_modes
from metriq_qudits.simulation.displaced_frame_simulator import DisplacedFrameSimulator
from metriq_qudits.simulation.rotating_frame_simulator import RotatingFrameSimulator


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify the displaced-frame simulator against the rotating frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--layers", type=int, default=8,
                   help="Number of ECD+R layers in the test circuit.")
    p.add_argument("--seed", type=int, default=1,
                   help="RNG seed for the random betas and rotations.")
    p.add_argument("--alpha-cd", type=float, default=6.0,
                   help="ECD transient displacement (peak photons ~ alpha_cd**2).")
    p.add_argument("--n-disp", type=int, default=45,
                   help="Displaced-frame cavity truncation (stays near vacuum).")
    p.add_argument("--n-rot", type=int, default=130,
                   help="Rotating-frame cavity truncation (holds the full displacement).")
    p.add_argument("--compare-levels", type=int, default=16,
                   help="Number of low Fock levels to compare populations over.")
    p.add_argument("--fid-tol", type=float, default=0.99,
                   help="Minimum cavity-state fidelity to pass.")
    p.add_argument("--pop-tol", type=float, default=1e-2,
                   help="Maximum allowed L-inf Fock-population difference to pass.")
    p.add_argument("--trunc-tol", type=float, default=1e-6,
                   help="Warn if either simulator's top-Fock population exceeds this.")
    return p.parse_args(argv)


def _build_pulse(layers, seed, alpha_cd):
    """Build one seeded-random multi-layer ECD+R circuit pulse."""
    rng = np.random.default_rng(seed)
    betas = (rng.normal(size=(layers, 1)) + 1j * rng.normal(size=(layers, 1))) * 0.9
    rotations = np.column_stack([rng.uniform(0, np.pi, layers + 1),
                                 rng.uniform(0, 2 * np.pi, layers + 1)])
    compiler = EchoedDisplacementCompiler(make_modes(1), make_ancilla())
    return compiler.compile_circuit(betas, rotations, peak_amplitude=alpha_cd,
                                    correct_cavity_phases=True)


def _final_cavity(sim, epsilon, omega, cavity_phase):
    result, alpha = sim.simulate(epsilon, omega)
    physical = sim.to_physical_frame(result.states[-1], alpha, cavity_phase=cavity_phase)
    cavity = physical.ptrace([0])
    populations = np.real(np.diag(cavity.full()))
    top_fock = float(populations[-3:].sum())
    return cavity, populations, top_fock


def _truncated(cavity, m):
    block = np.asarray(cavity.full())[:m, :m]
    return qt.Qobj(block / np.trace(block))


def main(argv=None) -> int:
    args = _parse_args(argv)
    mode = make_modes(1)[0]

    pulse = _build_pulse(args.layers, args.seed, args.alpha_cd)
    epsilon = pulse.cavity_drives[0]
    omega = pulse.ancilla_drive
    cavity_phase = pulse.final_cavity_phases[0]
    peak_photons = float(np.max(np.abs(np.cumsum(epsilon)) ** 2))
    print(f"layers={args.layers}  pulse length={len(epsilon)} ns  "
          f"peak |alpha|^2 ~ {peak_photons:.0f}")

    disp = DisplacedFrameSimulator(cavity_dim=args.n_disp, mode=mode, qubit_dim=3)
    rot = RotatingFrameSimulator(cavity_dim=args.n_rot, mode=mode, qubit_dim=3)

    cav_d, pops_d, top_d = _final_cavity(disp, epsilon, omega, cavity_phase)
    cav_r, pops_r, top_r = _final_cavity(rot, epsilon, omega, cavity_phase)
    print(f"displaced cavity_dim={args.n_disp:3d}  top-3 Fock pop={top_d:.2e}")
    print(f"rotating  cavity_dim={args.n_rot:3d}  top-3 Fock pop={top_r:.2e}")
    for name, top in (("displaced", top_d), ("rotating", top_r)):
        if top > args.trunc_tol:
            print(f"WARNING: {name} top-Fock population {top:.2e} exceeds "
                  f"{args.trunc_tol:.0e} — increase its cavity_dim.")

    k = args.compare_levels
    pop_linf = float(np.max(np.abs(pops_d[:k] - pops_r[:k])))
    m = min(args.n_disp, args.n_rot)
    fidelity = qt.fidelity(_truncated(cav_d, m), _truncated(cav_r, m))

    print(f"\nFock populations (first {k}):")
    print(f"  displaced: {np.array2string(pops_d[:k], precision=4, suppress_small=True)}")
    print(f"  rotating : {np.array2string(pops_r[:k], precision=4, suppress_small=True)}")
    print(f"  L-inf population diff = {pop_linf:.3e}  (tol {args.pop_tol:.0e})")
    print(f"  cavity state fidelity = {fidelity:.6f}  (tol {args.fid_tol})")

    passed = pop_linf < args.pop_tol and fidelity > args.fid_tol
    print(f"\nAGREEMENT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
