# metriq-qudits

## Installation

The code is currently tested with Python 3.12 and 3.13.

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/unitaryfoundation/metriq-qudits.git
   cd metriq-qudits
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment.

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install the package and its dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```

   This installs the standard CPU build of JAX. GPU users should follow the
   [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
   for their platform before installing the remaining requirements.

5. Confirm that the command is installed and view the available options:

   ```bash
   metriq-qudits --help
   ```

6. Run the smallest configured system without the full noise sweep:

   ```bash
   metriq-qudits --configs d4 --skip-sweep
   ```

Circuit compilation and pulse-level simulation are computationally expensive.
Set `N_JOBS` to run independent circuits in parallel:

```bash
N_JOBS=8 metriq-qudits --configs d4 --skip-sweep
```

On Windows PowerShell, set the same variable with `$env:N_JOBS = 8` before
running the Python command.

Generated artifacts are written to a visible `outputs/` directory:

```text
outputs/
├── calibration/
├── compiled_circuits/
├── pulses/
├── noiseless/
├── noise_sweeps/
└── plots/
```

To choose a different artifact root, use `--output-dir`:

```bash
metriq-qudits --output-dir /path/to/outputs --configs d4 --skip-sweep
```

The same default can be set with the `METRIQ_QUDITS_OUTPUT_DIR` environment
variable.

## Codebase overview

![ECD-QV pipeline](docs/pipeline_flow.png)

The command-line pipeline begins in
[`src/metriq_qudits/cli.py`](src/metriq_qudits/cli.py), which runs the following
stages:

1. **Gate parameter optimization.**
   [`compilation/compile.py`](src/metriq_qudits/compilation/compile.py)
   samples the target unitaries, selects the ECD circuit depth, and coordinates
   compilation. The optimizer itself is implemented in
   [`compilation/ecd_parameter_finder.py`](src/metriq_qudits/compilation/ecd_parameter_finder.py).

2. **Pulse construction.**
   [`pulses/pulse_stage.py`](src/metriq_qudits/pulses/pulse_stage.py) converts
   the compiled ECD parameters into physical control pulses using
   [`pulses/ecd_pulse_builder.py`](src/metriq_qudits/pulses/ecd_pulse_builder.py).

3. **Displaced-frame simulation.**
   [`simulation/sweep.py`](src/metriq_qudits/simulation/sweep.py) runs the
   noiseless calculation and optional T1/T2 sweep. The physical model and
   simulator are implemented in
   [`simulation/displaced_frame_simulator.py`](src/metriq_qudits/simulation/displaced_frame_simulator.py).

4. **Results and plots.**
   [`plot_results.py`](src/metriq_qudits/plot_results.py) loads the saved
   simulation results and generates the figures under `outputs/plots/`.

## References

- [Benchmarking the algorithmic reach of a high-Q cavity qudit](https://arxiv.org/abs/2408.13317) 
   - This paper is the first qudit benchmarking paper from Fermilab. They implement the same protocol and tests as this codebase, except the snap and displacement gateset is used instead of ecd and rotation gateset (what this codebase implements). 
- [Fast Universal Control of an Oscillator with Weak Dispersive Coupling to a Qubit](https://arxiv.org/abs/2111.06414)
   - The primary source for understanding ECD gates. See Fig. 1 for a nice summary, which descirbes the gate's pulse-level decomposition (Fig. 1c) and the ansatz we use (Fig. 1d); The ansatz is a k-layer sequence of alternating rotation and ECD gates. 
   - Table S1 is the source of the Hamiltonian parameters defined in `pulses/pulse_stage.py`: the dispersive coupling `CHI_KHZ` (χ/2π = 32.8 kHz), `CHI_PRIME_HZ` (χ′ = 2χ₀ = 3 Hz from the quoted χ₀/2π = 1.5 Hz), and `SELF_KERR_HZ` (K = 1 Hz). These parameters also feed into the displaced-frame simulation, where they are incorporated into the Hamiltonian in `simulation/displaced_frame_simulator.py` (Qutip) and `simulation/displaced_frame_simulator_dq.py` (Dynamiqs). 
- [Crosstalk-Robust Quantum Control in Multimode Bosonic Systems](https://arxiv.org/abs/2403.00275)
   - This paper provides the theory for the displaced frame Hamiltonian (Eq. B3-5) and its Lindblad (Eq. B6). 
   - The Hamiltonian is assembled in the `DisplacedFrameSimulator` class (`simulation/displaced_frame_simulator.py`) across `_static_hamiltonian`, `_diag_hamiltonian`, and `_offdiag_hamiltonian`, using the mode coefficients defined in `physics/displaced_frame_model.py`. The Lindblad dissipators (Eq. B6) are built in `_make_c_ops`. The Dynamiqs backend mirrors these in `DisplacedFrameSimulatorDQ` (`simulation/displaced_frame_simulator_dq.py`), with the dissipators in `_make_jump_ops`. 
   - The classical displaced-frame trajectory α(t) (Eq. B3) is solved in `physics/alpha_dynamics.py`. 
   - The spurious cavity phase corrections comes from Sec. II — the self-Kerr term φ_SK = 2K∫|α|²dt (Eq. 3) and the second-order dispersive term φ_2D = χ′∫|α|²dt (Eq. 5). They are computed in `_spurious_cavity_phase` (`ECDPulseBuilder` class, `pulses/ecd_pulse_builder.py`) and applied after each ECD gate in `ecd_circuit` when `correct_cavity_phases=True`. 

- [Metriq: A Collaborative Platform for Benchmarking Quantum Computers](https://arxiv.org/abs/2603.08680)
