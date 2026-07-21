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

## References

- [Benchmarking the algorithmic reach of a high-Q cavity qudit](https://arxiv.org/abs/2408.13317)
- [Fast Universal Control of an Oscillator with Weak Dispersive Coupling to a Qubit](https://arxiv.org/abs/2111.06414)
- [Ultracoherent superconducting cavity-based multiqudit platform with error-resilient control](https://arxiv.org/abs/2506.03286)
- [Crosstalk-Robust Quantum Control in Multimode Bosonic Systems](https://arxiv.org/abs/2403.00275)
- [Metriq: A Collaborative Platform for Benchmarking Quantum Computers](https://arxiv.org/abs/2603.08680)
