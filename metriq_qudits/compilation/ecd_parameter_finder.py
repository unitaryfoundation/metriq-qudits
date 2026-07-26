"""
Parameter optimization at the gate level.

All kets are laid out as transmon ⊗ cavity_0 ⊗ … ⊗ cavity_{m-1}:

    psi : (2 * N^m,) complex vector
          psi[:N^m]  — excited-state (|e⟩) block
          psi[N^m:]  — ground-state  (|g⟩) block
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from functools import partial

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from metriq_qudits.benchmark.gates import _ladder_ops, _embed_cavity, _ecd, _rotation_matrix
from metriq_qudits.benchmark.circuit_io import CompiledCircuit
from metriq_qudits.paths import output_dir

# Persistent XLA compilation cache. Compile workers are spawned as fresh
# processes that would otherwise each re-JIT the same (k, N) graphs; a shared
# on-disk cache compiles each graph once and reuses it across circuits, workers,
# and repeated runs. The results are unaffected — only compile latency.
_JAX_CACHE_DIR = output_dir() / "jax_cache"
os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def _mode_fock_probs(psi, num_modes: int, N: int) -> jax.Array:
    """Per-mode Fock populations, shape (num_modes, N), traced over the transmon
    and the other modes."""
    dim_cav = N ** num_modes
    joint   = (jnp.abs(psi[:dim_cav]) ** 2 + jnp.abs(psi[dim_cav:]) ** 2).reshape([N] * num_modes)
    return jnp.stack([
        joint.sum(axis=tuple(i for i in range(num_modes) if i != j))
        for j in range(num_modes)
    ])


@partial(jax.jit, static_argnames=("N", "n_penalize"))
def run_circuit(betas, rotations, N: int, n_penalize: int = 0):
    """Run the ECD+rotation circuit from |g⟩⊗|vac⟩.

    betas     : complex (k, num_modes) ECD displacement amplitudes
    rotations : (k*num_modes + 1, 2) [theta, phi] per rotation

    Returns (psi_g, penalty, max_boundary_pop):
      psi_g            — ground-state cavity block of the final state
      penalty          — accumulated exp-weighted population of the top
                         n_penalize Fock levels after each ECD
      max_boundary_pop — max top-Fock-level population reached after any ECD
                         (the boundary-leakage diagnostic)

    The k layers are executed with jax.lax.scan (rather than a Python loop) so
    the compiled graph is one layer body regardless of depth. The num_modes
    inner loop stays unrolled.
    """
    k, num_modes = betas.shape
    a, adag      = _ladder_ops(N)
    dim_cav      = N ** num_modes
    I_cav        = jnp.eye(dim_cav, dtype=jnp.complex128)
    weights      = jnp.exp(jnp.arange(N - n_penalize, N) + 1.0 - N)

    psi = jnp.zeros(2 * dim_cav, dtype=jnp.complex128).at[dim_cav].set(1.0)
    psi = jnp.kron(_rotation_matrix(rotations[0, 0], rotations[0, 1]), I_cav) @ psi

    # betas is already (k, num_modes). The in-loop rotations (all but the leading
    # one) group per layer as (k, num_modes, 2). scan slices the leading axis.
    layer_rotations = rotations[1:].reshape(k, num_modes, 2)

    def layer_step(carry, layer_inputs):
        psi, penalty, boundary = carry
        betas_l, rots_l = layer_inputs
        for j in range(num_modes):
            D_neg, D_pos = _ecd(betas_l[j], a, adag)
            D_neg_full   = _embed_cavity(D_neg, j, num_modes, N)
            D_pos_full   = _embed_cavity(D_pos, j, num_modes, N)
            psi_e, psi_g = psi[:dim_cav], psi[dim_cav:]
            psi          = jnp.concatenate([D_pos_full @ psi_g, D_neg_full @ psi_e])

            probs    = _mode_fock_probs(psi, num_modes, N)
            boundary = jnp.maximum(boundary, probs[:, -1].max())
            penalty  = penalty + (probs[:, N - n_penalize:] * weights).sum()

            psi = jnp.kron(_rotation_matrix(rots_l[j, 0], rots_l[j, 1]), I_cav) @ psi
        return (psi, penalty, boundary), None

    init = (psi, jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64))
    (psi, penalty, boundary), _ = jax.lax.scan(
        layer_step, init, (betas, layer_rotations))

    return psi[dim_cav:], penalty, boundary


def _unpack(params, k: int, num_modes: int):
    n_beta    = 2 * k * num_modes
    b         = params[:n_beta].reshape(k, num_modes, 2)
    betas     = b[..., 0] + 1j * b[..., 1]
    rotations = params[n_beta:].reshape(k * num_modes + 1, 2)
    return betas, rotations


# Batched multistart optimization (the ECD_control approach: many random
# initializations optimized simultaneously with Adam, following Eickbusch et al. 2022).
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
LOG_EPS               = 1e-12  # floor inside the log-objective
MIN_CONVERGED         = 10     # early-stop once this many members are below err_th
STABILITY_CANDIDATES  = 10     # converged members tried against the stability test


def _objective(p, psi_target, penalty_weight, k, num_modes, N, n_penalize):
    """log(composite loss): a monotone transform of the original
    infidelity + penalty objective that keeps gradients alive near convergence."""
    betas, rotations = _unpack(p, k, num_modes)
    psi_g, penalty, _ = run_circuit(betas, rotations, N, n_penalize)
    overlap_loss = 1.0 - jnp.abs(jnp.dot(psi_target.conj(), psi_g))
    return jnp.log(overlap_loss + penalty_weight * penalty + LOG_EPS)


@partial(jax.jit, static_argnames=("k", "num_modes", "N", "n_penalize", "n_steps"))
def _adam_chunk(params, m, v, t0, psi_target, penalty_weight, lr,
                k: int, num_modes: int, N: int, n_penalize: int, n_steps: int):
    """Run n_steps of Adam on a (B, P) parameter batch and return the updated state."""
    grad_fn = jax.vmap(jax.grad(_objective), in_axes=(0, None, None, None, None, None, None))

    def step(carry, i):
        params, m, v = carry
        g = grad_fn(params, psi_target, penalty_weight, k, num_modes, N, n_penalize)
        t = t0 + i + 1.0
        m = ADAM_B1 * m + (1.0 - ADAM_B1) * g
        v = ADAM_B2 * v + (1.0 - ADAM_B2) * g ** 2
        m_hat = m / (1.0 - ADAM_B1 ** t)
        v_hat = v / (1.0 - ADAM_B2 ** t)
        params = params - lr * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        return (params, m, v), None

    (params, m, v), _ = jax.lax.scan(step, (params, m, v), jnp.arange(n_steps))
    return params, m, v


@partial(jax.jit, static_argnames=("k", "num_modes", "N", "n_penalize"))
def _batched_eval(params, psi_target, k: int, num_modes: int, N: int, n_penalize: int):
    """Per-member (overlap infidelity, penalty) for a (B, P) batch."""
    def ev(p):
        betas, rotations = _unpack(p, k, num_modes)
        psi_g, penalty, _ = run_circuit(betas, rotations, N, n_penalize)
        return 1.0 - jnp.abs(jnp.dot(psi_target.conj(), psi_g)), penalty
    return jax.vmap(ev)(params)


def _pad(psi_haar, M: int, d: int, num_modes: int) -> np.ndarray:
    padded = np.zeros(M ** num_modes, dtype=complex)
    for idx, xs in enumerate(itertools.product(range(d), repeat=num_modes)):
        cav_idx = sum(x * M ** (num_modes - 1 - i) for i, x in enumerate(xs))
        padded[cav_idx] = psi_haar[idx]
    return padded / np.linalg.norm(padded)


@dataclass(frozen=True)
class OptimizerConfig:
    """Search hyperparameters for the ECD parameter optimizer.

    stability_th: when set, a converged candidate is accepted only if its max
    stability infidelity (replay at N + n_test_extra) stays below it. Setting it 
    to None returns the best converged candidate regardless (used by buffer calibration,
    which measures stability curves itself)."""

    n_penalize: int = 0
    penalty_weight: float = 0.0
    err_th: float = 0.01
    batch_size: int = 256
    n_steps: int = 2000
    check_every: int = 100
    lr: float = 0.01
    stability_th: float | None = None
    n_test_extra: tuple[int, ...] = tuple(range(1, 13))


@dataclass(frozen=True, eq=False)
class CompileJob:
    """A single picklable compile task for compile_circuit_worker."""

    d: int
    num_modes: int
    config: OptimizerConfig
    target: np.ndarray
    k_init: int
    k_max: int
    k_step: int
    N: int
    seed: int


class ECDParameterFinder:
    def __init__(self, d: int, num_modes: int, config: OptimizerConfig):
        self.d         = d
        self.num_modes = num_modes
        self.cfg       = config

    def _make_circuit(self, params, target_haar, k: int, N: int,
                      err: float, opt_trace, k_sweep=None) -> CompiledCircuit:
        betas, rotations = _unpack(np.asarray(params), k, self.num_modes)
        betas, rotations = np.array(betas), np.array(rotations)
        _, _, boundary   = run_circuit(betas, rotations, N)
        return CompiledCircuit(
            betas=betas,
            rotations=rotations,
            target_state=np.asarray(target_haar, dtype=complex),
            boundary_leakage=float(boundary),
            infidelity=float(err),
            optimization_trace=np.array(opt_trace),
            depth_sweep=k_sweep,
        )

    def _optimize_at_k(self, target_haar, k: int, N: int, rng, verbose: bool):
        """Batched Adam descent at fixed depth k: a batch of batch_size random
        initializations descends the log-composite objective simultaneously,
        stopping early once MIN_CONVERGED members are below err_th.
        Returns (params, infids, opt_trace)."""
        cfg        = self.cfg
        psi_target = jnp.array(_pad(target_haar, N, self.d, self.num_modes))
        B, m       = cfg.batch_size, self.num_modes
        static     = dict(k=k, num_modes=m, N=N, n_penalize=cfg.n_penalize)

        params = jnp.array(np.concatenate([
            rng.standard_normal((B, 2 * k * m)),
            rng.uniform(0, 2 * np.pi, (B, 2 * (k * m + 1))),
        ], axis=1))
        adam_m = jnp.zeros_like(params)
        adam_v = jnp.zeros_like(params)

        opt_trace, steps_done, infids = [], 0, None
        while steps_done < cfg.n_steps:
            chunk = min(cfg.check_every, cfg.n_steps - steps_done)
            params, adam_m, adam_v = _adam_chunk(
                params, adam_m, adam_v, float(steps_done),
                psi_target, cfg.penalty_weight, cfg.lr, n_steps=chunk, **static)
            steps_done += chunk

            infids, penalties = _batched_eval(params, psi_target, **static)
            infids, penalties = np.array(infids), np.array(penalties)
            objective = np.log(infids + cfg.penalty_weight * penalties + LOG_EPS)
            opt_trace.append(float(objective.min()))
            n_conv = int(np.sum(infids <= cfg.err_th))
            if verbose:
                print(f"      k={k}  step {steps_done:5d}/{cfg.n_steps}"
                      f"  best_err={infids.min():.2e}  converged={n_conv}/{B}",
                      flush=True)
            if n_conv >= min(MIN_CONVERGED, B):
                break

        return params, infids, opt_trace

    def _select(self, params, infids, opt_trace, target_haar, k: int, N: int,
                k_sweep=None, verbose: bool = False) -> CompiledCircuit | None:
        """Best converged batch member that passes the stability test (when
        stability_th is set). None if no member converges or none is stable."""
        candidates = [i for i in np.argsort(infids) if infids[i] <= self.cfg.err_th]
        for idx in candidates[:STABILITY_CANDIDATES]:
            circuit = self._make_circuit(params[idx], target_haar, k, N,
                                         infids[idx], opt_trace, k_sweep)
            if self.cfg.stability_th is None:
                return circuit
            stab = stability_infidelity(
                circuit, target_haar, [N + e for e in self.cfg.n_test_extra],
                self.d, self.num_modes)
            if float(np.nanmax(stab)) < self.cfg.stability_th:
                return circuit
            if verbose:
                print(f"      candidate err={infids[idx]:.2e} UNSTABLE"
                      f" (max stab {float(np.nanmax(stab)):.2e})", flush=True)
        return None

    def find_parameters(self, target_haar, k: int, N: int,
                        rng=None, verbose: bool = False) -> CompiledCircuit | None:
        """Optimize k-layer ECD parameters preparing target_haar in an N-level
        space at a single fixed depth."""
        rng = rng if rng is not None else np.random.default_rng()
        params, infids, opt_trace = self._optimize_at_k(target_haar, k, N, rng, verbose)
        return self._select(params, infids, opt_trace, target_haar, k, N,
                            verbose=verbose)

    def find_parameters_adaptive_k(self, target_haar, k_init: int, k_max: int, N: int,
                                   k_step: int = 1, rng=None,
                                   verbose: bool = False) -> CompiledCircuit | None:
        """Bottom-up minimum-depth sweep (Eickbusch et al. 2022, Fig. 2a): try
        k = k_init, k_init + k_step, …, recording the best infidelity at every
        depth. The returned circuit is compiled at the first depth with a
        stable, converged candidate and carries the whole sweep in k_sweep."""
        rng     = rng if rng is not None else np.random.default_rng()
        k_sweep = []
        for k in range(k_init, k_max + 1, k_step):
            params, infids, opt_trace = self._optimize_at_k(target_haar, k, N,
                                                            rng, verbose)
            k_sweep.append([float(k), float(infids.min())])
            circuit = self._select(params, infids, opt_trace, target_haar, k, N,
                                   k_sweep=np.array(k_sweep), verbose=verbose)
            if circuit is not None:
                return circuit
        return None

def stability_infidelity(circuit: CompiledCircuit, target_haar, N_test_values,
                         d: int, num_modes: int) -> np.ndarray:
    """Replay the circuit at larger truncations to detect dependence on the
    Fock-space boundary."""
    out = []
    for N_test in N_test_values:
        target      = _pad(target_haar, int(N_test), d, num_modes)
        psi_g, _, _ = run_circuit(circuit.betas, circuit.rotations, int(N_test))
        out.append(1.0 - float(np.abs(np.dot(target.conj(), np.array(psi_g)))))
    return np.array(out)


def compile_circuit_worker(job: CompileJob) -> CompiledCircuit | None:
    """Compile one target in a fresh process (picklable top-level function)."""
    finder = ECDParameterFinder(job.d, job.num_modes, job.config)
    return finder.find_parameters_adaptive_k(
        job.target, k_init=job.k_init, k_max=job.k_max, N=job.N,
        k_step=job.k_step, rng=np.random.default_rng(job.seed),
    )
