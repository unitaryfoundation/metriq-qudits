"""Low-level JAX/dynamiqs operator primitives for ECD circuits: Fock ladder
operators, displacement, ECD, rotation matrices, and multi-mode embedding.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import dynamiqs as dq


def _ladder_ops(N: int) -> tuple[jax.Array, jax.Array]:
    return dq.destroy(N).to_jax(), dq.create(N).to_jax()


def _displacement(alpha: complex, a: jax.Array, adag: jax.Array) -> jax.Array:
    return jax.scipy.linalg.expm(alpha * adag - jnp.conj(alpha) * a)


def _embed_cavity(op: jax.Array, j: int, num_modes: int, N: int) -> jax.Array:
    I_c    = dq.eye(N).to_jax()
    ops    = [op if i == j else I_c for i in range(num_modes)]
    result = ops[0]
    for o in ops[1:]:
        result = jnp.kron(result, o)
    return result


def _ecd(beta: complex, a: jax.Array, adag: jax.Array) -> tuple[jax.Array, jax.Array]:
    D_pos = _displacement(beta / 2, a, adag)
    D_neg = jnp.conj(D_pos).T
    return D_neg, D_pos


def _rotation_matrix(theta: float, phi: float) -> jax.Array:
    cos_half = jnp.cos(theta / 2)
    sin_half = jnp.sin(theta / 2)
    return jnp.array([
        [cos_half,                            -1j * jnp.exp(-1j * phi) * sin_half],
        [-1j * jnp.exp(1j * phi) * sin_half,  cos_half                           ],
    ], dtype=jnp.complex128)


def _snap(phases: jax.Array, psi: jax.Array) -> jax.Array:
    return jnp.exp(1j * phases) * psi
