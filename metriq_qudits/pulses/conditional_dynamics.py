from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DispersiveRates:
    chi: float = 0.0
    chi_prime: float = 0.0
    kerr: float = 0.0
    kappa: float = 0.0
    delta: float = 0.0


def _velocity(a, s, eps, r):
    n = a.real * a.real + a.imag * a.imag
    pull = -r.delta + r.kerr * n + s * (r.chi + r.chi_prime * n)
    return (1j * pull - 0.5 * r.kappa) * a - 1j * eps


def _rk4_step(a, s, e0, e1, e2, r, dt):
    k1 = _velocity(a, s, e0, r)
    k2 = _velocity(a + 0.5 * dt * k1, s, e1, r)
    k3 = _velocity(a + 0.5 * dt * k2, s, e1, r)
    k4 = _velocity(a + dt * k3, s, e2, r)
    return a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def evolve_pair(drive, rates, start=(0j, 0j), dt=1.0):
    eps = np.asarray(drive, dtype=complex)
    m = len(eps)
    traj_g = np.empty(m, dtype=complex)
    traj_e = np.empty(m, dtype=complex)
    if m == 0:
        return traj_g, traj_e
    a_g = complex(start[0])
    a_e = complex(start[1])
    traj_g[0] = a_g
    traj_e[0] = a_e
    for j in range(m - 1):
        e0 = complex(eps[j])
        e2 = complex(eps[j + 1])
        e1 = 0.5 * (e0 + e2)
        a_g = _rk4_step(a_g, 0.0, e0, e1, e2, rates, dt)
        a_e = _rk4_step(a_e, 1.0, e0, e1, e2, rates, dt)
        traj_g[j + 1] = a_g
        traj_e[j + 1] = a_e
    return traj_g, traj_e


def evolve_with_echoes(drive, echo_samples, rates, start=(0j, 0j), dt=1.0):
    eps = np.asarray(drive, dtype=complex)
    cuts = sorted({int(i) for i in echo_samples if 0 < int(i) < len(eps)})
    bounds = [0, *cuts, len(eps)]

    labeled_g = complex(start[0])
    labeled_e = complex(start[1])
    pieces_g, pieces_e = [], []
    swapped = False
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        segment = eps[lo:hi]
        if not swapped:
            seg_g, seg_e = evolve_pair(segment, rates, (labeled_g, labeled_e), dt)
        else:
            seg_e, seg_g = evolve_pair(segment, rates, (labeled_e, labeled_g), dt)
        pieces_g.append(seg_g)
        pieces_e.append(seg_e)
        if len(seg_g):
            labeled_g, labeled_e = seg_g[-1], seg_e[-1]
        swapped = not swapped
    return np.concatenate(pieces_g), np.concatenate(pieces_e)
