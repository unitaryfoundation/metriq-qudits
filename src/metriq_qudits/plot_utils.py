"""Plotting helpers for the single-qudit ECD benchmark.

Two groups of functions:

- Per-circuit *diagnostics* written during a pipeline run (pulse/trajectory/
  Wigner sanity checks), used by the optional ``diagnostics_dir`` hooks in the
  pulse and simulation stages.
- *Results* plots built after a run from the cached ``outputs/*`` NPZ files,
  driven by ``scripts/plot_results.py``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from metriq_qudits.benchmark.metrics import hog_ideal

METRICS = ("hog", "xeb", "fid")


def _save_diag_fig(out_dir: str, name: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pulse / trajectory diagnostics
# --------------------------------------------------------------------------- #

def plot_pulse(epsilon, ancilla_drive) -> None:
    """Cavity drive ε(t) and ancilla drive Ω(t), real/imag parts."""
    _, axs = plt.subplots(2, 1, figsize=(6, 5))
    axs[0].plot(np.real(epsilon), label="Real")
    axs[0].plot(np.imag(epsilon), label="Imag")
    axs[0].set_ylabel(r"$\epsilon(t)$", fontsize=16)
    axs[0].legend()
    axs[1].plot(np.real(ancilla_drive), label="I")
    axs[1].plot(np.imag(ancilla_drive), label="Q")
    axs[1].set_ylabel(r"$\Omega(t)$", fontsize=16)
    axs[1].set_xlabel("ns", fontsize=16)
    axs[1].legend()


def plot_trajectory_complex(alpha_g, alpha_e=None, ax=None, symmetric=False) -> None:
    """Semiclassical g/e conditional trajectory in the complex α plane."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.real(alpha_g), np.imag(alpha_g), label="g")
    ax.fill_between(np.real(alpha_g), np.imag(alpha_g), alpha=0.2)

    radius = np.max(np.abs(alpha_g))
    if alpha_e is not None:
        ax.plot(np.real(alpha_e), np.imag(alpha_e), label="e")
        ax.fill_between(np.real(alpha_e), np.imag(alpha_e), alpha=0.2)
        radius = max(radius, np.max(np.abs(alpha_e)))

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.sin(theta), radius * np.cos(theta), "--")
    if symmetric:
        lim = np.ceil(radius)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Semi-classical trajectory")


def save_conditional_trajectory_diag(alpha_g, alpha_e, out_dir, name, title) -> None:
    """One ECD gate's g/e trajectory: complex-plane path plus components vs time."""
    _, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_trajectory_complex(alpha_g, alpha_e, ax=ax_c, symmetric=True)
    ts = np.arange(len(alpha_g))
    ax_r.plot(ts, np.real(alpha_g), label="Re α_g")
    ax_r.plot(ts, np.imag(alpha_g), label="Im α_g")
    ax_r.plot(ts, np.real(alpha_e), "--", label="Re α_e")
    ax_r.plot(ts, np.imag(alpha_e), "--", label="Im α_e")
    ax_r.set_xlabel("time [ns]")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, frameon=False)
    ax_r.set_title("trajectory components")
    _save_diag_fig(out_dir, name, title)


def _wigner_on_ax(ax, state, rg=3.0, num=100):
    import qutip as qt
    import matplotlib

    vec = np.linspace(-rg, rg, num)
    w = qt.wigner(state, vec, vec)
    lim = np.abs(w).max()
    norm = matplotlib.colors.Normalize(-lim, lim)
    pl = ax.contourf(vec, vec, w, 100, cmap=matplotlib.cm.RdBu, norm=norm)
    ax.set_aspect("equal")
    ax.set_xlabel("Re α")
    ax.set_ylabel("Im α")
    return pl


def _target_state(psi_target, N_cav):
    """Embed a d-level target amplitude vector into an N_cav Fock ket."""
    import qutip as qt

    amps = np.zeros(N_cav, dtype=complex)
    amps[: len(psi_target)] = psi_target
    amps /= np.linalg.norm(amps)
    return qt.Qobj(amps.reshape(N_cav, 1))


def save_state_diagnostics(alpha, rho_final, out_dir,
                           psi_target=None, N_cav=None, circuit_idx=0) -> None:
    """Per-circuit noiseless sanity visuals: the drive-frame trajectory α(t) with
    its transient photon number, and the final cavity Wigner function (target vs
    achieved when a target is given)."""
    ci = circuit_idx
    _, (ax_t, ax_n) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_t.plot(np.real(alpha), np.imag(alpha), lw=1)
    ax_t.set_aspect("equal")
    ax_t.grid(True, alpha=0.3)
    ax_t.set_xlabel("Re α")
    ax_t.set_ylabel("Im α")
    ax_t.set_title("drive-frame trajectory α(t)")
    ax_n.plot(np.abs(alpha) ** 2)
    ax_n.set_xlabel("time [ns]")
    ax_n.set_ylabel("|α(t)|²")
    ax_n.grid(True, alpha=0.3)
    ax_n.set_title("transient photon number")
    _save_diag_fig(os.path.join(out_dir, "trajectories"), f"c{ci:02d}.png",
                   f"circuit {ci}")

    rho_cav = rho_final.ptrace(0)
    n_bar = float(np.real(rho_cav.diag() @ np.arange(rho_cav.shape[0])))
    rg = max(3.0, np.sqrt(max(n_bar, 0.0)) + 2)
    if psi_target is not None and N_cav is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _wigner_on_ax(axes[0], _target_state(psi_target, N_cav), rg=rg)
        axes[0].set_title("target (ideal)")
        pl = _wigner_on_ax(axes[1], rho_cav, rg=rg)
        axes[1].set_title("achieved (noiseless)")
        fig.colorbar(pl, ax=axes, shrink=0.8)
        _save_diag_fig(os.path.join(out_dir, "wigner"), f"c{ci:02d}.png",
                       f"Wigner: target vs achieved, circuit {ci}")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        pl = _wigner_on_ax(ax, rho_cav, rg=rg)
        fig.colorbar(pl, ax=ax)
        _save_diag_fig(os.path.join(out_dir, "wigner"), f"c{ci:02d}.png",
                       f"final state Wigner, circuit {ci}")


# --------------------------------------------------------------------------- #
# Results plots (from cached npz)
# --------------------------------------------------------------------------- #

def plot_metrics_vs_dim(records, out_path: str) -> None:
    """Noiseless HOG/XEB/FID versus qudit dimension d, with the ideal Haar-mean
    HOG reference. ``records`` is a list of dicts with keys d, hog, xeb, fid."""
    records = sorted(records, key=lambda r: r["d"])
    d = np.array([r["d"] for r in records])
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, marker in zip(METRICS, ("o", "s", "^")):
        ax.plot(d, [r[metric] for r in records], marker + "-", label=metric.upper())
    ax.plot(d, [hog_ideal(int(x)) for x in d], "k--", lw=1, label="HOG ideal (Haar)")
    ax.set_xlabel("qudit dimension d")
    ax.set_ylabel("metric")
    ax.set_title("Noiseless benchmark metrics vs dimension")
    ax.set_xticks(d)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_noise_heatmaps(record, out_path: str) -> None:
    """HOG/XEB/FID heatmaps over the T1×T2 grid for one config. ``record`` has
    keys d, T1_us, T2_us, and a T1×T2 array per metric."""
    T1, T2 = record["T1_us"], record["T2_us"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, METRICS):
        grid = np.asarray(record[metric], dtype=float)
        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1,
                       extent=[T1[0], T1[-1], T2[0], T2[-1]])
        ax.set_xlabel("T1 [µs]")
        ax.set_ylabel("T2 [µs]")
        ax.set_title(f"{metric.upper()}  (d={record['d']})")
        fig.colorbar(im, ax=ax, shrink=0.85)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
