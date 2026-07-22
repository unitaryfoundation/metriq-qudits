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


def plot_compile_summary(compiled_path: str, out_path: str, err_th: float = 0.01) -> None:
    """Stage-1 compile quality for one config, from the compiled-circuits NPZ.

    Four panels: per-circuit final infidelity and boundary leakage (both colored
    by the accepted depth k), the |β| distribution across all circuits/layers
    (the ECD amplitudes that drive pulse duration), and the batched-optimizer
    convergence traces. Skips silently for an empty cache.
    """
    data = np.load(compiled_path)
    err = data["err"]
    leakage = data["boundary_leakage"]
    k_per = data["k_per_circuit"]
    n_circuits = len(err)
    if n_circuits == 0:
        return
    n_attempted = int(data["n_attempted"]) if "n_attempted" in data.files else None

    beta_mags = np.concatenate([
        np.abs(data["betas"][i, :ki].ravel()) for i, ki in enumerate(k_per)
    ])
    beta_mags = beta_mags[beta_mags > 1e-3]  # drop zero-padding and skipped gates

    k_values = sorted({int(k) for k in k_per})
    palette = plt.cm.tab10.colors
    k_color = {k: palette[i % len(palette)] for i, k in enumerate(k_values)}
    index = np.arange(n_circuits)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax_err, ax_leak, ax_beta, ax_conv = axes.flat

    for k in k_values:
        mask = k_per == k
        ax_err.semilogy(index[mask], err[mask], "o", ms=5, color=k_color[k],
                        label=f"k={k}" if len(k_values) > 1 else None)
    ax_err.axhline(err_th, color="darkred", ls="--", lw=1.2, label=f"err_th={err_th:g}")
    ax_err.set_xlabel("circuit index")
    ax_err.set_ylabel("compile infidelity")
    ax_err.set_title("optimizer final error per circuit")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which="both", alpha=0.3)

    for k in k_values:
        mask = k_per == k
        ax_leak.semilogy(index[mask], np.maximum(leakage[mask], 1e-12), "o", ms=5,
                         color=k_color[k])
    ax_leak.set_xlabel("circuit index")
    ax_leak.set_ylabel("max top-Fock population")
    ax_leak.set_title("boundary leakage per circuit")
    ax_leak.grid(True, which="both", alpha=0.3)

    ax_beta.hist(beta_mags, bins=30, color="tab:blue", alpha=0.75)
    ax_beta.set_xlabel(r"$|\beta|$")
    ax_beta.set_ylabel("count")
    ax_beta.set_title(f"ECD amplitudes, all circuits/layers "
                      f"(median {np.median(beta_mags):.2f})")
    ax_beta.grid(True, alpha=0.3)

    if "opt_trace" in data.files and data["opt_trace"].size:
        stride = int(data["trace_stride"]) if "trace_stride" in data.files else 1
        for i, row in enumerate(data["opt_trace"]):
            trace = np.exp(row[~np.isnan(row)])  # stored log-objective -> objective
            steps = stride * np.arange(1, len(trace) + 1)
            ax_conv.semilogy(steps, trace, color=k_color[int(k_per[i])], alpha=0.4, lw=1)
        ax_conv.axhline(err_th, color="darkred", ls="--", lw=1.2)
        ax_conv.set_xlabel("optimizer step")
        ax_conv.set_ylabel("batch-best objective")
        ax_conv.set_title("optimizer convergence")
        ax_conv.grid(True, which="both", alpha=0.3)
    else:
        ax_conv.text(0.5, 0.5, "no opt_trace in cache\n(recompile with --overwrite)",
                     ha="center", va="center", fontsize=9, color="dimgray")
        ax_conv.set_axis_off()

    survived = f"{n_circuits}/{n_attempted}" if n_attempted else str(n_circuits)
    fig.suptitle(f"compile quality  ({os.path.basename(compiled_path)})  "
                 f"circuits: {survived}", fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
