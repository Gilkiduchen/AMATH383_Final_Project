"""ODE analysis figures and tables for the AMATH 383 seizure project.

This script is self-contained and writes all outputs to
final_project_seizure_dynamics/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import expit


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Params:
    tau_e: float = 1.0
    tau_i: float = 2.0
    w_ee: float = 10.0
    w_ei: float = 9.0
    w_ie: float = 10.0
    w_ii: float = 2.0
    q: float = 0.5
    a_e: float = 1.5
    a_i: float = 1.5
    theta_e: float = 3.5
    theta_i: float = 3.5


def sigmoid(u: np.ndarray | float, a: float, theta: float) -> np.ndarray | float:
    return expit(a * (u - theta))


def rhs(y: np.ndarray, p_drive: float, params: Params) -> np.ndarray:
    e, i = y
    u_e = params.w_ee * e - params.w_ei * i + p_drive
    u_i = params.w_ie * e - params.w_ii * i + params.q
    s_e = sigmoid(u_e, params.a_e, params.theta_e)
    s_i = sigmoid(u_i, params.a_i, params.theta_i)
    de = (-e + (1.0 - e) * s_e) / params.tau_e
    di = (-i + (1.0 - i) * s_i) / params.tau_i
    return np.asarray([de, di], dtype=float)


def jacobian(y: np.ndarray, p_drive: float, params: Params) -> np.ndarray:
    e, i = y
    u_e = params.w_ee * e - params.w_ei * i + p_drive
    u_i = params.w_ie * e - params.w_ii * i + params.q
    s_e = sigmoid(u_e, params.a_e, params.theta_e)
    s_i = sigmoid(u_i, params.a_i, params.theta_i)
    sp_e = params.a_e * s_e * (1.0 - s_e)
    sp_i = params.a_i * s_i * (1.0 - s_i)

    return np.asarray(
        [
            [
                (-1.0 - s_e + (1.0 - e) * sp_e * params.w_ee) / params.tau_e,
                ((1.0 - e) * sp_e * (-params.w_ei)) / params.tau_e,
            ],
            [
                ((1.0 - i) * sp_i * params.w_ie) / params.tau_i,
                (-1.0 - s_i + (1.0 - i) * sp_i * (-params.w_ii)) / params.tau_i,
            ],
        ],
        dtype=float,
    )


def find_equilibria(p_drive: float, params: Params) -> list[dict[str, float]]:
    guesses = np.array(
        [[e, i] for e in np.linspace(0.02, 0.98, 8) for i in np.linspace(0.02, 0.98, 8)]
    )
    roots: list[np.ndarray] = []

    for guess in guesses:
        sol = root(lambda y: rhs(y, p_drive, params), guess)
        if not sol.success:
            continue
        y = np.clip(sol.x, 0.0, 1.0)
        if np.linalg.norm(rhs(y, p_drive, params)) > 1e-7:
            continue
        if not any(np.linalg.norm(y - old) < 1e-4 for old in roots):
            roots.append(y)

    rows = []
    for y in sorted(roots, key=lambda arr: (arr[0], arr[1])):
        eig = np.linalg.eigvals(jacobian(y, p_drive, params))
        max_real = float(np.max(np.real(eig)))
        rows.append(
            {
                "P": float(p_drive),
                "E_star": float(y[0]),
                "I_star": float(y[1]),
                "eig1_real": float(np.real(eig[0])),
                "eig1_imag": float(np.imag(eig[0])),
                "eig2_real": float(np.real(eig[1])),
                "eig2_imag": float(np.imag(eig[1])),
                "max_real_eig": max_real,
                "stable": bool(max_real < 0.0),
            }
        )
    return rows


def simulate(p_drive: float, params: Params, y0: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        lambda _t, y: rhs(y, p_drive, params),
        (0.0, 80.0),
        np.asarray(y0, dtype=float),
        max_step=0.05,
    )
    return sol.t, sol.y


def transition_intervals(eq_df: pd.DataFrame) -> list[tuple[float, float]]:
    """Return grid-level P intervals where the tracked equilibrium is locally unstable."""
    unstable_p = sorted(eq_df.loc[~eq_df["stable"], "P"].unique())
    if not unstable_p:
        return []

    step = float(np.median(np.diff(sorted(eq_df["P"].unique()))))
    intervals = []
    start = unstable_p[0]
    prev = unstable_p[0]
    for p_drive in unstable_p[1:]:
        if abs(p_drive - prev) > step * 1.5:
            intervals.append((float(start), float(prev)))
            start = p_drive
        prev = p_drive
    intervals.append((float(start), float(prev)))
    return intervals


def plot_sweep_and_trajectories(eq_df: pd.DataFrame, params: Params) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.3), constrained_layout=True)

    stable = eq_df[eq_df["stable"]]
    unstable = eq_df[~eq_df["stable"]]
    axes[0].scatter(stable["P"], stable["E_star"], s=10, label="stable", color="#1f77b4")
    axes[0].scatter(unstable["P"], unstable["E_star"], s=10, label="unstable", color="#d62728")
    axes[0].set_xlabel("excitability drive P")
    axes[0].set_ylabel("equilibrium excitation E*")
    axes[0].set_title("Equilibrium sweep")
    axes[0].legend(frameon=False)

    axes[1].scatter(stable["P"], stable["max_real_eig"], s=10, color="#1f77b4", label="stable")
    axes[1].scatter(unstable["P"], unstable["max_real_eig"], s=10, color="#d62728", label="unstable")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("excitability drive P")
    axes[1].set_ylabel("largest real eigenvalue")
    axes[1].set_title("Local stability")
    axes[1].legend(frameon=False)

    for begin, end in transition_intervals(eq_df):
        for ax in axes[:2]:
            ax.axvspan(begin, end, color="#d62728", alpha=0.08, lw=0)

    for p_drive, color, label in [
        (0.8, "#2ca02c", "stable low P"),
        (2.0, "#ff7f0e", "unstable interval"),
        (3.2, "#9467bd", "restabilized high P"),
    ]:
        t, y = simulate(p_drive, params, y0=(0.05, 0.05))
        axes[2].plot(t, y[0], color=color, label=f"E(t), {label}")
        axes[2].plot(t, y[1], color=color, linestyle="--", alpha=0.85, label=f"I(t), {label}")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("activity")
    axes[2].set_title("Example trajectories")
    axes[2].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.savefig(OUT_DIR / "ode_bifurcation_stability_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_plane(params: Params, p_drive: float = 2.0) -> None:
    e_grid, i_grid = np.meshgrid(np.linspace(0.0, 0.8, 25), np.linspace(0.0, 0.8, 25))
    de = np.zeros_like(e_grid)
    di = np.zeros_like(i_grid)
    speed = np.zeros_like(e_grid)
    for idx in np.ndindex(e_grid.shape):
        v = rhs(np.asarray([e_grid[idx], i_grid[idx]]), p_drive, params)
        de[idx], di[idx] = v
        speed[idx] = np.linalg.norm(v)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.streamplot(e_grid, i_grid, de, di, color=np.log1p(speed), cmap="viridis", density=1.2)
    for row in find_equilibria(p_drive, params):
        color = "#1f77b4" if row["stable"] else "#d62728"
        marker = "o" if row["stable"] else "x"
        ax.scatter(row["E_star"], row["I_star"], color=color, marker=marker, s=70, zorder=3)
    ax.set_xlabel("excitatory activity E")
    ax.set_ylabel("inhibitory activity I")
    ax.set_title(f"Wilson-Cowan phase plane in unstable interval, P = {p_drive:.2f}")
    fig.savefig(OUT_DIR / "ode_phase_plane.png", dpi=220)
    plt.close(fig)


def plot_model_diagram() -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.axis("off")
    node_style = dict(boxstyle="round,pad=0.35", fc="#f4f6f8", ec="#2f3b45", lw=1.2)
    ax.text(0.28, 0.55, "Excitatory\npopulation E(t)", ha="center", va="center", bbox=node_style, fontsize=11)
    ax.text(0.72, 0.55, "Inhibitory\npopulation I(t)", ha="center", va="center", bbox=node_style, fontsize=11)
    ax.annotate("", xy=(0.58, 0.60), xytext=(0.42, 0.60), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"))
    ax.annotate("", xy=(0.42, 0.48), xytext=(0.58, 0.48), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#d62728"))
    ax.annotate("", xy=(0.28, 0.73), xytext=(0.28, 0.91), arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"))
    ax.text(0.28, 0.95, "drive P", ha="center", va="bottom", fontsize=11)
    ax.text(0.50, 0.66, "excites", ha="center", va="bottom", fontsize=9, color="#1f77b4")
    ax.text(0.50, 0.39, "inhibits", ha="center", va="top", fontsize=9, color="#d62728")
    ax.text(0.5, 0.12, "For the chosen parameters, intermediate drive produces loss of local stability.", ha="center", fontsize=10)
    fig.savefig(OUT_DIR / "ode_ei_model_diagram.png", dpi=220)
    plt.close(fig)


def main() -> None:
    params = Params()
    eq_rows = []
    for p_drive in np.linspace(0.0, 8.0, 201):
        eq_rows.extend(find_equilibria(float(p_drive), params))
    eq_df = pd.DataFrame(eq_rows)
    eq_df.to_csv(OUT_DIR / "ode_equilibrium_sweep.csv", index=False)

    plot_sweep_and_trajectories(eq_df, params)
    plot_phase_plane(params, p_drive=2.0)
    plot_model_diagram()

    intervals = transition_intervals(eq_df)
    crossing_rows = eq_df.iloc[np.argsort(np.abs(eq_df["max_real_eig"].to_numpy()))[:6]].copy()
    crossing_rows.to_csv(OUT_DIR / "ode_stability_crossing_candidates.csv", index=False)

    summary = {
        "n_equilibrium_rows": int(len(eq_df)),
        "n_stable": int(eq_df["stable"].sum()),
        "n_unstable": int((~eq_df["stable"]).sum()),
        "p_min": float(eq_df["P"].min()),
        "p_max": float(eq_df["P"].max()),
        "unstable_intervals": intervals,
        "unstable_intervals_grid": intervals,
        "grid_resolution_note": "The unstable interval bounds are grid-level values from the chosen P sweep; actual crossings lie between neighboring grid points.",
        "interpretation": "For these parameters the sweep tracks one equilibrium that loses local stability over a finite P interval; the complex eigenvalues at the crossings suggest a Hopf-like oscillatory transition, not a saddle-node/hysteresis transition.",
    }
    pd.Series(summary).to_json(OUT_DIR / "ode_run_summary.json", indent=2)
    print(f"wrote ODE outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
