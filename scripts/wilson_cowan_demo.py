"""Starter Wilson-Cowan bifurcation-style demo for the AMATH 383 project.

The script sweeps an excitability parameter P, finds equilibria from
multiple initial guesses, classifies local stability by Jacobian eigenvalues,
and saves a simple figure under ../outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import expit


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


def sigmoid(u: float | np.ndarray, a: float, theta: float) -> float | np.ndarray:
    return expit(a * (u - theta))


def rhs_vec(y: np.ndarray, p_drive: float, params: Params) -> np.ndarray:
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

    dfee = (-1.0 - s_e + (1.0 - e) * sp_e * params.w_ee) / params.tau_e
    dfei = ((1.0 - e) * sp_e * (-params.w_ei)) / params.tau_e
    dgie = ((1.0 - i) * sp_i * params.w_ie) / params.tau_i
    dgii = (-1.0 - s_i + (1.0 - i) * sp_i * (-params.w_ii)) / params.tau_i
    return np.asarray([[dfee, dfei], [dgie, dgii]], dtype=float)


def find_equilibria(p_drive: float, params: Params) -> list[tuple[np.ndarray, bool, float]]:
    guesses = np.array(
        [[e, i] for e in np.linspace(0.02, 0.98, 6) for i in np.linspace(0.02, 0.98, 6)]
    )
    equilibria: list[np.ndarray] = []

    for guess in guesses:
        sol = root(lambda y: rhs_vec(y, p_drive, params), guess)
        if not sol.success:
            continue
        y = np.clip(sol.x, 0.0, 1.0)
        if np.linalg.norm(rhs_vec(y, p_drive, params)) > 1e-6:
            continue
        if not any(np.linalg.norm(y - old) < 1e-4 for old in equilibria):
            equilibria.append(y)

    classified = []
    for y in sorted(equilibria, key=lambda v: (v[0], v[1])):
        eig = np.linalg.eigvals(jacobian(y, p_drive, params))
        max_real = float(np.max(np.real(eig)))
        classified.append((y, max_real < 0.0, max_real))
    return classified


def simulate(p_drive: float, params: Params, y0: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        lambda t, y: rhs_vec(y, p_drive, params),
        t_span=(0.0, 80.0),
        y0=np.asarray(y0, dtype=float),
        max_step=0.05,
        dense_output=False,
    )
    return sol.t, sol.y


def main() -> None:
    params = Params()
    p_values = np.linspace(0.0, 8.0, 161)

    stable_p, stable_e = [], []
    unstable_p, unstable_e = [], []
    max_real_by_p = []

    for p_drive in p_values:
        eqs = find_equilibria(float(p_drive), params)
        if eqs:
            max_real_by_p.append((p_drive, min(item[2] for item in eqs)))
        for y, stable, _max_real in eqs:
            if stable:
                stable_p.append(p_drive)
                stable_e.append(y[0])
            else:
                unstable_p.append(p_drive)
                unstable_e.append(y[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].scatter(stable_p, stable_e, s=10, label="stable equilibrium", color="#1f77b4")
    axes[0].scatter(unstable_p, unstable_e, s=10, label="unstable equilibrium", color="#d62728")
    axes[0].set_xlabel("excitability drive P")
    axes[0].set_ylabel("equilibrium excitation E*")
    axes[0].set_title("Equilibrium sweep")
    axes[0].legend(frameon=False)

    for p_drive, color, label in [(2.0, "#2ca02c", "low drive"), (5.5, "#9467bd", "high drive")]:
        t, y = simulate(p_drive, params, y0=(0.05, 0.05))
        axes[1].plot(t, y[0], color=color, label=f"E(t), {label}")
        axes[1].plot(t, y[1], color=color, linestyle="--", label=f"I(t), {label}")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("activity")
    axes[1].set_title("Example trajectories")
    axes[1].legend(frameon=False, fontsize=8)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wilson_cowan_bifurcation_demo.png"
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
