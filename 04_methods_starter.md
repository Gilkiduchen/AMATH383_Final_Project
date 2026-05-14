# Methods Starter

## Model 1: Wilson-Cowan Excitatory-Inhibitory ODE

Use normalized population activities:

- `E(t)`: excitatory activity, between 0 and 1.
- `I(t)`: inhibitory activity, between 0 and 1.

Equations:

```text
dE/dt = F(E, I) = [-E + (1 - E) S_E(u_E)] / tau_E
dI/dt = G(E, I) = [-I + (1 - I) S_I(u_I)] / tau_I

u_E = w_EE E - w_EI I + P
u_I = w_IE E - w_II I + Q
S_X(u) = 1 / (1 + exp[-a_X (u - theta_X)])
```

Interpretation:

- Larger `P` means higher excitatory drive or excitability.
- Larger `w_EI` means stronger inhibitory control of excitation.
- A seizure-like state corresponds to high `E`, high model output, or loss of stability of a low-activity equilibrium.

## Equilibrium Conditions

Equilibria satisfy:

```text
F(E*, I*) = 0
G(E*, I*) = 0
```

These will usually be solved numerically for different values of `P`.

## Jacobian

Let:

```text
S_E' = a_E S_E(u_E) [1 - S_E(u_E)]
S_I' = a_I S_I(u_I) [1 - S_I(u_I)]
```

Then:

```text
dF/dE = [-1 - S_E + (1 - E) S_E' w_EE] / tau_E
dF/dI = [(1 - E) S_E' (-w_EI)] / tau_E

dG/dE = [(1 - I) S_I' w_IE] / tau_I
dG/dI = [-1 - S_I + (1 - I) S_I' (-w_II)] / tau_I
```

At each equilibrium, compute eigenvalues of:

```text
J = [[dF/dE, dF/dI],
     [dG/dE, dG/dI]]
```

If all eigenvalues have negative real parts, the equilibrium is locally stable. If at least one eigenvalue has positive real part, it is unstable. A transition can occur when the largest real part crosses zero or when stable/unstable equilibria collide.

## Model 2: Saddle-Node Normal Form Fallback

If the Wilson-Cowan model is too detailed for fitting, use:

```text
dx/dt = mu + x^2 - c x^3
```

where:

- `x(t)` is a scalar seizure activity index.
- `mu` is a slowly changing excitability parameter.
- `c > 0` prevents unbounded growth in numerical simulations.

The simpler form still allows equilibrium and stability analysis. It also makes it easier to explain the threshold idea.

## Data-To-Model Mapping

EEG features are not the same as ODE state variables. Treat them as observables:

- Excitability proxies: beta power, total power, line length, RMS.
- Synchrony/complexity proxies: sample entropy, fuzzy entropy, spectral entropy, Hjorth complexity.
- Frequency-shift proxies: spectral slope, beta peak frequency, band ratios.

Possible empirical transition index:

```text
R(t) = zscore(line_length) + zscore(betaP) - zscore(spec_entropy)
```

This is only a heuristic. It should be presented as a visualization tool, not a clinical risk score.

## Consistency Index

For explanation methods `m = 1, ..., M`, compute a feature importance ranking and top-k feature set for each method.

Rank agreement:

```text
CI_rank = mean_pairwise normalized Spearman rho
normalized rho = (rho + 1) / 2
```

Top-k agreement:

```text
CI_topk = mean_pairwise Jaccard(top_k_i, top_k_j)
```

Combined index:

```text
CI(alpha) = alpha * CI_rank + (1 - alpha) * CI_topk
```

Interpretation:

- High and alpha-stable CI: explanation methods mostly agree.
- Low or alpha-sensitive CI: feature importance depends strongly on the explanation method.
- High top-k but lower rank CI: methods agree on core drivers but not full ordering.

