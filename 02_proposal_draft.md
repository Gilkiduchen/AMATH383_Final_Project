# Superseded Proposal Draft

This is an earlier working draft retained for project history. Use `05_project_proposal_submission.md` or `05_project_proposal_submission.tex` for the current proposal. After initial results, the generated classifier should be described as ictal-window detection, not seizure prediction. The current Wilson-Cowan parameter sweep shows a Hopf-like loss of local stability over a finite excitability interval, not a separate stable high-activity seizure equilibrium.

## Title

**Seizure Onset as a Dynamical Transition: ODE Bifurcation Modeling with Explainable ML Reliability Checks**

## Problem And Motivation

Epileptic seizures are sudden transitions in brain electrical activity. From a mathematical modeling perspective, this makes seizure onset a natural candidate for nonlinear dynamical systems analysis: a brain state may remain stable under normal conditions, then cross a threshold where a seizure-like state becomes stable or unavoidable. This project studies seizure onset as a dynamical transition using a low-dimensional ODE model and compares the model's transition behavior with EEG features from the CHB-MIT scalp EEG dataset.

The project also connects to my existing EEG preprocessing and classification codebase. That code extracts EEG window features and trains lightweight classifiers. However, a classifier alone does not explain why a transition happens. The goal here is to combine a mechanistic ODE model with data-driven feature analysis: the ODE gives a mathematical explanation of transition behavior, while the ML pipeline checks whether extracted EEG features and feature explanations are consistent with the model.

## Research Questions

1. Can a low-dimensional excitatory-inhibitory ODE model produce a seizure-like transition through a change in an excitability parameter?
2. What do equilibria, Jacobian eigenvalues, and parameter sweeps reveal about the stability boundary between normal and seizure-like activity?
3. Do CHB-MIT EEG features such as band power, line length, spectral slope, entropy, and Hjorth features change in a way that is consistent with approaching a transition?
4. Are the data-driven explanations of ictal-window detection stable across feature-selection and explanation methods, as measured by a Consistency Index?

## Simplifications And Assumptions

The brain is far too complex to model at the cellular level in this course project. I will simplify it to a neural population model with one excitatory activity variable `E(t)` and one inhibitory activity variable `I(t)`. A seizure-like state will be interpreted as a high-excitation or high-synchrony regime, not as a full clinical seizure description. EEG features will be treated as indirect observables of the underlying neural state. The analysis will focus on qualitative transition behavior rather than individualized clinical prediction.

If the two-variable model is too difficult to fit directly to data, I will use it for mechanistic analysis and use a simpler one-dimensional saddle-node normal form as a fallback model for the transition indicator.

## Mathematical Model

The main model will be a Wilson-Cowan style excitatory-inhibitory system:

```text
dE/dt = [-E + (1 - E) S_E(w_EE E - w_EI I + P)] / tau_E
dI/dt = [-I + (1 - I) S_I(w_IE E - w_II I + Q)] / tau_I
```

Here `E(t)` and `I(t)` represent normalized excitatory and inhibitory population activity. The functions `S_E` and `S_I` are sigmoid response functions. Parameters such as `P`, `Q`, and the coupling weights control the balance between excitation and inhibition. The main bifurcation-style experiment will vary `P`, interpreted as external drive or excitability, and track equilibria and eigenvalues.

The mathematical work will include:

- Finding equilibria by solving `dE/dt = 0` and `dI/dt = 0`.
- Computing the Jacobian matrix at equilibria.
- Classifying local stability from eigenvalues.
- Sweeping the excitability parameter to identify threshold-like transitions.
- Simulating trajectories below, near, and above the transition.

## Data And Computation Plan

The empirical component will use the local CHB-MIT dataset in `data/chbmit/1.0.0`. The local copy contains 686 EDF recordings and 141 seizure-labeled EDF records, with 198 parsed seizure events. The sampling rate is 256 Hz.

I will reuse the existing feature pipeline where possible:

- 8-feature compact representation from `src/data/extractFeature_pd.py`.
- Expanded 25-feature bank from `src/data/extractFeature_pd_bank.py`.
- Candidate features include band powers, relative band powers, spectral entropy, spectral slope, beta peak frequency, RMS, line length, zero-crossing rate, and Hjorth features.

Windows will be labeled by their time relation to seizure intervals. The simplest plan is to compare feature trajectories in baseline, preictal, and ictal windows for a small subset of CHB-MIT subjects. The project will not depend on training a perfect classifier; the important goal is to connect observed feature changes to the ODE transition picture.

## Interpretability Reliability Plan

I will adapt the CI/RFECV framework from Chang, Chen, and Chen (2026). For multiple feature-importance outputs, the Consistency Index is:

```text
CI(alpha) = alpha * CI_rank + (1 - alpha) * CI_topk
```

where `CI_rank` measures Spearman rank agreement between explanation methods and `CI_topk` measures Jaccard overlap among the top-k features. I will compute CI across methods such as linear SVM weights, random-forest importance, and permutation importance, depending on what is feasible in the repo. RFECV or repeated feature selection will be used to check whether a compact feature set keeps the same core drivers.

## Expected Results

I expect the ODE model to show a threshold-like transition as excitability increases. Near the transition, the model should show loss of local stability and seizure-like oscillatory behavior. In the EEG features, I expect seizure-near windows to show changes in features related to spectral power, signal complexity, and line length. I also expect that CI may show only partial agreement across explanation methods, meaning the exact set of important seizure-related drivers can be method-dependent.

## Risks And Fallbacks

If direct fitting of ODE parameters to EEG features is weak, I will present the ODE as a mechanistic model and use data only for qualitative comparison. If the Wilson-Cowan model is too complicated for the final timeline, I will analyze a one-dimensional saddle-node normal form:

```text
dx/dt = mu + x^2 - c x^3
```

where `x(t)` is a seizure activity index and `mu` is a slowly varying excitability parameter. This fallback still supports equilibrium, stability, and bifurcation analysis.

## Timeline

- May 14-16, 2026: finalize model choice, proposal draft, and source list.
- May 17-18, 2026: finish and submit the 2-page proposal. The guideline says "Tuesday May 18th," but May 18, 2026 is Monday, so I will confirm the exact Canvas deadline.
- May 19-24, 2026: run ODE simulations, equilibrium/eigenvalue sweeps, and create first figures.
- May 25-29, 2026: extract selected CHB-MIT feature trajectories and compute preliminary feature-importance/CI results.
- May 30-June 4, 2026: write final paper or prepare presentation slides.
- June 5, 2026: final paper/slides due according to the course schedule.
- June 8, 2026: discussion slot if choosing the paper option.

## References

- Jirsa, V. K., et al. (2014). "On the nature of seizure dynamics." Brain. https://doi.org/10.1093/brain/awu133
- Meijer, H. G. E., et al. (2015). "Modeling focal epileptic activity in the Wilson-Cowan model with depolarization block." Journal of Mathematical Neuroscience. https://doi.org/10.1186/s13408-015-0019-4
- Negahbani, E., et al. (2015). "Noise-induced precursors of state transitions in the stochastic Wilson-Cowan model." Journal of Mathematical Neuroscience. https://doi.org/10.1186/s13408-015-0021-x
- Guttag, J. (2010). CHB-MIT Scalp EEG Database, PhysioNet. https://doi.org/10.13026/C2K01R
- Shoeb, A. H., and Guttag, J. V. (2010). "Application of machine learning to epileptic seizure detection." ICML. https://physionet.org/physiobank/database/chbmit/shoeb-icml-2010.pdf
- Chang, C.-C., Chen, Y., and Chen, C.-Y. (2026). "Towards decision-ready explainable machine learning for water quality management using a consistency index and cross-validated feature-selection framework." Environmental Research. https://doi.org/10.1016/j.envres.2026.124207
