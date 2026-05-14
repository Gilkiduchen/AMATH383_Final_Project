# Research Brief

## Project Choice

Recommended title:

**Seizure Onset as a Dynamical Transition: ODE Bifurcation Modeling with Explainable ML Reliability Checks**

Main idea:

Use a low-dimensional neural population ODE to model transition from normal/preictal activity to seizure-like activity, then compare the model's transition indicators with features and classifier outputs from the existing CHB-MIT EEG pipeline. Use the Consistency Index (CI) and RFECV idea from the uploaded Environmental Research paper to test whether feature explanations are stable enough to support interpretation.

## Course Fit

From `d:\Download\Final Project Guidelines(1).docx`, the project should:

- Be personally interesting.
- Use mathematical techniques from AMATH 383, especially mechanistic modeling.
- Focus on modeling rather than treating data collection as the main effort.
- Include references from journals/books.
- For the paper/presentation, cover problem description, simplifications, mathematical model, solution, results/discussion, improvements, conclusions, and references.

This topic fits because the core work is ODE modeling, equilibria, Jacobians, stability, parameter sweeps, and bifurcation-style interpretation. The data and ML parts support the model rather than replacing it.

## Local Repo Inventory

The current repo already provides useful project material:

- Dataset: local CHB-MIT copy at `data/chbmit/1.0.0`.
- Local dataset summary:
  - 24 summary files.
  - 686 EDF files.
  - 141 seizure-labeled EDF records.
  - 198 parsed seizure events.
  - About 193.52 minutes of seizure duration.
  - CHB-MIT summaries report 256 Hz sampling and common 23-channel scalp EEG montages.
- Existing CHB-MIT-style training direction: `train.py` uses SVM-style classification and group-aware splitting.
- Existing feature extraction:
  - `src/data/extractFeature_pd.py`: 8 compact features: sample entropy approximation, fuzzy entropy approximation, skew, kurtosis, delta/theta/alpha/beta band powers.
  - `src/data/extractFeature_pd_bank.py`: 25-feature bank including band powers, relative band powers, band ratios, spectral entropy, spectral slope, beta peak frequency, RMS, line length, zero-crossing rate, and Hjorth features.
- Existing feature selection:
  - `select_features_rf_pd.py` provides a repeated random-forest feature-selection workflow with group splitting, top-k selection, and stable feature frequency.
  - Existing top-k output keeps: `hjorth_complexity`, `spec_slope_4_40`, `betaP`, `hjorth_mobility`, `sampleEn_approx`, `beta_peak_freq`, `totalP_0p5_40`, `line_length`.

These features are not direct ODE state variables, but they can be used as observable proxies for excitability, synchronization, spectral concentration, and signal complexity.

## Literature Takeaways

1. Seizure onset can be modeled as a bifurcation.

Jirsa et al. introduced the Epileptor and argued that seizure onset and offset can be understood as mathematical events, with the common class involving saddle-node onset and homoclinic offset. This directly supports using stability and bifurcation analysis as the AMATH core.

2. A low-dimensional model is defensible.

The Epileptor uses a small set of state variables across multiple time scales. For a course project, a reduced Wilson-Cowan excitatory-inhibitory system or a one-dimensional saddle-node normal form is a reasonable simplification if the assumptions are clearly stated.

3. Wilson-Cowan gives a course-friendly ODE.

Wilson-Cowan models excitatory and inhibitory population activity with coupled nonlinear differential equations. Epilepsy-specific Wilson-Cowan work shows that bifurcation analysis can produce high-activity seizure-like equilibria and propagation behavior.

4. Critical slowing can connect ODE theory to EEG features.

Near bifurcation, theory predicts changes in variance, autocorrelation, and spectral behavior. This is useful because the repo already computes signal features such as line length, spectral slope, band power, entropy, and Hjorth mobility/complexity.

5. The CI/RFECV paper provides a transferable interpretability method.

The uploaded Environmental Research paper defines a CI combining rank agreement and top-k overlap:

`CI(alpha) = alpha * CI_rank + (1 - alpha) * CI_topk`

where `CI_rank` is based on Spearman rank agreement and `CI_topk` is based on Jaccard overlap of top-k features. That method can be transferred from water quality management to EEG seizure modeling by replacing water-quality variables with EEG features.

6. The uploaded Water Research X paper is less central but useful as a modeling precedent.

The ammonia-nitrogen paper shows a practical ML early-warning workflow on noisy real-world process data. It is useful background for prediction framing, but the CI/RFECV paper is the more important reference for this AMATH project.

## Proposed Research Questions

1. Can a two-variable excitatory-inhibitory ODE reproduce a qualitative transition from stable normal activity to seizure-like high activity?
2. Which ODE parameters create the transition, and what do the equilibria/eigenvalues show near the transition?
3. Do EEG features from the local pipeline show trends consistent with the ODE transition indicator near seizure onset?
4. Are the ML feature explanations stable across methods and feature-selection choices, as measured by CI(alpha)?

## Working Contribution

The final project does not need to claim a medically valid seizure prediction system. The defensible contribution is:

- Formulate a biologically interpretable ODE model.
- Analyze stability and bifurcation behavior.
- Use CHB-MIT windows as empirical context.
- Use CI/RFECV to discuss whether data-driven feature importance is reliable.

## Key References

- Jirsa, V. K., et al. (2014). "On the nature of seizure dynamics." Brain. https://doi.org/10.1093/brain/awu133
- Saggio, M. L., and Jirsa, V. (2024). "Bifurcations and bursting in the Epileptor." PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1011903
- Chizhov, A. V., et al. (2018). "Minimal model of interictal and ictal discharges 'Epileptor-2'." PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1006186
- Meijer, H. G. E., et al. (2015). "Modeling focal epileptic activity in the Wilson-Cowan model with depolarization block." Journal of Mathematical Neuroscience. https://doi.org/10.1186/s13408-015-0019-4
- Negahbani, E., et al. (2015). "Noise-induced precursors of state transitions in the stochastic Wilson-Cowan model." Journal of Mathematical Neuroscience. https://doi.org/10.1186/s13408-015-0021-x
- Guttag, J. (2010). CHB-MIT Scalp EEG Database, PhysioNet. https://doi.org/10.13026/C2K01R
- Shoeb, A. H., and Guttag, J. V. (2010). "Application of machine learning to epileptic seizure detection." ICML. https://physionet.org/physiobank/database/chbmit/shoeb-icml-2010.pdf
- Chang, C.-C., Chen, Y., and Chen, C.-Y. (2026). "Towards decision-ready explainable machine learning for water quality management using a consistency index and cross-validated feature-selection framework." Environmental Research. https://doi.org/10.1016/j.envres.2026.124207
- Tang, C. Y. J., Chen, Y., Wu, P.-H., Chang, C.-C., and Yu, C.-P. (2025). "Enhanced prediction of ammonia nitrogen levels in reverse osmosis brine from a full-scale water reclamation plant using machine learning." Water Research X. https://doi.org/10.1016/j.wroa.2025.100384

