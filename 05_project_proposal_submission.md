# Seizure Onset as a Dynamical Transition: ODE Bifurcation Modeling with Explainable ML Reliability Checks

## Abstract

Epileptic seizures can be viewed as sudden transitions in the electrical activity of the brain. This makes seizure onset a natural problem for continuous mathematical modeling: a neural system may remain near a stable normal state, then cross a parameter threshold where a high-activity seizure-like state appears or becomes stable. In this project, I will model seizure onset using a low-dimensional excitatory-inhibitory ordinary differential equation (ODE) system, analyze equilibria and local stability, and use parameter sweeps to identify transition behavior. I will then compare the qualitative predictions of this mechanistic model with EEG features extracted from the CHB-MIT scalp EEG dataset, using my existing seizure prediction codebase as the computational starting point. As a secondary component, I will adapt a Consistency Index framework from explainable machine learning to ask whether different feature-importance methods agree on the most important EEG features near seizure onset. The goal is not to build a clinical seizure predictor, but to understand how a continuous model can explain seizure-like transitions and how data-derived features can support or challenge that explanation.

## Description Of The Problem

Epilepsy is a neurological disorder in which abnormal brain activity can produce seizures. From a modeling perspective, one of the central questions is why brain activity can shift rapidly from apparently normal dynamics to a seizure state. Purely data-driven classifiers can sometimes detect seizure windows from EEG signals, but they do not by themselves explain the mathematical mechanism of the transition. A mechanistic ODE model can help answer a different kind of question: under what changes in excitation, inhibition, or external drive does a stable normal state lose stability or give way to a seizure-like state?

This question is closely related to topics in AMATH 383. The project will use systems of differential equations, nonlinear dynamics, equilibria, Jacobian matrices, eigenvalue-based stability analysis, and bifurcation-style parameter sweeps. The real-world problem is complex, but the modeling question can be made course-appropriate by simplifying brain activity to interacting neural populations. Specifically, I will model the average activity of an excitatory population and an inhibitory population. A seizure-like state will be represented mathematically as a high-excitation or high-synchrony regime rather than as a complete clinical description of a seizure.

The project will use the CHB-MIT scalp EEG database as empirical context. A local copy of the dataset is already available in my repository, with 686 EDF recordings, 141 seizure-labeled recordings, and 198 parsed seizure events. The existing codebase contains feature extraction and lightweight machine-learning components, including EEG band powers, entropy approximations, spectral features, line length, and Hjorth features. These features are not direct measurements of ODE state variables, but they can serve as observable summaries of the EEG signal. For example, band power and line length may reflect changes in signal amplitude or rhythmic activity, while spectral entropy and Hjorth complexity may reflect changes in signal organization.

The main question of the project is:

**Can a low-dimensional ODE model explain seizure onset as a stability transition, and do EEG features from seizure data show qualitative patterns consistent with that transition?**

I will also study a secondary interpretability question:

**When different machine-learning explanation methods rank EEG features, do they agree on the same core seizure-related drivers?**

This second question will use the Consistency Index idea from Chang, Chen, and Chen (2026), which combines rank agreement and top-k feature overlap. This part supports the main modeling question by checking whether data-driven feature interpretations are stable enough to be used as evidence.

## Brief Description Of Proposed Methods

The primary mathematical model will be a Wilson-Cowan style excitatory-inhibitory system. Let \(E(t)\) represent normalized excitatory population activity and \(I(t)\) represent normalized inhibitory population activity. I will use the ODE system

\[
\frac{dE}{dt} =
\frac{-E + (1-E)S_E(w_{EE}E - w_{EI}I + P)}{\tau_E},
\]

\[
\frac{dI}{dt} =
\frac{-I + (1-I)S_I(w_{IE}E - w_{II}I + Q)}{\tau_I}.
\]

Here \(S_E\) and \(S_I\) are sigmoid response functions, \(\tau_E\) and \(\tau_I\) are time constants, \(w_{EE}\), \(w_{EI}\), \(w_{IE}\), and \(w_{II}\) describe coupling strengths, and \(P\) is an excitability or external-drive parameter. The main parameter sweep will vary \(P\). Low values of \(P\) are expected to correspond to normal activity, while sufficiently high values may produce high-excitation behavior.

The mathematical analysis will proceed in four steps. First, I will solve for equilibria by finding points \((E^\*, I^\*)\) where \(dE/dt = 0\) and \(dI/dt = 0\). Second, I will compute the Jacobian matrix at each equilibrium. Third, I will classify local stability using the real parts of the Jacobian eigenvalues. Fourth, I will sweep the excitability parameter \(P\) and plot equilibrium excitation levels against \(P\). This will produce a bifurcation-style diagram showing where stable and unstable behavior changes. I will also simulate trajectories below, near, and above the transition to visualize the difference between normal and seizure-like model behavior.

The key assumptions are that population-level activity can be approximated by two continuous variables, that seizure onset can be represented as a change in stability, and that the parameter \(P\) summarizes multiple biological factors such as external input, excitability, or reduced effective inhibition. These assumptions are simplifications. The model does not include detailed neuron physiology, spatial propagation, patient-specific anatomy, medication effects, or full EEG electrode geometry. I will state these limitations clearly in the final project.

If the Wilson-Cowan model is too complex to connect cleanly to data, I will use a one-dimensional saddle-node normal form as a fallback:

\[
\frac{dx}{dt} = \mu + x^2 - cx^3,
\]

where \(x(t)\) is a seizure-activity index, \(\mu\) is a slowly varying excitability parameter, and \(c > 0\) limits growth. This fallback still allows equilibrium and stability analysis while giving a simpler threshold model.

For the empirical component, I will select a small number of CHB-MIT subjects and compare baseline, preictal, and ictal windows. I will reuse existing feature extraction code where possible. Candidate features include delta, theta, alpha, and beta band powers; spectral entropy; spectral slope; beta peak frequency; line length; RMS; and Hjorth mobility and complexity. I will plot feature trajectories around seizure onset and compare them qualitatively with the ODE model's transition indicator. I do not plan to claim that the ODE is directly fitted to every EEG channel; the data comparison will be used to test whether the model's qualitative transition story is plausible.

For the interpretability component, I will compute feature-importance rankings from feasible methods such as linear model coefficients, random-forest importance, and permutation importance. Then I will calculate

\[
CI(\alpha) = \alpha CI_{\text{rank}} + (1-\alpha)CI_{\text{topk}},
\]

where \(CI_{\text{rank}}\) is based on Spearman rank agreement and \(CI_{\text{topk}}\) is based on Jaccard overlap among top-k feature sets. A high and stable CI would suggest that different explanation methods identify similar EEG drivers. A low or highly alpha-sensitive CI would suggest that the interpretation depends strongly on the chosen explanation method.

## Potential Implications And Limitations

If successful, this project will show how seizure onset can be studied as a dynamical systems problem rather than only as a classification task. The expected final result is a set of plots and explanations showing how a change in an excitability parameter can alter stability and produce seizure-like activity. The EEG feature analysis will provide a reality check: if features such as power, line length, or entropy change near seizure onset, those changes can be discussed in relation to the ODE transition.

The main limitation is that a two-variable ODE is a strong simplification of brain dynamics. It cannot capture all mechanisms of epilepsy, and it should not be interpreted as a clinical prediction model. The project will therefore emphasize mathematical explanation, assumptions, and qualitative comparison rather than medical claims.

## Bibliography

Chang, C.-C., Chen, Y., & Chen, C.-Y. (2026). Towards decision-ready explainable machine learning for water quality management using a consistency index and cross-validated feature-selection framework. *Environmental Research, 297*, 124207. https://doi.org/10.1016/j.envres.2026.124207

Guttag, J. (2010). *CHB-MIT Scalp EEG Database*. PhysioNet. https://doi.org/10.13026/C2K01R

Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014). On the nature of seizure dynamics. *Brain, 137*(8), 2210-2230. https://doi.org/10.1093/brain/awu133

Meijer, H. G. E., Eissa, T. L., Kiewiet, B., Neuman, J. F., Schevon, C. A., Emerson, R. G., Goodman, R. R., McKhann, G. M., Marcuccilli, C. J., Tryba, A. K., Cowan, J. D., van Gils, S. A., & van Drongelen, W. (2015). Modeling focal epileptic activity in the Wilson-Cowan model with depolarization block. *Journal of Mathematical Neuroscience, 5*, 7. https://doi.org/10.1186/s13408-015-0019-4

Shoeb, A. H., & Guttag, J. V. (2010). Application of machine learning to epileptic seizure detection. In *Proceedings of the 27th International Conference on Machine Learning*. https://physionet.org/physiobank/database/chbmit/shoeb-icml-2010.pdf

