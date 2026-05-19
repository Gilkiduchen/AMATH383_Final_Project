# Project Task List

## Phase 1: Scope And Approval

- [ ] Confirm with instructor that this self-nominated topic is approved.
- [ ] Decide whether the final deliverable will be a paper or presentation.
- [ ] Confirm the proposal deadline because the guideline says "Tuesday May 18th," while May 18, 2026 is Monday.
- [x] Finalize one primary model: Wilson-Cowan ODE.
- [x] Keep one fallback model: one-dimensional saddle-node normal form.

## Phase 2: Literature And Proposal

- [x] Read final project guidelines.
- [x] Identify main mechanistic seizure dynamics references.
- [x] Identify how the uploaded CI/RFECV paper transfers to EEG interpretation.
- [x] Inspect local EEG preprocessing/classification repo and available features.
- [x] Convert `02_proposal_draft.md` into the final 2-page submission.
- [ ] Ask instructor whether the proposal should emphasize paper option or presentation option.

## Phase 3: ODE Model

- [x] Define final variables, parameters, units/scales, and assumptions.
- [x] Write the Wilson-Cowan equations cleanly in LaTeX.
- [x] Derive the Jacobian matrix.
- [x] Compute equilibria numerically for a grid of excitability values.
- [x] Classify equilibria using eigenvalues.
- [x] Generate a bifurcation-style figure: excitability parameter vs equilibrium excitation.
- [x] Simulate time traces below, near, and above transition.
- [x] Interpret the transition in terms of excitation-inhibition balance.

## Phase 4: CHB-MIT Feature Link

- [x] Select a small, feasible subject subset: chb01, chb03, and chb05.
- [x] Define baseline, preictal, and ictal windows.
- [x] Reuse existing feature extraction where possible.
- [x] Plot feature trajectories around seizure onset.
- [x] Compare features to model variables or transition indicators.
- [x] Keep claims qualitative unless fitting is reliable.

## Phase 5: ML And CI/RFECV Reliability

- [x] Train or reuse a lightweight classifier such as linear SVM, logistic regression, random forest, or MLP.
- [x] Use group-aware splitting to avoid leakage.
- [x] Compute at least two feature-importance rankings:
  - Linear model weights or coefficients.
  - Random forest importance.
  - Permutation importance.
- [x] Optionally add RFECV or repeated top-k feature selection.
- [x] Compute `CI_rank`, `CI_topk`, and `CI(alpha)`.
- [x] Plot CI sensitivity over `alpha`.
- [x] Interpret whether the same core EEG features appear across methods.

## Phase 6: Final Figures

- [x] Model diagram: excitatory-inhibitory feedback loop.
- [x] Phase-plane or vector-field plot.
- [x] Bifurcation-style equilibrium/eigenvalue plot.
- [x] Example ODE time traces.
- [x] EEG feature trajectories around onset.
- [x] Classifier metric table, if used.
- [x] CI alpha-sensitivity curve.

## Phase 7: Paper Or Presentation

- [ ] Write abstract.
- [ ] Write problem description and motivation.
- [ ] Write simplifications and assumptions.
- [ ] Write mathematical model section.
- [ ] Write solution/stability/bifurcation section.
- [ ] Write data and interpretability reliability section.
- [ ] Write results and discussion.
- [ ] Write limitations and possible improvements.
- [ ] Add references.
- [ ] Check figure captions and notation.
