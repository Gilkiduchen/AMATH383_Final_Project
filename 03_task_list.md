# Project Task List

## Phase 1: Scope And Approval

- [ ] Confirm with instructor that this self-nominated topic is approved.
- [ ] Decide whether the final deliverable will be a paper or presentation.
- [ ] Confirm the proposal deadline because the guideline says "Tuesday May 18th," while May 18, 2026 is Monday.
- [ ] Finalize one primary model: Wilson-Cowan ODE.
- [ ] Keep one fallback model: one-dimensional saddle-node normal form.

## Phase 2: Literature And Proposal

- [x] Read final project guidelines.
- [x] Identify main mechanistic seizure dynamics references.
- [x] Identify how the uploaded CI/RFECV paper transfers to EEG interpretation.
- [x] Inspect local seizure prediction repo and available features.
- [ ] Convert `02_proposal_draft.md` into the final 2-page submission.
- [ ] Ask instructor whether the proposal should emphasize paper option or presentation option.

## Phase 3: ODE Model

- [ ] Define final variables, parameters, units/scales, and assumptions.
- [ ] Write the Wilson-Cowan equations cleanly in LaTeX.
- [ ] Derive the Jacobian matrix.
- [ ] Compute equilibria numerically for a grid of excitability values.
- [ ] Classify equilibria using eigenvalues.
- [ ] Generate a bifurcation-style figure: excitability parameter vs equilibrium excitation.
- [ ] Simulate time traces below, near, and above transition.
- [ ] Interpret the transition in terms of excitation-inhibition balance.

## Phase 4: CHB-MIT Feature Link

- [ ] Select a small, feasible subject subset, such as chb01 and chb03.
- [ ] Define baseline, preictal, and ictal windows.
- [ ] Reuse existing feature extraction where possible.
- [ ] Plot feature trajectories around seizure onset.
- [ ] Compare features to model variables or transition indicators.
- [ ] Keep claims qualitative unless fitting is reliable.

## Phase 5: ML And CI/RFECV Reliability

- [ ] Train or reuse a lightweight classifier such as linear SVM, logistic regression, random forest, or MLP.
- [ ] Use group-aware splitting to avoid leakage.
- [ ] Compute at least two feature-importance rankings:
  - Linear model weights or coefficients.
  - Random forest importance.
  - Permutation importance.
- [ ] Optionally add RFECV or repeated top-k feature selection.
- [ ] Compute `CI_rank`, `CI_topk`, and `CI(alpha)`.
- [ ] Plot CI sensitivity over `alpha`.
- [ ] Interpret whether the same core EEG features appear across methods.

## Phase 6: Final Figures

- [ ] Model diagram: excitatory-inhibitory feedback loop.
- [ ] Phase-plane or vector-field plot.
- [ ] Bifurcation-style equilibrium/eigenvalue plot.
- [ ] Example ODE time traces.
- [ ] EEG feature trajectories around onset.
- [ ] Classifier metric table, if used.
- [ ] CI alpha-sensitivity curve.

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

