# Current Project Status And Next Steps

Generated on May 19, 2026.

## Completed Work

### Proposal Materials

- Created a polished proposal in Markdown: `05_project_proposal_submission.md`.
- Created a LaTeX proposal version: `05_project_proposal_submission.tex`.
- Read and incorporated the proposal rubric requirements.

### ODE Modeling

Code:

- `scripts/project_ode_analysis.py`
- `scripts/wilson_cowan_demo.py`

Outputs:

- `outputs/ode_equilibrium_sweep.csv`
- `outputs/ode_bifurcation_stability_trajectories.png`
- `outputs/ode_phase_plane.png`
- `outputs/ode_ei_model_diagram.png`
- `outputs/ode_run_summary.json`

What it currently does:

- Defines a Wilson-Cowan excitatory-inhibitory ODE model.
- Sweeps the excitability drive parameter `P`.
- Solves equilibria numerically.
- Classifies local stability using Jacobian eigenvalues.
- Simulates stable low-drive, unstable-interval, and restabilized high-drive trajectories.
- Creates a phase-plane/vector-field figure and a simple model diagram.
- Current interpretation: for the chosen parameters, the sweep tracks one equilibrium that loses local stability over approximately `P = 1.64` to `P = 2.80` on the chosen sweep grid. The actual crossings lie between neighboring grid points. The complex eigenvalues at the stability crossings suggest a Hopf-like oscillatory transition, not a saddle-node/hysteresis transition.

### CHB-MIT Data And Classifier Pipeline

Code:

- `scripts/project_chbmit_pipeline.py`
- `scripts/consistency_index.py`

Main command used:

```powershell
python final_project_seizure_dynamics\scripts\project_chbmit_pipeline.py --force-extract
```

The script was then rerun without `--force-extract` to regenerate model outputs and figures from cached features.

Scope:

- Subjects: `chb01`, `chb03`, `chb05`.
- Selected EDF records: 51.
- Window length: 5 seconds.
- Step size: 5 seconds.
- Extracted windows: 35,679.
- Phase counts:
  - baseline: 34,241
  - preictal: 1,143
  - ictal: 295

Models trained:

- Logistic regression with robust scaling and class weighting.
- Random forest with class weighting.

Model target:

- `ictal_label`, where seizure-overlapping windows are positive.

Split strategy:

- Group-aware split by EDF record to avoid placing windows from the same recording in both train and test. This is record-held-out, not subject-held-out.

Key metrics:

- Logistic regression: balanced accuracy 0.915, recall 0.875, precision 0.115, average precision 0.743, ROC AUC 0.979.
- Random forest: balanced accuracy 0.905, recall 0.833, precision 0.190, average precision 0.736, ROC AUC 0.977.

Interpretation:

- Both models detect many ictal windows, but precision remains low because the data are highly imbalanced.
- Random forest gives fewer false positives than logistic regression in this subset.
- These are working project results, not full-dataset benchmark claims.
- The false-positive-rate column is false-positive 5-second windows per non-ictal hour, not event-level clinical false alarms per hour.
- The post hoc event-level false-alarm estimate counts contiguous predicted-positive non-ictal windows within a record as one event.
- Feature medians show strong ictal separability but limited evidence of gradual preictal separation; preictal medians remain close to baseline for the reported energy features.
- Preictal positive-call rates are higher than baseline rates for the current models, but preictal windows are still labeled negative, so this is only qualitative onset-near context.
- Preictal labels are record-local in this implementation; windows in neighboring EDF records are not assigned time-to-next-seizure labels across file boundaries.

### Feature Importance And CI

Outputs:

- `outputs/chbmit_feature_importance.csv`
- `outputs/chbmit_ci_alpha_sweep.csv`
- `outputs/chbmit_ci_summary.json`
- `outputs/chbmit_top_feature_importance.png`
- `outputs/chbmit_ci_alpha_sensitivity.png`

Top permutation-importance features in the current run:

- `mean_rms`
- `mean_thetaP`
- `mean_hjorth_activity`
- `mean_alphaP`
- `mean_totalP_0p5_40`

CI result at alpha = 0.50:

- CI = 0.432
- CI_rank = 0.569
- CI_topk = 0.295

Interpretation:

- There is moderate global rank agreement but weaker top-k agreement across logistic coefficients, random-forest impurity importance, and random-forest permutation importance.
- This supports a useful final-paper discussion: explanation methods agree partly, but the exact core feature set is method-dependent.
- CI uses absolute values only for signed logistic-regression coefficients; random-forest and permutation importances are treated as nonnegative scores.

### RFECV Extension

Code:

- `scripts/project_rfecv_extension.py`

Outputs:

- `08_rfecv_extension_results.md`
- `outputs/chbmit_rfecv_feature_ranking.csv`
- `outputs/chbmit_rfecv_selected_features.txt`
- `outputs/chbmit_rfecv_model_metrics.csv`
- `outputs/chbmit_rfecv_cv_results.csv`
- `outputs/chbmit_rfecv_curve.png`
- `outputs/chbmit_rfecv_test_predictions.csv`
- `outputs/chbmit_rfecv_refit_feature_importance.csv`
- `outputs/chbmit_rfecv_selected_ci_alpha_sweep.csv`
- `outputs/chbmit_rfecv_selected_ci_summary.json`
- `outputs/chbmit_ci_full_vs_rfecv_selected.png`
- `outputs/chbmit_all_model_metrics.csv`
- `outputs/chbmit_all_model_confusion_matrices.png`
- `models/chbmit_rfecv_logistic_regression.joblib`

Result:

- RFECV selected 22 of 50 features.
- RFECV logistic regression test balanced accuracy: 0.931.
- RFECV logistic regression recall: 0.917.
- RFECV logistic regression precision: 0.100.
- RFECV logistic regression average precision: 0.730.
- RFECV logistic regression ROC AUC: 0.983.
- RFECV increased recall but also increased false-positive windows compared with the random forest.
- RFECV-refit selected-feature CI at alpha = 0.50 is 0.407, compared with 0.432 on the full feature set.
- RFECV improved balanced accuracy and recall, but feature selection did not improve explanation consistency after refitting on the selected feature subset.
- This selected-feature CI is recomputed after refitting logistic-regression and random-forest models on only the RFECV-selected feature subset.
- RFECV cross-validation now includes imputation and scaling inside the estimator pipeline.

## Figure Notes For Final Paper

- `outputs/ode_bifurcation_stability_trajectories.png`: describe the unstable interval as grid-resolution dependent.
- `outputs/ode_phase_plane.png`: explain that the red X marks an unstable equilibrium at `P = 2.00`, and streamlines illustrate oscillatory behavior near the instability.
- `outputs/chbmit_feature_trajectories.png`: features are robust-z scaled and clipped to +/-8 for visualization.
- `outputs/chbmit_all_model_confusion_matrices.png`: use the printed counts and metrics table as the main evidence because true negatives dominate the color scale.
- `outputs/chbmit_rfecv_curve.png`: note that CV balanced accuracy varies only slightly across many feature counts.

## Important Output Files

- `06_chbmit_model_results.md`: generated summary of the CHB-MIT run.
- `outputs/chbmit_features.csv`: extracted window-level feature table.
- `outputs/chbmit_selected_records.csv`: records used for this scoped run.
- `outputs/chbmit_model_metrics.csv`: classifier metrics.
- `outputs/chbmit_all_model_metrics.csv`: combined metric table for logistic regression, random forest, and RFECV logistic regression.
- `outputs/chbmit_test_predictions.csv`: record/window-level test predictions and probabilities.
- `outputs/chbmit_rfecv_test_predictions.csv`: RFECV record/window-level test predictions and probabilities.
- `outputs/chbmit_feature_trajectories.png`: median feature trajectories around seizure onset.
- `outputs/chbmit_ci_full_vs_rfecv_selected.png`: full-feature vs RFECV-refit selected-feature CI comparison.
- `08_rfecv_extension_results.md`: RFECV summary and selected feature list.
- `models/chbmit_logistic_regression.joblib`: trained logistic model.
- `models/chbmit_random_forest.joblib`: trained random forest model.
- `models/chbmit_rfecv_logistic_regression.joblib`: trained RFECV logistic model.

## Remaining Work

- Decide whether RFECV should be central in the final paper or presented as an interpretability/reliability extension.
- Improve the written final report by connecting ODE states to EEG features more explicitly.
- Add citations and figure captions to the final paper.
- Consider rerunning the pipeline on more subjects if there is time and the compute cost is acceptable.
- Clearly state that this is not a clinical seizure prediction model.
