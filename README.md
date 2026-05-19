# AMATH 383 Seizure Dynamics Final Project

Proposed topic: **Seizure onset as a dynamical transition: ODE bifurcation modeling with explainable ML reliability checks**.

This folder collects the working materials for the course final project. The project is designed so the main AMATH 383 contribution is mechanistic continuous modeling, while the CHB-MIT feature pipeline provides qualitative empirical context and the CI/RFECV analysis checks interpretability reliability.

## Files

- `01_research_brief.md`: literature, local-repo inventory, and why the topic fits the course.
- `02_proposal_draft.md`: starter 2-page proposal text.
- `03_task_list.md`: concrete checklist from topic approval to final paper or presentation.
- `04_methods_starter.md`: ODE equations, stability plan, CI formula, and data-to-model mapping.
- `05_project_proposal_submission.md`: polished 2-4 page proposal aligned with the assignment prompt and rubric.
- `05_project_proposal_submission.tex`: LaTeX version of the polished proposal.
- `06_chbmit_model_results.md`: generated results summary from the CHB-MIT subset run.
- `07_current_status_and_next_steps.md`: current work log and next-step guide.
- `08_rfecv_extension_results.md`: RFECV feature-selection extension results.
- `requirements_project.txt`: package versions used for the current generated outputs.
- `scripts/project_ode_analysis.py`: full ODE analysis script for equilibrium, stability, phase-plane, and diagram outputs.
- `scripts/project_chbmit_pipeline.py`: self-contained CHB-MIT EDF reader, feature extraction, classifier training, feature importance, and CI pipeline.
- `scripts/project_rfecv_extension.py`: RFECV feature-selection extension using cached CHB-MIT features.
- `scripts/wilson_cowan_demo.py`: small ODE/bifurcation starter script.
- `scripts/consistency_index.py`: reusable CI computation helper.

## Date Note

The guideline file says the 2-page proposal is due "Tuesday May 18th." In the 2026 calendar, May 18, 2026 is Monday. The safest action is to confirm the exact Canvas deadline with the instructor, but treat **May 18, 2026** as the working deadline.

## Reproducibility Note

The project folder contains cached features and generated outputs, but not the raw CHB-MIT EDF files. To rerun feature extraction, place the raw CHB-MIT dataset at `data/chbmit/1.0.0` relative to the repository root, then run:

```powershell
python final_project_seizure_dynamics\scripts\project_chbmit_pipeline.py --force-extract
python final_project_seizure_dynamics\scripts\project_rfecv_extension.py
```

If only reviewing the project logic and results, the cached `outputs/chbmit_features.csv` and result CSVs are sufficient.

Cached features avoid EDF extraction, but rerunning the pipeline still retrains models and recomputes feature importance, so it may take several minutes.

`outputs/chbmit_run_config.json` stores the local absolute raw-data path only as diagnostic metadata; cache validation ignores that absolute path so cached-feature review can work on another machine. The selected-record CSV itself stores EDF paths relative to `data/chbmit/1.0.0`.

## Interpretation Corrections

The current classifier is an **ictal-window detector**, not a seizure prediction or early-warning model. Preictal windows are included as descriptive context and are treated as negative examples in the classifier.

Preictal labels are record-local in this implementation; windows in neighboring EDF records are not assigned time-to-next-seizure labels across file boundaries.

For the current Wilson-Cowan parameters, the ODE sweep shows one equilibrium that loses local stability over a finite excitability interval. The tracked equilibrium is unstable over approximately `P = 1.64` to `P = 2.80` on the chosen sweep grid, with actual crossings lying between neighboring grid points. The stability crossings are Hopf-like because the relevant eigenvalues are complex; the current generated ODE result should not be described as a saddle-node/hysteresis transition or as emergence of a separate stable seizure equilibrium.
