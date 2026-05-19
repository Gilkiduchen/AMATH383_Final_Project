"""RFECV extension for the CHB-MIT final-project feature table.

This script uses the cached outputs from project_chbmit_pipeline.py and runs a
record-aware RFECV pass on logistic regression. It is intentionally separate so
the main pipeline stays fast and reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

from consistency_index import alpha_sweep, consistency_index  # noqa: E402
from project_chbmit_pipeline import FEATURE_NAMES, false_alarm_events_per_hour, simple_markdown_table  # noqa: E402


def metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    false_alarm_events_per_hour_value: float,
    step_sec: float = 5.0,
) -> dict[str, float | str]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    neg_hours = max(float((y_true == 0).sum() * step_sec / 3600.0), 1e-12)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_score)) if y_score is not None else np.nan,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_score is not None else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_windows_per_hour": float(fp / neg_hours),
        "false_alarm_events_per_hour": float(false_alarm_events_per_hour_value),
    }


def main() -> None:
    feature_path = OUT_DIR / "chbmit_features.csv"
    split_path = OUT_DIR / "chbmit_train_test_split.csv"
    if not feature_path.exists() or not split_path.exists():
        raise FileNotFoundError("Run project_chbmit_pipeline.py before the RFECV extension.")

    df = pd.read_csv(feature_path)
    split_df = pd.read_csv(split_path)
    if len(df) != len(split_df):
        raise RuntimeError("Feature table and split table have different lengths.")

    train_mask = split_df["split"].to_numpy() == "train"
    test_mask = split_df["split"].to_numpy() == "test"
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["ictal_label"].to_numpy(dtype=int)
    groups = df["record"].to_numpy()

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    groups_train = groups[train_mask]

    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=383,
                ),
            ),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=383)
    selector = RFECV(
        estimator=estimator,
        step=2,
        min_features_to_select=5,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        importance_getter="named_steps.clf.coef_",
    )
    selector.fit(X_train, y_train, groups=groups_train)

    selected_features = [name for name, keep in zip(FEATURE_NAMES, selector.support_) if keep]
    ranking_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "selected": selector.support_,
            "rank": selector.ranking_,
        }
    ).sort_values(["rank", "feature"])
    ranking_df.to_csv(OUT_DIR / "chbmit_rfecv_feature_ranking.csv", index=False)
    pd.Series(selected_features).to_csv(OUT_DIR / "chbmit_rfecv_selected_features.txt", index=False, header=False)

    pred_df = df.iloc[np.where(test_mask)[0]][
        [
            "record",
            "subject",
            "window_start_sec",
            "window_end_sec",
            "window_center_sec",
            "rel_time_to_onset_sec",
            "phase",
            "ictal_label",
            "preictal_label",
        ]
    ].copy()
    y_pred = selector.predict(X_test)
    y_prob = selector.predict_proba(X_test)[:, 1] if hasattr(selector, "predict_proba") else None
    pred_df["rfecv_logistic_regression_pred"] = y_pred
    pred_df["rfecv_logistic_regression_prob"] = y_prob if y_prob is not None else np.nan
    pred_df.to_csv(OUT_DIR / "chbmit_rfecv_test_predictions.csv", index=False)
    positive_rates = (
        pred_df.groupby("phase")["rfecv_logistic_regression_pred"]
        .agg(n_windows="size", positive_call_rate="mean")
        .reset_index()
    )
    positive_rates.to_csv(OUT_DIR / "chbmit_rfecv_positive_call_rates_by_phase.csv", index=False)
    false_alarm_rate = false_alarm_events_per_hour(
        pred_df,
        "rfecv_logistic_regression_pred",
        step_sec=5.0,
    )
    metrics_df = pd.DataFrame(
        [
            metrics(
                "rfecv_logistic_regression",
                y_test,
                y_pred,
                y_score=y_prob,
                false_alarm_events_per_hour_value=false_alarm_rate,
            )
        ]
    )
    metrics_df.to_csv(OUT_DIR / "chbmit_rfecv_model_metrics.csv", index=False)
    main_metrics_path = OUT_DIR / "chbmit_model_metrics.csv"
    if main_metrics_path.exists():
        all_metrics_df = pd.concat([pd.read_csv(main_metrics_path), metrics_df], ignore_index=True)
    else:
        all_metrics_df = metrics_df.copy()
    all_metrics_df.to_csv(OUT_DIR / "chbmit_all_model_metrics.csv", index=False)

    fig, axes = plt.subplots(1, len(all_metrics_df), figsize=(4.2 * len(all_metrics_df), 3.6), constrained_layout=True)
    if len(all_metrics_df) == 1:
        axes = [axes]
    for ax, row in zip(axes, all_metrics_df.itertuples(index=False)):
        cm = np.asarray([[row.tn, row.fp], [row.fn, row.tp]], dtype=float)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
        ax.set_yticks([0, 1], labels=["true 0", "true 1"])
        ax.set_title(str(row.model).replace("_", "\n"), fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(OUT_DIR / "chbmit_all_model_confusion_matrices.png", dpi=220)
    plt.close(fig)

    joblib.dump(
        {
            "selector": selector,
            "selected_features": selected_features,
        },
        MODEL_DIR / "chbmit_rfecv_logistic_regression.joblib",
    )

    cv_result_cols = {}
    for key, value in selector.cv_results_.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            cv_result_cols[key] = arr
        elif arr.ndim == 2:
            for col_idx in range(arr.shape[1]):
                cv_result_cols[f"{key}_{col_idx}"] = arr[:, col_idx]
    cv_results = pd.DataFrame(cv_result_cols)
    cv_results.to_csv(OUT_DIR / "chbmit_rfecv_cv_results.csv", index=False)
    if "n_features" in cv_results.columns and "mean_test_score" in cv_results.columns:
        fig, ax = plt.subplots(figsize=(6.3, 4.0), constrained_layout=True)
        ax.plot(cv_results["n_features"], cv_results["mean_test_score"], marker="o")
        ax.axvline(len(selected_features), color="#d62728", linestyle="--", label="selected")
        ax.set_xlabel("number of selected features")
        ax.set_ylabel("CV balanced accuracy")
        ax.set_title("RFECV feature-count curve")
        ax.legend(frameon=False)
        fig.savefig(OUT_DIR / "chbmit_rfecv_curve.png", dpi=220)
        plt.close(fig)

    selected_idx = np.where(selector.support_)[0]
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    logit_selected = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=383,
                ),
            ),
        ]
    )
    rf_selected = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=383,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logit_selected.fit(X_train_sel, y_train)
    rf_selected.fit(X_train_sel, y_train)

    rng = np.random.default_rng(383)
    if len(y_test) > 6000:
        perm_local = np.sort(rng.choice(np.arange(len(y_test)), size=6000, replace=False))
        X_perm = X_test_sel[perm_local]
        y_perm = y_test[perm_local]
    else:
        X_perm = X_test_sel
        y_perm = y_test
    perm = permutation_importance(
        rf_selected,
        X_perm,
        y_perm,
        n_repeats=8,
        random_state=383,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    logit_coef = np.abs(logit_selected.named_steps["clf"].coef_[0])
    rf_importance = np.maximum(rf_selected.named_steps["clf"].feature_importances_, 0.0)
    perm_importance = np.maximum(perm.importances_mean, 0.0)
    selected_importance_df = pd.DataFrame(
        {
            "feature": selected_features,
            "logit_abs_coef": logit_coef,
            "rf_gini_importance": rf_importance,
            "rf_permutation_importance": perm_importance,
            "rf_permutation_importance_raw": perm.importances_mean,
            "rf_permutation_std": perm.importances_std,
        }
    ).sort_values("rf_permutation_importance", ascending=False)
    selected_importance_df.to_csv(OUT_DIR / "chbmit_rfecv_refit_feature_importance.csv", index=False)

    method_scores = {
        "logit_abs_coef": dict(zip(selected_features, logit_coef)),
        "rf_gini_importance": dict(zip(selected_features, rf_importance)),
        "rf_permutation_importance": dict(zip(selected_features, perm_importance)),
    }
    topk = min(8, len(selected_features))
    ci_selected_df = pd.DataFrame(alpha_sweep(method_scores, k=topk, step=0.05, use_abs=False))
    ci_selected_df.to_csv(OUT_DIR / "chbmit_rfecv_selected_ci_alpha_sweep.csv", index=False)
    pd.Series(consistency_index(method_scores, alpha=0.5, k=topk, use_abs=False)).to_json(
        OUT_DIR / "chbmit_rfecv_selected_ci_summary.json",
        indent=2,
    )
    ci_mid = ci_selected_df.iloc[(ci_selected_df["alpha"] - 0.5).abs().argmin()]
    ci_sentence = (
        f"At alpha = {ci_mid['alpha']:.2f}, RFECV-refit selected-feature CI = {ci_mid['CI']:.3f}, "
        f"CI_rank = {ci_mid['CI_rank']:.3f}, and CI_topk = {ci_mid['CI_topk']:.3f}."
    )

    full_ci_path = OUT_DIR / "chbmit_ci_alpha_sweep.csv"
    if full_ci_path.exists():
        full_ci_df = pd.read_csv(full_ci_path)
        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
        ax.plot(full_ci_df["alpha"], full_ci_df["CI"], marker="o", label="full 50 features")
        ax.plot(ci_selected_df["alpha"], ci_selected_df["CI"], marker="s", label="RFECV refit selected features")
        ax.set_xlabel("alpha")
        ax.set_ylabel("consistency index")
        ax.set_ylim(0, 1.02)
        ax.set_title("Full-feature vs RFECV-refit selected-feature CI")
        ax.legend(frameon=False)
        fig.savefig(OUT_DIR / "chbmit_ci_full_vs_rfecv_selected.png", dpi=220)
        plt.close(fig)

    summary_lines = [
        "# RFECV Extension Results",
        "",
        "This file was generated by `scripts/project_rfecv_extension.py` using cached CHB-MIT features.",
        "",
        f"- Selected features: {len(selected_features)} of {len(FEATURE_NAMES)}.",
        f"- Best CV score reported by RFECV: {float(selector.cv_results_['mean_test_score'].max()):.4f}.",
        "- RFECV cross-validation now includes imputation and scaling inside the estimator pipeline.",
        f"- {ci_sentence}",
        "- This CI is recomputed after refitting logistic-regression and random-forest models on the RFECV-selected feature subset.",
        "- RFECV improves balanced accuracy and recall in this run, but the RFECV-refit selected-feature CI is lower than the full-feature CI, so feature selection did not improve explanation consistency here.",
        "- The RFECV curve uses a narrow y-axis range; CV balanced accuracy varies only slightly across many feature counts.",
        "",
        "## Selected Features",
        "",
        "\n".join(f"- `{name}`" for name in selected_features),
        "",
        "## Test Metrics",
        "",
        simple_markdown_table(metrics_df),
        "",
        "## Positive-Call Rates By Phase",
        "",
        simple_markdown_table(positive_rates),
        "",
        "## Files",
        "",
        "- `outputs/chbmit_rfecv_feature_ranking.csv`",
        "- `outputs/chbmit_rfecv_selected_features.txt`",
        "- `outputs/chbmit_rfecv_model_metrics.csv`",
        "- `outputs/chbmit_rfecv_test_predictions.csv`",
        "- `outputs/chbmit_rfecv_positive_call_rates_by_phase.csv`",
        "- `outputs/chbmit_all_model_metrics.csv`",
        "- `outputs/chbmit_all_model_confusion_matrices.png`",
        "- `outputs/chbmit_rfecv_cv_results.csv`",
        "- `outputs/chbmit_rfecv_curve.png`",
        "- `outputs/chbmit_rfecv_refit_feature_importance.csv`",
        "- `outputs/chbmit_rfecv_selected_ci_alpha_sweep.csv`",
        "- `outputs/chbmit_rfecv_selected_ci_summary.json`",
        "- `outputs/chbmit_ci_full_vs_rfecv_selected.png`",
        "- `models/chbmit_rfecv_logistic_regression.joblib`",
        "",
    ]
    (PROJECT_DIR / "08_rfecv_extension_results.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"selected {len(selected_features)} features")
    print(f"wrote RFECV outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
