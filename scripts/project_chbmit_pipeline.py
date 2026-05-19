"""CHB-MIT feature extraction, model training, and CI analysis.

The project environment does not currently have mne or pyedflib installed, so
this script includes a minimal EDF reader for the CHB-MIT EDF files. It keeps a
manageable default scope: seizure files plus neighboring context files for
chb01, chb03, and chb05.

Outputs are written under final_project_seizure_dynamics/outputs and models.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
OUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

from consistency_index import alpha_sweep, consistency_index  # noqa: E402


SELECTED_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "FZ-CZ",
    "CZ-PZ",
]

BASE_FEATURE_NAMES = [
    "sampleEn_approx",
    "fuzzyEn_approx",
    "skew",
    "kurt",
    "deltaP",
    "thetaP",
    "alphaP",
    "betaP",
    "totalP_0p5_40",
    "deltaRel",
    "thetaRel",
    "alphaRel",
    "betaRel",
    "theta_beta_ratio",
    "beta_alpha_ratio",
    "alpha_theta_ratio",
    "spec_entropy",
    "spec_slope_4_40",
    "beta_peak_freq",
    "rms",
    "line_length",
    "zcr",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
]

FEATURE_NAMES = [f"mean_{name}" for name in BASE_FEATURE_NAMES] + [
    f"max_{name}" for name in BASE_FEATURE_NAMES
]

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


@dataclass(frozen=True)
class EdfHeader:
    header_bytes: int
    n_records: int
    record_duration: float
    labels: list[str]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    samples_per_record: np.ndarray


def _decode_ascii(raw: bytes) -> str:
    return raw.decode("ascii", errors="ignore").strip()


def _read_str_array(handle, width: int, count: int) -> list[str]:
    return [_decode_ascii(handle.read(width)) for _ in range(count)]


def _read_float_array(handle, width: int, count: int) -> np.ndarray:
    vals = []
    for _ in range(count):
        text = _decode_ascii(handle.read(width))
        vals.append(float(text) if text else np.nan)
    return np.asarray(vals, dtype=float)


def _read_int_array(handle, width: int, count: int) -> np.ndarray:
    vals = []
    for _ in range(count):
        text = _decode_ascii(handle.read(width))
        vals.append(int(float(text)) if text else 0)
    return np.asarray(vals, dtype=int)


def read_edf_header(path: Path) -> EdfHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(_decode_ascii(fixed[184:192]))
        n_records = int(float(_decode_ascii(fixed[236:244])))
        record_duration = float(_decode_ascii(fixed[244:252]))
        n_signals = int(_decode_ascii(fixed[252:256]))

        labels = _read_str_array(handle, 16, n_signals)
        handle.read(80 * n_signals)  # transducer
        handle.read(8 * n_signals)  # physical dimension
        physical_min = _read_float_array(handle, 8, n_signals)
        physical_max = _read_float_array(handle, 8, n_signals)
        digital_min = _read_float_array(handle, 8, n_signals)
        digital_max = _read_float_array(handle, 8, n_signals)
        handle.read(80 * n_signals)  # prefiltering
        samples_per_record = _read_int_array(handle, 8, n_signals)
        handle.read(32 * n_signals)  # reserved

    if n_records < 0:
        bytes_per_record = int(samples_per_record.sum()) * 2
        n_records = int((path.stat().st_size - header_bytes) // bytes_per_record)

    return EdfHeader(
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration=record_duration,
        labels=labels,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
        samples_per_record=samples_per_record,
    )


def read_edf_selected(path: Path, selected_channels: Iterable[str]) -> tuple[np.ndarray, float, list[str]]:
    header = read_edf_header(path)
    selected_set = {c.upper() for c in selected_channels}
    indices = []
    seen_labels: set[str] = set()
    for i, label in enumerate(header.labels):
        label_key = label.upper()
        if label_key in selected_set and label_key not in seen_labels:
            indices.append(i)
            seen_labels.add(label_key)
    if not indices:
        indices = list(range(min(8, len(header.labels))))

    samples = header.samples_per_record
    record_len = int(samples.sum())
    offsets = np.concatenate([[0], np.cumsum(samples)[:-1]])
    expected_values = header.n_records * record_len

    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        raw = np.fromfile(handle, dtype="<i2", count=expected_values)

    if raw.size != expected_values:
        usable_records = raw.size // record_len
        raw = raw[: usable_records * record_len]
    else:
        usable_records = header.n_records

    raw_records = raw.reshape(usable_records, record_len)
    signals = []
    names = []
    fs_values = []
    for idx in indices:
        nsamp = int(samples[idx])
        start = int(offsets[idx])
        dig = raw_records[:, start : start + nsamp].reshape(-1).astype(float)
        denom = header.digital_max[idx] - header.digital_min[idx]
        if denom == 0:
            phys = dig
        else:
            scale = (header.physical_max[idx] - header.physical_min[idx]) / denom
            phys = (dig - header.digital_min[idx]) * scale + header.physical_min[idx]
        signals.append(phys)
        names.append(header.labels[idx])
        fs_values.append(nsamp / header.record_duration)

    min_len = min(len(sig) for sig in signals)
    data = np.column_stack([sig[:min_len] for sig in signals])
    fs = float(np.median(fs_values))
    return data, fs, names


def parse_summary_intervals(summary_path: Path) -> dict[str, list[tuple[float, float]]]:
    intervals_by_file: dict[str, list[tuple[float, float]]] = {}
    current_file: str | None = None
    pending_start: float | None = None

    for line in summary_path.read_text(errors="ignore").splitlines():
        m_file = re.search(r"File Name:\s*(\S+)", line)
        if m_file:
            current_file = m_file.group(1)
            intervals_by_file.setdefault(current_file, [])
            pending_start = None
            continue

        if current_file is None:
            continue

        m_start = re.search(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s+seconds", line)
        if m_start:
            pending_start = float(m_start.group(1))
            continue

        m_end = re.search(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s+seconds", line)
        if m_end and pending_start is not None:
            intervals_by_file[current_file].append((pending_start, float(m_end.group(1))))
            pending_start = None

    return intervals_by_file


def parse_subject_list(subjects: str) -> list[str]:
    parsed = []
    for token in subjects.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            parsed.append(f"chb{int(token):02d}")
        elif token.lower().startswith("chb"):
            parsed.append(token.lower())
        else:
            raise ValueError(f"Unrecognized subject token: {token}")
    return parsed


def choose_records(data_dir: Path, subjects: list[str], baseline_files_per_subject: int = 4) -> pd.DataFrame:
    records = [line.strip() for line in (data_dir / "RECORDS").read_text().splitlines() if line.strip()]
    seizure_records = {
        line.strip()
        for line in (data_dir / "RECORDS-WITH-SEIZURES").read_text().splitlines()
        if line.strip()
    }

    selected: dict[str, str] = {}
    for subject in subjects:
        subject_records = [r for r in records if r.startswith(f"{subject}/") and r.endswith(".edf")]
        subject_seizure_records = [r for r in subject_records if r in seizure_records]
        subject_set = set(subject_records)

        for rec in subject_seizure_records:
            idx = subject_records.index(rec)
            for j in [idx - 1, idx, idx + 1]:
                if 0 <= j < len(subject_records):
                    selected[subject_records[j]] = "seizure_or_neighbor"

        n_added_baseline = 0
        for rec in subject_records:
            if rec in seizure_records or rec in selected:
                continue
            if rec not in subject_set:
                continue
            selected[rec] = "baseline_context"
            n_added_baseline += 1
            if n_added_baseline >= baseline_files_per_subject:
                break

    rows = []
    for rec, reason in sorted(selected.items()):
        rows.append(
            {
                "record": rec,
                "subject": rec.split("/")[0],
                "edf_path": rec,
                "is_seizure_record": rec in seizure_records,
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


def _band_power(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(pxx[mask], freqs[mask]))


def _fft_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    if len(x) < 8:
        return np.array([], dtype=float), np.array([], dtype=float)
    window = np.hanning(len(x))
    denom = fs * np.sum(window**2)
    fft = np.fft.rfft(x * window)
    pxx = (np.abs(fft) ** 2) / max(denom, 1e-12)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return freqs, pxx


def _hjorth(x: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float) - np.mean(x)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = float(np.var(x)) + eps
    var1 = float(np.var(dx)) + eps
    var2 = float(np.var(ddx)) + eps
    mobility = float(np.sqrt(var1 / var0))
    complexity = float(np.sqrt(var2 / var1) / (mobility + eps))
    return var0, mobility, complexity


def extract_channel_features(sig: np.ndarray, fs: float) -> np.ndarray:
    eps = 1e-12
    raw = np.asarray(sig, dtype=float)
    x = raw - np.mean(raw)
    z = _zscore(raw)
    dz = np.diff(z)

    sample_en = float(np.log(np.std(dz) + eps))
    fuzzy_en = float(-np.log((np.linalg.norm(z[:-1] - z[1:]) / max(len(z), 1)) + eps))
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)

    freqs, pxx = _fft_psd(x, fs)
    delta_p = _band_power(freqs, pxx, *EEG_BANDS["delta"])
    theta_p = _band_power(freqs, pxx, *EEG_BANDS["theta"])
    alpha_p = _band_power(freqs, pxx, *EEG_BANDS["alpha"])
    beta_p = _band_power(freqs, pxx, *EEG_BANDS["beta"])
    total_p = _band_power(freqs, pxx, 0.5, 40.0)
    total_safe = total_p + eps

    delta_rel = delta_p / total_safe
    theta_rel = theta_p / total_safe
    alpha_rel = alpha_p / total_safe
    beta_rel = beta_p / total_safe
    theta_beta_ratio = theta_p / (beta_p + eps)
    beta_alpha_ratio = beta_p / (alpha_p + eps)
    alpha_theta_ratio = alpha_p / (theta_p + eps)

    if freqs.size == 0:
        spec_entropy = 0.0
        spec_slope = 0.0
        beta_peak_freq = 0.0
    else:
        m_0_40 = (freqs >= 0.5) & (freqs <= 40.0)
        p_0_40 = pxx[m_0_40]
        if p_0_40.size:
            pn = p_0_40 / (p_0_40.sum() + eps)
            spec_entropy = float(-(pn * np.log(pn + eps)).sum() / np.log(len(pn) + eps))
        else:
            spec_entropy = 0.0

        m_4_40 = (freqs >= 4.0) & (freqs <= 40.0)
        if np.any(m_4_40):
            lf = np.log(freqs[m_4_40] + eps)
            lp = np.log(pxx[m_4_40] + eps)
            spec_slope = float(np.polyfit(lf, lp, 1)[0])
        else:
            spec_slope = 0.0

        m_beta = (freqs >= 13.0) & (freqs < 30.0)
        beta_peak_freq = float(freqs[m_beta][np.argmax(pxx[m_beta])]) if np.any(m_beta) else 0.0

    rms = float(np.sqrt(np.mean(x**2)))
    line_length = float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    zcr = float(np.mean(np.diff(np.signbit(x)) != 0)) if len(x) > 1 else 0.0
    hj_activity, hj_mobility, hj_complexity = _hjorth(x)

    return np.nan_to_num(
        np.asarray(
            [
                sample_en,
                fuzzy_en,
                skew,
                kurt,
                delta_p,
                theta_p,
                alpha_p,
                beta_p,
                total_p,
                delta_rel,
                theta_rel,
                alpha_rel,
                beta_rel,
                theta_beta_ratio,
                beta_alpha_ratio,
                alpha_theta_ratio,
                spec_entropy,
                spec_slope,
                beta_peak_freq,
                rms,
                line_length,
                zcr,
                hj_activity,
                hj_mobility,
                hj_complexity,
            ],
            dtype=float,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def extract_window_features(window_data: np.ndarray, fs: float) -> np.ndarray:
    channel_features = np.vstack(
        [extract_channel_features(window_data[:, ch], fs) for ch in range(window_data.shape[1])]
    )
    return np.concatenate([np.mean(channel_features, axis=0), np.max(channel_features, axis=0)])


def interval_overlap(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any((start < stop and end > begin) for begin, stop in intervals)


def phase_for_window(
    start: float,
    end: float,
    intervals: list[tuple[float, float]],
    preictal_sec: float,
) -> str:
    if interval_overlap(start, end, intervals):
        return "ictal"
    for begin, _stop in intervals:
        if end <= begin and (begin - end) <= preictal_sec:
            return "preictal"
    return "baseline"


def rel_time_to_nearest_onset(center: float, intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return np.nan
    starts = np.asarray([begin for begin, _stop in intervals], dtype=float)
    idx = int(np.argmin(np.abs(center - starts)))
    return float(center - starts[idx])


def extract_features_for_records(
    selected_records: pd.DataFrame,
    data_dir: Path,
    win_sec: float,
    step_sec: float,
    preictal_sec: float,
) -> pd.DataFrame:
    all_rows = []
    interval_cache: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for rec_idx, rec_row in selected_records.reset_index(drop=True).iterrows():
        raw_path = Path(str(rec_row["edf_path"]))
        edf_path = raw_path if raw_path.is_absolute() else data_dir / raw_path
        subject = rec_row["subject"]
        summary_path = data_dir / subject / f"{subject}-summary.txt"
        if subject not in interval_cache:
            interval_cache[subject] = parse_summary_intervals(summary_path)
        intervals = interval_cache[subject].get(edf_path.name, [])

        print(f"[{rec_idx + 1}/{len(selected_records)}] extracting {edf_path.name}")
        data, fs, channels = read_edf_selected(edf_path, SELECTED_CHANNELS)
        win_samples = int(round(win_sec * fs))
        step_samples = int(round(step_sec * fs))
        n_samples = data.shape[0]

        if n_samples < win_samples:
            continue

        for start_idx in range(0, n_samples - win_samples + 1, step_samples):
            end_idx = start_idx + win_samples
            start_sec = start_idx / fs
            end_sec = end_idx / fs
            center_sec = 0.5 * (start_sec + end_sec)
            feats = extract_window_features(data[start_idx:end_idx, :], fs)
            phase = phase_for_window(start_sec, end_sec, intervals, preictal_sec)
            row = {
                "record": f"{subject}/{edf_path.name}",
                "subject": subject,
                "window_start_sec": start_sec,
                "window_end_sec": end_sec,
                "window_center_sec": center_sec,
                "rel_time_to_onset_sec": rel_time_to_nearest_onset(center_sec, intervals),
                "phase": phase,
                "ictal_label": int(phase == "ictal"),
                "preictal_label": int(phase == "preictal"),
                "is_seizure_record": bool(intervals),
                "n_channels_used": len(channels),
                "channels_used": "|".join(channels),
            }
            row.update({name: float(value) for name, value in zip(FEATURE_NAMES, feats)})
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def choose_group_split(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(200):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups))
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
            return train_idx, test_idx
    raise RuntimeError("Could not find a group split containing both classes in train and test.")


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray | None) -> float:
    if y_score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray | None) -> float:
    if y_score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def false_alarm_events_per_hour(pred_df: pd.DataFrame, pred_col: str, step_sec: float) -> float:
    non_ictal = pred_df[pred_df["ictal_label"] == 0].copy()
    non_ictal_hours = max(float(len(non_ictal) * step_sec / 3600.0), 1e-12)
    events = 0
    for _record, g in non_ictal.sort_values(["record", "window_start_sec"]).groupby("record"):
        prev_positive = False
        prev_end = None
        for row in g.itertuples(index=False):
            positive = int(getattr(row, pred_col)) == 1
            start = float(getattr(row, "window_start_sec"))
            if positive and (not prev_positive or prev_end is None or start - prev_end > step_sec * 1.5):
                events += 1
            prev_positive = positive
            prev_end = float(getattr(row, "window_end_sec"))
    return float(events / non_ictal_hours)


def metric_row(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step_sec: float,
    y_score: np.ndarray | None = None,
    false_alarm_events_per_hour_value: float | None = None,
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
        "average_precision": _safe_average_precision(y_true, y_score),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_windows_per_hour": float(fp / neg_hours),
        "false_alarm_events_per_hour": float(false_alarm_events_per_hour_value)
        if false_alarm_events_per_hour_value is not None
        else np.nan,
    }


def train_models(feature_df: pd.DataFrame, step_sec: float, seed: int, topk: int) -> dict[str, object]:
    y = feature_df["ictal_label"].to_numpy(dtype=int)
    groups = feature_df["record"].to_numpy()
    X = feature_df[FEATURE_NAMES].to_numpy(dtype=float)
    train_idx, test_idx = choose_group_split(y, groups, test_size=0.30, seed=seed)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logit = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    rf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logit.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    test_pred_df = feature_df.iloc[test_idx][
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
    metrics = []
    for name, model in [("logistic_regression", logit), ("random_forest", rf)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_pred_df[f"{name}_pred"] = y_pred
        test_pred_df[f"{name}_prob"] = y_prob if y_prob is not None else np.nan
        events_per_hour = false_alarm_events_per_hour(test_pred_df, f"{name}_pred", step_sec)
        metrics.append(
            metric_row(
                name,
                y_test,
                y_pred,
                step_sec=step_sec,
                y_score=y_prob,
                false_alarm_events_per_hour_value=events_per_hour,
            )
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DIR / "chbmit_model_metrics.csv", index=False)
    test_pred_df.to_csv(OUT_DIR / "chbmit_test_predictions.csv", index=False)
    joblib.dump(logit, MODEL_DIR / "chbmit_logistic_regression.joblib")
    joblib.dump(rf, MODEL_DIR / "chbmit_random_forest.joblib")

    logit_coef = logit.named_steps["clf"].coef_[0]
    rf_importance = rf.named_steps["clf"].feature_importances_

    if len(test_idx) > 6000:
        rng = np.random.default_rng(seed)
        perm_local = np.sort(rng.choice(np.arange(len(test_idx)), size=6000, replace=False))
        X_perm = X_test[perm_local]
        y_perm = y_test[perm_local]
    else:
        X_perm = X_test
        y_perm = y_test

    perm = permutation_importance(
        rf,
        X_perm,
        y_perm,
        n_repeats=8,
        random_state=seed,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "logit_abs_coef": np.abs(logit_coef),
            "rf_gini_importance": rf_importance,
            "rf_permutation_importance": np.maximum(perm.importances_mean, 0.0),
            "rf_permutation_importance_raw": perm.importances_mean,
            "rf_permutation_std": perm.importances_std,
        }
    )
    for col in ["logit_abs_coef", "rf_gini_importance", "rf_permutation_importance"]:
        importance_df[f"{col}_rank"] = importance_df[col].rank(ascending=False, method="min")
    importance_df = importance_df.sort_values("rf_permutation_importance", ascending=False)
    importance_df.to_csv(OUT_DIR / "chbmit_feature_importance.csv", index=False)

    method_scores = {
        "logit_abs_coef": dict(zip(FEATURE_NAMES, np.abs(logit_coef))),
        "rf_gini_importance": dict(zip(FEATURE_NAMES, np.maximum(rf_importance, 0.0))),
        "rf_permutation_importance": dict(zip(FEATURE_NAMES, np.maximum(perm.importances_mean, 0.0))),
    }
    ci_rows = alpha_sweep(method_scores, k=topk, step=0.05, use_abs=False)
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(OUT_DIR / "chbmit_ci_alpha_sweep.csv", index=False)
    pd.Series(consistency_index(method_scores, alpha=0.5, k=topk, use_abs=False)).to_json(
        OUT_DIR / "chbmit_ci_summary.json",
        indent=2,
    )

    split_df = feature_df[["record", "subject", "phase", "ictal_label"]].copy()
    split_df["split"] = "unused"
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[test_idx, "split"] = "test"
    split_df.to_csv(OUT_DIR / "chbmit_train_test_split.csv", index=False)

    split_summary = (
        split_df.groupby(["split", "subject", "record"])["ictal_label"]
        .agg(windows="size", ictal_windows="sum")
        .reset_index()
    )
    split_summary.to_csv(OUT_DIR / "chbmit_split_record_summary.csv", index=False)

    return {
        "metrics": metrics_df,
        "importance": importance_df,
        "ci": ci_df,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "y_test": y_test,
        "rf_pred": rf.predict(X_test),
        "test_predictions": test_pred_df,
        "split_summary": split_summary,
    }


def plot_class_balance(feature_df: pd.DataFrame) -> None:
    counts = feature_df["phase"].value_counts().reindex(["baseline", "preictal", "ictal"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.bar(counts.index, counts.values, color=["#7f8c8d", "#ff7f0e", "#d62728"])
    ax.set_ylabel("number of windows")
    ax.set_title("CHB-MIT extracted window phases")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    fig.savefig(OUT_DIR / "chbmit_class_balance.png", dpi=220)
    plt.close(fig)


def plot_importances(importance_df: pd.DataFrame) -> None:
    top = importance_df.sort_values("rf_permutation_importance", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    ax.barh(top["feature"], top["rf_permutation_importance"], color="#2c7fb8")
    ax.set_xlabel("permutation importance, balanced accuracy drop")
    ax.set_title("Top EEG features by random-forest permutation importance")
    fig.savefig(OUT_DIR / "chbmit_top_feature_importance.png", dpi=220)
    plt.close(fig)


def plot_ci(ci_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(ci_df["alpha"], ci_df["CI"], marker="o", label="CI(alpha)")
    ax.axhline(float(ci_df["CI_rank"].iloc[0]), color="#d62728", linestyle="--", label="CI_rank")
    ax.axhline(float(ci_df["CI_topk"].iloc[0]), color="#2ca02c", linestyle=":", label="CI_topk")
    ax.set_xlabel("alpha")
    ax.set_ylabel("consistency index")
    ax.set_ylim(0, 1.02)
    ax.set_title("Explanation consistency across feature-importance methods")
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "chbmit_ci_alpha_sensitivity.png", dpi=220)
    plt.close(fig)


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if not np.isfinite(mad) or mad < 1e-12:
        return series * 0.0
    return (series - med) / (1.4826 * mad)


def plot_feature_trajectories(feature_df: pd.DataFrame, step_sec: float) -> None:
    plot_df = feature_df[
        feature_df["is_seizure_record"]
        & feature_df["rel_time_to_onset_sec"].between(-600, 180, inclusive="both")
    ].copy()
    if plot_df.empty:
        return

    feature_choices = [
        "max_betaP",
        "max_line_length",
        "mean_spec_entropy",
        "max_hjorth_complexity",
    ]
    for name in feature_choices:
        plot_df[f"z_{name}"] = robust_z(plot_df[name].astype(float)).clip(-8, 8)
    plot_df["transition_index"] = (
        plot_df["z_max_betaP"] + plot_df["z_max_line_length"] - plot_df["z_mean_spec_entropy"]
    )
    feature_choices_with_index = feature_choices + ["transition_index"]

    plot_df["time_bin"] = (np.round(plot_df["rel_time_to_onset_sec"] / step_sec) * step_sec).astype(int)
    grouped = plot_df.groupby("time_bin")[[f"z_{x}" for x in feature_choices] + ["transition_index"]].median()

    fig, axes = plt.subplots(len(feature_choices_with_index), 1, figsize=(8.5, 9.0), sharex=True, constrained_layout=True)
    for ax, name in zip(axes, feature_choices_with_index):
        col = f"z_{name}" if name in feature_choices else name
        ax.plot(grouped.index, grouped[col], color="#1f77b4", lw=1.6)
        ax.axvline(0, color="#d62728", linestyle="--", lw=1)
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_ylabel(name.replace("_", "\n"), fontsize=8)
    axes[-1].set_xlabel("seconds relative to seizure onset")
    axes[0].set_title("Median feature trajectories (robust-z, clipped to +/-8)")
    fig.savefig(OUT_DIR / "chbmit_feature_trajectories.png", dpi=220)
    plt.close(fig)


def plot_confusion(metrics_df: pd.DataFrame) -> None:
    row = metrics_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
    cm = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.3, 3.8), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    ax.set_title(f"Confusion matrix: {row['model']}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT_DIR / "chbmit_confusion_matrix.png", dpi=220)
    plt.close(fig)


def simple_markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    """Small markdown table formatter that avoids requiring tabulate."""
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}g}"
        return str(value)

    cols = list(df.columns)
    rows = [[fmt(value) for value in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def feature_cache_config(args: argparse.Namespace, subjects: list[str], data_dir: Path) -> dict[str, object]:
    return {
        "feature_cache_version": 2,
        "data_dir_resolved": str(data_dir.resolve()),
        "subjects": args.subjects,
        "subjects_parsed": subjects,
        "win_sec": float(args.win_sec),
        "step_sec": float(args.step_sec),
        "preictal_sec": float(args.preictal_sec),
        "baseline_files_per_subject": int(args.baseline_files_per_subject),
        "selected_channels": SELECTED_CHANNELS,
        "feature_names": FEATURE_NAMES,
    }


def cache_config_matches(old: dict[str, object], new: dict[str, object]) -> bool:
    keys = [
        "feature_cache_version",
        "subjects_parsed",
        "win_sec",
        "step_sec",
        "preictal_sec",
        "baseline_files_per_subject",
        "selected_channels",
        "feature_names",
    ]
    return all(old.get(key) == new.get(key) for key in keys)


def selected_records_from_feature_cache(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record, g in feature_df.groupby("record", sort=True):
        subject = str(g["subject"].iloc[0])
        rows.append(
            {
                "record": record,
                "subject": subject,
                "edf_path": record,
                "is_seizure_record": bool(g["is_seizure_record"].any()),
                "selection_reason": "reconstructed_from_feature_cache",
            }
        )
    return pd.DataFrame(rows)


def write_results_summary(
    args: argparse.Namespace,
    selected_records: pd.DataFrame,
    feature_df: pd.DataFrame,
    analysis: dict[str, object],
) -> None:
    metrics_df: pd.DataFrame = analysis["metrics"]  # type: ignore[assignment]
    importance_df: pd.DataFrame = analysis["importance"]  # type: ignore[assignment]
    ci_df: pd.DataFrame = analysis["ci"]  # type: ignore[assignment]

    top_features = importance_df.sort_values("rf_permutation_importance", ascending=False).head(10)
    ci_mid = ci_df.iloc[(ci_df["alpha"] - 0.5).abs().argmin()]
    phase_counts = feature_df["phase"].value_counts().to_dict()
    split_summary: pd.DataFrame = analysis["split_summary"]  # type: ignore[assignment]
    test_seizure_records = split_summary[
        (split_summary["split"] == "test") & (split_summary["ictal_windows"] > 0)
    ][["subject", "record", "windows", "ictal_windows"]]
    key_features = ["mean_rms", "mean_thetaP", "mean_hjorth_activity", "mean_totalP_0p5_40"]
    phase_medians = (
        feature_df.groupby("phase")[key_features]
        .median()
        .reindex(["baseline", "preictal", "ictal"])
        .reset_index()
    )
    phase_medians.to_csv(OUT_DIR / "chbmit_phase_feature_medians.csv", index=False)
    pred_df: pd.DataFrame = analysis["test_predictions"]  # type: ignore[assignment]
    positive_rate_rows = []
    for model_name in ["logistic_regression", "random_forest"]:
        pred_col = f"{model_name}_pred"
        if pred_col not in pred_df.columns:
            continue
        for phase in ["baseline", "preictal", "ictal"]:
            phase_df = pred_df[pred_df["phase"] == phase]
            positive_rate_rows.append(
                {
                    "model": model_name,
                    "phase": phase,
                    "n_windows": int(len(phase_df)),
                    "positive_call_rate": float(phase_df[pred_col].mean()) if len(phase_df) else np.nan,
                }
            )
    positive_rates = pd.DataFrame(positive_rate_rows)
    positive_rates.to_csv(OUT_DIR / "chbmit_positive_call_rates_by_phase.csv", index=False)

    lines = [
        "# CHB-MIT Modeling Run Results",
        "",
        "This file was generated by `scripts/project_chbmit_pipeline.py`.",
        "",
        "## Scope",
        "",
        f"- Subjects: `{args.subjects}`.",
        f"- Window length: {args.win_sec:g} s.",
        f"- Step size: {args.step_sec:g} s.",
        f"- Preictal window definition: {args.preictal_sec:g} s before seizure onset.",
        f"- Selected EDF records: {len(selected_records)}.",
        f"- Extracted windows: {len(feature_df)}.",
        f"- Phase counts: `{phase_counts}`.",
        "",
        "The run uses seizure EDF records, neighboring context records, and a few baseline records per subject. This is a working subset for the final project, not a claim of full CHB-MIT benchmarking.",
        "",
        "This is an ictal-window detection task. Preictal windows are descriptive context and are treated as negative examples by the classifier.",
        "",
        "Preictal labels are record-local in this implementation; windows in neighboring EDF records are not assigned time-to-next-seizure labels across file boundaries.",
        "",
        "## Split Notes",
        "",
        "The split is record-held-out, not subject-held-out. This avoids placing windows from the same EDF recording in both train and test, but it does not test generalization to unseen patients.",
        "",
        "Test seizure records:",
        "",
        simple_markdown_table(test_seizure_records),
        "",
        "Feature medians by phase:",
        "",
        simple_markdown_table(phase_medians),
        "",
        "Positive-call rates by phase:",
        "",
        simple_markdown_table(positive_rates),
        "",
        "## Model Metrics",
        "",
        simple_markdown_table(metrics_df),
        "",
        "The classifier target is `ictal_label`, so seizure-overlapping windows are positive and all other windows are negative. The false-positive-window column reports false-positive 5-second windows per non-ictal hour. The false-alarm-event column uses a simple post hoc definition in which contiguous false-positive non-ictal windows within a record count as one event.",
        "",
        "## Figure Notes",
        "",
        "`outputs/chbmit_feature_trajectories.png` uses robust-z scaled features clipped to +/-8 for visualization. The first panel may hit the clipping limit, so the figure is best interpreted as a qualitative trajectory shape rather than an absolute magnitude comparison.",
        "",
        "`outputs/chbmit_confusion_matrix.png` prints raw counts. Because true negatives dominate the color scale, the metrics table is the clearer source for positive-class performance.",
        "",
        "## Top Features",
        "",
        simple_markdown_table(
            top_features[
                ["feature", "logit_abs_coef", "rf_gini_importance", "rf_permutation_importance"]
            ]
        ),
        "",
        "## Consistency Index",
        "",
        f"At alpha = {ci_mid['alpha']:.2f}, CI = {ci_mid['CI']:.3f}, CI_rank = {ci_mid['CI_rank']:.3f}, and CI_topk = {ci_mid['CI_topk']:.3f}.",
        "",
        "Interpretation: CI combines global ranking agreement with top-k feature overlap across logistic-regression coefficients, random-forest impurity importance, and random-forest permutation importance. Because these come from different model families as well as different importance definitions, disagreement reflects both model-family and explanation-method dependence.",
        "",
        "## Generated Files",
        "",
        "- `outputs/chbmit_selected_records.csv`",
        "- `outputs/chbmit_features.csv`",
        "- `outputs/chbmit_model_metrics.csv`",
        "- `outputs/chbmit_test_predictions.csv`",
        "- `outputs/chbmit_split_record_summary.csv`",
        "- `outputs/chbmit_phase_feature_medians.csv`",
        "- `outputs/chbmit_positive_call_rates_by_phase.csv`",
        "- `outputs/chbmit_feature_importance.csv`",
        "- `outputs/chbmit_ci_alpha_sweep.csv`",
        "- `outputs/chbmit_class_balance.png`",
        "- `outputs/chbmit_feature_trajectories.png`",
        "- `outputs/chbmit_top_feature_importance.png`",
        "- `outputs/chbmit_ci_alpha_sensitivity.png`",
        "- `outputs/chbmit_confusion_matrix.png`",
        "- `models/chbmit_logistic_regression.joblib`",
        "- `models/chbmit_random_forest.joblib`",
        "",
        "## Caution",
        "",
        "These results are for mathematical-modeling support and proposal development. They should not be interpreted as a medically validated seizure prediction system.",
        "",
    ]
    (PROJECT_DIR / "06_chbmit_model_results.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CHB-MIT project feature/model pipeline.")
    parser.add_argument("--data-dir", default=str(REPO_DIR / "data" / "chbmit" / "1.0.0"))
    parser.add_argument("--subjects", default="chb01,chb03,chb05")
    parser.add_argument("--win-sec", type=float, default=5.0)
    parser.add_argument("--step-sec", type=float, default=5.0)
    parser.add_argument("--preictal-sec", type=float, default=300.0)
    parser.add_argument("--baseline-files-per-subject", type=int, default=4)
    parser.add_argument("--seed", type=int, default=383)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    subjects = parse_subject_list(args.subjects)

    feature_path = OUT_DIR / "chbmit_features.csv"
    meta_path = OUT_DIR / "chbmit_run_config.json"
    selected_records_path = OUT_DIR / "chbmit_selected_records.csv"
    config = feature_cache_config(args, subjects, data_dir)

    if feature_path.exists() and not args.force_extract:
        if not meta_path.exists():
            raise RuntimeError(
                f"{feature_path} exists but {meta_path} is missing. "
                "Rerun with --force-extract to rebuild the feature cache."
            )
        old_config = json.loads(meta_path.read_text(encoding="utf-8"))
        if not cache_config_matches(old_config, config):
            raise RuntimeError(
                "Cached feature configuration does not match current arguments. "
                "Rerun with --force-extract to rebuild outputs/chbmit_features.csv."
            )
        print(f"loading cached features from {feature_path}")
        feature_df = pd.read_csv(feature_path)
        if selected_records_path.exists():
            selected_records = pd.read_csv(selected_records_path)
        else:
            selected_records = selected_records_from_feature_cache(feature_df)
            selected_records.to_csv(selected_records_path, index=False)
    else:
        selected_records = choose_records(
            data_dir,
            subjects,
            baseline_files_per_subject=args.baseline_files_per_subject,
        )
        selected_records.to_csv(selected_records_path, index=False)
        feature_df = extract_features_for_records(
            selected_records,
            data_dir=data_dir,
            win_sec=args.win_sec,
            step_sec=args.step_sec,
            preictal_sec=args.preictal_sec,
        )
        feature_df.to_csv(feature_path, index=False)
        meta_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if feature_df["ictal_label"].nunique() < 2:
        raise RuntimeError("Only one class found after feature extraction; cannot train classifier.")

    analysis = train_models(feature_df, step_sec=args.step_sec, seed=args.seed, topk=args.topk)
    plot_class_balance(feature_df)
    plot_feature_trajectories(feature_df, step_sec=args.step_sec)
    plot_importances(analysis["importance"])  # type: ignore[arg-type]
    plot_ci(analysis["ci"])  # type: ignore[arg-type]
    plot_confusion(analysis["metrics"])  # type: ignore[arg-type]
    write_results_summary(args, selected_records, feature_df, analysis)

    print(f"wrote CHB-MIT outputs to {OUT_DIR}")
    print(f"wrote result summary to {PROJECT_DIR / '06_chbmit_model_results.md'}")


if __name__ == "__main__":
    main()
