"""Consistency Index helper for feature-importance explanations.

This implements the CI idea used in the uploaded XML/RFECV paper:

    CI(alpha) = alpha * CI_rank + (1 - alpha) * CI_topk

Inputs are dictionaries mapping explanation-method names to feature scores.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr


FeatureScores = Mapping[str, Mapping[str, float]]


def _feature_union(method_scores: FeatureScores) -> list[str]:
    features: set[str] = set()
    for scores in method_scores.values():
        features.update(scores.keys())
    return sorted(features)


def _score_vector(scores: Mapping[str, float], features: list[str]) -> np.ndarray:
    return np.asarray([float(scores.get(f, 0.0)) for f in features], dtype=float)


def _top_k(scores: Mapping[str, float], k: int, use_abs: bool = True) -> set[str]:
    def key(item: tuple[str, float]) -> tuple[float, str]:
        value = abs(float(item[1])) if use_abs else float(item[1])
        return value, item[0]

    ranked = sorted(scores.items(), key=key, reverse=True)
    return {name for name, _ in ranked[: max(1, int(k))]}


def ci_rank(method_scores: FeatureScores, use_abs: bool = True) -> float:
    """Mean pairwise normalized Spearman rank agreement in [0, 1]."""
    names = list(method_scores.keys())
    if len(names) < 2:
        return 1.0

    features = _feature_union(method_scores)
    vals = []
    for a, b in combinations(names, 2):
        va = _score_vector(method_scores[a], features)
        vb = _score_vector(method_scores[b], features)
        if use_abs:
            va = np.abs(va)
            vb = np.abs(vb)
        rho = spearmanr(va, vb).correlation
        if not np.isfinite(rho):
            rho = 0.0
        vals.append((float(rho) + 1.0) / 2.0)
    return float(np.mean(vals))


def ci_topk(method_scores: FeatureScores, k: int = 8, use_abs: bool = True) -> float:
    """Mean pairwise Jaccard overlap among top-k feature sets."""
    names = list(method_scores.keys())
    if len(names) < 2:
        return 1.0

    vals = []
    for a, b in combinations(names, 2):
        sa = _top_k(method_scores[a], k=k, use_abs=use_abs)
        sb = _top_k(method_scores[b], k=k, use_abs=use_abs)
        denom = len(sa | sb)
        vals.append(1.0 if denom == 0 else len(sa & sb) / denom)
    return float(np.mean(vals))


def consistency_index(
    method_scores: FeatureScores,
    *,
    alpha: float = 0.5,
    k: int = 8,
    use_abs: bool = True,
) -> dict[str, float]:
    """Return CI components for one alpha value."""
    alpha = float(alpha)
    rank = ci_rank(method_scores, use_abs=use_abs)
    topk = ci_topk(method_scores, k=k, use_abs=use_abs)
    ci = alpha * rank + (1.0 - alpha) * topk
    return {"CI": float(ci), "CI_rank": rank, "CI_topk": topk, "alpha": alpha}


def alpha_sweep(
    method_scores: FeatureScores,
    *,
    k: int = 8,
    step: float = 0.05,
    use_abs: bool = True,
) -> list[dict[str, float]]:
    """Compute CI(alpha) from 0 to 1."""
    alphas = np.arange(0.0, 1.0 + step / 2.0, step)
    return [
        consistency_index(method_scores, alpha=float(a), k=k, use_abs=use_abs)
        for a in alphas
    ]


if __name__ == "__main__":
    demo_scores = {
        "svm": {"betaP": 0.8, "line_length": 0.7, "spec_entropy": 0.2},
        "rf": {"betaP": 0.9, "line_length": 0.5, "hjorth_complexity": 0.4},
        "perm": {"line_length": 0.9, "betaP": 0.6, "spec_slope_4_40": 0.5},
    }
    for row in alpha_sweep(demo_scores, k=2, step=0.25):
        print(row)

