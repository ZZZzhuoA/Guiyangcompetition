from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float


def stratified_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for fold_idx, sample_idx in enumerate(idx):
            folds[fold_idx % n_splits].append(int(sample_idx))

    all_idx = np.arange(len(y))
    splits = []
    for fold in folds:
        valid_idx = np.array(sorted(fold), dtype=int)
        train_idx = np.setdiff1d(all_idx, valid_idx)
        splits.append((train_idx, valid_idx))
    return splits


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(scores))


def cross_validate(models: list[object], x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> list[EvaluationResult]:
    splits = stratified_folds(y)
    results = []
    for model in models:
        accs = []
        f1s = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], y[train_idx], metric1[train_idx], metric2[train_idx])
            pred = fitted.predict(x[valid_idx])
            accs.append(float(np.mean(pred == y[valid_idx])))
            f1s.append(macro_f1(y[valid_idx], pred))
        results.append(
            EvaluationResult(
                model=model.name,
                accuracy_mean=float(np.mean(accs)),
                accuracy_std=float(np.std(accs)),
                macro_f1_mean=float(np.mean(f1s)),
                macro_f1_std=float(np.std(f1s)),
            )
        )
    return results
