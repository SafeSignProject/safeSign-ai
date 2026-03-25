"""평가 지표 모듈 – 분류, 다중 레이블, 회귀 지표 통합."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
)


def compute_clause_type_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Head A: Clause type classification 지표."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_risk_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Head B: Multi-label risk classification 지표."""
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_risk_score_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, score_names: List[str]
) -> Dict[str, float]:
    """Head C: Risk score regression 지표."""
    result: Dict[str, float] = {}
    for i, name in enumerate(score_names):
        t, p = y_true[:, i], y_pred[:, i]
        result[f"{name}_mae"] = float(mean_absolute_error(t, p))
        result[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(t, p)))
    # 전체 MAE
    result["overall_mae"] = float(mean_absolute_error(y_true, y_pred))
    result["overall_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return result


def format_metrics_report(
    type_metrics: Dict[str, float],
    label_metrics: Dict[str, float],
    score_metrics: Dict[str, float],
) -> str:
    """지표를 사람이 읽기 좋은 형태로 포매팅."""
    lines = [
        "=" * 60,
        "  Clause Type Classification (Head A)",
        "=" * 60,
    ]
    for k, v in type_metrics.items():
        lines.append(f"  {k:20s}: {v:.4f}")

    lines += [
        "",
        "=" * 60,
        "  Risk Label Classification (Head B)",
        "=" * 60,
    ]
    for k, v in label_metrics.items():
        lines.append(f"  {k:20s}: {v:.4f}")

    lines += [
        "",
        "=" * 60,
        "  Risk Score Regression (Head C)",
        "=" * 60,
    ]
    for k, v in score_metrics.items():
        lines.append(f"  {k:25s}: {v:.4f}")

    return "\n".join(lines)
