"""멀티태스크 손실 함수 – classification + multi-label + regression 통합."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


class MultiTaskLoss(nn.Module):
    """3-head 통합 손실 함수.

    - Head A: CrossEntropy (or Focal) for clause type
    - Head B: BCEWithLogitsLoss for risk labels
    - Head C: MSELoss for risk scores
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        num_clause_types: int,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w_type = loss_weights.get("clause_type", 1.0)
        self.w_label = loss_weights.get("risk_label", 1.0)
        self.w_score = loss_weights.get("risk_score", 0.5)

        if use_focal_loss:
            self.type_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif class_weights is not None:
            self.type_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.type_loss_fn = nn.CrossEntropyLoss()

        self.label_loss_fn = nn.BCEWithLogitsLoss()
        self.score_loss_fn = nn.MSELoss()

    def forward(
        self,
        clause_type_logits: torch.Tensor,
        risk_label_logits: torch.Tensor,
        risk_scores_pred: torch.Tensor,
        clause_type_target: torch.Tensor,
        risk_labels_target: torch.Tensor,
        risk_scores_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_type = self.type_loss_fn(clause_type_logits, clause_type_target)
        loss_label = self.label_loss_fn(risk_label_logits, risk_labels_target)
        loss_score = self.score_loss_fn(risk_scores_pred, risk_scores_target)

        total = self.w_type * loss_type + self.w_label * loss_label + self.w_score * loss_score

        return {
            "total": total,
            "clause_type_loss": loss_type,
            "risk_label_loss": loss_label,
            "risk_score_loss": loss_score,
        }
