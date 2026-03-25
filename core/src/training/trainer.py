"""학습 루프 – PyTorch 기반 Trainer with early stopping, checkpoint, eval."""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.models.clause_risk_model import ClauseRiskDetector
from src.training.losses import MultiTaskLoss
from src.utils.metrics import (
    compute_clause_type_metrics,
    compute_risk_label_metrics,
    compute_risk_score_metrics,
    format_metrics_report,
)
from src.utils.helpers import get_device


class Trainer:
    """멀티태스크 학습 Trainer."""

    def __init__(
        self,
        model: ClauseRiskDetector,
        criterion: MultiTaskLoss,
        cfg,
        logger=None,
    ):
        self.model = model
        self.criterion = criterion
        self.cfg = cfg
        self.device = get_device()
        self.logger = logger

        self.model.to(self.device)
        self.criterion.to(self.device)

        # Optimizer: backbone vs. heads 학습률 분리
        backbone_params = list(model.sentence_encoder.backbone.parameters())
        head_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("sentence_encoder.backbone")
        ]
        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": cfg.training.backbone_lr},
                {"params": head_params, "lr": cfg.training.learning_rate},
            ],
            weight_decay=cfg.training.weight_decay,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

        os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.paths.log_dir, exist_ok=True)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_losses = {"total": 0.0, "clause_type_loss": 0.0, "risk_label_loss": 0.0, "risk_score_loss": 0.0}
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentence_mask = batch["sentence_mask"].to(self.device)
            clause_type = batch["clause_type"].to(self.device)
            risk_labels = batch["risk_labels"].to(self.device)
            risk_scores = batch["risk_scores"].to(self.device)

            outputs = self.model(input_ids, attention_mask, sentence_mask)

            losses = self.criterion(
                clause_type_logits=outputs["clause_type_logits"],
                risk_label_logits=outputs["risk_label_logits"],
                risk_scores_pred=outputs["risk_scores"],
                clause_type_target=clause_type,
                risk_labels_target=risk_labels,
                risk_scores_target=risk_scores,
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
            self.optimizer.step()

            for k in total_losses:
                total_losses[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, risk_threshold: float = 0.5
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """검증/테스트 평가 – loss + metrics 반환."""
        self.model.eval()
        total_losses = {"total": 0.0, "clause_type_loss": 0.0, "risk_label_loss": 0.0, "risk_score_loss": 0.0}
        n_batches = 0

        all_type_true, all_type_pred = [], []
        all_label_true, all_label_pred = [], []
        all_score_true, all_score_pred = [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentence_mask = batch["sentence_mask"].to(self.device)
            clause_type = batch["clause_type"].to(self.device)
            risk_labels = batch["risk_labels"].to(self.device)
            risk_scores = batch["risk_scores"].to(self.device)

            outputs = self.model(input_ids, attention_mask, sentence_mask)

            losses = self.criterion(
                clause_type_logits=outputs["clause_type_logits"],
                risk_label_logits=outputs["risk_label_logits"],
                risk_scores_pred=outputs["risk_scores"],
                clause_type_target=clause_type,
                risk_labels_target=risk_labels,
                risk_scores_target=risk_scores,
            )

            for k in total_losses:
                total_losses[k] += losses[k].item()
            n_batches += 1

            # Predictions
            type_pred = outputs["clause_type_logits"].argmax(dim=-1).cpu().numpy()
            label_pred = (torch.sigmoid(outputs["risk_label_logits"]) > risk_threshold).int().cpu().numpy()
            score_pred = outputs["risk_scores"].cpu().numpy()

            all_type_true.append(clause_type.cpu().numpy())
            all_type_pred.append(type_pred)
            all_label_true.append(risk_labels.cpu().numpy().astype(int))
            all_label_pred.append(label_pred)
            all_score_true.append(risk_scores.cpu().numpy())
            all_score_pred.append(score_pred)

        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

        type_m = compute_clause_type_metrics(np.concatenate(all_type_true), np.concatenate(all_type_pred))
        label_m = compute_risk_label_metrics(np.concatenate(all_label_true), np.concatenate(all_label_pred))
        score_m = compute_risk_score_metrics(
            np.concatenate(all_score_true), np.concatenate(all_score_pred), self.cfg.risk_score_names
        )

        return avg_losses, type_m, label_m, score_m

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """전체 학습 루프."""
        epochs = self.cfg.training.epochs
        patience = self.cfg.training.early_stopping_patience
        freeze_epochs = self.cfg.model.freeze_backbone_epochs

        self._log(f"Training for max {epochs} epochs on {self.device}")
        self._log(f"Backbone frozen for first {freeze_epochs} epochs")

        if freeze_epochs > 0:
            self.model.freeze_backbone()

        for epoch in range(1, epochs + 1):
            # Unfreeze backbone after N epochs
            if epoch == freeze_epochs + 1:
                self.model.unfreeze_backbone()
                self._log("Backbone unfrozen!")

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, type_m, label_m, score_m = self.evaluate(val_loader)

            self._log(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss['total']:.4f} | "
                f"Val Loss: {val_loss['total']:.4f} | "
                f"Type F1: {type_m['macro_f1']:.4f} | "
                f"Risk F1: {label_m['macro_f1']:.4f} | "
                f"Score MAE: {score_m['overall_mae']:.4f}"
            )

            # Early stopping / checkpointing
            if val_loss["total"] < self.best_val_loss:
                self.best_val_loss = val_loss["total"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    self._log(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                    break

        self._log(f"Training complete. Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        path = os.path.join(self.cfg.paths.checkpoint_dir, "last_model.pt")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.cfg.paths.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)
            self._log(f"  Best model saved (epoch {epoch})")

    def load_best_model(self) -> None:
        path = os.path.join(self.cfg.paths.checkpoint_dir, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._log(f"Loaded best model from epoch {checkpoint['epoch']}")
