#!/usr/bin/env python3
"""학습 스크립트 – 데이터 생성 → 모델 학습 → 평가 → 체크포인트 저장."""
from __future__ import annotations

import sys
import os
import argparse

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.utils.helpers import set_seed, setup_logging
from src.utils.metrics import format_metrics_report
from src.datasets.toy_data_generator import generate_toy_dataset
from src.datasets.clause_dataset import create_dataloaders
from src.models.clause_risk_model import build_model
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Clause Risk Detector")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--generate-data", action="store_true", default=False, help="Generate toy dataset")
    args = parser.parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    logger = setup_logging(cfg.paths.log_dir)

    # --- Data Generation ---
    if args.generate_data or not os.path.exists(cfg.data.train_path):
        logger.info("Generating toy dataset...")
        generate_toy_dataset(
            num_samples=200,
            seed=cfg.training.seed,
            output_dir="data",
            split_ratios=tuple(cfg.data.train_val_test_split),
        )

    # --- DataLoaders ---
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg, cfg.model.backbone)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # --- Model ---
    logger.info("Building model...")
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # --- Loss ---
    loss_weights = cfg.loss_weights.to_dict() if hasattr(cfg.loss_weights, 'to_dict') else {
        "clause_type": cfg.loss_weights.clause_type,
        "risk_label": cfg.loss_weights.risk_label,
        "risk_score": cfg.loss_weights.risk_score,
    }
    criterion = MultiTaskLoss(
        loss_weights=loss_weights,
        num_clause_types=len(cfg.clause_types),
        use_focal_loss=cfg.loss_options.use_focal_loss,
        focal_alpha=cfg.loss_options.focal_alpha,
        focal_gamma=cfg.loss_options.focal_gamma,
    )

    # --- Train ---
    trainer = Trainer(model, criterion, cfg, logger)
    trainer.fit(train_loader, val_loader)

    # --- Final Evaluation ---
    logger.info("=" * 60)
    logger.info("Final evaluation on test set:")
    trainer.load_best_model()
    test_loss, type_m, label_m, score_m = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss['total']:.4f}")
    report = format_metrics_report(type_m, label_m, score_m)
    logger.info("\n" + report)

    # Save report
    report_path = os.path.join(cfg.paths.output_dir, "eval_report.txt")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
