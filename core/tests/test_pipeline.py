#!/usr/bin/env python3
"""기본 단위 테스트 – 모델 구조, 데이터셋, 추론 파이프라인 검증."""
from __future__ import annotations

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def cfg():
    from src.utils.config_loader import load_config
    return load_config()


@pytest.fixture(scope="module")
def toy_data_dir(cfg):
    """임시 디렉토리에 toy 데이터 생성."""
    from src.datasets.toy_data_generator import generate_toy_dataset
    tmpdir = tempfile.mkdtemp()
    generate_toy_dataset(num_samples=30, seed=42, output_dir=tmpdir)
    return tmpdir


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def test_config_loads(self, cfg):
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "clause_types")
        assert hasattr(cfg, "training")

    def test_clause_types_defined(self, cfg):
        assert len(cfg.clause_types) >= 5
        assert "termination" in cfg.clause_types

    def test_risk_labels_defined(self, cfg):
        assert "legal_violation" in cfg.risk_labels
        assert "unfair_clause" in cfg.risk_labels


# ============================================================
# Sentence Splitter Tests
# ============================================================

class TestSentenceSplitter:
    def test_basic_split(self):
        from src.utils.sentence_splitter import split_sentences
        text = "보증금은 5천만원으로 한다. 임차인은 계약 시 납부한다."
        sents = split_sentences(text)
        assert len(sents) >= 1

    def test_empty_input(self):
        from src.utils.sentence_splitter import split_sentences
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        from src.utils.sentence_splitter import split_sentences
        sents = split_sentences("본 계약은 민법에 따른다")
        assert len(sents) == 1


# ============================================================
# Dataset Tests
# ============================================================

class TestDataset:
    def test_toy_data_generation(self, toy_data_dir):
        for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            path = os.path.join(toy_data_dir, name)
            assert os.path.exists(path), f"{name} not found"
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) > 0

    def test_dataset_loading(self, cfg, toy_data_dir):
        from src.datasets.clause_dataset import ClauseDataset
        ds = ClauseDataset(
            data_path=os.path.join(toy_data_dir, "train.jsonl"),
            tokenizer_name=cfg.model.backbone,
            clause_type_list=cfg.clause_types,
            risk_label_list=cfg.risk_labels,
            risk_score_names=cfg.risk_score_names,
            max_sentences=cfg.model.max_sentences_per_clause,
            max_tokens=cfg.model.max_tokens_per_sentence,
        )
        assert len(ds) > 0

        sample = ds[0]
        assert "input_ids" in sample
        assert sample["input_ids"].shape == (cfg.model.max_sentences_per_clause, cfg.model.max_tokens_per_sentence)
        assert sample["sentence_mask"].shape == (cfg.model.max_sentences_per_clause,)
        assert sample["risk_labels"].shape == (len(cfg.risk_labels),)
        assert sample["risk_scores"].shape == (len(cfg.risk_score_names),)


# ============================================================
# Model Tests
# ============================================================

class TestModel:
    def test_model_builds(self, cfg):
        from src.models.clause_risk_model import build_model
        model = build_model(cfg)
        assert model is not None

    def test_model_forward(self, cfg):
        from src.models.clause_risk_model import build_model
        model = build_model(cfg)
        model.eval()

        B, S, T = 2, cfg.model.max_sentences_per_clause, cfg.model.max_tokens_per_sentence
        input_ids = torch.randint(0, 1000, (B, S, T))
        attention_mask = torch.ones(B, S, T, dtype=torch.long)
        sentence_mask = torch.ones(B, S)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, sentence_mask)

        assert "clause_type_logits" in outputs
        assert outputs["clause_type_logits"].shape == (B, len(cfg.clause_types))
        assert outputs["risk_label_logits"].shape == (B, len(cfg.risk_labels))
        assert outputs["risk_scores"].shape == (B, len(cfg.risk_score_names))
        assert outputs["gate_weights"].shape == (B, S)

        # gate weights should be non-negative
        assert (outputs["gate_weights"] >= 0).all()

    def test_freeze_unfreeze(self, cfg):
        from src.models.clause_risk_model import build_model
        model = build_model(cfg)

        model.freeze_backbone()
        for p in model.sentence_encoder.backbone.parameters():
            assert not p.requires_grad

        model.unfreeze_backbone()
        for p in model.sentence_encoder.backbone.parameters():
            assert p.requires_grad


# ============================================================
# Loss Tests
# ============================================================

class TestLoss:
    def test_multitask_loss(self, cfg):
        from src.training.losses import MultiTaskLoss
        loss_weights = {
            "clause_type": cfg.loss_weights.clause_type,
            "risk_label": cfg.loss_weights.risk_label,
            "risk_score": cfg.loss_weights.risk_score,
        }
        criterion = MultiTaskLoss(
            loss_weights=loss_weights,
            num_clause_types=len(cfg.clause_types),
        )

        B = 4
        type_logits = torch.randn(B, len(cfg.clause_types))
        label_logits = torch.randn(B, len(cfg.risk_labels))
        score_pred = torch.sigmoid(torch.randn(B, len(cfg.risk_score_names)))
        type_target = torch.randint(0, len(cfg.clause_types), (B,))
        label_target = torch.randint(0, 2, (B, len(cfg.risk_labels))).float()
        score_target = torch.rand(B, len(cfg.risk_score_names))

        losses = criterion(type_logits, label_logits, score_pred, type_target, label_target, score_target)
        assert "total" in losses
        assert losses["total"].item() > 0


# ============================================================
# Metrics Tests
# ============================================================

class TestMetrics:
    def test_clause_type_metrics(self):
        import numpy as np
        from src.utils.metrics import compute_clause_type_metrics
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2])
        m = compute_clause_type_metrics(y_true, y_pred)
        assert "accuracy" in m
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_risk_label_metrics(self):
        import numpy as np
        from src.utils.metrics import compute_risk_label_metrics
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        m = compute_risk_label_metrics(y_true, y_pred)
        assert "micro_f1" in m

    def test_risk_score_metrics(self):
        import numpy as np
        from src.utils.metrics import compute_risk_score_metrics
        y_true = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        y_pred = np.array([[0.15, 0.25, 0.35, 0.45], [0.55, 0.65, 0.75, 0.85]])
        m = compute_risk_score_metrics(y_true, y_pred, ["a", "b", "c", "d"])
        assert "overall_mae" in m


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
