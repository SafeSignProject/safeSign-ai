"""독소조항 탐지 모델 – Sentence Encoder + Gated Aggregation + Multi-Task Heads.

Architecture:
    1. 입력: 조항 텍스트 → 문장 리스트 → 토큰 시퀀스
    2. 문장 인코딩: Transformer backbone (koELECTRA) → CLS pooling → sentence embedding
    3. 게이트 기반 중요도: 각 문장에 대해 0~1 gate score 계산
    4. 조항 벡터: gate-weighted sum으로 clause representation 생성
    5. 멀티태스크 출력:
       - Head A: Clause Type (softmax single-label)
       - Head B: Risk Labels (sigmoid multi-label)
       - Head C: Risk Scores (sigmoid regression, 0~1)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SentenceEncoder(nn.Module):
    """Transformer backbone으로 문장 임베딩 생성.

    [CLS] 토큰의 hidden state를 sentence vector로 사용.
    """

    def __init__(self, model_name: str, hidden_dim: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size
        # projection layer (backbone dim != hidden_dim일 때)
        self.proj = nn.Linear(backbone_dim, hidden_dim) if backbone_dim != hidden_dim else nn.Identity()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch * num_sents, seq_len)
            attention_mask: (batch * num_sents, seq_len)
        Returns:
            sentence_embeddings: (batch * num_sents, hidden_dim)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.proj(cls_output)


class GatedAggregation(nn.Module):
    """문장 중요도 게이트 → 가중합으로 조항 벡터 생성.

    각 문장 벡터에 대해 MLP → sigmoid로 0~1 gate 값 산출.
    gate 값으로 가중 평균을 수행하여 단일 clause vector 생성.
    """

    def __init__(self, hidden_dim: int, gate_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sentence_vectors: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sentence_vectors: (batch, num_sents, hidden_dim)
            sentence_mask: (batch, num_sents)  1=실제 문장, 0=패딩
        Returns:
            clause_vector: (batch, hidden_dim)
            gate_weights: (batch, num_sents)  각 문장의 중요도 가중치
        """
        gates = self.gate_net(sentence_vectors).squeeze(-1)  # (batch, num_sents)

        # 패딩 문장의 gate를 0으로 마스킹
        gates = gates * sentence_mask

        # 가중합 (정규화)
        gate_sum = gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
        gate_weights = gates / gate_sum  # (batch, num_sents)

        # weighted sum
        clause_vector = torch.bmm(
            gate_weights.unsqueeze(1), sentence_vectors
        ).squeeze(1)  # (batch, hidden_dim)

        return clause_vector, gate_weights


class ClauseTypeHead(nn.Module):
    """Head A: 조항 유형 분류 (single-label softmax)."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, clause_vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(clause_vector)  # (batch, num_classes) – logits


class RiskLabelHead(nn.Module):
    """Head B: 위험 레이블 분류 (multi-label sigmoid)."""

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, clause_vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(clause_vector)  # (batch, num_labels) – logits (sigmoid in loss)


class RiskScoreHead(nn.Module):
    """Head C: 위험 점수 회귀 (0~1 sigmoid)."""

    def __init__(self, hidden_dim: int, num_scores: int, dropout: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_scores),
            nn.Sigmoid(),
        )

    def forward(self, clause_vector: torch.Tensor) -> torch.Tensor:
        return self.regressor(clause_vector)  # (batch, num_scores) – 0~1


class ClauseRiskDetector(nn.Module):
    """독소조항 탐지 통합 모델.

    Pipeline:
        input_ids, attention_mask (batch, num_sents, seq_len)
        → SentenceEncoder → (batch, num_sents, hidden_dim)
        → GatedAggregation → clause_vector (batch, hidden_dim), gate_weights
        → 3 Task Heads → clause_type_logits, risk_label_logits, risk_scores
    """

    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int,
        gate_hidden_dim: int,
        num_clause_types: int,
        num_risk_labels: int,
        num_risk_scores: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(backbone_name, hidden_dim)
        self.gated_aggregation = GatedAggregation(hidden_dim, gate_hidden_dim, dropout)
        self.clause_type_head = ClauseTypeHead(hidden_dim, num_clause_types, dropout)
        self.risk_label_head = RiskLabelHead(hidden_dim, num_risk_labels, dropout)
        self.risk_score_head = RiskScoreHead(hidden_dim, num_risk_scores, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, num_sents, seq_len)
            attention_mask: (batch, num_sents, seq_len)
            sentence_mask: (batch, num_sents)
        Returns:
            dict with keys:
                clause_type_logits, risk_label_logits, risk_scores,
                gate_weights, sentence_vectors
        """
        batch_size, num_sents, seq_len = input_ids.shape

        # Flatten for encoder: (batch * num_sents, seq_len)
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Sentence encoding
        sent_embeds = self.sentence_encoder(flat_ids, flat_mask)  # (B*S, H)
        sent_embeds = sent_embeds.view(batch_size, num_sents, -1)  # (B, S, H)

        # Gated aggregation
        clause_vector, gate_weights = self.gated_aggregation(sent_embeds, sentence_mask)

        # Multi-task heads
        clause_type_logits = self.clause_type_head(clause_vector)
        risk_label_logits = self.risk_label_head(clause_vector)
        risk_scores = self.risk_score_head(clause_vector)

        return {
            "clause_type_logits": clause_type_logits,
            "risk_label_logits": risk_label_logits,
            "risk_scores": risk_scores,
            "gate_weights": gate_weights,
            "sentence_vectors": sent_embeds,
        }

    def freeze_backbone(self) -> None:
        """Backbone 파라미터 freeze (초기 학습 안정화용)."""
        for param in self.sentence_encoder.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Backbone 파라미터 unfreeze."""
        for param in self.sentence_encoder.backbone.parameters():
            param.requires_grad = True


def build_model(cfg) -> ClauseRiskDetector:
    """Config 기반으로 모델 빌드."""
    return ClauseRiskDetector(
        backbone_name=cfg.model.backbone,
        hidden_dim=cfg.model.hidden_dim,
        gate_hidden_dim=cfg.model.gate_hidden_dim,
        num_clause_types=len(cfg.clause_types),
        num_risk_labels=len(cfg.risk_labels),
        num_risk_scores=len(cfg.risk_score_names),
        dropout=cfg.model.dropout,
    )
