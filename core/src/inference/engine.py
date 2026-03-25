"""추론 엔진 – 단일/배치 조항 예측 + 중요 문장 추출 + JSON 출력."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from src.models.clause_risk_model import ClauseRiskDetector
from src.utils.sentence_splitter import split_sentences
from src.utils.helpers import get_device


class ClauseInferenceEngine:
    """학습된 모델로 조항 위험도를 추론하는 엔진."""

    def __init__(
        self,
        model: ClauseRiskDetector,
        tokenizer_name: str,
        clause_types: List[str],
        risk_labels: List[str],
        risk_score_names: List[str],
        max_sentences: int = 10,
        max_tokens: int = 128,
        top_k_sentences: int = 3,
        risk_label_threshold: float = 0.5,
    ):
        self.device = get_device()
        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.clause_types = clause_types
        self.risk_labels = risk_labels
        self.risk_score_names = risk_score_names
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.top_k = top_k_sentences
        self.threshold = risk_label_threshold

    def _prepare_input(self, clause_text: str) -> dict:
        """단일 조항을 모델 입력 형태로 변환."""
        sentences = split_sentences(clause_text)
        sentences = sentences[: self.max_sentences]
        num_sents = len(sentences)

        input_ids_list, attention_mask_list = [], []
        for sent in sentences:
            enc = self.tokenizer(
                sent,
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attention_mask_list.append(enc["attention_mask"].squeeze(0))

        pad_input = torch.zeros(self.max_tokens, dtype=torch.long)
        pad_mask = torch.zeros(self.max_tokens, dtype=torch.long)
        while len(input_ids_list) < self.max_sentences:
            input_ids_list.append(pad_input.clone())
            attention_mask_list.append(pad_mask.clone())

        input_ids = torch.stack(input_ids_list).unsqueeze(0).to(self.device)
        attention_mask = torch.stack(attention_mask_list).unsqueeze(0).to(self.device)
        sentence_mask = torch.zeros(1, self.max_sentences, device=self.device)
        sentence_mask[0, :num_sents] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_mask": sentence_mask,
            "sentences": sentences,
            "num_sents": num_sents,
        }

    @torch.no_grad()
    def predict_single(self, clause_text: str) -> Dict:
        """단일 조항 추론 → JSON dict 반환."""
        prepared = self._prepare_input(clause_text)
        outputs = self.model(
            prepared["input_ids"],
            prepared["attention_mask"],
            prepared["sentence_mask"],
        )

        # Head A: Clause Type
        type_probs = torch.softmax(outputs["clause_type_logits"], dim=-1)[0]
        type_idx = type_probs.argmax().item()
        clause_type_pred = self.clause_types[type_idx]
        clause_type_conf = type_probs[type_idx].item()

        # Head B: Risk Labels
        label_probs = torch.sigmoid(outputs["risk_label_logits"])[0]
        risk_labels_pred = {
            name: round(label_probs[i].item(), 4) for i, name in enumerate(self.risk_labels)
        }

        # Head C: Risk Scores
        scores = outputs["risk_scores"][0]
        risk_scores_pred = {
            name: round(scores[i].item(), 4) for i, name in enumerate(self.risk_score_names)
        }

        # 중요 문장 추출
        gate_weights = outputs["gate_weights"][0].cpu()
        sentences = prepared["sentences"]
        num_sents = prepared["num_sents"]

        sent_weights = [(sentences[i], gate_weights[i].item()) for i in range(num_sents)]
        sent_weights.sort(key=lambda x: x[1], reverse=True)
        top_k = min(self.top_k, num_sents)
        important_sentences = [
            {"text": text, "weight": round(w, 4)} for text, w in sent_weights[:top_k]
        ]

        return {
            "clause_text": clause_text,
            "clause_type_pred": clause_type_pred,
            "clause_type_confidence": round(clause_type_conf, 4),
            "risk_labels_pred": risk_labels_pred,
            "risk_scores": risk_scores_pred,
            "important_sentences": important_sentences,
        }

    def predict_batch(self, clause_texts: List[str]) -> List[Dict]:
        """여러 조항 배치 추론."""
        return [self.predict_single(text) for text in clause_texts]

    def predict_to_json(self, clause_text: str) -> str:
        """추론 결과를 JSON 문자열로 반환."""
        result = self.predict_single(clause_text)
        return json.dumps(result, ensure_ascii=False, indent=2)


def load_inference_engine(cfg, checkpoint_path: Optional[str] = None) -> ClauseInferenceEngine:
    """Config와 체크포인트에서 추론 엔진 로드."""
    from src.models.clause_risk_model import build_model

    model = build_model(cfg)
    device = get_device()

    ckpt_path = checkpoint_path or cfg.inference.checkpoint_path
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
        except FileNotFoundError:
            print(f"Warning: Checkpoint {ckpt_path} not found. Using random weights.")

    return ClauseInferenceEngine(
        model=model,
        tokenizer_name=cfg.model.backbone,
        clause_types=cfg.clause_types,
        risk_labels=cfg.risk_labels,
        risk_score_names=cfg.risk_score_names,
        max_sentences=cfg.model.max_sentences_per_clause,
        max_tokens=cfg.model.max_tokens_per_sentence,
        top_k_sentences=cfg.inference.top_k_sentences,
        risk_label_threshold=cfg.inference.risk_label_threshold,
    )
