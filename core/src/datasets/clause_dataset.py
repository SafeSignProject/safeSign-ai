"""조항 단위 데이터셋 – 문장 분리 + 토크나이징 + 배치 콜레이트."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.utils.sentence_splitter import split_sentences


class ClauseDataset(Dataset):
    """JSONL 기반 조항 데이터셋.

    각 샘플은:
    - 조항 텍스트 → 문장 분리 → 문장별 토큰화
    - clause_type(단일 레이블), risk_labels(다중 레이블), risk_scores(회귀 타겟)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        clause_type_list: List[str],
        risk_label_list: List[str],
        risk_score_names: List[str],
        max_sentences: int = 10,
        max_tokens: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.clause_type_list = clause_type_list
        self.risk_label_list = risk_label_list
        self.risk_score_names = risk_score_names
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens

        self.type2idx = {t: i for i, t in enumerate(clause_type_list)}
        self.samples = self._load(data_path)

    def _load(self, path: str) -> List[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        clause_text = sample["clause_text"]

        # --- 문장 분리 ---
        sentences = split_sentences(clause_text)
        sentences = sentences[: self.max_sentences]
        num_sents = len(sentences)

        # --- 문장별 토큰화 ---
        input_ids_list = []
        attention_mask_list = []

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

        # 부족한 문장은 패딩
        pad_input = torch.zeros(self.max_tokens, dtype=torch.long)
        pad_mask = torch.zeros(self.max_tokens, dtype=torch.long)
        while len(input_ids_list) < self.max_sentences:
            input_ids_list.append(pad_input.clone())
            attention_mask_list.append(pad_mask.clone())

        input_ids = torch.stack(input_ids_list)          # (max_sentences, max_tokens)
        attention_mask = torch.stack(attention_mask_list)  # (max_sentences, max_tokens)
        sentence_mask = torch.zeros(self.max_sentences, dtype=torch.float)
        sentence_mask[:num_sents] = 1.0

        # --- Labels ---
        clause_type_idx = self.type2idx.get(sample["clause_type"], self.type2idx.get("other", 0))
        risk_labels = torch.tensor(sample["risk_labels"], dtype=torch.float)
        risk_scores = torch.tensor(
            [sample.get(name, 0.0) for name in self.risk_score_names],
            dtype=torch.float,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_mask": sentence_mask,
            "clause_type": torch.tensor(clause_type_idx, dtype=torch.long),
            "risk_labels": risk_labels,
            "risk_scores": risk_scores,
            "clause_text": clause_text,
            "sentences": sentences,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """커스텀 collate – 텍스트 필드는 리스트로, 텐서는 스택."""
    result: Dict = {}
    tensor_keys = ["input_ids", "attention_mask", "sentence_mask", "clause_type", "risk_labels", "risk_scores"]
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch])
    result["clause_text"] = [b["clause_text"] for b in batch]
    result["sentences"] = [b["sentences"] for b in batch]
    return result


def create_dataloaders(
    cfg,
    tokenizer_name: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Config 기반으로 train/val/test DataLoader 생성."""
    common_kwargs = dict(
        tokenizer_name=tokenizer_name,
        clause_type_list=cfg.clause_types,
        risk_label_list=cfg.risk_labels,
        risk_score_names=cfg.risk_score_names,
        max_sentences=cfg.model.max_sentences_per_clause,
        max_tokens=cfg.model.max_tokens_per_sentence,
    )

    train_ds = ClauseDataset(data_path=cfg.data.train_path, **common_kwargs)
    val_ds = ClauseDataset(data_path=cfg.data.val_path, **common_kwargs)
    test_ds = ClauseDataset(data_path=cfg.data.test_path, **common_kwargs)

    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
