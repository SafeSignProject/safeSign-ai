"""더미 데이터 생성기 – 파이프라인 테스트용 임대차 계약서 조항 데이터 자동 생성."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======================================================================
# 조항 유형별 예시 텍스트 템플릿
# ======================================================================
CLAUSE_TEMPLATES: Dict[str, List[Tuple[str, dict]]] = {
    "contract_period": [
        (
            "본 계약의 임대차 기간은 2024년 3월 1일부터 2026년 2월 28일까지 2년으로 한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.05, 0.05]},
        ),
        (
            "임대차 기간은 계약일로부터 1년으로 하며, 임차인의 사정에 의한 기간 단축은 인정하지 아니한다. 기간 만료 전 퇴거 시 잔여 기간 임대료 전액을 위약금으로 납부하여야 한다.",
            {"risk_labels": [0, 1, 0], "scores": [0.3, 0.8, 0.2, 0.7]},
        ),
        (
            "계약 기간은 별도 협의에 의하며, 정해진 기간 없이 임대인이 정하는 바에 따른다.",
            {"risk_labels": [1, 1, 1], "scores": [0.8, 0.7, 0.9, 0.85]},
        ),
    ],
    "deposit": [
        (
            "보증금은 금 오천만원(50,000,000원)으로 하며 계약 시 일천만원, 입주 시 사천만원을 납부한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.1, 0.07]},
        ),
        (
            "보증금은 어떠한 경우에도 반환하지 아니한다.",
            {"risk_labels": [1, 1, 0], "scores": [0.95, 0.95, 0.1, 0.95]},
        ),
        (
            "보증금 반환 시기는 임대인이 신규 임차인을 구한 이후로 한다. 신규 임차인을 구하지 못한 경우 임차인은 보증금 반환을 청구할 수 없다.",
            {"risk_labels": [1, 1, 1], "scores": [0.7, 0.85, 0.6, 0.8]},
        ),
    ],
    "rent_fee": [
        (
            "월 임대료는 금 오십만원(500,000원)으로 하며 매월 25일까지 임대인 지정 계좌로 납부한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.05, 0.05]},
        ),
        (
            "임대료는 임대인의 판단에 따라 사전 통보 없이 인상할 수 있으며, 임차인은 이에 이의를 제기할 수 없다.",
            {"risk_labels": [1, 1, 0], "scores": [0.85, 0.9, 0.2, 0.85]},
        ),
        (
            "월 임대료를 3일 이상 연체할 경우 연 20%의 연체이자가 부과된다.",
            {"risk_labels": [0, 1, 0], "scores": [0.4, 0.7, 0.1, 0.55]},
        ),
    ],
    "termination": [
        (
            "임대인 또는 임차인은 계약 기간 만료 1개월 전 서면 통지로 계약을 해지할 수 있다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.1, 0.07]},
        ),
        (
            "임대인은 별도의 사유 없이 언제든지 본 계약을 해지할 수 있으며, 임차인은 통보 수령 후 7일 이내에 퇴거하여야 한다.",
            {"risk_labels": [1, 1, 0], "scores": [0.9, 0.9, 0.15, 0.9]},
        ),
        (
            "임차인의 계약 해지 시 보증금의 30%를 위약금으로 공제한다.",
            {"risk_labels": [0, 1, 0], "scores": [0.3, 0.75, 0.15, 0.6]},
        ),
    ],
    "renewal": [
        (
            "계약 만료 시 양 당사자 합의에 의해 재계약할 수 있다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.05, 0.05]},
        ),
        (
            "임차인의 계약갱신요구권은 본 계약에서 배제한다.",
            {"risk_labels": [1, 1, 0], "scores": [0.95, 0.85, 0.1, 0.9]},
        ),
    ],
    "restoration": [
        (
            "임차인은 계약 종료 시 임차 목적물을 원상복구하여 반환한다. 통상적인 사용에 따른 마모는 제외한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.1, 0.07]},
        ),
        (
            "임차인은 자연 마모를 포함한 일체의 손상을 원상복구하여야 하며, 복구 비용은 전액 임차인이 부담한다. 복구 수준은 임대인이 판단한다.",
            {"risk_labels": [0, 1, 1], "scores": [0.3, 0.85, 0.7, 0.75]},
        ),
    ],
    "repair": [
        (
            "임대인은 건물의 주요 구조 및 설비의 수선 의무를 진다. 소모품 교체는 임차인이 부담한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.1, 0.1, 0.08]},
        ),
        (
            "건물 내 모든 수리 및 유지보수는 사유를 불문하고 임차인이 전액 부담한다.",
            {"risk_labels": [1, 1, 0], "scores": [0.7, 0.85, 0.15, 0.75]},
        ),
    ],
    "damages": [
        (
            "임차인의 고의 또는 과실로 인한 손해는 임차인이 배상한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.1, 0.1, 0.08]},
        ),
        (
            "임대 목적물에 발생하는 모든 손해에 대해 원인을 불문하고 임차인이 배상 책임을 진다.",
            {"risk_labels": [0, 1, 1], "scores": [0.3, 0.8, 0.6, 0.7]},
        ),
    ],
    "penalty": [
        (
            "계약 위반 시 상대방에게 보증금의 10%에 해당하는 위약금을 지급한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.1, 0.15, 0.1, 0.12]},
        ),
        (
            "임차인이 본 계약의 어떠한 조항이라도 위반할 경우 보증금 전액을 몰수하며, 별도의 손해배상을 청구할 수 있다.",
            {"risk_labels": [1, 1, 0], "scores": [0.8, 0.9, 0.2, 0.85]},
        ),
    ],
    "deposit_return": [
        (
            "임대인은 계약 종료 후 14일 이내에 보증금을 반환한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.05, 0.05]},
        ),
        (
            "보증금 반환 시기는 임대인의 재정 상황에 따라 유동적으로 결정되며, 구체적인 반환 일자를 보장하지 않는다.",
            {"risk_labels": [1, 1, 1], "scores": [0.7, 0.8, 0.85, 0.8]},
        ),
    ],
    "special_terms": [
        (
            "특약사항: 반려동물 사육 불가. 입주 전 도배 및 장판 교체 임대인 부담.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.1, 0.1, 0.08]},
        ),
        (
            "특약사항: 임차인은 본 계약과 관련된 어떠한 분쟁에 대해서도 소송을 제기할 수 없으며, 임대인의 결정에 따른다.",
            {"risk_labels": [1, 1, 0], "scores": [0.95, 0.9, 0.2, 0.92]},
        ),
        (
            "특약사항: 임차인은 계약 기간 중 제3자에게 전대할 수 없다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.1, 0.05, 0.07]},
        ),
    ],
    "defect": [
        (
            "임차인은 입주 시 목적물의 하자를 점검하고, 발견된 하자를 7일 이내에 임대인에게 통보하여야 한다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.1, 0.1, 0.08]},
        ),
        (
            "입주 후 발견된 하자에 대해서는 임대인이 일체의 책임을 지지 아니한다.",
            {"risk_labels": [1, 1, 0], "scores": [0.7, 0.8, 0.15, 0.7]},
        ),
    ],
    "other": [
        (
            "본 계약에 명시되지 않은 사항은 민법 및 주택임대차보호법에 따른다.",
            {"risk_labels": [0, 0, 0], "scores": [0.05, 0.05, 0.05, 0.05]},
        ),
        (
            "본 계약에 정하지 아니한 사항은 임대인의 결정에 따르며, 임차인은 이에 동의한 것으로 간주한다.",
            {"risk_labels": [0, 1, 1], "scores": [0.4, 0.8, 0.7, 0.7]},
        ),
    ],
}


def generate_toy_dataset(
    num_samples: int = 200,
    seed: int = 42,
    output_dir: Optional[str] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> List[dict]:
    """더미 학습 데이터를 생성하여 JSONL로 저장."""
    random.seed(seed)

    samples: List[dict] = []
    clause_types = list(CLAUSE_TEMPLATES.keys())

    for i in range(num_samples):
        ctype = random.choice(clause_types)
        template_text, template_meta = random.choice(CLAUSE_TEMPLATES[ctype])

        # 약간의 변형 추가
        noise_scores = [
            max(0.0, min(1.0, s + random.gauss(0, 0.05)))
            for s in template_meta["scores"]
        ]

        sample = {
            "clause_id": f"clause_{i:04d}",
            "contract_id": f"contract_{i // 5:03d}",
            "clause_text": template_text,
            "clause_type": ctype,
            "risk_labels": template_meta["risk_labels"],
            "legality_score": round(noise_scores[0], 3),
            "unfairness_score": round(noise_scores[1], 3),
            "ambiguity_score": round(noise_scores[2], 3),
            "overall_risk": round(noise_scores[3], 3),
        }
        samples.append(sample)

    random.shuffle(samples)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n = len(samples)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train : n_train + n_val],
            "test": samples[n_train + n_val :],
        }

        for name, data in splits.items():
            path = out / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for s in data:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"  {name}: {len(data)} samples -> {path}")

    return samples


if __name__ == "__main__":
    print("Generating toy dataset...")
    generate_toy_dataset(num_samples=200, output_dir="data")
    print("Done!")
