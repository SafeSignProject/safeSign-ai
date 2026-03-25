#!/usr/bin/env python3
"""추론 스크립트 – 학습된 모델로 조항 위험도 분석."""
from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.inference.engine import load_inference_engine


SAMPLE_CLAUSES = [
    "임대인은 별도의 사유 없이 언제든지 본 계약을 해지할 수 있으며, 임차인은 통보 수령 후 7일 이내에 퇴거하여야 한다.",
    "보증금은 금 오천만원(50,000,000원)으로 하며 계약 시 일천만원, 입주 시 사천만원을 납부한다.",
    "임대료는 임대인의 판단에 따라 사전 통보 없이 인상할 수 있으며, 임차인은 이에 이의를 제기할 수 없다.",
    "본 계약에 명시되지 않은 사항은 민법 및 주택임대차보호법에 따른다.",
    "특약사항: 임차인은 본 계약과 관련된 어떠한 분쟁에 대해서도 소송을 제기할 수 없으며, 임대인의 결정에 따른다.",
]


def main():
    parser = argparse.ArgumentParser(description="Clause Risk Inference")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--text", type=str, default=None, help="Single clause text")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = load_inference_engine(cfg, args.checkpoint)

    if args.text:
        clauses = [args.text]
    else:
        print("No --text provided. Using sample clauses.\n")
        clauses = SAMPLE_CLAUSES

    results = engine.predict_batch(clauses)

    # 출력
    for i, result in enumerate(results):
        print(f"\n{'='*70}")
        print(f"[조항 {i+1}]")
        print(f"텍스트: {result['clause_text'][:80]}...")
        print(f"유형: {result['clause_type_pred']} (확신도: {result['clause_type_confidence']:.2%})")
        print(f"위험 레이블: {result['risk_labels_pred']}")
        print(f"위험 점수: {result['risk_scores']}")
        print(f"중요 문장:")
        for s in result["important_sentences"]:
            print(f"  [{s['weight']:.3f}] {s['text'][:60]}")

    # JSON 저장
    output_path = args.output or os.path.join("outputs", "inference_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
