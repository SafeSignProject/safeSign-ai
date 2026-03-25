#!/usr/bin/env python3
"""비ML 컴포넌트 검증 스크립트 – torch 없이도 실행 가능."""
from __future__ import annotations

import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    print("[1/5] Config 로딩 테스트...")
    from src.utils.config_loader import load_config
    cfg = load_config()
    assert len(cfg.clause_types) == 13, f"Expected 13 clause types, got {len(cfg.clause_types)}"
    assert len(cfg.risk_labels) == 3
    assert len(cfg.risk_score_names) == 4
    assert cfg.model.backbone == "monologg/koelectra-base-v3-discriminator"
    print("  ✓ Config 정상 로딩")
    return cfg


def test_sentence_splitter():
    print("[2/5] 문장 분리 테스트...")
    from src.utils.sentence_splitter import split_sentences

    # 빈 입력
    assert split_sentences("") == []
    assert split_sentences("   ") == []

    # 단일 문장
    result = split_sentences("본 계약은 민법에 따른다")
    assert len(result) == 1

    # 복수 문장
    result = split_sentences("보증금은 5천만원으로 한다. 잔금은 입주일에 지급한다.")
    assert len(result) >= 2, f"Expected >=2 sentences, got {len(result)}"

    # 복잡한 조항
    complex_text = (
        "임차인은 자연 마모를 포함한 일체의 손상을 원상복구하여야 하며, "
        "복구 비용은 전액 임차인이 부담한다. 복구 수준은 임대인이 판단한다."
    )
    result = split_sentences(complex_text)
    assert len(result) >= 2

    print(f"  ✓ 문장 분리 정상 ({len(result)} sentences)")


def test_data_generation():
    print("[3/5] 데이터 생성 테스트...")
    from src.datasets.toy_data_generator import generate_toy_dataset
    import tempfile

    tmpdir = tempfile.mkdtemp()
    samples = generate_toy_dataset(num_samples=50, output_dir=tmpdir, seed=42)
    assert len(samples) == 50

    for split in ["train", "val", "test"]:
        path = os.path.join(tmpdir, f"{split}.jsonl")
        assert os.path.exists(path), f"{split}.jsonl missing"
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) > 0
        sample = json.loads(lines[0])
        required_keys = ["clause_id", "contract_id", "clause_text", "clause_type",
                        "risk_labels", "legality_score", "unfairness_score",
                        "ambiguity_score", "overall_risk"]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    print(f"  ✓ 50개 샘플 생성, JSONL 스키마 검증 완료")


def test_data_distribution():
    print("[4/5] 데이터 분포 분석...")
    from src.datasets.toy_data_generator import generate_toy_dataset
    from collections import Counter

    samples = generate_toy_dataset(num_samples=200, seed=42)
    type_dist = Counter(s["clause_type"] for s in samples)
    risk_dist = Counter(tuple(s["risk_labels"]) for s in samples)

    print(f"  조항 유형 분포 (상위 5):")
    for ctype, count in type_dist.most_common(5):
        print(f"    {ctype}: {count}")

    risky_count = sum(1 for s in samples if any(s["risk_labels"]))
    safe_count = len(samples) - risky_count
    print(f"  위험 조항: {risky_count}, 안전 조항: {safe_count}")
    print(f"  ✓ 데이터 분포 확인 완료")


def test_metrics():
    print("[5/5] 평가 지표 테스트...")
    import numpy as np
    from src.utils.metrics import (
        compute_clause_type_metrics,
        compute_risk_label_metrics,
        compute_risk_score_metrics,
        format_metrics_report,
    )

    # Type metrics
    type_m = compute_clause_type_metrics(
        np.array([0, 1, 2, 0, 1]), np.array([0, 1, 2, 0, 2])
    )
    assert 0 <= type_m["accuracy"] <= 1

    # Label metrics
    label_m = compute_risk_label_metrics(
        np.array([[1, 0, 1], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, 1, 0]]),
    )
    assert "micro_f1" in label_m

    # Score metrics
    score_m = compute_risk_score_metrics(
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        np.array([[0.15, 0.25, 0.35, 0.45]]),
        ["legality", "unfairness", "ambiguity", "overall"],
    )
    assert score_m["overall_mae"] < 0.1

    # Report formatting
    report = format_metrics_report(type_m, label_m, score_m)
    assert "Head A" in report

    print(f"  ✓ 모든 지표 함수 정상")


def main():
    print("=" * 60)
    print("  독소조항 탐지 파이프라인 - 컴포넌트 검증")
    print("=" * 60)
    print()

    tests = [test_config, test_sentence_splitter, test_data_generation, test_data_distribution, test_metrics]
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"  결과: {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
