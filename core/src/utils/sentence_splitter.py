"""문장 분리 유틸리티 – 한국어 법률/계약 텍스트에 특화."""
from __future__ import annotations

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """계약서 조항 텍스트를 문장 단위로 분리.

    규칙:
    1. '. ', '다. ', '요. ' 등 한국어 종결 패턴 기반 분리
    2. 번호매김(①②, 1., 가. 등) 기준 분리
    3. 빈 줄 기준 분리
    4. 단, 1문장 이하인 경우 그대로 반환
    """
    if not text or not text.strip():
        return []

    # 줄바꿈으로 먼저 분리
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    sentences: List[str] = []
    for line in lines:
        # 번호매김 패턴 분리: ①, ②, 1., 가. 등
        parts = re.split(r"(?<=[.!?])\s+(?=[①-⑳㉠-㉻\d가-힣]\.|[①-⑳])", line)
        for part in parts:
            # 한국어 종결어미 기반 분리
            subs = re.split(r"(?<=[다요음함됨임])[\.\s]+(?=[가-힣①-⑳\d])", part)
            for s in subs:
                s = s.strip()
                if s:
                    sentences.append(s)

    # 너무 짧은 조각 병합 (5자 미만)
    merged: List[str] = []
    for s in sentences:
        if merged and len(s) < 5:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    return merged if merged else [text.strip()]
