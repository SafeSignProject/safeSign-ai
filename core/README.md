[README.md](https://github.com/user-attachments/files/26235832/README.md)
# 주거용 임대차 계약서 독소조항 탐지 시스템

## Deep Learning Pipeline v1.0

주거용 임대차 계약서에서 조항 단위로 위험을 탐지하는 딥러닝 파이프라인입니다.
조항 유형 분류, 위험 유형 탐지, 위험 점수 산출, 중요 문장 추출까지 하나의 모델로 수행합니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **대상** | 주거용 임대차 계약서 |
| **사용자** | 임차인 (계약 체결 전 사전 점검) |
| **입력** | 조항 단위 텍스트 (향후 PDF/이미지 OCR 확장 가능) |
| **출력** | 조항 유형, 위험 레이블, 위험 점수, 중요 문장, 추론 JSON |

### 핵심 특징
- **조항 단위 분석**: 문장 단독이 아닌 조항 맥락에서 위험 판단
- **문장 중요도 게이팅**: 조항 내 모든 문장이 동일 중요도가 아님을 반영
- **멀티태스크 학습**: 유형 분류 + 위험 탐지 + 점수 회귀를 동시 학습
- **확장 가능 설계**: config 기반으로 클래스/가중치/모델 변경 용이

---

## 2. 모델 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                    입력 (Input)                      │
│  조항 텍스트 → 문장 분리 → 문장별 토큰화              │
│  shape: (batch, num_sentences, seq_len)              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            1. 문장 인코딩 (Sentence Encoder)          │
│                                                      │
│  각 문장 → Transformer backbone (koELECTRA)          │
│  → [CLS] 토큰의 hidden state 추출                    │
│  → Linear projection → sentence embedding            │
│                                                      │
│  출력: (batch, num_sentences, hidden_dim=768)        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│     2. 게이트 기반 중요도 계산 (Gated Aggregation)     │
│                                                      │
│  각 문장 벡터 → MLP(768 → 256 → 1)                   │
│  → Sigmoid → gate score (0~1)                        │
│                                                      │
│  패딩 문장은 mask로 gate=0 처리                       │
│  정규화된 gate 값으로 가중합 수행                      │
│                                                      │
│  출력: gate_weights (batch, num_sentences)            │
│        → 중요 문장 추출에도 활용                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         3. 조항 벡터 생성 (Clause Vector)              │
│                                                      │
│  clause_vec = Σ (gate_weight_i × sentence_vec_i)     │
│                                                      │
│  gate 가중합으로 단일 조항 representation 생성          │
│  출력: (batch, hidden_dim=768)                        │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Head A     │ │   Head B     │ │   Head C     │
│ Clause Type  │ │ Risk Labels  │ │ Risk Scores  │
│              │ │              │ │              │
│ MLP → 13cls │ │ MLP → 3cls  │ │ MLP → 4val  │
│ softmax     │ │ sigmoid     │ │ sigmoid     │
│ (단일 레이블) │ │ (다중 레이블) │ │ (0~1 회귀)  │
└──────────────┘ └──────────────┘ └──────────────┘
```

### 설계 근거

**왜 문장 단위 인코딩 + 게이트 집계인가?**

독소조항 여부는 조항 내 특정 문장에 의해 결정되는 경우가 많습니다.
예를 들어 "보증금은 5천만원으로 한다"는 안전하지만, 같은 조항 내
"어떠한 경우에도 반환하지 아니한다"가 위험의 핵심입니다.
게이트 메커니즘이 이 문장에 높은 가중치를 부여하여 조항 벡터에 반영하고,
동시에 사용자에게 "왜 위험한지" 근거 문장을 제시할 수 있습니다.

**왜 멀티태스크인가?**

조항 유형(Head A)은 위험 판단의 맥락 정보가 됩니다.
"보증금 조항"과 "위약금 조항"에서 동일 문구도 위험도가 다를 수 있으며,
공유 representation을 통해 이러한 상호작용을 학습합니다.

---

## 3. 데이터 포맷

### JSONL 스키마

```json
{
  "clause_id": "clause_0001",
  "contract_id": "contract_001",
  "clause_text": "보증금은 어떠한 경우에도 반환하지 아니한다.",
  "clause_type": "deposit",
  "risk_labels": [1, 1, 0],
  "legality_score": 0.95,
  "unfairness_score": 0.95,
  "ambiguity_score": 0.10,
  "overall_risk": 0.95
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `clause_id` | str | 조항 고유 ID |
| `contract_id` | str | 계약서 ID |
| `clause_text` | str | 조항 전체 텍스트 |
| `clause_type` | str | 조항 유형 (13개 클래스) |
| `risk_labels` | List[int] | [legal_violation, unfair_clause, ambiguity] |
| `legality_score` | float | 법적 위반 점수 (0~1) |
| `unfairness_score` | float | 불공정성 점수 (0~1) |
| `ambiguity_score` | float | 모호성 점수 (0~1) |
| `overall_risk` | float | 종합 위험 점수 (0~1) |

### 조항 유형 클래스 (13개)

`contract_period`, `deposit`, `rent_fee`, `termination`, `renewal`,
`restoration`, `repair`, `damages`, `penalty`, `deposit_return`,
`special_terms`, `defect`, `other`

### 위험 레이블 클래스 (3개)

- `legal_violation`: 법률 위반 (주택임대차보호법 등)
- `unfair_clause`: 불공정 조항 (일방적 권리 제한)
- `ambiguity`: 모호한 표현 (해석 여지)

---

## 4. 학습 방법

### 환경 준비

```bash
pip install -r requirements.txt
```

필수 패키지: `torch>=2.0`, `transformers>=4.30`, `scikit-learn`, `pyyaml`, `fastapi`, `uvicorn`

### 데이터 생성 & 학습

```bash
# 1. toy 데이터 생성 + 학습 (한 번에)
cd project/
python scripts/train.py --generate-data

# 2. 커스텀 config로 학습
python scripts/train.py --config configs/default_config.yaml
```

### 학습 설정 변경

`configs/default_config.yaml`에서 수정:

```yaml
training:
  epochs: 30
  batch_size: 8
  learning_rate: 2.0e-5
  early_stopping_patience: 5

loss_weights:
  clause_type: 1.0
  risk_label: 1.0
  risk_score: 0.5
```

### Loss 구성

| Head | Loss Function | 대안 |
|------|--------------|------|
| A. Clause Type | CrossEntropyLoss | FocalLoss (config에서 전환) |
| B. Risk Labels | BCEWithLogitsLoss | - |
| C. Risk Scores | MSELoss | - |

총 Loss = w_A × L_A + w_B × L_B + w_C × L_C (가중치 config에서 조절)

### Class Imbalance 대응

- `loss_options.use_focal_loss: true`로 Focal Loss 활성화
- `loss_options.use_class_weights: true`로 클래스 가중치 적용
- 평가 기준: Macro F1 (소수 클래스 성능 반영)

---

## 5. 추론 방법

### CLI 추론

```bash
# 샘플 데이터로 추론
python scripts/inference.py

# 단일 조항 추론
python scripts/inference.py --text "보증금은 어떠한 경우에도 반환하지 아니한다."

# 결과 파일 지정
python scripts/inference.py --output outputs/my_results.json
```

### 추론 결과 JSON 형식

```json
{
  "clause_text": "임대인은 별도의 사유 없이 언제든지 본 계약을 해지할 수 있으며...",
  "clause_type_pred": "termination",
  "clause_type_confidence": 0.91,
  "risk_labels_pred": {
    "legal_violation": 0.87,
    "unfair_clause": 0.92,
    "ambiguity": 0.15
  },
  "risk_scores": {
    "legality_score": 0.85,
    "unfairness_score": 0.90,
    "ambiguity_score": 0.18,
    "overall_risk": 0.88
  },
  "important_sentences": [
    {
      "text": "임대인은 별도의 사유 없이 언제든지 본 계약을 해지할 수 있으며",
      "weight": 0.72
    },
    {
      "text": "임차인은 통보 수령 후 7일 이내에 퇴거하여야 한다",
      "weight": 0.28
    }
  ]
}
```

---

## 6. API 실행 방법

### 서버 시작

```bash
# 프로젝트 루트에서
cd project/
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/predict_clause` | 단일 조항 분석 |
| POST | `/predict_clauses` | 배치 조항 분석 |

### 요청 예시

```bash
# 단일 조항
curl -X POST http://localhost:8000/predict_clause \
  -H "Content-Type: application/json" \
  -d '{"clause_text": "보증금은 어떠한 경우에도 반환하지 아니한다."}'

# 배치 조항
curl -X POST http://localhost:8000/predict_clauses \
  -H "Content-Type: application/json" \
  -d '{"clauses": ["조항1 텍스트...", "조항2 텍스트..."]}'
```

### Swagger UI

서버 실행 후 `http://localhost:8000/docs`에서 인터랙티브 API 문서 확인 가능.

---

## 7. 프로젝트 구조

```
project/
├── configs/
│   └── default_config.yaml      # 모든 설정 (모델, 학습, 데이터, API)
├── data/                         # 학습/검증/테스트 JSONL 데이터
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── src/
│   ├── datasets/
│   │   ├── clause_dataset.py    # PyTorch Dataset + DataLoader
│   │   └── toy_data_generator.py # 더미 데이터 생성기
│   ├── models/
│   │   └── clause_risk_model.py # 핵심 모델 (Encoder + Gate + Heads)
│   ├── training/
│   │   ├── losses.py            # 멀티태스크 Loss (Focal Loss 포함)
│   │   └── trainer.py           # 학습 루프 + 체크포인트 + 평가
│   ├── inference/
│   │   └── engine.py            # 추론 엔진 (단일/배치 + JSON 출력)
│   ├── api/
│   │   └── app.py               # FastAPI 서버
│   └── utils/
│       ├── config_loader.py     # YAML config dot-access 래퍼
│       ├── helpers.py           # 시드 고정, 디바이스, 로깅
│       ├── metrics.py           # 평가 지표 (F1, MAE 등)
│       └── sentence_splitter.py # 한국어 문장 분리
├── scripts/
│   ├── train.py                 # 학습 실행 스크립트
│   ├── inference.py             # 추론 실행 스크립트
│   └── validate_components.py   # 비ML 컴포넌트 검증
├── tests/
│   └── test_pipeline.py         # pytest 기반 테스트
├── checkpoints/                  # 모델 체크포인트
├── logs/                         # 학습 로그
├── outputs/                      # 추론 결과, 평가 리포트
├── requirements.txt
└── README.md
```

---

## 8. 향후 확장 포인트

### 모델 확장
- **Advanced Gate**: 현재 MLP 게이트 → self-attention 기반 inter-sentence attention으로 확장
- **Clause-Class Similarity**: clause embedding과 class embedding의 유사도 기반 분류
- **Cross-Clause Context**: 계약서 전체 맥락을 반영하는 document-level attention
- **Auxiliary Features**: 조항 유형 보조 feature를 clause vector에 concat하여 risk prediction 보강

### 데이터 확장
- **실제 계약서 데이터**: 법률 전문가 라벨링 데이터 확보
- **Data Augmentation**: 동의어 치환, 문장 순서 셔플 등
- **Active Learning**: 모델 불확실성 기반 라벨링 우선순위 지정

### 입력 확장
- **PDF 파서**: PyMuPDF / pdfplumber 기반 PDF 텍스트 추출
- **OCR 파이프라인**: Tesseract / CLOVA OCR → 텍스트 추출 → 조항 세그멘테이션
- **Clause Segmentation**: 전체 문서에서 조항 단위 분리 모듈

### 출력 확장
- **위험 근거 설명 생성**: LLM 기반 위험 이유 자연어 설명
- **보완 특약 추천**: 위험 유형별 보호 조항 추천
- **체크리스트 생성**: 계약 전 확인 사항 자동 생성
- **하이라이트 시각화**: 프론트엔드 연동 위험 조항 시각 표시

---

## 9. 현재 한계

1. **Toy 데이터**: 200개 더미 데이터로는 실제 성능을 기대할 수 없음. 실제 계약서 데이터 확보 필수.
2. **Backbone 제한**: koELECTRA는 일반 한국어에 사전학습되어 법률 도메인 특화가 부족함. 법률 코퍼스 추가 사전학습(domain adaptation) 권장.
3. **문장 분리 규칙 기반**: 현재 regex 기반으로 정교하지 않음. KSS 등 라이브러리 도입 또는 모델 기반 분리 권장.
4. **Gate 해석성**: gate weight가 반드시 "설명 가능성"을 보장하지 않음. attention 기반 해석에는 한계가 있으므로 SHAP/LIME 등 추가 해석 기법 고려.
5. **단일 조항 독립 분석**: 현재 조항 간 상호작용은 고려하지 않음. 계약서 전체 맥락이 중요한 경우가 있음.
6. **위험 점수 절대값 신뢰도**: 학습 데이터 양과 질에 따라 점수 자체의 절대적 의미 부여는 어려움. 상대적 비교로 활용 권장.

---

## 퀵 스타트

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 컴포넌트 검증 (torch 없이도 가능)
python scripts/validate_components.py

# 3. 데이터 생성 + 학습
python scripts/train.py --generate-data

# 4. 추론
python scripts/inference.py

# 5. API 서버
uvicorn src.api.app:app --port 8000
```
