"""FastAPI 추론 API – 조항 위험도 분석 엔드포인트."""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config_loader import load_config
from src.inference.engine import load_inference_engine

# ============================================================
# Pydantic Schemas
# ============================================================

class ClauseRequest(BaseModel):
    clause_text: str = Field(..., description="분석할 조항 텍스트")


class ClauseBatchRequest(BaseModel):
    clauses: List[str] = Field(..., description="분석할 조항 텍스트 리스트")


class ImportantSentence(BaseModel):
    text: str
    weight: float


class ClauseResponse(BaseModel):
    clause_text: str
    clause_type_pred: str
    clause_type_confidence: float
    risk_labels_pred: dict
    risk_scores: dict
    important_sentences: List[ImportantSentence]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# App & Engine
# ============================================================

app = FastAPI(
    title="임대차 계약서 독소조항 탐지 API",
    description="주거용 임대차 계약서 조항의 위험도를 분석합니다.",
    version="0.1.0",
)

# 전역 엔진 (startup에서 초기화)
_engine = None


@app.on_event("startup")
async def startup():
    global _engine
    try:
        cfg = load_config()
        _engine = load_inference_engine(cfg)
        print("Inference engine loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load engine: {e}")
        print("API will start but predictions will fail until model is available.")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=_engine is not None)


@app.post("/predict_clause", response_model=ClauseResponse)
async def predict_clause(request: ClauseRequest):
    """단일 조항 위험도 분석."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = _engine.predict_single(request.clause_text)
        return ClauseResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_clauses", response_model=List[ClauseResponse])
async def predict_clauses(request: ClauseBatchRequest):
    """배치 조항 위험도 분석."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        results = _engine.predict_batch(request.clauses)
        return [ClauseResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
