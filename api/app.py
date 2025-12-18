"""
FastAPI service for flight delay classification.
Loads the trained model and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from flightdelays_pipeline import build_preprocessing, make_estimator_for_name


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# 你在 02/03 里保存的最终模型文件名（按你现在的命名）
MODEL_PATH = Path("/app/models/global_best_flightdelays_optuna.pkl")

# 你最终用于训练/预测的特征列（要和 Streamlit & 02/03 一致）
REQUIRED_COLUMNS = [
    "schedtime",
    "distance",
    "weather",
    "dayweek",
    "daymonth",
    "flightnumber",
    "carrier",
    "origin",
    "dest",
]

app = FastAPI(
    title="Flight Delay Classification API",
    description="FastAPI service for predicting whether a flight is delayed (binary classification).",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    if hasattr(m, "named_steps"):
        print(f"  Pipeline steps: {list(m.named_steps.keys())}")
    return m


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"✗ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"  Error: {e}")
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Prediction request with list of instances (dicts of features).
    """
    instances: List[Dict[str, Any]] = Field(..., description="List of feature dicts for prediction")

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "schedtime": 930,
                        "distance": 2475,
                        "weather": 0,
                        "dayweek": 3,
                        "daymonth": 15,
                        "flightnumber": 1234,
                        "carrier": "AA",
                        "origin": "JFK",
                        "dest": "LAX",
                    }
                ]
            }
        }


class PredictionItem(BaseModel):
    """
    Return both predicted label and (if available) probability of delay.
    - pred: 0/1
    - label: "ontime"/"delayed"
    - prob_delayed: optional float (P(y=1))
    """
    pred: int
    label: str
    prob_delayed: Optional[float] = None


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"pred": 1, "label": "delayed", "prob_delayed": 0.73}
                ],
                "count": 1,
            }
        }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Flight Delay Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
        "model_path": str(MODEL_PATH),
        "required_columns": REQUIRED_COLUMNS,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided. Please provide at least one instance.",
        )

    # 1) Convert to DataFrame
    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format. Could not convert to DataFrame: {e}",
        )

    # 2) Validate columns
    missing = set(REQUIRED_COLUMNS) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    # (可选) 只保留需要的列，避免用户多传字段导致顺序/类型混乱
    X = X[REQUIRED_COLUMNS].copy()

    # 3) Predict class
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    # 4) Predict probability if supported
    prob_delayed: Optional[List[float]] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # proba shape: (n, 2) => [:, 1] is P(y=1)
            prob_delayed = [float(p) for p in proba[:, 1]]
        except Exception:
            prob_delayed = None

    # 5) Build response items
    items: List[PredictionItem] = []
    for i, p in enumerate(preds):
        p_int = int(p)
        label = "delayed" if p_int == 1 else "ontime"
        item = PredictionItem(
            pred=p_int,
            label=label,
            prob_delayed=(prob_delayed[i] if prob_delayed is not None else None),
        )
        items.append(item)

    return PredictResponse(predictions=items, count=len(items))


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Flight Delay Classification API - Starting Up")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print(f"Required columns: {REQUIRED_COLUMNS}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")
