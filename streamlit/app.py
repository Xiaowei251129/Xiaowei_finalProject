import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Flight Delay Classification", page_icon="✈️", layout="centered")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# 你04生成的schema文件：flightdelays_schema.json
# Docker里建议把它挂到 /app/xiaowei_data/flightdelays_schema.json
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "/app/xiaowei_data/flightdelays_schema.json"))

# API_URL is set in docker-compose environment
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

schema = load_schema(SCHEMA_PATH)
numerical_features: Dict[str, Dict[str, float]] = schema.get("numerical", {})
categorical_features: Dict[str, Dict[str, Any]] = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _coerce_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _pretty_label(name: str) -> str:
    return name.replace("_", " ").title()

def _guess_step(min_val: float, max_val: float) -> float:
    r = max_val - min_val
    if r <= 10:
        return 1.0
    if r <= 100:
        return 1.0
    if r <= 1000:
        return 5.0
    if r <= 10000:
        return 10.0
    return 50.0

def _extract_prob(pred_obj: Any) -> Optional[float]:
    """
    Try to extract P(delayed) from different possible API return formats.
    Supports:
      - {"label": "delayed", "prob_delayed": 0.73}
      - {"pred": 1, "proba": [0.27, 0.73]}
      - {"prediction": "delayed", "probability": 0.73}
      - [0, 1] etc.
    """
    if isinstance(pred_obj, dict):
        for k in ("prob_delayed", "p_delayed", "probability", "proba_delayed"):
            if k in pred_obj:
                return _coerce_float(pred_obj[k], None)  # type: ignore
        # proba vector
        if "proba" in pred_obj and isinstance(pred_obj["proba"], list) and len(pred_obj["proba"]) >= 2:
            return _coerce_float(pred_obj["proba"][1], None)  # type: ignore
    return None

def _extract_label(pred_obj: Any) -> str:
    """
    Normalize prediction to a human-friendly label.
    Supports:
      - "delayed"/"ontime"
      - 1/0
      - {"label": "..."} / {"prediction": "..."} / {"pred": 1}
    """
    if isinstance(pred_obj, dict):
        for k in ("label", "prediction"):
            if k in pred_obj:
                return str(pred_obj[k])
        if "pred" in pred_obj:
            v = pred_obj["pred"]
            if isinstance(v, (int, float)):
                return "delayed" if int(v) == 1 else "ontime"
            return str(v)

    if isinstance(pred_obj, (int, float)):
        return "delayed" if int(pred_obj) == 1 else "ontime"

    return str(pred_obj)

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("✈️ Flight Delay Classification App")
st.write(f"This app sends your inputs to the FastAPI backend at **{API_BASE_URL}** for prediction.")

st.header("Input Features")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Numerical Features
# -----------------------------------------------------------------------------
st.subheader("Numerical Features")

# 对你的列做一个“更合理”的输入控件类型设置
# schedtime 常见是 0-2359；dayweek 1-7；daymonth 1-31；flightnumber 通常整数
INT_LIKE = {"schedtime", "dayweek", "daymonth", "flightnumber"}
# distance/weather 可能是连续或半连续
SLIDER_FEATURES = {"schedtime", "distance", "weather", "dayweek", "daymonth", "flightnumber"}

for feature_name, stats in numerical_features.items():
    min_val = _coerce_float(stats.get("min", 0.0))
    max_val = _coerce_float(stats.get("max", 100.0))
    mean_val = _coerce_float(stats.get("mean", (min_val + max_val) / 2))
    median_val = _coerce_float(stats.get("median", mean_val))
    default_val = median_val

    label = _pretty_label(feature_name)
    help_text = f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Median: {median_val:.2f}"

    if feature_name in SLIDER_FEATURES:
        step = 1.0 if feature_name in INT_LIKE else (_guess_step(min_val, max_val) / 10.0)

        val = st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=float(step),
            help=help_text,
            key=feature_name,
        )
        # 对“看起来应该是整数”的列强转为 int，避免API侧类型不一致
        user_input[feature_name] = int(round(val)) if feature_name in INT_LIKE else float(val)
    else:
        step = 1.0 if feature_name in INT_LIKE else _guess_step(min_val, max_val)
        val = st.number_input(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=float(step),
            help=help_text,
            key=feature_name,
        )
        user_input[feature_name] = int(round(val)) if feature_name in INT_LIKE else float(val)

# -----------------------------------------------------------------------------
# Categorical Features
# -----------------------------------------------------------------------------
st.subheader("Categorical Features")

for feature_name, info in categorical_features.items():
    unique_values: List[str] = info.get("unique_values", [])
    value_counts: Dict[str, int] = info.get("value_counts", {})

    if not unique_values:
        continue

    # 默认选最常见的值（更接近“真实输入分布”）
    if value_counts:
        default_value = max(value_counts, key=value_counts.get)
    else:
        default_value = unique_values[0]

    try:
        default_idx = unique_values.index(default_value)
    except ValueError:
        default_idx = 0

    user_input[feature_name] = st.selectbox(
        _pretty_label(feature_name),
        options=unique_values,
        index=default_idx,
        key=feature_name,
        help=f"Distribution: {value_counts}",
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Delay", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling API for prediction..."):
        try:
            resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request to API failed: {e}")
        else:
            if resp.status_code != 200:
                st.error(f"❌ API error: HTTP {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                preds = data.get("predictions", [])

                if not preds:
                    st.warning("⚠️ No predictions returned from API.")
                else:
                    pred0 = preds[0]
                    label = _extract_label(pred0)
                    prob_delayed = _extract_prob(pred0)

                    st.success("✅ Prediction successful!")
                    st.subheader("Prediction Result")

                    # 展示类别
                    label_norm = label.lower().strip()
                    if label_norm in {"1", "delayed", "delay", "true", "yes"}:
                        st.metric("Predicted Class", "DELAYED")
                    elif label_norm in {"0", "ontime", "on time", "no"}:
                        st.metric("Predicted Class", "ON TIME")
                    else:
                        st.metric("Predicted Class", label)

                    # 展示概率（如果API返回）
                    if prob_delayed is not None:
                        st.metric("Probability of Delay (P=1)", f"{prob_delayed:.3f}")

                    # 展示输入
                    with st.expander("📋 View Input Summary"):
                        st.json(user_input)

st.markdown("---")
st.caption(
    f"📁 Schema: `{SCHEMA_PATH}`  \n"
    f"🌐 API: `{API_BASE_URL}`  \n"
    f"🎯 Endpoint: `{PREDICT_ENDPOINT}`"
)
