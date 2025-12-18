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

# -----------------------
# Numerical (int) features
# -----------------------
st.subheader("Numerical Features")

def int_slider(name, min_v, max_v, default_v, help_text=""):
    return st.slider(
        name.replace("_", " ").title(),
        min_value=int(min_v),
        max_value=int(max_v),
        value=int(default_v),
        step=1,
        help=help_text,
        key=name,
    )

def int_select(name, options, default_v=None, help_text=""):
    label = name.replace("_", " ").title()
    if default_v is None:
        default_v = options[0]
    default_idx = options.index(default_v) if default_v in options else 0
    return st.selectbox(label, options=options, index=default_idx, help=help_text, key=name)

# 1) schedtime: 强制 0~2359
sched_stats = numerical_features.get("schedtime", {"min": 0, "max": 2359, "median": 1200})
user_input["schedtime"] = int_slider(
    "schedtime",
    0, 2359,
    sched_stats.get("median", 1200),
    help_text="Scheduled departure time in HHMM (0~2359)."
)

# 2) distance: int slider（范围来自 schema）
dist_stats = numerical_features.get("distance", {"min": 0, "max": 3000, "median": 500})
user_input["distance"] = int_slider(
    "distance",
    dist_stats.get("min", 0),
    dist_stats.get("max", 3000),
    dist_stats.get("median", 500),
    help_text=f"Min/Max from data: {dist_stats.get('min')}~{dist_stats.get('max')}"
)

# 3) dayweek: 1~7
user_input["dayweek"] = int_select("dayweek", list(range(1, 8)), default_v=1)

# 4) daymonth: 1~31
user_input["daymonth"] = int_select("daymonth", list(range(1, 32)), default_v=15)

# 5) flightnumber: int input（有些范围很大，用 number_input 更合适）
fn_stats = numerical_features.get("flightnumber", {"min": 1, "max": 9999, "median": 1000})
user_input["flightnumber"] = st.number_input(
    "Flightnumber",
    min_value=int(fn_stats.get("min", 1)),
    max_value=int(fn_stats.get("max", 9999)),
    value=int(fn_stats.get("median", 1000)),
    step=1,
    help="Flight number (integer).",
    key="flightnumber",
)

# -----------------------
# Categorical features
# -----------------------
st.subheader("Categorical Features")

# weather: 0/1 下拉框
weather_info = categorical_features.get("weather", {})
weather_options = weather_info.get("unique_values", [0, 1])
# 兼容 JSON 里可能存成字符串
weather_options = [int(x) for x in weather_options]

weather_label_map = {0: "0 (No weather delay)", 1: "1 (Weather delay)"}
weather_display = [weather_label_map.get(x, str(x)) for x in weather_options]
weather_choice = st.selectbox("Weather", options=weather_display, index=0, key="weather")
user_input["weather"] = 0 if weather_choice.startswith("0") else 1

# carrier/origin/dest：正常 selectbox
for feature_name in ["carrier", "origin", "dest"]:
    info = categorical_features.get(feature_name, {})
    unique_values = info.get("unique_values", [])
    value_counts = info.get("value_counts", {})

    if not unique_values:
        st.warning(f"Missing schema for {feature_name}.")
        continue

    default_value = max(value_counts, key=value_counts.get) if value_counts else unique_values[0]
    default_idx = unique_values.index(default_value) if default_value in unique_values else 0

    user_input[feature_name] = st.selectbox(
        feature_name.title(),
        options=unique_values,
        index=default_idx,
        key=feature_name,
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
                probs = data.get("probabilities", None)  # 如果你的 API 有返回概率

                if not preds:
                    st.warning("⚠️ No predictions returned from API.")
                else:
                    pred0 = preds[0]

                    # 1) 先抽概率（优先用 pred0 里带的；否则用 API 单独返回的 probabilities）
                    prob_delayed = _extract_prob(pred0)
                    if prob_delayed is None and isinstance(probs, list) and len(probs) > 0:
                        prob_delayed = _coerce_float(probs[0], None)  # type: ignore

                    # 2) 再抽 label（如果 API 给了字符串 label 就直接用）
                    label_raw = _extract_label(pred0).strip().lower()

                    # 3) 决策：优先用概率（因为你已经看到概率在变），否则用 label/raw 数字
                    if prob_delayed is not None:
                        pred_class = 1 if float(prob_delayed) >= 0.5 else 0
                    else:
                        # label_raw 可能是 "delayed"/"ontime"/"on time"/"0"/"1"
                        if label_raw in ("delayed", "delay", "1"):
                            pred_class = 1
                        elif label_raw in ("ontime", "on time", "on_time", "0"):
                            pred_class = 0
                        else:
                            # 最后兜底：如果 pred0 是数字
                            pred_class = int(pred0) if isinstance(pred0, (int, float)) else 0

                    pred_label = "DELAYED" if pred_class == 1 else "ON TIME"

                    st.success("✅ Prediction successful!")
                    st.subheader("Prediction Result")
                    st.metric("Predicted Class", pred_label)

                    if prob_delayed is not None:
                        st.metric("Probability of Delay (P=1)", f"{float(prob_delayed):.3f}")

                    with st.expander("📋 View Input Summary"):
                        st.json(user_input)




st.markdown("---")
st.caption(
    f"📁 Schema: `{SCHEMA_PATH}`  \n"
    f"🌐 API: `{API_BASE_URL}`  \n"
    f"🎯 Endpoint: `{PREDICT_ENDPOINT}`"
)
