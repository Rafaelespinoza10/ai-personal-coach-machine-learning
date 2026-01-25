import sys
from pathlib import Path

# Add project src for engineer_features
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

try:
    from utils.functions import engineer_features
except ImportError:
    engineer_features = None


def _preprocess_stress_input(features: Dict[str, Any], model_server) -> pd.DataFrame:
    """
    Build base row from features, apply engineer_features, then preprocess
    (impute + OHE) using fitted preprocessors. Transform only, no fit.
    """
    s = model_server.stress
    prep = s.get("preprocessors")
    if prep is None:
        raise RuntimeError(
            "Stress preprocessors not found. Re-train the main model with "
            "'use_selected=False' (all features) to generate preprocessors.pkl, "
            "then restart the API."
        )

    base_columns = prep.get("base_columns", [])
    base_numeric = prep.get("base_numeric", [])
    base_categorical = prep.get("base_categorical", [])
    numeric_cols = prep["numeric_cols"]
    categorical_cols = prep["categorical_cols"]
    numeric_imputer = prep["numeric_imputer"]
    categorical_imputer = prep["categorical_imputer"]
    ohe = prep["ohe"]

    # Build base 1-row DataFrame (same columns as X_train)
    base_row = {}
    for col in base_columns:
        val = features.get(col, None)
        if col in base_categorical:
            base_row[col] = "" if val is None else str(val).strip()
        else:
            n = pd.to_numeric(val, errors="coerce")
            base_row[col] = np.nan if pd.isna(n) else float(n)
    df_base = pd.DataFrame([base_row])

    # Engineer features (adds health_stress_index, sleep_efficiency, etc. when base cols exist)
    if engineer_features is not None:
        df_fe = engineer_features(df_base)
    else:
        df_fe = df_base.copy()

    # Ensure we have numeric_cols + categorical_cols; fill missing with 0 / ""
    for c in numeric_cols:
        if c not in df_fe.columns:
            df_fe[c] = 0.0
    for c in categorical_cols:
        if c not in df_fe.columns:
            df_fe[c] = ""

    # Numeric: same order as training
    numeric_values = []
    for col in numeric_cols:
        val = df_fe[col].iloc[0] if col in df_fe.columns else 0.0
        n = pd.to_numeric(val, errors="coerce")
        numeric_values.append(np.nan if pd.isna(n) else float(n))
    numeric_arr = np.array(numeric_values, dtype=float).reshape(1, -1)
    numeric_arr = np.nan_to_num(numeric_arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        numeric_imputed = numeric_imputer.transform(numeric_arr)
    except AttributeError as e:
        if "_fill_dtype" in str(e):
            numeric_imputed = numeric_arr
        else:
            raise
    df_numeric = pd.DataFrame(numeric_imputed, columns=numeric_cols)

    if ohe is not None and len(categorical_cols) > 0:
        stats = (
            categorical_imputer.statistics_
            if categorical_imputer is not None and hasattr(categorical_imputer, "statistics_")
            else [""] * len(categorical_cols)
        )
        cat_vals = []
        for i, col in enumerate(categorical_cols):
            val = df_fe[col].iloc[0] if col in df_fe.columns else ""
            v = "" if val is None else str(val).strip()
            fill = stats[i] if i < len(stats) else ""
            cat_vals.append(v if v else fill)
        cat_arr = np.array(cat_vals, dtype=object).reshape(1, -1)
        df_cat_in = pd.DataFrame(cat_arr, columns=categorical_cols)
        cat_enc = ohe.transform(df_cat_in)
        cat_names = ohe.get_feature_names_out(categorical_cols)
        df_cat = pd.DataFrame(cat_enc, columns=cat_names)
        df_processed = pd.concat(
            [df_numeric.reset_index(drop=True), df_cat.reset_index(drop=True)],
            axis=1,
        )
    else:
        df_processed = df_numeric.copy()

    return df_processed


def predict(model_server, features: Dict[str, Any]):
    if model_server.stress is None:
        raise RuntimeError("Stress model not loaded")

    s = model_server.stress
    prep = s.get("preprocessors")
    if prep is None:
        raise RuntimeError(
            "Stress preprocessors not found. Re-train the main model with "
            "'use_selected=False' (all features) to generate preprocessors.pkl, "
            "then restart the API."
        )

    model = s["model"]
    scaler = s["scaler"]
    mapping = s["config"].get("target_mapping", {"0": "Low", "1": "Medium+High"})

    df = _preprocess_stress_input(features, model_server)

    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    pred = int(model.predict(df)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])

    label = mapping.get(str(pred), "Medium+High" if pred == 1 else "Low")
    meta = _format_stress_for_human(label, proba)
    return {
        "stress_level": label,
        "probability": proba,
        "binary_class": pred,
        "message": meta["message"],
        "recommendation": meta["recommendation"],
        "display_metadata": meta["display_metadata"],
    }


def _format_stress_for_human(stress_level: str, probability: Optional[float]) -> dict:
    """Mensaje, recomendación y metadatos para UI según nivel de estrés y probabilidad."""
    is_low = stress_level == "Low"
    prob = probability if probability is not None else (0.3 if is_low else 0.7)

    if is_low:
        title = "Nivel de estrés bajo"
        color = "#4CAF50"
        icon = "check_circle"
        message = (
            "Tu nivel de estrés se encuentra en un rango bajo. "
            "Sigue cuidando tu sueño, actividad física y momentos de desconexión."
        )
        recommendation = (
            "Mantén tus rutinas de autocuidado y evita sobrecargarte. "
            "Pequeñas pausas durante el día ayudan a sostener este equilibrio."
        )
    else:
        if prob < 0.6:
            title = "Tendencia a estrés moderado o alto"
            color = "#FF9800"
            icon = "warning"
            message = (
                "El análisis sugiere una tendencia a estrés moderado o alto. "
                "Puede ser buen momento para revisar sueño, carga de trabajo y espacios de descanso."
            )
            recommendation = (
                "Prioriza el descanso y la actividad física ligera. "
                "Organiza tareas por prioridad y reserva tiempo para desconectar. "
                "Si lo necesitas, busca apoyo en tu entorno o en un profesional."
            )
        else:
            title = "Nivel de estrés moderado o alto"
            color = "#F44336"
            icon = "error"
            message = (
                "Se detecta un nivel de estrés moderado o alto. "
                "Es importante atenderlo: el sueño, la actividad y el manejo de la carga influyen mucho."
            )
            recommendation = (
                "Revisa tu sueño, horarios y momentos de descanso. "
                "Reduce la sobrecarga donde sea posible y mantén movimiento diario. "
                "Si el estrés te limita o afecta tu día a día, considera hablar con un profesional."
            )

    display_metadata = {"title": title, "color": color, "icon": icon}
    return {
        "message": message,
        "recommendation": recommendation,
        "display_metadata": display_metadata,
    }
