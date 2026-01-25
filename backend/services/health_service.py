import numpy as np
import pandas as pd
from typing import Dict, List

def preprocess_mental_health_input(raw_data: dict, model_server):
    """
    Convert raw request data to a full processed DataFrame (numeric + OHE),
    using fitted preprocessors from training. Transform only, no fit.
    Selection of features happens after scaling in predict().
    """
    mh = model_server.mental_health
    prep = mh.get("preprocessors")
    if prep is None:
        raise RuntimeError(
            "Mental health preprocessors not found. Re-run mental health model "
            "training to generate preprocessors.pkl, then restart the API."
        )

    numeric_cols = prep["numeric_cols"]
    categorical_cols = prep["categorical_cols"]
    numeric_imputer = prep["numeric_imputer"]
    categorical_imputer = prep["categorical_imputer"]
    ohe = prep["ohe"]

    numeric_values = []
    for col in numeric_cols:
        val = raw_data.get(col, None)
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
            if categorical_imputer is not None
            and hasattr(categorical_imputer, "statistics_")
            else [""] * len(categorical_cols)
        )
        categorical_values = []
        for i, col in enumerate(categorical_cols):
            val = raw_data.get(col, None)
            v = "" if val is None else str(val).strip()
            fill = stats[i] if i < len(stats) else ""
            categorical_values.append(v if v else fill)
        cat_arr = np.array(categorical_values, dtype=object).reshape(1, -1)
        df_cat_input = pd.DataFrame(cat_arr, columns=categorical_cols)
        cat_encoded = ohe.transform(df_cat_input)
        cat_names = ohe.get_feature_names_out(categorical_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=cat_names)
        df_processed = pd.concat(
            [df_numeric.reset_index(drop=True), df_cat.reset_index(drop=True)],
            axis=1,
        )
    else:
        df_processed = df_numeric.copy()

    return df_processed


def interpret_indicator(name: str, score: float) -> Dict:
    """
    Interpreta un indicador de salud mental de manera profesional.
    Retorna nivel, severidad, descripción, recomendación y metadatos visuales.
    """
    # Definir umbrales profesionales (basados en escalas comunes de salud mental)
    thresholds = {
        "depression_level": {"bajo": 2.0, "moderado": 4.0, "alto": 6.0},
        "sleep_issues": {"bajo": 2.5, "moderado": 4.5, "alto": 6.5},
        "easily_distracted": {"bajo": 2.0, "moderado": 4.0, "alto": 6.0},
        "worry_level": {"bajo": 2.0, "moderado": 4.0, "alto": 6.0},
        "concentration_difficulty": {"bajo": 2.5, "moderado": 4.5, "alto": 6.5},
    }
    
    # Obtener umbrales para este indicador
    thresh = thresholds.get(name, {"bajo": 2.0, "moderado": 4.0, "alto": 6.0})
    
    # Determinar nivel
    if score < thresh["bajo"]:
        level = "Bajo"
        severity = "Baja"
        color = "#4CAF50"  # Verde
        icon = "check_circle"
        descriptions = {
            "depression_level": f"El nivel de depresión ({score:.2f}) se encuentra en un rango bajo. Esto indica un estado emocional relativamente estable.",
            "sleep_issues": f"Los problemas de sueño ({score:.2f}) están en un nivel bajo. El patrón de descanso parece ser adecuado.",
            "easily_distracted": f"La facilidad de distracción ({score:.2f}) es baja. La capacidad de mantener el foco está bien desarrollada.",
            "worry_level": f"El nivel de preocupación ({score:.2f}) es bajo. La ansiedad relacionada con preocupaciones está bien controlada.",
            "concentration_difficulty": f"La dificultad de concentración ({score:.2f}) es baja. La capacidad de mantener la atención está funcionando bien.",
        }
        recommendations = {
            "depression_level": "Mantén hábitos saludables y actividades que te generen bienestar. Continúa con tu rutina actual.",
            "sleep_issues": "Sigue manteniendo buenos hábitos de sueño. Establece horarios regulares y evita pantallas antes de dormir.",
            "easily_distracted": "Continúa con estrategias que te ayudan a mantener el foco. Considera técnicas de mindfulness para fortalecer aún más la atención.",
            "worry_level": "Mantén técnicas de manejo de ansiedad que ya estés usando. La práctica regular de relajación puede ser beneficiosa.",
            "concentration_difficulty": "Sigue con las estrategias que te funcionan. Considera ejercicios de atención plena para mantener esta capacidad.",
        }
    elif score < thresh["moderado"]:
        level = "Moderado"
        severity = "Moderada"
        color = "#FF9800"  # Naranja
        icon = "warning"
        descriptions = {
            "depression_level": f"El nivel de depresión ({score:.2f}) se encuentra en un rango moderado. Se recomienda monitoreo y estrategias de apoyo.",
            "sleep_issues": f"Los problemas de sueño ({score:.2f}) están en un nivel moderado. Puede haber interrupciones o dificultades ocasionales en el descanso.",
            "easily_distracted": f"La facilidad de distracción ({score:.2f}) es moderada. Puede haber momentos donde mantener el foco requiere más esfuerzo.",
            "worry_level": f"El nivel de preocupación ({score:.2f}) es moderado. La ansiedad puede estar presente pero aún manejable.",
            "concentration_difficulty": f"La dificultad de concentración ({score:.2f}) es moderada. Puede haber desafíos ocasionales para mantener la atención sostenida.",
        }
        recommendations = {
            "depression_level": "Considera aumentar actividades que generen bienestar, mantener conexiones sociales y buscar apoyo si persiste. La actividad física regular puede ser beneficiosa.",
            "sleep_issues": "Establece una rutina de sueño más estricta. Evita cafeína y pantallas 2 horas antes de dormir. Considera técnicas de relajación antes de acostarte.",
            "easily_distracted": "Implementa técnicas de gestión del tiempo como Pomodoro. Reduce estímulos externos durante tareas importantes. Practica mindfulness regularmente.",
            "worry_level": "Practica técnicas de respiración y relajación. Considera escribir tus preocupaciones y desafiar pensamientos negativos. La actividad física puede ayudar a reducir la ansiedad.",
            "concentration_difficulty": "Divide tareas grandes en pasos más pequeños. Elimina distracciones del entorno. Considera ejercicios de atención plena y descansos regulares.",
        }
    elif score < thresh["alto"]:
        level = "Alto"
        severity = "Alta"
        color = "#F44336"  # Rojo
        icon = "error"
        descriptions = {
            "depression_level": f"El nivel de depresión ({score:.2f}) se encuentra en un rango alto. Se recomienda atención profesional y estrategias de apoyo inmediatas.",
            "sleep_issues": f"Los problemas de sueño ({score:.2f}) están en un nivel alto. Puede haber dificultades significativas para descansar adecuadamente.",
            "easily_distracted": f"La facilidad de distracción ({score:.2f}) es alta. Mantener el foco puede ser un desafío constante que afecta la productividad.",
            "worry_level": f"El nivel de preocupación ({score:.2f}) es alto. La ansiedad puede estar interfiriendo significativamente con el bienestar diario.",
            "concentration_difficulty": f"La dificultad de concentración ({score:.2f}) es alta. Puede haber problemas significativos para mantener la atención en tareas.",
        }
        recommendations = {
            "depression_level": "Es importante buscar apoyo profesional. Considera terapia cognitivo-conductual o apoyo psicológico. Mantén conexiones sociales y actividades que te generen placer. Si tienes pensamientos de autolesión, busca ayuda inmediata.",
            "sleep_issues": "Consulta con un especialista en sueño. Establece una higiene del sueño estricta. Considera técnicas de relajación avanzadas o terapia cognitivo-conductual para insomnio. Evita estimulantes y pantallas antes de dormir.",
            "easily_distracted": "Considera evaluación profesional para TDAH o problemas de atención. Implementa estrategias estructuradas de gestión del tiempo. Reduce significativamente estímulos externos. Terapia ocupacional puede ser beneficiosa.",
            "worry_level": "Busca apoyo profesional para manejo de ansiedad. Terapia cognitivo-conductual puede ser muy efectiva. Practica técnicas de relajación diariamente. Considera reducir consumo de cafeína y estimulantes.",
            "concentration_difficulty": "Consulta con un profesional para evaluación. Considera técnicas de entrenamiento de atención y terapia ocupacional. Establece un entorno de trabajo libre de distracciones. Descansos frecuentes y técnicas de mindfulness pueden ayudar.",
        }
    else:
        level = "Muy Alto"
        severity = "Muy Alta"
        color = "#9C27B0"  # Púrpura
        icon = "priority_high"
        descriptions = {
            "depression_level": f"El nivel de depresión ({score:.2f}) se encuentra en un rango muy alto. Se requiere atención profesional inmediata y estrategias de apoyo intensivas.",
            "sleep_issues": f"Los problemas de sueño ({score:.2f}) están en un nivel muy alto. Puede haber dificultades severas que requieren intervención profesional.",
            "easily_distracted": f"La facilidad de distracción ({score:.2f}) es muy alta. Los problemas de atención pueden estar afectando significativamente la funcionalidad diaria.",
            "worry_level": f"El nivel de preocupación ({score:.2f}) es muy alto. La ansiedad puede estar causando interferencia significativa en la vida diaria.",
            "concentration_difficulty": f"La dificultad de concentración ({score:.2f}) es muy alta. Puede haber problemas severos que requieren evaluación profesional.",
        }
        recommendations = {
            "depression_level": "Busca ayuda profesional inmediata. Contacta con un psicólogo o psiquiatra. Si tienes pensamientos de autolesión o suicidio, llama a una línea de crisis inmediatamente. No estás solo y hay ayuda disponible.",
            "sleep_issues": "Consulta urgentemente con un especialista en medicina del sueño. Puede requerirse evaluación médica completa. Considera terapia cognitivo-conductual para insomnio. La falta de sueño puede afectar gravemente la salud.",
            "easily_distracted": "Busca evaluación profesional inmediata. Puede ser necesario considerar evaluación para TDAH u otros trastornos de atención. Terapia y posiblemente tratamiento médico pueden ser necesarios.",
            "worry_level": "Busca ayuda profesional urgente para manejo de ansiedad. Terapia cognitivo-conductual o medicación pueden ser necesarias. Practica técnicas de relajación diariamente. Si la ansiedad es incapacitante, considera atención de emergencia.",
            "concentration_difficulty": "Consulta urgentemente con un profesional. Puede requerirse evaluación neurológica o psicológica. Terapia ocupacional y posiblemente tratamiento médico pueden ser necesarios. No ignores estos síntomas.",
        }
    
    return {
        "level": level,
        "severity": severity,
        "description": descriptions.get(name, f"El indicador {name} tiene un valor de {score:.2f}."),
        "recommendation": recommendations.get(name, "Considera consultar con un profesional de salud mental."),
        "color": color,
        "icon": icon,
    }


def format_mental_health_response(raw_scores: Dict[str, float]) -> Dict:
    """
    Formatea la respuesta de salud mental con interpretaciones profesionales,
    resumen general y recomendaciones.
    """
    indicators = {}
    priority_areas = []
    high_severity_count = 0
    
    for name, score in raw_scores.items():
        interpretation = interpret_indicator(name, score)
        indicators[name] = {
            "score": score,
            "interpretation": interpretation
        }
        
        if interpretation["level"] in ["Moderado", "Alto", "Muy Alto"]:
            priority_areas.append(name)
            if interpretation["level"] in ["Alto", "Muy Alto"]:
                high_severity_count += 1
    
    if high_severity_count >= 3:
        overall_assessment = "Se detectan múltiples áreas con niveles altos o muy altos que requieren atención profesional inmediata."
        overall_severity = "Alta"
    elif high_severity_count >= 2:
        overall_assessment = "Hay varias áreas que muestran niveles elevados. Se recomienda monitoreo cercano y consideración de apoyo profesional."
        overall_severity = "Moderada-Alta"
    elif len(priority_areas) >= 3:
        overall_assessment = "Se identifican múltiples áreas que requieren atención. Se recomienda implementar estrategias de apoyo y considerar consulta profesional."
        overall_severity = "Moderada"
    elif len(priority_areas) >= 1:
        overall_assessment = "Hay algunas áreas que requieren atención. Se recomienda monitoreo y estrategias de apoyo preventivas."
        overall_severity = "Leve-Moderada"
    else:
        overall_assessment = "Los indicadores se encuentran en rangos saludables. Continúa con hábitos que promuevan el bienestar."
        overall_severity = "Baja"
    
    general_recommendations = [
        "Mantén un registro regular de tu bienestar mental para identificar patrones.",
        "Establece rutinas de autocuidado que incluyan actividad física, sueño adecuado y tiempo para actividades placenteras.",
        "Mantén conexiones sociales significativas y busca apoyo cuando lo necesites.",
        "Considera técnicas de mindfulness o meditación para mejorar el bienestar general.",
    ]
    
    if high_severity_count >= 2:
        general_recommendations.insert(0, "Se recomienda encarecidamente consultar con un profesional de salud mental para evaluación y apoyo.")
    
    response = {
        **raw_scores,  
        "indicators": indicators,
        "overall_assessment": overall_assessment,
        "overall_severity": overall_severity,
        "priority_areas": priority_areas,
        "general_recommendations": general_recommendations,
    }
    
    return response


def predict(model_server, raw_data: dict):
    """Predict mental health from raw user data."""
    if model_server.mental_health is None:
        raise RuntimeError("Mental health model not loaded")

    mh = model_server.mental_health
    model = mh["model"]
    scaler = mh["scaler"]
    targets = mh["config"].get(
        "targets",
        [
            "depression_level",
            "sleep_issues",
            "easily_distracted",
            "worry_level",
            "concentration_difficulty",
        ],
    )

    df_full = preprocess_mental_health_input(raw_data, model_server)

    if scaler is not None:
        df_scaled = pd.DataFrame(
            scaler.transform(df_full),
            columns=df_full.columns,
        )
    else:
        df_scaled = df_full

    selected_features = mh["selected_features"]
    df_final = df_scaled.reindex(columns=selected_features, fill_value=0.0)

    preds = model.predict(df_final)[0]
    raw_scores = {targets[i]: float(preds[i]) for i in range(len(targets))}
    
    return format_mental_health_response(raw_scores)
