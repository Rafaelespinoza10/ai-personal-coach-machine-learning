# 🤖 AI Personal Performance Coach

Sistema de IA que analiza hábitos, rutinas y productividad para predecir fatiga, bajo rendimiento y riesgo de burnout. Incluye **análisis de sentimiento**, **indicadores de salud mental** (uso de redes sociales) y **nivel de estrés**, con recomendaciones y metadatos para apps (títulos, colores, iconos).

## 📋 Tabla de Contenidos

- [Características](#-características)
- [API REST](#-api-rest)
- [Arquitectura](#-arquitectura)
- [Requisitos e instalación](#-requisitos-e-instalación)
- [Uso](#-uso)
- [Modelos y entrenamiento](#-modelos-y-entrenamiento)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Próximos pasos](#-próximos-pasos)

## ✨ Características

- **API REST (FastAPI)**: Endpoints para predicciones en tiempo real.
- **Análisis de sentimiento**: Clasificación de emociones en texto (joyful, sad, scared, peaceful, mad, powerful) con explicaciones, palabras clave y metadatos para UI.
- **Salud mental**: Predicción de indicadores (depresión, sueño, distracción, preocupación, concentración) a partir de uso de redes sociales. Respuesta con interpretaciones, severidad y recomendaciones.
- **Estrés**: Clasificación binaria Low vs Medium+High con mensajes y recomendaciones adaptadas a la probabilidad.
- **Feature engineering**: Features compuestas para el modelo de estrés (health stress index, sleep efficiency, etc.).
- **Pipeline completo**: EDA en notebooks, entrenamiento en `src/models/`, API en `backend/`.

## 🚀 API REST

### Levantar la API

Desde la raíz del proyecto, con el entorno activado:

```bash
cd backend
uvicorn main:app --reload
```

La API corre en `http://localhost:8000`. Documentación interactiva: `http://localhost:8000/docs`.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/sentiment/analyze` | Análisis de emoción en texto |
| `POST` | `/mental-health/predict` | Indicadores de salud mental (uso redes sociales) |
| `POST` | `/stress/predict` | Nivel de estrés (Low / Medium+High) |

### Ejemplos de uso

**1. Sentimiento** — `POST /sentiment/analyze`

```json
{ "text": "Hoy me siento muy bien, todo salió increíble" }
```

Respuesta: `predicted_emotion`, `confidence`, `explanation`, `key_words`, `all_emotions_scores`, `display_metadata` (title, color, icon).

**2. Salud mental** — `POST /mental-health/predict`

```json
{
  "age": 25,
  "gender": "Male",
  "relationship_status": "Single",
  "occupation_status": "University Student",
  "organization": null,
  "platforms": "Facebook, Instagram, YouTube",
  "daily_usage_time": "Between 2 and 3 hours",
  "daily_usage_hours": 2.5,
  "num_platforms": 3,
  "usage_without_purpose": 3,
  "distraction_level": 2,
  "restlessness": 2,
  "social_comparison": 2,
  "comparison_feelings": 1,
  "validation_seeking": 2,
  "interest_fluctuation": 2,
  "social_media_addiction_score": null,
  "mental_health_risk_score": null,
  "digital_wellbeing_score": null
}
```

Respuesta: scores por indicador, `indicators` (interpretación por indicador), `overall_assessment`, `overall_severity`, `priority_areas`, `general_recommendations`.

**3. Estrés** — `POST /stress/predict`

```json
{
  "features": {
    "age": 30,
    "gender": "Male",
    "sleep_quality_norm": 0.7,
    "sleep_quality": 6,
    "sleep_duration": 7,
    "physical_activity": 4,
    "heart_rate": 72,
    "Occupation": "Engineer",
    "diet_type": "Balanced",
    "exercise_level": "Moderate"
  }
}
```

Las claves de `features` deben coincidir con las columnas base del dataset unificado (sin `dataset_source` ni `stress_level_norm`). Respuesta: `stress_level`, `probability`, `binary_class`, `message`, `recommendation`, `display_metadata`.

## 🏗️ Arquitectura

```
ai_personal_performance_coach/
├── backend/                 # API FastAPI
│   ├── main.py              # App y routers
│   ├── models_loader.py     # Carga de modelos (.pkl, configs)
│   ├── schemas.py           # Pydantic (request/response)
│   ├── controllers/         # Rutas por servicio
│   ├── services/            # Lógica (sentiment, health, stress)
│   └── helpers/             # Stopwords, keywords, etc.
├── src/
│   ├── models/              # Entrenamiento
│   │   ├── main/            # Modelo estrés (unified dataset)
│   │   ├── mental_health/   # Salud mental (redes sociales)
│   │   ├── sentiment/       # Análisis de emociones
│   │   └── analysis/        # SHAP
│   └── utils/               # preprocessing, constants, etc.
├── models/                  # Artefactos guardados
│   ├── mental_health/       # model_config, (pkl en .gitignore)
│   ├── sentiment/           # config, (pkl en .gitignore)
│   └── preprocessors.pkl    # Para stress (si use_selected=False)
├── datasets/
│   └── final/               # Unified dataset, metadata, validación
└── notebooks/               # EDA e integración
```

## 📦 Requisitos e instalación

- **Python 3.8+**
- **Dependencias**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`, `pydantic`. Opcional: `shap`, `matplotlib`, `seaborn`, `jupyter` para EDA y SHAP.

```bash
git clone https://github.com/tu-usuario/ai_personal_performance_coach.git
cd ai_personal_performance_coach
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic
```

## 💻 Uso

### 1. Entrenar modelos

Los artefactos (`.pkl`, `scaler`, `preprocessors`, etc.) se guardan en `models/`. Si no existen, entrena primero:

- **Estrés (main)**  
  Dataset unificado en `datasets/final/01_unified_dataset.csv`.  
  ```bash
  python -m src.models.main.main_model
  ```  
  Con `use_selected=False` se genera `models/preprocessors.pkl`, requerido por la API de estrés.

- **Salud mental**  
  ```bash
  python -m src.models.mental_health.mental_health_model
  ```  
  Genera `models/mental_health/` (modelo, scaler, selected_features, preprocessors, config).

- **Sentimiento**  
  ```bash
  python -m src.models.sentiment.sentiment_analysis_model
  ```  
  Genera `models/sentiment/` (model_explainer, config, etc.).

### 2. Ejecutar la API

```bash
cd backend
uvicorn main:app --reload
```

Prueba los endpoints con Postman, `curl` o la UI en `/docs`.

### 3. EDA y análisis

Notebooks en `notebooks/` (integración, EDA por dataset, sentiment, salud mental). SHAP en `src/models/analysis/shap_analysis.py`.

## 🧠 Modelos y entrenamiento

| Modelo | Salida | Uso en API |
|--------|--------|------------|
| **Sentiment** | Emoción (6 clases) | `/sentiment/analyze` |
| **Mental health** | 5 indicadores (depresión, sueño, distracción, preocupación, concentración) | `/mental-health/predict` |
| **Stress (main)** | Binario Low / Medium+High | `/stress/predict` |

- **Estrés**: SVM (u otro según `training_results.json`), sobre dataset unificado + feature engineering. Requiere `preprocessors.pkl` si usas all-features.
- **Salud mental**: XGBoost multi-output, preprocesamiento con OHE e imputers guardados.
- **Sentimiento**: RandomForest + LinearSVC (explicador), TF-IDF, keywords y override por confianza baja.

## 📁 Estructura del proyecto

- `backend/`: API, carga de modelos, controllers, services, schemas.
- `src/models/`: Scripts de entrenamiento (main, mental_health, sentiment, analysis).
- `src/utils/`: `preprocess_data`, `engineer_features`, constantes, etc.
- `models/`: Configs y, si corres entrenamiento, artefactos (.pkl). Los `.pkl` y otros binarios suelen estar en `.gitignore`.
- `datasets/final/`: Dataset unificado, columnas, resúmenes, validación.
- `notebooks/`: EDA e integración de datos.

## 🎯 Próximos pasos

- [ ] Cliente Flutter/móvil para consumir la API
- [ ] Tests unitarios y de integración para la API
- [ ] CI/CD (entrenamiento + despliegue)
- [ ] Dashboard para métricas y predicciones
- [ ] Series temporales o tendencias de bienestar

## 📝 Notas

- Los CSV grandes y los `.pkl` de modelos suelen estar en `.gitignore`. Regenera modelos con los scripts de `src/models/`.
- Para `/stress/predict` hace falta `models/preprocessors.pkl` (entrenar main model con `use_selected=False`).
- Variables de entorno sensibles (`.env`) no se versionan.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

**Desarrollado por el Ing. Alejandro Rafael Moreno Espinoza**  
*Hecho para mejorar bienestar y rendimiento personal*
