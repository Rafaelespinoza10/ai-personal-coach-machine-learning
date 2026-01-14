# 🤖 AI Personal Performance Coach

Sistema de inteligencia artificial que analiza hábitos, rutinas y productividad para predecir fatiga, bajo rendimiento y riesgo de burnout, ofreciendo recomendaciones personalizadas para mejorar el bienestar y rendimiento.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Modelos Implementados](#-modelos-implementados)
- [Análisis SHAP](#-análisis-shap)
- [Feature Engineering](#-feature-engineering)
- [Próximos Pasos](#-próximos-pasos)
- [Contribuciones](#-contribuciones)

## ✨ Características

- **Análisis Multidimensional**: Integra datos de sueño, actividad física, estado emocional, salud mental y niveles de estrés
- **Predicción Binaria**: Clasifica el nivel de estrés como "Low" o "Medium+High" para facilitar la interpretación
- **Feature Engineering**: Crea features compuestas que mejoran la capacidad predictiva del modelo
- **Feature Selection**: Selecciona automáticamente las top 10 features más importantes
- **Explicabilidad**: Utiliza SHAP (SHapley Additive exPlanations) para explicar las predicciones del modelo
- **Pipeline Completo**: Desde EDA hasta entrenamiento y análisis de modelos

## 🏗️ Arquitectura del Proyecto

```
ai_personal_performance_coach/
├── datasets/
│   ├── raw/              # Datasets originales
│   ├── processed/        # Datasets limpiados
│   └── final/            # Dataset unificado
├── models/               # Modelos entrenados y resultados
│   └── shap_analysis/    # Visualizaciones SHAP
├── notebooks/            # Análisis exploratorio (EDA)
│   ├── 01_main_model_data_integration.ipynb
│   ├── 02_EDA_sleep_health.ipynb
│   ├── 03_EDA_emotional_monitoring_dataset.ipynb
│   ├── 04_EDA_mental_health_lifestyle_dataset.ipynb
│   └── 05_EDA_stress_level_dataset.ipynb
└── src/
    ├── models/
    │   ├── main/
    │   │   └── main_model.py      # Script de entrenamiento
    │   └── analysis/
    │       └── shap_analysis.py   # Análisis SHAP
    └── utils/
        ├── constants.py           # Constantes del proyecto
        ├── functions.py           # Funciones utilitarias
        └── __init__.py            # Barrel file
```

## 📦 Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost (opcional, pero recomendado)
- shap (para análisis de explicabilidad)
- matplotlib
- seaborn
- jupyter (para notebooks)

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/ai_personal_performance_coach.git
cd ai_personal_performance_coach
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

## 💻 Uso

### 1. Preparar los Datos

Coloca tus datasets en `datasets/raw/`:
- `01_sleep_health_lifestyle.csv`
- `02_emotional_monitoring_dataset_with_target.csv`
- `03_mental_health_lifestyle.csv`
- `04_stress_level_dataset.csv`

### 2. Análisis Exploratorio (EDA)

Ejecuta los notebooks en orden:
1. `01_main_model_data_integration.ipynb` - Integración de datasets
2. `02_EDA_sleep_health.ipynb` - Análisis de datos de sueño
3. `03_EDA_emotional_monitoring_dataset.ipynb` - Análisis emocional
4. `04_EDA_mental_health_lifestyle_dataset.ipynb` - Análisis de salud mental
5. `05_EDA_stress_level_dataset.ipynb` - Análisis de niveles de estrés

### 3. Entrenar el Modelo

```bash
python src/models/main/main_model.py
```

Este script:
- Aplica feature engineering
- Entrena múltiples modelos (Random Forest, Gradient Boosting, SVM, Neural Network, XGBoost)
- Selecciona las top 10 features más importantes
- Re-entrena con features seleccionadas
- Guarda el mejor modelo en `models/best_model.pkl`

### 4. Análisis SHAP (Explicabilidad)

```bash
python src/models/analysis/shap_analysis.py
```

Genera visualizaciones en `models/shap_analysis/`:
- `shap_summary_bar.png` - Importancia de features
- `shap_summary_dot.png` - Distribución de SHAP values
- `shap_waterfall_instance0.png` - Explicación de una instancia
- `shap_feature_importance.csv` - Tabla de importancia
- `shap_force_plot_instance*.html` - Gráficos interactivos

## 📊 Resultados

### Mejor Modelo: SVM (Support Vector Machine)

**Métricas de Rendimiento:**
- **Test Accuracy**: 76.4%
- **Test F1 Score**: 85.2%
- **Test Precision**: 74.1%
- **Test Recall**: 100.0%
- **CV Accuracy**: 75.6% (±0.5%)

**Clasificación Binaria:**
- **Clase 0 (Low)**: Bajo nivel de estrés
- **Clase 1 (Medium+High)**: Nivel de estrés medio o alto

### Comparación de Modelos

| Modelo | Test Accuracy | F1 Score | CV Mean |
|--------|--------------|----------|---------|
| **SVM** | **76.4%** | **85.2%** | **75.6%** |
| Random Forest | 76.2% | 85.0% | 75.6% |
| Gradient Boosting | 76.0% | 84.8% | 75.5% |
| XGBoost | 75.7% | 84.7% | 75.3% |
| Neural Network | 75.1% | 84.2% | 75.0% |

## 🔍 Análisis SHAP

El análisis SHAP revela las features más importantes para predecir el nivel de estrés. Las visualizaciones generadas muestran:

- **Importancia de Features**: Qué variables tienen mayor impacto en las predicciones
- **Distribución de SHAP Values**: Cómo cada feature afecta las predicciones
- **Explicaciones Individuales**: Por qué el modelo predice un nivel de estrés específico para cada instancia

**Top Features (por importancia SHAP):**
1. `stress_level` - Nivel de estrés reportado
2. `sleep_quality` - Calidad del sueño
3. `physical_activity` - Nivel de actividad física
4. `heart_rate` - Frecuencia cardíaca
5. `cortisol_level` - Nivel de cortisol
6. Features engineered (ratios e interacciones)

## 🧪 Modelos Implementados

1. **Random Forest** - Ensemble de árboles de decisión
2. **Gradient Boosting** - Boosting secuencial
3. **SVM (RBF Kernel)** - Support Vector Machine con kernel RBF ⭐ Mejor modelo
4. **Neural Network (MLP)** - Perceptrón multicapa
5. **XGBoost** - Gradient boosting optimizado

## 📈 Feature Engineering

El proyecto incluye creación automática de features compuestas:

- **Health Stress Index**: `sleep_quality - physical_activity / 10`
- **Sleep Efficiency**: `sleep_quality / (sleep_duration + 1)`
- **HR Activity Ratio**: `heart_rate / (physical_activity + 1)`
- **Stress Activity Balance**: `stress_level / (physical_activity + 1)`
- **Sleep Stress Interaction**: `sleep_quality * stress_level`
- **Physiological Stress Score**: `cortisol_level * 10 + heart_rate / 10`

## 🎯 Próximos Pasos

- [ ] Implementar API REST para predicciones en tiempo real
- [ ] Agregar análisis de series temporales para predicción de tendencias
- [ ] Integrar procesamiento de texto para análisis de diarios emocionales
- [ ] Crear dashboard interactivo para visualización de resultados
- [ ] Implementar sistema de recomendaciones personalizadas
- [ ] Agregar tests unitarios y de integración
- [ ] Documentación de API con Swagger/OpenAPI

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Notas

- Los datasets grandes (CSV) y modelos entrenados (PKL) están excluidos del repositorio por tamaño
- Los modelos se pueden regenerar ejecutando `main_model.py`
- Los resultados de entrenamiento se guardan en `models/training_results.json`

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

---

**Desarrollado por el Ingeniero Alejandro Rafael Moreno Espinoza**

*Desarrollado con ❤️ para mejorar el bienestar y rendimiento personal*

