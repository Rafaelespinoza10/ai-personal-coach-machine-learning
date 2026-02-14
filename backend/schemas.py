from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to analyze for sentiment")

class DisplayMetadata(BaseModel):
    title: str
    color: str
    icon: str

class SentimentResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    explanation: str
    key_words: List[str]
    all_emotions_scores: Dict[str, float]
    display_metadata: Optional[DisplayMetadata] = None
    text_processed: Optional[str] = None

class MentalHealthRequest(BaseModel):
    age: float = Field(..., description="Edad del usuario")
    gender: str = Field(..., description="Género: Male, Female, Non-binary, etc.")
    relationship_status: str = Field(..., description="Estado civil: Single, Married, etc.")
    occupation_status: str = Field(..., description="Ocupación: University Student, Salaried Worker, etc.")
    organization: Optional[str] = Field(None, description="Organización")
    platforms: str = Field(..., description="Plataformas separadas por comas: 'Facebook, Instagram, YouTube'")
    daily_usage_time: str = Field(..., description="Tiempo de uso diario: 'Between 2 and 3 hours', etc.")
    daily_usage_hours: float = Field(..., description="Horas de uso diario")
    num_platforms: float = Field(..., description="Número de plataformas")
    usage_without_purpose: float = Field(..., description="Uso sin propósito (escala 1-5)")
    distraction_level: float = Field(..., description="Nivel de distracción (escala 1-5)")
    restlessness: float = Field(..., description="Inquietud (escala 1-5)")
    social_comparison: float = Field(..., description="Comparación social (escala 1-5)")
    comparison_feelings: float = Field(..., description="Sentimientos de comparación (escala 1-5)")
    validation_seeking: float = Field(..., description="Búsqueda de validación (escala 1-5)")
    interest_fluctuation: float = Field(..., description="Fluctuación de interés (escala 1-5)")
    social_media_addiction_score: Optional[float] = Field(None, description="Score de adicción (opcional)")
    mental_health_risk_score: Optional[float] = Field(None, description="Score de riesgo (opcional)")
    digital_wellbeing_score: Optional[float] = Field(None, description="Score de bienestar digital (opcional)")

class IndicatorInterpretation(BaseModel):
    """Interpretación profesional de un indicador de salud mental"""
    level: str  
    severity: str  
    description: str  
    recommendation: str  
    color: str  
    icon: str  

class MentalHealthIndicator(BaseModel):
    """Indicador individual con score e interpretación"""
    score: float
    interpretation: IndicatorInterpretation

class MentalHealthResponse(BaseModel):
    """Respuesta completa con scores e interpretaciones profesionales"""
    depression_level: float
    sleep_issues: float
    easily_distracted: float
    worry_level: float
    concentration_difficulty: float
    
    indicators: Dict[str, MentalHealthIndicator]
    
    overall_assessment: str
    overall_severity: str
    priority_areas: List[str]  
    general_recommendations: List[str]  

class StressRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="unified features for stress prediction (numeric + categorical, matching unified dataset base columns)")

class StressResponse(BaseModel):
    stress_level: str
    probability: Optional[float] = None
    binary_class: int
    message: str = Field(..., description="Interpretación en lenguaje natural")
    recommendation: str = Field(..., description="Recomendación breve y accionable")
    display_metadata: Optional[DisplayMetadata] = None


# Auth
class UserCreate(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    email: str = Field(...)
    password: str = Field(...)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: Optional[str] = None