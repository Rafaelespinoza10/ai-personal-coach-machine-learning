from fastapi import APIRouter, HTTPException
from schemas import MentalHealthRequest, MentalHealthResponse
from models_loader import model_server
import services.health_service as health_service

router = APIRouter(prefix="/mental-health", tags=["Mental Health"])

@router.post("/predict", response_model=MentalHealthResponse)
def predict_mental_health(req: MentalHealthRequest):
    """
    Predict mental health indicators based on social media usage.
    Accepts raw user data (age, gender, platforms, etc.)
    """
    try:
        raw_data = req.dict()
        out = health_service.predict(model_server, raw_data)
        return MentalHealthResponse(**out)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")