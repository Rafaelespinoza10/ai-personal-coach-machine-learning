
from fastapi import APIRouter, HTTPException
from backend.schemas import StressRequest, StressResponse
from backend.models_loader import model_server
from backend.services import stress_service

router = APIRouter(prefix="/stress", tags=["Stress"])

@router.post("/predict", response_model=StressResponse)
def predict_stress(request: StressRequest) -> StressResponse:
    try:
        output = stress_service.predict(model_server, request.features)
        return StressResponse(**output)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")