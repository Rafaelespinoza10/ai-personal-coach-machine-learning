from fastapi import APIRouter, HTTPException
from backend.schemas import SentimentRequest, SentimentResponse
from backend.models_loader import model_server
from backend.services import sentiment_service

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

@router.post("/analyze", response_model=SentimentResponse)
def analyze_emotion(req: SentimentRequest):
    try:
        out = sentiment_service.analyze(model_server, req.text)
        return SentimentResponse(**out)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))