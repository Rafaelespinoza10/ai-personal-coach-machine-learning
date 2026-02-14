import json
from fastapi import APIRouter, Depends, Header, HTTPException
from backend.schemas import SentimentRequest, SentimentResponse
from backend.models_loader import model_server
from backend.services import sentiment_service
from backend.config.database import get_db

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

@router.post("/analyze", response_model=SentimentResponse)
def analyze_emotion(
    req: SentimentRequest,
    conn=Depends(get_db),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    try:
        out = sentiment_service.analyze(model_server, req.text)
        response = SentimentResponse(**out)
        try:
            input_payload = {"text": req.text}
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_events (event_type, session_id, input_payload, output_payload)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        "sentiment",
                        x_session_id,
                        json.dumps(input_payload),
                        json.dumps(out),
                    ),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] No se pudo guardar evento sentiment: {e}")
        return response
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))