import json
from fastapi import APIRouter, Depends, HTTPException
from backend.schemas import SentimentRequest, SentimentResponse
from backend.models_loader import model_server
from backend.services import sentiment_service
from backend.config.database import get_db
from backend.dependencies import get_current_user_id_required, get_session_id

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

@router.post("/analyze", response_model=SentimentResponse)
def analyze_emotion(
    req: SentimentRequest,
    conn=Depends(get_db),
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user_id_required),
):
    try:
        out = sentiment_service.analyze(model_server, req.text)
        response = SentimentResponse(**out)
        try:
            print("[DB] Guardando evento sentiment...", flush=True)
            input_payload = {"text": req.text}
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_events (event_type, session_id, user_id, input_payload, output_payload)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        "sentiment",
                        session_id,
                        user_id,
                        json.dumps(input_payload),
                        json.dumps(out),
                    ),
                )
            conn.commit()
            print("[DB] Evento sentiment guardado.", flush=True)
        except Exception as e:
            conn.rollback()
            import traceback
            print(f"[DB] No se pudo guardar evento sentiment: {e}", flush=True)
            traceback.print_exc()
        return response
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))