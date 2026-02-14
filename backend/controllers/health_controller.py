import json
from fastapi import APIRouter, Depends, Header, HTTPException
from backend.schemas import MentalHealthRequest, MentalHealthResponse
from backend.models_loader import model_server
from backend.services import health_service
from backend.config.database import get_db

router = APIRouter(prefix="/mental-health", tags=["Mental Health"])

@router.post("/predict", response_model=MentalHealthResponse)
def predict_mental_health(
    req: MentalHealthRequest,
    conn=Depends(get_db),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    try:
        raw_data = req.dict()
        out = health_service.predict(model_server, raw_data)
        response = MentalHealthResponse(**out)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_events (event_type, session_id, input_payload, output_payload)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        "mental_health",
                        x_session_id,
                        json.dumps(raw_data),
                        json.dumps(out),
                    ),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] No se pudo guardar evento mental_health: {e}")
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")