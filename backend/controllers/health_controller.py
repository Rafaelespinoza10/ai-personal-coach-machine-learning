import json
from fastapi import APIRouter, Depends, HTTPException
from backend.schemas import MentalHealthRequest, MentalHealthResponse
from backend.models_loader import model_server
from backend.services import health_service
from backend.config.database import get_db
from backend.dependencies import get_current_user_id_required, get_session_id

router = APIRouter(prefix="/mental-health", tags=["Mental Health"])

@router.post("/predict", response_model=MentalHealthResponse)
def predict_mental_health(
    req: MentalHealthRequest,
    conn=Depends(get_db),
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user_id_required),
):
    try:
        raw_data = req.dict()
        out = health_service.predict(model_server, raw_data)
        response = MentalHealthResponse(**out)
        try:
            print("[DB] Guardando evento mental_health...", flush=True)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_events (event_type, session_id, user_id, input_payload, output_payload)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        "mental_health",
                        session_id,
                        user_id,
                        json.dumps(raw_data),
                        json.dumps(out),
                    ),
                )
            conn.commit()
            print("[DB] Evento mental_health guardado.", flush=True)
        except Exception as e:
            conn.rollback()
            import traceback
            print(f"[DB] No se pudo guardar evento mental_health: {e}", flush=True)
            traceback.print_exc()
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")