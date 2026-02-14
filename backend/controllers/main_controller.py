import json
from fastapi import APIRouter, Depends, HTTPException
from backend.schemas import StressRequest, StressResponse
from backend.models_loader import model_server
from backend.services import stress_service
from backend.config.database import get_db
from backend.dependencies import get_current_user_id_required, get_session_id

router = APIRouter(prefix="/stress", tags=["Stress"])

@router.post("/predict", response_model=StressResponse)
def predict_stress(
    request: StressRequest,
    conn=Depends(get_db),
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_current_user_id_required),
) -> StressResponse:
    try:
        output = stress_service.predict(model_server, request.features)
        response = StressResponse(**output)
        try:
            print("[DB] Guardando evento stress...", flush=True)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_events (event_type, session_id, user_id, input_payload, output_payload)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        "stress",
                        session_id,
                        user_id,
                        json.dumps(request.features),
                        json.dumps(output),
                    ),
                )
            conn.commit()
            print("[DB] Evento stress guardado.", flush=True)
        except Exception as e:
            conn.rollback()
            import traceback
            print(f"[DB] No se pudo guardar evento stress: {e}", flush=True)
            traceback.print_exc()
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")