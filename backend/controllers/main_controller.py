import json
from fastapi import APIRouter, Depends, Header, HTTPException
from backend.schemas import StressRequest, StressResponse
from backend.models_loader import model_server
from backend.services import stress_service
from backend.config.database import get_db

router = APIRouter(prefix="/stress", tags=["Stress"])

@router.post("/predict", response_model=StressResponse)
def predict_stress(
    request: StressRequest,
    conn=Depends(get_db),
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
) -> StressResponse:
    try:
        output = stress_service.predict(model_server, request.features)
        response = StressResponse(**output)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_events (event_type, session_id, input_payload, output_payload)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        "stress",
                        x_session_id,
                        json.dumps(request.features),
                        json.dumps(output),
                    ),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] No se pudo guardar evento stress: {e}")
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")