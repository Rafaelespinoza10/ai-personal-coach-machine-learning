"""Dependencias compartidas: auth opcional u obligatorio por JWT."""
import uuid
from fastapi import Header, HTTPException, Request
from backend.services.auth_service import decode_token


def get_session_id(request: Request) -> str:
    """
    Lee X-Session-Id del request (insensible a mayúsculas).
    Si no viene, devuelve un UUID para que siempre se guarde un valor.
    """
    raw = request.headers.get("x-session-id") or request.headers.get("X-Session-Id")
    if raw and raw.strip():
        return raw.strip()
    return str(uuid.uuid4())


def get_current_user_id_required(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-Api-Key"),
) -> str:
    """
    Exige Bearer token; si no hay o es inválido, responde 401.
    Acepta Authorization: Bearer <token> o X-Api-Key: <token>.
    """
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token and x_api_key:
        token = x_api_key.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Bearer token requerido")
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    return user_id


def get_current_user_id_optional(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="X-Api-Key"),
) -> str | None:
    """
    Extrae user_id del JWT si viene en Authorization: Bearer <token> o en X-Api-Key.
    Si no hay token o es inválido, retorna None (las rutas siguen funcionando para anónimos).
    """
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token and x_api_key:
        token = x_api_key.strip()
    if not token:
        return None
    return decode_token(token)
