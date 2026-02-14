from fastapi import APIRouter, Depends, HTTPException
from backend.schemas import UserCreate, UserLogin, TokenResponse, UserResponse
from backend.config.database import get_db
from backend.services import auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse)
def register(data: UserCreate, conn=Depends(get_db)):
    """Registra un usuario y devuelve JWT (access_token)."""
    existing = auth_service.get_user_by_email(conn, data.email)
    if existing:
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    user_id, _ = auth_service.create_user(conn, data.email, data.password)
    access_token = auth_service.create_access_token(user_id)
    return TokenResponse(access_token=access_token, user_id=user_id)


@router.post("/login", response_model=TokenResponse)
def login(data: UserLogin, conn=Depends(get_db)):
    """Login con email/password; devuelve JWT (access_token)."""
    user = auth_service.get_user_by_email(conn, data.email)
    if not user:
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")
    user_id, password_hash = user
    if not auth_service.verify_password(data.password, password_hash):
        raise HTTPException(status_code=401, detail="Email o contraseña incorrectos")
    access_token = auth_service.create_access_token(user_id)
    return TokenResponse(access_token=access_token, user_id=user_id)
