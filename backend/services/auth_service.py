import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from backend.config.settings import settings

# bcrypt limita la contraseña a 72 bytes
_MAX_PASSWORD_BYTES = 72


def _to_bytes(password: str) -> bytes:
    b = password.encode("utf-8")
    return b[: _MAX_PASSWORD_BYTES] if len(b) > _MAX_PASSWORD_BYTES else b


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_to_bytes(plain), hashed.encode("utf-8"))


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload.get("sub")
    except Exception:
        return None


def create_user(conn, email: str, password: str) -> tuple[str, str]:
    """Crea usuario, retorna (user_id, password_hash)."""
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s)",
            (user_id, email.strip().lower(), password_hash),
        )
    conn.commit()
    return user_id, password_hash


def get_user_by_email(conn, email: str) -> tuple[str, str] | None:
    """Retorna (user_id, password_hash) o None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, password_hash FROM users WHERE email = %s",
            (email.strip().lower(),),
        )
        row = cur.fetchone()
    if not row:
        return None
    return (str(row[0]), row[1])
