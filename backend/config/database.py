"""
Conexión a PostgreSQL con SQL puro (psycopg2). Sin ORM.
"""
from typing import Optional
from urllib.parse import urlparse

from psycopg2.pool import ThreadedConnectionPool

from backend.config.settings import settings


def _parse_db_url(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": (parsed.path or "/").lstrip("/") or "ai_performance_coach",
        "user": parsed.username,
        "password": parsed.password,
    }


_pool: Optional[ThreadedConnectionPool] = None


def get_pool() -> ThreadedConnectionPool:
    """Obtiene el pool de conexiones (creado bajo demanda)."""
    global _pool
    if _pool is None:
        url = settings.database_url_fixed
        if not url:
            raise RuntimeError("DATABASE_URL no está configurada")
        params = _parse_db_url(url)
        _pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **params,
        )
    return _pool


def get_db():
    """
    Dependencia FastAPI: entrega una conexión para SQL puro.
    Uso: def route(conn = Depends(get_db)): cur = conn.cursor(); cur.execute("SELECT 1"); ...
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        pool.putconn(conn)


def close_pool():
    """Cierra el pool (llamar en shutdown de la app)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
