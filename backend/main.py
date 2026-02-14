from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Depends, HTTPException
from backend.models_loader import model_server
from backend.controllers import sentiment_controller, health_controller, main_controller
from backend.config.database import close_pool, get_db, get_pool
from backend.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_server.load_all_models()
    url = settings.database_url_fixed
    host = urlparse(url).hostname if url else "?"
    print(f"[DB] DATABASE_URL host: {host}")
    yield
    close_pool()


app = FastAPI(title="AI Performance Coach API", lifespan=lifespan)

app.include_router(sentiment_controller.router)
app.include_router(health_controller.router)
app.include_router(main_controller.router)


@app.get("/")
async def root():
    return {"message": "AI Personal Performance Coach API is Running"}


@app.get("/health/db")
def health_db():
    """Prueba la conexión a la base de datos. Devuelve 200 si conecta, 503 si no."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return {"status": "ok", "database": "connected"}
        finally:
            conn.close()
            pool.putconn(conn)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unreachable. Set DATABASE_URL in .env.local to your Render URL. Error: {e}",
        )