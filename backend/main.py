from fastapi import FastAPI
from models_loader import model_server
from controllers import sentiment_controller, health_controller, main_controller

app = FastAPI(title="AI Performance Coach API")

@app.on_event("startup")
async def startup():
    model_server.load_all_models()

app.include_router(sentiment_controller.router)
app.include_router(health_controller.router) 
app.include_router(main_controller.router)

@app.get("/")
async def root():
    return {"message": "AI Personal Performance Coach API is Running"}