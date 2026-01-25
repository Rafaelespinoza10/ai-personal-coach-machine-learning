# src/backend/models_loader.py
import pickle
import json
from pathlib import Path

MODELS_PATH = Path(__file__).resolve().parent.parent / "models"
class ModelServer:
    def __init__(self):
        self.sentiment = None     
        self.mental_health = None  
        self.stress = None         

    def load_all_models(self):
        loaded = []

        p = MODELS_PATH / "sentiment" / "model_explainer.pkl"
        if p.exists():
            with open(p, "rb") as f:
                self.sentiment = pickle.load(f)
            loaded.append("sentiment")
        else:
            self.sentiment = None

        # Mental health
        mh_model = MODELS_PATH / "mental_health" / "best_mental_health_model.pkl"
        mh_scaler = MODELS_PATH / "mental_health" / "scaler.pkl"
        mh_feats = MODELS_PATH / "mental_health" / "selected_features.pkl"
        mh_prep = MODELS_PATH / "mental_health" / "preprocessors.pkl"
        mh_config = MODELS_PATH / "mental_health" / "model_config.json"
        if mh_model.exists():
            with open(mh_model, "rb") as f:
                model = pickle.load(f)
            scaler = pickle.load(open(mh_scaler, "rb")) if mh_scaler.exists() else None
            feats = pickle.load(open(mh_feats, "rb")) if mh_feats.exists() else []
            preprocessors = pickle.load(open(mh_prep, "rb")) if mh_prep.exists() else None
            config = json.load(open(mh_config)) if mh_config.exists() else {}
            self.mental_health = {
                "model": model,
                "scaler": scaler,
                "selected_features": feats,
                "preprocessors": preprocessors,
                "config": config,
            }
            loaded.append("mental_health")
        else:
            self.mental_health = None

        # Stress (main)
        stress_model = MODELS_PATH / "best_model.pkl"
        stress_scaler = MODELS_PATH / "scaler.pkl"
        stress_prep = MODELS_PATH / "preprocessors.pkl"
        stress_config = MODELS_PATH / "training_results.json"
        if stress_model.exists():
            with open(stress_model, "rb") as f:
                model = pickle.load(f)
            scaler = None
            if stress_scaler.exists():
                with open(stress_scaler, "rb") as f:
                    scaler = pickle.load(f)
            preprocessors = None
            if stress_prep.exists():
                with open(stress_prep, "rb") as f:
                    preprocessors = pickle.load(f)
            config = {}
            if stress_config.exists():
                config = json.load(open(stress_config)).get("config", {})
            self.stress = {"model": model, "scaler": scaler, "preprocessors": preprocessors, "config": config}
            loaded.append("stress")
        else:
            self.stress = None

        print("Models loaded:", loaded)
        return loaded

model_server = ModelServer()