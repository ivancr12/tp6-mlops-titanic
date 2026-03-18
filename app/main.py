from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd
import numpy as np
from app.schemas import TitanicFeatures, PredictionResponse, HealthResponse
import os

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API para predecir supervivencia en el Titanic",
    version="1.0.0"
)

# Usar el nuevo modelo
MODEL_PATH = "models/modelo_titanic_v2.joblib"
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Modelo cargado exitosamente")
        else:
            print(f"❌ No se encontró el modelo en {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    return HealthResponse(
        status="healthy",
        model_version="v2.0",
        model_type="RandomForestClassifier (Titanic)"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TitanicFeatures):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    try:
        input_data = pd.DataFrame([features.dict()])
        
        # Hacer la predicción
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        survival_prob = probability[1] if len(probability) > 1 else probability[0]
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Sobrevive" if prediction == 1 else "No sobrevive",
            probability=float(survival_prob)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en la predicción: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "Titanic Survival Prediction API",
        "docs": "/docs",
        "health": "/health"
    }
