"""
app.py
------
A small FastAPI service that loads model.joblib and serves irrigation
need predictions.

Run locally:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test UI.
"""

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Irrigation Need Prediction API")

MODEL_PATH = "model.joblib"
_model = None  # loaded lazily so `import app` doesn't fail before training


def get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python train_model.py` first.",
            )
    return _model


class IrrigationInput(BaseModel):
    Soil_Type: Literal["Clay", "Silt", "Sandy", "Loamy"]
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Temperature_C: float
    Humidity: float
    Rainfall_mm: float
    Sunlight_Hours: float
    Wind_Speed_kmh: float
    Crop_Type: Literal["Wheat", "Maize", "Cotton", "Rice", "Sugarcane", "Potato"]
    Crop_Growth_Stage: Literal["Vegetative", "Flowering", "Harvest", "Sowing"]
    Season: Literal["Rabi", "Zaid", "Kharif"]
    Irrigation_Type: Literal["Rainfed", "Canal", "Drip", "Sprinkler"]
    Water_Source: Literal["Reservoir", "Groundwater", "River", "Rainwater"]
    Field_Area_hectare: float
    Mulching_Used: Literal["Yes", "No"]
    Previous_Irrigation_mm: float
    Region: Literal["South", "Central", "North", "East", "West"]

    class Config:
        json_schema_extra = {
            "example": {
                "Soil_Type": "Clay",
                "Soil_pH": 6.5,
                "Soil_Moisture": 25.0,
                "Organic_Carbon": 0.6,
                "Electrical_Conductivity": 1.5,
                "Temperature_C": 30.0,
                "Humidity": 55.0,
                "Rainfall_mm": 500.0,
                "Sunlight_Hours": 8.0,
                "Wind_Speed_kmh": 10.0,
                "Crop_Type": "Wheat",
                "Crop_Growth_Stage": "Vegetative",
                "Season": "Rabi",
                "Irrigation_Type": "Drip",
                "Water_Source": "Groundwater",
                "Field_Area_hectare": 2.0,
                "Mulching_Used": "Yes",
                "Previous_Irrigation_mm": 20.0,
                "Region": "North",
            }
        }


class PredictionOutput(BaseModel):
    irrigation_need: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: IrrigationInput):
    model = get_model()
    row = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(row)[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()  # CatBoost returns a nested array, not a scalar
    proba = model.predict_proba(row)[0]
    # Most models expose classes_ on the final pipeline step; models that needed
    # label-encoding during training (see train_model.py) expose it directly.
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = model.named_steps["model"].classes_
    return PredictionOutput(
        irrigation_need=prediction,
        probabilities={cls: round(float(p), 4) for cls, p in zip(classes, proba)},
    )
