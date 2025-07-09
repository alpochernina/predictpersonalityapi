import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal
import pandas as pd
from src.inference import PersonalityPredictor
from typing import Optional

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

predictor: Optional[PersonalityPredictor] = None

try:
    predictor = PersonalityPredictor(
        model_path="model/xgb_model.pkl",
        encoder_path="model/xgb_encoders.pkl"
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load model.")
    raise RuntimeError("Model loading failed.") from e

class InputData(BaseModel):
    Time_spent_Alone: Optional[float] = Field(default=None, ge=0, le=11, description="Hours spent alone daily (0–11).")
    Stage_fear: Optional[Literal["Yes", "No"]] = Field(default=None, description="Presence of stage fright (Yes/No).")
    Social_event_attendance: Optional[float] = Field(default=None, ge=0, le=10, description="Frequency of social events (0–10).")
    Going_outside: Optional[float] = Field(default=None, ge=0, le=7, description="Frequency of going outside (0–7).")
    Drained_after_socializing: Optional[Literal["Yes", "No"]] = Field(default=None, description="Feeling drained after socializing (Yes/No).")
    Friends_circle_size: Optional[float] = Field(default=None, ge=0, le=15, description="Number of close friends (0–15).")
    Post_frequency: Optional[float] = Field(default=None, ge=0, le=10, description="Social media post frequency (0–10).")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logging.info(f"Incoming request: {request.method} {request.url} Body: {body.decode('utf-8')}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception occurred.")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.get("/")
async def root():
    return {"message": "Personality prediction API is up and running!"}

@app.post("/predict")
def predict(data: List[InputData]):
    df = pd.DataFrame([item.dict() for item in data])
    prediction = predictor.predict(df)
    return {"prediction": prediction}
