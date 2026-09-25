from pathlib import Path
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "retail_demand_model.pkl"

model_bundle = joblib.load(MODEL_FILE)

model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
features = model_bundle["features"]


app = FastAPI(
    title="Retail Demand Forecasting API",
    description="API for predicting retail store demand",
    version="1.0.0"
)


prediction_counter = Counter(
    "retail_prediction_requests_total",
    "Total number of prediction requests"
)

prediction_latency = Histogram(
    "retail_prediction_latency_seconds",
    "Prediction request latency in seconds"
)

prediction_error_counter = Counter(
    "retail_prediction_errors_total",
    "Total number of prediction errors"
)


class PredictionRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    Year: int
    Month: int
    Day: int
    WeekOfYear: int
    IsWeekend: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: int
    Promo2SinceWeek: float
    Promo2SinceYear: float
    PromoInterval: str
    Sales_Lag_1: float
    Sales_Lag_7: float
    Sales_Lag_14: float
    Rolling_Mean_7: float
    Rolling_Mean_14: float


@app.get("/")
def home():
    return {
        "message": "Retail Demand Forecasting API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.time()

    try:
        input_data = pd.DataFrame(
            [request.model_dump()]
        )

        input_data = input_data[features]

        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)[0]

        prediction_counter.inc()

        prediction_latency.observe(
            time.time() - start_time
        )

        return {
            "predicted_sales": round(float(prediction), 2)
        }

    except Exception:
        prediction_error_counter.inc()
        raise