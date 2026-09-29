# ============================================================
# E6 — FASTAPI MODEL SERVING
# Sleep Debt Multiclass Classification
# ============================================================

from contextlib import asynccontextmanager
from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "model" / "model.pkl"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"


# ============================================================
# 2. GLOBAL MODEL OBJECTS
# ============================================================
#
# They are loaded once when the application starts.
# They are NOT reloaded for every prediction request.
# ============================================================

model = None
metadata = None


# ============================================================
# 3. LOAD MODEL AT APPLICATION STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global metadata

    try:
        model = joblib.load(MODEL_PATH)

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        print("Model loaded successfully.")
        print(f"Model: {metadata['model_name']}")

    except Exception as exc:
        model = None
        metadata = None

        print(
            f"ERROR loading model: {exc}"
        )

    yield


# ============================================================
# 4. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Sleep Debt Classification API",
    description=(
        "API for predicting sleep debt category using "
        "a supervised XGBoost multiclass classification model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 5. INPUT SCHEMA
# ============================================================
#
# These are the SAME 10 raw variables used during training.
#
# We do NOT manually encode categorical variables here.
# The serialized model.pkl contains:
#
#   preprocessing
#       +
#   XGBoost classifier
#
# ============================================================

class SleepDebtInput(BaseModel):

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Age of the person",
    )

    gender: str = Field(
        ...,
        min_length=1,
        description="Gender category",
    )

    occupation_type: str = Field(
        ...,
        min_length=1,
        description="Occupation type",
    )

    chronotype: str = Field(
        ...,
        min_length=1,
        description="Chronotype category",
    )

    bedtime_phone_minutes: float = Field(
        ...,
        ge=0,
        description="Minutes of phone use around bedtime",
    )

    primary_bedtime_app: str = Field(
        ...,
        min_length=1,
        description="Primary application used at bedtime",
    )

    screen_brightness_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Screen brightness percentage",
    )

    blue_light_filter_active: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if blue-light filter is active, otherwise 0",
    )

    caffeine_post_5pm_mg: float = Field(
        ...,
        ge=0,
        description="Caffeine consumed after 5 PM in milligrams",
    )

    physical_activity_min: float = Field(
        ...,
        ge=0,
        description="Daily physical activity in minutes",
    )


# ============================================================
# 6. OUTPUT SCHEMA
# ============================================================

class PredictionResponse(BaseModel):

    prediction: str
    prediction_code: int
    probability: float
    probabilities: dict[str, float]
    model_version: str


# ============================================================
# 7. HELPER FUNCTION
# ============================================================

def predict_one(
    input_data: SleepDebtInput,
) -> PredictionResponse:

    if model is None or metadata is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not available.",
        )

    try:

        # --------------------------------------------
        # Convert Pydantic input into one-row DataFrame
        # --------------------------------------------

        row = pd.DataFrame(
            [input_data.model_dump()]
        )

        # Enforce the exact feature order used in training
        row = row[
            metadata["features"]
        ]

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        prediction_code = int(
            model.predict(row)[0]
        )

        probabilities_array = (
            model.predict_proba(row)[0]
        )

        prediction_label = (
            metadata["int_to_class"][
                str(prediction_code)
            ]
        )

        probability = float(
            probabilities_array[
                prediction_code
            ]
        )

        # --------------------------------------------
        # Probability for every class
        # --------------------------------------------

        probabilities = {
            metadata["int_to_class"][str(i)]:
                round(float(prob), 6)

            for i, prob
            in enumerate(probabilities_array)
        }

        return PredictionResponse(
            prediction=prediction_label,
            prediction_code=prediction_code,
            probability=round(
                probability,
                6,
            ),
            probabilities=probabilities,
            model_version="1.0.0",
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while generating the prediction."
            ),
        ) from exc


# ============================================================
# 8. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Sleep Debt Classification API",

        "documentation":
            "/docs",
    }


# ============================================================
# 9. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok"
            if model is not None
            else "error",

        "model_loaded":
            model is not None,
    }


# ============================================================
# 10. MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    if metadata is None:
        raise HTTPException(
            status_code=500,
            detail="Model metadata is not available.",
        )

    return {

        "model_name":
            metadata["model_name"],

        "model_type":
            metadata["model_type"],

        "problem_type":
            metadata["problem_type"],

        "target":
            metadata["target"],

        "model_version":
            "1.0.0",

        "python_version":
            metadata["python_version"],

        "sklearn_version":
            metadata["sklearn_version"],

        "xgboost_version":
            metadata["xgboost_version"],

        "features":
            metadata["features"],

        "classes":
            metadata["class_names"],

        "metrics":
            metadata["metrics"],
    }


# ============================================================
# 11. SINGLE PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    input_data: SleepDebtInput,
):

    return predict_one(
        input_data
    )


# ============================================================
# 12. BATCH PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict-batch",
    response_model=list[PredictionResponse],
)
def predict_batch(
    inputs: list[SleepDebtInput],
):

    if len(inputs) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "The batch must contain "
                "at least one observation."
            ),
        )

    return [
        predict_one(item)
        for item in inputs
    ]