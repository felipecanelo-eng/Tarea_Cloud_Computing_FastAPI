# ============================================================
# E6 — FASTAPI MODEL SERVING
# Sleep Debt Multiclass Classification
# ============================================================

from contextlib import asynccontextmanager
from pathlib import Path
from enum import Enum
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

model = None
metadata = None


# ============================================================
# 3. VALID CATEGORICAL VALUES
# ============================================================
#
# These values come directly from the categories present
# in the training dataset.
#
# Using Enum prevents unseen categorical values from reaching
# the model. FastAPI/Pydantic will return HTTP 422 automatically.
# ============================================================

class Gender(str, Enum):
    FEMALE = "Female"
    MALE = "Male"
    NON_BINARY = "Non-Binary"


class OccupationType(str, Enum):
    CORPORATE = "Corporate 9-to-5"
    FREELANCE = "Freelance / Creative"
    HEALTHCARE = "Healthcare / Shift Worker"
    REMOTE_TECH = "Remote Tech"
    STUDENT = "Student"


class Chronotype(str, Enum):
    INTERMEDIATE = "Intermediate"
    MORNING_LARK = "Morning Lark"
    NIGHT_OWL = "Night Owl"


class BedtimeApp(str, Enum):
    INSTAGRAM_REDDIT = "Instagram / Reddit"
    MESSAGING_CHAT = "Messaging / Chat"
    NEWS_READING = "News / Reading"
    STREAMING = "Streaming (Netflix/Hulu)"
    TIKTOK_REELS = "TikTok / Reels"
    YOUTUBE = "YouTube"


# ============================================================
# 4. LOAD MODEL AT APPLICATION STARTUP
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
# 5. CREATE FASTAPI APPLICATION
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
# 6. INPUT SCHEMA
# ============================================================

class SleepDebtInput(BaseModel):

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Age of the person",
    )

    gender: Gender = Field(
        ...,
        description="Gender category",
    )

    occupation_type: OccupationType = Field(
        ...,
        description="Occupation type",
    )

    chronotype: Chronotype = Field(
        ...,
        description="Chronotype category",
    )

    bedtime_phone_minutes: float = Field(
        ...,
        ge=0,
        description="Minutes of phone use around bedtime",
    )

    primary_bedtime_app: BedtimeApp = Field(
        ...,
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
# 7. OUTPUT SCHEMA
# ============================================================

class PredictionResponse(BaseModel):

    prediction: str
    prediction_code: int
    probability: float
    probabilities: dict[str, float]
    model_version: str


# ============================================================
# 8. HELPER FUNCTION
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

        # ----------------------------------------------------
        # Convert validated Pydantic input into plain JSON
        # values.
        #
        # mode="json" is important because Enum values become
        # their original strings before reaching scikit-learn.
        # ----------------------------------------------------

        row = pd.DataFrame(
            [
                input_data.model_dump(
                    mode="json"
                )
            ]
        )

        # Exact feature order used during training
        row = row[
            metadata["features"]
        ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Probability for every class
        # ----------------------------------------------------

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
# 9. ROOT ENDPOINT
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
# 10. HEALTH ENDPOINT
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
# 11. MODEL INFORMATION ENDPOINT
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
# 12. SINGLE PREDICTION ENDPOINT
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
# 13. BATCH PREDICTION ENDPOINT
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