# ============================================================
# E7 — AUTOMATED API TESTS
# ============================================================

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# TEST CLIENT
# ============================================================

@pytest.fixture(scope="module")
def client():
    """
    TestClient used as a context manager so that the FastAPI
    lifespan event runs and loads model.pkl before the tests.
    """
    with TestClient(app) as test_client:
        yield test_client


# ============================================================
# VALID EXAMPLES
# ============================================================

VALID_INPUT_1 = {
    "age": 31,
    "gender": "Female",
    "occupation_type": "Remote Tech",
    "chronotype": "Night Owl",
    "bedtime_phone_minutes": 135,
    "primary_bedtime_app": "TikTok / Reels",
    "screen_brightness_pct": 82,
    "blue_light_filter_active": 0,
    "caffeine_post_5pm_mg": 140,
    "physical_activity_min": 25
}


VALID_INPUT_2 = {
    "age": 42,
    "gender": "Male",
    "occupation_type": "Corporate 9-to-5",
    "chronotype": "Morning Lark",
    "bedtime_phone_minutes": 30,
    "primary_bedtime_app": "News / Reading",
    "screen_brightness_pct": 35,
    "blue_light_filter_active": 1,
    "caffeine_post_5pm_mg": 0,
    "physical_activity_min": 60
}


# ============================================================
# 1. HEALTH
# ============================================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ============================================================
# 2. MODEL INFO
# ============================================================

def test_model_info(client):

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "XGBClassifier"
    assert data["problem_type"] == "multiclass_classification"

    assert len(data["features"]) == 10
    assert len(data["classes"]) == 4

    assert "f1_macro" in data["metrics"]
    assert "balanced_accuracy" in data["metrics"]


# ============================================================
# 3. SINGLE PREDICTION
# ============================================================

def test_predict(client):

    response = client.post(
        "/predict",
        json=VALID_INPUT_1,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "prediction_code" in data
    assert "probability" in data
    assert "probabilities" in data

    assert 0 <= data["probability"] <= 1

    assert len(data["probabilities"]) == 4

    total_probability = sum(
        data["probabilities"].values()
    )

    assert abs(total_probability - 1.0) < 0.01


# ============================================================
# 4. BATCH PREDICTION
# ============================================================

def test_predict_batch(client):

    response = client.post(
        "/predict-batch",
        json=[
            VALID_INPUT_1,
            VALID_INPUT_2,
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    for prediction in data:

        assert "prediction" in prediction
        assert "probability" in prediction

        assert (
            0
            <= prediction["probability"]
            <= 1
        )


# ============================================================
# 5. INVALID CATEGORY -> 422
# ============================================================

def test_invalid_category_returns_422(client):

    invalid_input = VALID_INPUT_1.copy()

    invalid_input["gender"] = "INVALID_CATEGORY"

    response = client.post(
        "/predict",
        json=invalid_input,
    )

    assert response.status_code == 422