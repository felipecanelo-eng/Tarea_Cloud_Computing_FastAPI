# ============================================================
# E1 + E2
# TRAINING, EVALUATION AND SERIALIZATION
# SUPERVISED MULTICLASS CLASSIFICATION WITH XGBOOST
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "bedtime_screentime_sleep_debt.csv"
)

MODEL_DIR = ROOT_DIR / "model"

MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("DATASET")
print("========================================")
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n========================================")
print("DATA QUALITY")
print("========================================")

print("\nMissing values by column:")
print(df.isnull().sum())

print(
    "\nTotal duplicated rows:",
    df.duplicated().sum()
)

print("\nData types:")
print(df.dtypes)

# ============================================================
# 3. DEFINE TARGET
# ============================================================

TARGET = "sleep_debt_category"


# ============================================================
# 4. SELECT PREDICTORS
# ============================================================
#
# We intentionally use variables available before/during
# bedtime.
#
# We EXCLUDE post-sleep variables such as:
#
# - sleep_latency_min
# - total_sleep_hours
# - deep_sleep_pct
# - rem_sleep_pct
# - morning_alarm_snoozes
# - next_day_fatigue_score
#
# because they are outcomes of the sleep episode and may
# directly encode sleep_debt_category, generating leakage.
#
# user_id is also excluded because it is only an identifier.
# ============================================================

FEATURES = [
    "age",
    "gender",
    "occupation_type",
    "chronotype",
    "bedtime_phone_minutes",
    "primary_bedtime_app",
    "screen_brightness_pct",
    "blue_light_filter_active",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
]

X = df[FEATURES].copy()
y_original = df[TARGET].copy()


# ============================================================
# 5. DEFINE CATEGORICAL AND NUMERICAL VARIABLES
# ============================================================

CATEGORICAL_FEATURES = [
    "gender",
    "occupation_type",
    "chronotype",
    "primary_bedtime_app",
]

NUMERICAL_FEATURES = [
    "age",
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "blue_light_filter_active",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
]


# ============================================================
# 6. TARGET ENCODING
# ============================================================
#
# XGBoost expects numeric class labels.
#
# The mapping is saved in metadata.json so the API can later
# convert the numeric prediction back to its original label.
# ============================================================

CLASS_NAMES = sorted(
    y_original.unique().tolist()
)

CLASS_TO_INT = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

INT_TO_CLASS = {
    idx: class_name
    for class_name, idx in CLASS_TO_INT.items()
}

y = y_original.map(CLASS_TO_INT)


print("\n========================================")
print("TARGET DISTRIBUTION")
print("========================================")

target_distribution = (
    y_original
    .value_counts()
    .rename_axis("class")
    .reset_index(name="count")
)

target_distribution["percentage"] = (
    target_distribution["count"]
    / len(df)
    * 100
)

print(target_distribution.to_string(index=False))


print("\nClass mapping:")

for class_name, class_id in CLASS_TO_INT.items():
    print(f"{class_id}: {class_name}")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================
#
# Stratification preserves the class proportions in both
# train and test sets.
#
# random_state=42 guarantees reproducibility.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print(f"Train rows: {len(X_train)}")
print(f"Test rows:  {len(X_test)}")


# ============================================================
# 8. PREPROCESSING
# ============================================================
#
# Categorical:
#   - fill missing values with most frequent category
#   - OneHotEncoder
#
# Numerical:
#   - fill missing values with median
#
# Scaling is NOT required for XGBoost because it is a
# tree-based algorithm.
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES,
        ),
    ]
)


# ============================================================
# 9. XGBOOST CLASSIFIER
# ============================================================

classifier = XGBClassifier(
    objective="multi:softprob",
    num_class=len(CLASS_NAMES),

    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,

    subsample=0.90,
    colsample_bytree=0.90,

    eval_metric="mlogloss",

    random_state=RANDOM_STATE,
    n_jobs=-1,

    tree_method="hist",
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================
#
# IMPORTANT:
#
# model.pkl contains:
#
# raw input
#     ↓
# imputation
#     ↓
# one-hot encoding
#     ↓
# XGBoost
#
# We are NOT saving preprocessing separately.
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            classifier,
        ),
    ]
)


# ============================================================
# 11. CLASS BALANCING
# ============================================================
#
# The target is imbalanced:
#
# Moderate Debt is much more frequent than Severe Sleep Debt.
#
# Balanced sample weights prevent the largest class from
# dominating training.
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST")
print("========================================")

pipeline.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)

print("Training completed.")


# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_proba = pipeline.predict_proba(X_test)


# ============================================================
# 14. METRICS
# ============================================================
#
# Main metrics:
#
# F1 MACRO
# --------
# Gives equal importance to every class.
# Appropriate because the target is imbalanced.
#
# BALANCED ACCURACY
# -----------------
# Average recall across classes.
# Prevents the majority class from dominating the metric.
#
# We additionally report:
# - accuracy
# - multiclass ROC-AUC OvR macro
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred,
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
)

roc_auc_macro = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="macro",
)


METRICS = {
    "accuracy": round(float(accuracy), 4),
    "balanced_accuracy": round(
        float(balanced_accuracy),
        4,
    ),
    "f1_macro": round(
        float(f1_macro),
        4,
    ),
    "roc_auc_ovr_macro": round(
        float(roc_auc_macro),
        4,
    ),
}


print("\n========================================")
print("MODEL METRICS")
print("========================================")

for metric_name, value in METRICS.items():
    print(f"{metric_name}: {value}")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# 17. SERIALIZE COMPLETE PIPELINE
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH,
)


print("\n========================================")
print("MODEL SERIALIZATION")
print("========================================")

print(f"Model saved to:")
print(MODEL_PATH)


# ============================================================
# 18. CREATE METADATA.JSON
# ============================================================

metadata = {

    "model_name":
        "Sleep Debt XGBoost Classifier",

    "model_type":
        "XGBClassifier",

    "problem_type":
        "multiclass_classification",

    "target":
        TARGET,

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "python_version":
        platform.python_version(),

    "sklearn_version":
        sklearn.__version__,

    "xgboost_version":
        xgboost.__version__,

    "pandas_version":
        pd.__version__,

    "random_state":
        RANDOM_STATE,

    "test_size":
        TEST_SIZE,

    "dataset_rows":
        int(len(df)),

    "train_rows":
        int(len(X_train)),

    "test_rows":
        int(len(X_test)),

    "features":
        FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "numerical_features":
        NUMERICAL_FEATURES,

    "class_names":
        CLASS_NAMES,

    "class_to_int":
        CLASS_TO_INT,

    "int_to_class": {
        str(k): v
        for k, v in INT_TO_CLASS.items()
    },

    "target_distribution": {
        str(k): int(v)
        for k, v
        in y_original.value_counts().items()
    },

    "metrics":
        METRICS,

    "metric_justification": {

        "f1_macro":
            (
                "Selected because the target classes are "
                "imbalanced and macro F1 gives equal "
                "importance to each class."
            ),

        "balanced_accuracy":
            (
                "Selected because it averages recall "
                "across classes and prevents the majority "
                "class from dominating evaluation."
            ),

        "roc_auc_ovr_macro":
            (
                "Additional probability-based multiclass "
                "metric using one-vs-rest evaluation."
            ),
    },

    "excluded_columns": {

        "user_id":
            "Identifier; no predictive meaning.",

        "sleep_latency_min":
            "Post-bedtime sleep outcome; excluded to reduce leakage.",

        "total_sleep_hours":
            "Direct sleep outcome strongly associated with the target; excluded to reduce target leakage.",

        "deep_sleep_pct":
            "Post-sleep physiological outcome.",

        "rem_sleep_pct":
            "Post-sleep physiological outcome.",

        "morning_alarm_snoozes":
            "Observed after the sleep episode.",

        "next_day_fatigue_score":
            "Observed after sleep and strongly related to sleep debt.",
    },
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False,
    )


print("\nMetadata saved to:")
print(METADATA_PATH)


# ============================================================
# 19. VERIFY SERIALIZED MODEL
# ============================================================

loaded_pipeline = joblib.load(
    MODEL_PATH
)

verification_predictions = loaded_pipeline.predict(
    X_test.iloc[:5]
)


print("\n========================================")
print("SERIALIZATION CHECK")
print("========================================")

print(
    "Reloaded model predictions:",
    verification_predictions.tolist(),
)

print("\nPipeline steps:")

for name, step in loaded_pipeline.named_steps.items():
    print(
        f"- {name}: "
        f"{type(step).__name__}"
    )


print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print(f"\nE1 -> train.py")
print(f"E2 -> model/model.pkl")
print(f"Metadata -> model/metadata.json")
