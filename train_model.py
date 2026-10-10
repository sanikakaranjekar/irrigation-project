"""
train_model.py
----------------
Trains and compares SEVEN classifiers on the irrigation dataset, then saves
the best-performing one as the production model.

Models compared:
    Logistic Regression, Decision Tree, K-Nearest Neighbors, SVM (RBF),
    Random Forest, XGBoost, LightGBM, CatBoost

Run:
    python train_model.py

Produces:
    model.joblib              -> best pipeline (preprocessing + model), used by app.py
    metrics.json               -> best model's detailed report (kept for backward compatibility)
    model_comparison.json      -> accuracy / macro-F1 for every model trained
"""

import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_PATH = "data/irrigation_prediction.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
COMPARISON_PATH = "model_comparison.json"
TARGET = "Irrigation_Need"

NUMERIC_FEATURES = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATEGORICAL_FEATURES = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, random_state=42, verbose=False, auto_class_weights="Balanced",
        ),
    }


# XGBoost's sklearn wrapper needs numeric labels + sample_weight instead of class_weight
SAMPLE_WEIGHT_MODELS = {"XGBoost"}


class LabelDecodingPipeline:
    """Wraps a fitted pipeline trained on encoded labels so predict() returns strings."""

    def __init__(self, pipeline, encoder):
        self.pipeline = pipeline
        self.encoder = encoder
        self.named_steps = pipeline.named_steps
        self.classes_ = encoder.classes_  # so app.py can read labels regardless of which model won

    def predict(self, X):
        return self.encoder.inverse_transform(self.pipeline.predict(X))

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    label_encoder = LabelEncoder().fit(y_train)
    y_train_enc = label_encoder.transform(y_train)
    sample_weights = compute_sample_weight("balanced", y_train_enc)

    results = {}
    best_name, best_pipeline, best_f1 = None, None, -1.0

    for name, estimator in get_models().items():
        print(f"\nTraining {name} ...")
        t0 = time.time()
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])

        if name in SAMPLE_WEIGHT_MODELS:
            pipeline.fit(X_train, y_train_enc, model__sample_weight=sample_weights)
            y_pred = label_encoder.inverse_transform(pipeline.predict(X_test))
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        elapsed = time.time() - t0
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"  accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  ({elapsed:.1f}s)")

        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "train_seconds": round(elapsed, 2),
            "report": report,
        }

        if macro_f1 > best_f1:
            best_name = name
            best_f1 = macro_f1
            best_pipeline = (
                LabelDecodingPipeline(pipeline, label_encoder)
                if name in SAMPLE_WEIGHT_MODELS
                else pipeline
            )

    with open(COMPARISON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(best_pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "accuracy": results[best_name]["accuracy"],
                "macro_f1": results[best_name]["macro_f1"],
                "report": results[best_name]["report"],
            },
            f, indent=2,
        )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by macro-F1)")
    print("=" * 60)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["macro_f1"]):
        print(f"{name:22s} acc={r['accuracy']*100:5.2f}%   macro_f1={r['macro_f1']*100:5.2f}%")
    print(f"\nBest model: {best_name}  -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
