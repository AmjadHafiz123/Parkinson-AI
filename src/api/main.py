from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap"
    / "random_forest_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "parkinson_patient_features.csv"
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Parkinson AI API",
    description="Patient-level Parkinson's disease prediction API",
    version="1.0.0",
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_PATH)
dataset = pd.read_csv(DATA_PATH)

FEATURE_COLUMNS = [
    column
    for column in dataset.columns
    if column not in ["patient_id", "label"]
]

explainer = shap.TreeExplainer(model)


# ============================================================
# VALIDATION
# ============================================================

if len(FEATURE_COLUMNS) != model.n_features_in_:
    raise RuntimeError(
        f"Model expects {model.n_features_in_} features, "
        f"but dataset contains {len(FEATURE_COLUMNS)}."
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "application": "Parkinson AI",
        "status": "running",
        "model": "Random Forest",
        "features": model.n_features_in_,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "dataset_loaded": True,
    }


# ============================================================
# LIST PATIENTS
# ============================================================

@app.get("/patients")
def get_patients():
    patients = dataset["patient_id"].tolist()

    return {
        "count": len(patients),
        "patients": patients,
    }


# ============================================================
# PREDICT PATIENT
# ============================================================

@app.get("/predict/{patient_id}")
def predict_patient(patient_id: int):

    # --------------------------------------------------------
    # Find patient
    # --------------------------------------------------------

    patient = dataset[
        dataset["patient_id"] == patient_id
    ]

    if patient.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found.",
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = patient[FEATURE_COLUMNS].copy()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(features)[0]
    )

    probabilities = model.predict_proba(features)[0]

    healthy_probability = float(probabilities[0])
    parkinson_probability = float(probabilities[1])

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = explainer.shap_values(features)
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        patient_shap = shap_values[0, :, 1]

    elif shap_values.ndim == 2:
        if shap_values.shape[1] == 2:
            patient_shap = shap_values[:, 1]
        else:
            patient_shap = shap_values[0]

    elif shap_values.ndim == 1:
        patient_shap = shap_values

    else:
        raise RuntimeError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    patient_shap = np.asarray(
        patient_shap
    ).flatten()

    # --------------------------------------------------------
    # Explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": features.iloc[0].values,
            "shap_value": patient_shap,
            "absolute_shap": np.abs(patient_shap),
        }
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False,
    )

    top_features = explanation.head(10)

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "patient_id": patient_id,

        "prediction": (
            "Parkinson's"
            if prediction == 1
            else "Healthy"
        ),

        "prediction_class": prediction,

        "healthy_probability": round(
            healthy_probability,
            4,
        ),

        "parkinson_probability": round(
            parkinson_probability,
            4,
        ),

        "top_features": [
            {
                "feature": row["feature"],
                "value": float(row["value"]),
                "shap_value": float(row["shap_value"]),
            }
            for _, row in top_features.iterrows()
        ],
    }