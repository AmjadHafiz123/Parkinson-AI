from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

from pathlib import Path

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
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")
print(f"Expected features: {model.n_features_in_}")


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

dataset = pd.read_csv(DATA_PATH)

FEATURE_COLUMNS = [
    column
    for column in dataset.columns
    if column not in ["patient_id", "label"]
]

X = dataset[FEATURE_COLUMNS]

print(f"Dataset loaded: {DATA_PATH}")
print(f"Features found: {len(FEATURE_COLUMNS)}")


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

if len(FEATURE_COLUMNS) != model.n_features_in_:
    raise ValueError(
        f"Feature mismatch! "
        f"Model expects {model.n_features_in_} features, "
        f"but dataset contains {len(FEATURE_COLUMNS)} features."
    )


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_patient(patient_features):
    """
    Predict Parkinson's probability for one patient.

    Parameters
    ----------
    patient_features : pandas.DataFrame
        One row containing the 112 model features.

    Returns
    -------
    dict
        Prediction, probabilities and SHAP explanations.
    """

    if not isinstance(patient_features, pd.DataFrame):
        raise TypeError(
            "patient_features must be a pandas DataFrame"
        )

    if len(patient_features) != 1:
        raise ValueError(
            "patient_features must contain exactly one patient"
        )

    # --------------------------------------------------------
    # Ensure correct feature order
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in patient_features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    patient_features = patient_features[FEATURE_COLUMNS].copy()

    # --------------------------------------------------------
    # Check number of features
    # --------------------------------------------------------

    if patient_features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Expected {model.n_features_in_} features, "
            f"but received {patient_features.shape[1]}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(model.predict(patient_features)[0])

    probabilities = model.predict_proba(patient_features)[0]

    healthy_probability = float(probabilities[0])
    parkinson_probability = float(probabilities[1])

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_values = explainer.shap_values(patient_features)

    shap_values = np.asarray(shap_values)

    print(f"SHAP output shape: {shap_values.shape}")

    # SHAP output handling for binary Random Forest
    #
    # Depending on SHAP version/output format:
    #
    # (1, 112, 2) -> sample, feature, class
    # (1, 112)    -> sample, feature
    # (112, 2)    -> feature, class
    # (112,)      -> feature

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
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    patient_shap = np.asarray(patient_shap).flatten()

    if len(patient_shap) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"SHAP feature mismatch: "
            f"{len(patient_shap)} SHAP values for "
            f"{len(FEATURE_COLUMNS)} features."
        )

    # --------------------------------------------------------
    # Build explanation table
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_value": patient_features.iloc[0].values,
            "shap_value": patient_shap,
            "absolute_shap": np.abs(patient_shap),
        }
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )

    top_features = explanation.head(10)

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "prediction_label": (
            "Parkinson's"
            if prediction == 1
            else "Healthy"
        ),
        "healthy_probability": healthy_probability,
        "parkinson_probability": parkinson_probability,
        "top_features": top_features[
            [
                "feature",
                "feature_value",
                "shap_value",
            ]
        ].to_dict(orient="records"),
    }


# ============================================================
# TEST USING A REAL PATIENT
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PARKINSON AI INFERENCE TEST")
    print("=" * 60)

    # Use patient 1 from the processed dataset
    patient = dataset.iloc[[0]]

    patient_id = patient["patient_id"].iloc[0]
    actual_label = patient["label"].iloc[0]

    features = patient[FEATURE_COLUMNS]

    result = predict_patient(features)

    print(f"\nPatient ID: {patient_id}")

    print(
        "Actual label: "
        + ("Parkinson's" if actual_label == 1 else "Healthy")
    )

    print(
        f"Prediction: {result['prediction_label']}"
    )

    print(
        f"Healthy probability: "
        f"{result['healthy_probability']:.4f}"
    )

    print(
        f"Parkinson's probability: "
        f"{result['parkinson_probability']:.4f}"
    )

    print("\nTop contributing features:")

    for item in result["top_features"]:

        print(
            f"  {item['feature']:<30} "
            f"value={item['feature_value']:.6f} "
            f"SHAP={item['shap_value']:+.6f}"
        )

    print("\n" + "=" * 60)
    print("INFERENCE TEST COMPLETE")
    print("=" * 60)