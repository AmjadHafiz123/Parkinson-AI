from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/parkinson_patient_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 70)
    print("PARKINSON'S EXPLAINABLE AI - SHAP ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        columns=["patient_id", "label"]
    )

    y = df["label"]

    print("\nDataset:")
    print(f"Patients: {len(df)}")
    print(f"Features: {X.shape[1]}")

    # -----------------------------------------------------
    # Train final Random Forest
    # -----------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X,
        y
    )

    print("Training complete.")

    # -----------------------------------------------------
    # SHAP Tree Explainer
    # -----------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.TreeExplainer(
        model
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X
    )

    print("SHAP calculation complete.")

    # -----------------------------------------------------
    # Handle SHAP output format
    # -----------------------------------------------------

    if isinstance(shap_values, list):

        # Binary classification:
        # class 1 = Parkinson's
        shap_parkinson = shap_values[1]

    else:

        shap_parkinson = shap_values

        # Some SHAP versions return:
        # samples x features x classes

        if shap_parkinson.ndim == 3:

            shap_parkinson = shap_parkinson[:, :, 1]

    # -----------------------------------------------------
    # Global feature importance
    # -----------------------------------------------------

    print("\nCalculating global feature importance...")

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": (
                abs(shap_parkinson).mean(axis=0)
            ),
        }
    )

    feature_importance = feature_importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    print("\nTop 20 SHAP Features:")

    print(
        feature_importance.head(20).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Save feature importance
    # -----------------------------------------------------

    importance_file = (
        OUTPUT_DIR /
        "shap_feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_file,
        index=False
    )

    print(
        f"\nSaved:\n{importance_file}"
    )

    # -----------------------------------------------------
    # SHAP bar plot
    # -----------------------------------------------------

    print("\nCreating SHAP feature importance plot...")

    plt.figure()

    shap.summary_plot(
        shap_parkinson,
        X,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    bar_plot = (
        OUTPUT_DIR /
        "shap_feature_importance.png"
    )

    plt.savefig(
        bar_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{bar_plot}"
    )

    # -----------------------------------------------------
    # SHAP beeswarm plot
    # -----------------------------------------------------

    print("\nCreating SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_parkinson,
        X,
        show=False
    )

    plt.tight_layout()

    summary_plot = (
        OUTPUT_DIR /
        "shap_summary.png"
    )

    plt.savefig(
        summary_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{summary_plot}"
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_file = (
        OUTPUT_DIR /
        "random_forest_model.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"\nModel saved:\n{model_file}"
    )

    # -----------------------------------------------------
    # Individual patient explanation
    # -----------------------------------------------------

    patient_index = 0

    patient_id = df.iloc[
        patient_index
    ]["patient_id"]

    actual_label = df.iloc[
        patient_index
    ]["label"]

    patient_features = X.iloc[
        [patient_index]
    ]

    prediction = model.predict(
        patient_features
    )[0]

    probability = model.predict_proba(
        patient_features
    )[0, 1]

    patient_shap = shap_parkinson[
        patient_index
    ]

    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_value": patient_features.iloc[0].values,
            "shap_value": patient_shap,
            "absolute_shap": abs(patient_shap),
        }
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )

    explanation_file = (
        OUTPUT_DIR /
        f"patient_{patient_id}_explanation.csv"
    )

    explanation.to_csv(
        explanation_file,
        index=False
    )

    # -----------------------------------------------------
    # Print individual explanation
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("INDIVIDUAL PATIENT EXPLANATION")
    print("=" * 70)

    print(
        f"\nPatient ID: {patient_id}"
    )

    print(
        f"Actual label: {actual_label}"
    )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Parkinson's probability: "
        f"{probability:.4f}"
    )

    print("\nTop contributing features:")

    print(
        explanation.head(10)[
            [
                "feature",
                "feature_value",
                "shap_value",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nExplanation saved:\n"
        f"{explanation_file}"
    )

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()