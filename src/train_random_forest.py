from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


INPUT_FILE = Path(
    "data/processed/parkinson_patient_features.csv"
)


def main():

    print("=" * 70)
    print("PARKINSON'S RANDOM FOREST MODEL")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    X = df.drop(
        columns=["patient_id", "label"]
    )

    y = df["label"]

    print("\nDataset:")
    print(f"Patients: {len(df)}")
    print(f"Features: {X.shape[1]}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split:")
    print(f"Training patients: {len(X_train)}")
    print(f"Testing patients: {len(X_test)}")

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RANDOM FOREST PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy:           {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy:  {balanced_accuracy:.4f}"
    )

    print(
        f"Precision:          {precision:.4f}"
    )

    print(
        f"Recall:             {recall:.4f}"
    )

    print(
        f"F1 Score:           {f1:.4f}"
    )

    print(
        f"ROC-AUC:            {roc_auc:.4f}"
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Healthy",
                "Parkinson's",
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print("\nTop 20 Important Features:")

    print(
        importance.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()