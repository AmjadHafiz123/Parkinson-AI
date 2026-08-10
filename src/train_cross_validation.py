from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


INPUT_FILE = Path(
    "data/processed/parkinson_patient_features.csv"
)


def main():

    print("=" * 70)
    print("PARKINSON'S PATIENT-LEVEL 5-FOLD CROSS-VALIDATION")
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
    # 5-fold stratified CV
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"FOLD {fold}")
        print("-" * 70)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print(
            f"Training patients: {len(X_train)}"
        )

        print(
            f"Testing patients: {len(X_test)}"
        )

        # -----------------------------------------------------
        # Random Forest
        # -----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        print("\nTraining Random Forest...")

        model.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------------------
        # Predictions
        # -----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred,
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability,
        )

        print(
            f"Accuracy:           {accuracy:.4f}"
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

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

    # ---------------------------------------------------------
    # Results table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Mean and standard deviation
    # ---------------------------------------------------------

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    summary = []

    for metric in metric_columns:

        summary.append(
            {
                "metric": metric,
                "mean": results_df[metric].mean(),
                "std": results_df[metric].std(),
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    for _, row in summary_df.iterrows():

        print(
            f"{row['metric']:20s} "
            f"{row['mean']:.4f} "
            f"+/- "
            f"{row['std']:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_file = Path(
        "data/processed/cross_validation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nResults saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()