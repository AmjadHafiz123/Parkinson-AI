from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
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


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path(
    "data/processed/parkinson_features.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("PARKINSON'S BASELINE MODEL")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset:")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print(
    f"Unique patients: {df['patient_id'].nunique()}"
)


# ============================================================
# Prepare features
# ============================================================

# Remove identifiers / non-numeric columns
X = df.drop(
    columns=[
        "patient_id",
        "task",
        "wrist",
        "label",
    ]
)

y = df["label"]

groups = df["patient_id"]


print("\nFeature columns:")
print(len(X.columns))

print("\nClass distribution:")
print(y.value_counts().sort_index())


# ============================================================
# Patient-level train/test split
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]


# ============================================================
# Verify no patient leakage
# ============================================================

train_patients = set(
    groups_train.unique()
)

test_patients = set(
    groups_test.unique()
)

overlap = train_patients.intersection(
    test_patients
)

print("\n" + "=" * 70)
print("PATIENT SPLIT")
print("=" * 70)

print(
    f"\nTraining patients: {len(train_patients)}"
)

print(
    f"Testing patients: {len(test_patients)}"
)

print(
    f"Patient overlap: {len(overlap)}"
)

if len(overlap) > 0:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED!"
    )

print(
    "\n✓ No patient appears in both "
    "training and testing."
)


# ============================================================
# Pipeline
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# Train
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# Evaluation
# ============================================================

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


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy:           {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
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


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Healthy",
            "Parkinson's"
        ],
        zero_division=0,
    )
)


print("\n" + "=" * 70)
print("BASELINE MODEL COMPLETE")
print("=" * 70)