from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/parkinson_features.csv"
)

OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = (
    OUTPUT_DIR / "parkinson_patient_features.csv"
)


def main():

    print("=" * 70)
    print("CREATING PATIENT-LEVEL DATASET")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print("\nRecording-level dataset:")
    print(f"Rows: {len(df):,}")
    print(f"Patients: {df['patient_id'].nunique()}")

    # ---------------------------------------------------------
    # Numerical feature columns
    # ---------------------------------------------------------

    identifier_columns = [
        "patient_id",
        "task",
        "wrist",
        "label",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in identifier_columns
    ]

    print(
        f"\nNumerical features: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # Patient-level aggregation
    # ---------------------------------------------------------

    print("\nAggregating recordings per patient...")

    patient_mean = (
        df.groupby("patient_id")[feature_columns]
        .mean()
        .add_suffix("_mean")
    )

    patient_std = (
        df.groupby("patient_id")[feature_columns]
        .std()
        .add_suffix("_std")
    )

    # ---------------------------------------------------------
    # Patient labels
    # ---------------------------------------------------------

    patient_labels = (
        df.groupby("patient_id")["label"]
        .first()
        .to_frame()
    )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    patient_df = pd.concat(
        [
            patient_mean,
            patient_std,
            patient_labels,
        ],
        axis=1,
    )

    patient_df = patient_df.reset_index()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    patient_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATIENT DATASET CREATED")
    print("=" * 70)

    print(
        f"\nPatients: {len(patient_df):,}"
    )

    print(
        f"Columns: {len(patient_df.columns):,}"
    )

    print("\nClass distribution:")

    print(
        patient_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")

    print(
        patient_df.isna().sum().sum()
    )

    print("\nOutput:")
    print(OUTPUT_FILE.resolve())

    print("\nPreview:")

    print(
        patient_df.head().to_string()
    )


if __name__ == "__main__":
    main()