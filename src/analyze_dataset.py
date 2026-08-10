from pathlib import Path
import pandas as pd


DATA_FILE = Path(
    "data/processed/parkinson_features.csv"
)


def main():

    print("=" * 70)
    print("PARKINSON FEATURE DATASET ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # Patient counts
    # ---------------------------------------------------------

    print("\nUnique patients:")
    print(df["patient_id"].nunique())

    print("\nPatients by class:")

    patient_labels = (
        df[["patient_id", "label"]]
        .drop_duplicates()
    )

    print(
        patient_labels["label"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Record counts
    # ---------------------------------------------------------

    print("\nRecordings by class:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Recordings per patient
    # ---------------------------------------------------------

    recordings_per_patient = (
        df.groupby("patient_id")
        .size()
    )

    print("\nRecordings per patient:")

    print(
        recordings_per_patient.describe()
    )

    # ---------------------------------------------------------
    # Tasks
    # ---------------------------------------------------------

    print("\nMovement tasks:")

    print(
        sorted(df["task"].unique())
    )

    print(
        "\nNumber of movement tasks:",
        df["task"].nunique()
    )

    # ---------------------------------------------------------
    # Wrist
    # ---------------------------------------------------------

    print("\nWrist distribution:")

    print(
        df["wrist"].value_counts()
    )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing)

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    print("\nDuplicate rows:")

    print(
        df.duplicated().sum()
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()