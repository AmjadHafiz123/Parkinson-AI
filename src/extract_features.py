from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path(
    "data/pads-parkinsons-disease-smartwatch-dataset-1.0.0"
    "/pads-parkinsons-disease-smartwatch-dataset-1.0.0"
)

TIMESERIES_DIR = DATA_ROOT / "movement" / "timeseries"
LABEL_FILE = DATA_ROOT / "preprocessed" / "file_list.csv"

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "parkinson_features.csv"


# ============================================================
# Sensor feature extraction
# ============================================================

def extract_features(file_path):
    """
    Extract statistical features from one PADS
    smartwatch time-series recording.
    """

    # PADS files contain 7 numerical columns without headers
    df = pd.read_csv(file_path, header=None)

    if df.shape[1] != 7:
        raise ValueError(
            f"Unexpected number of columns in {file_path}: "
            f"{df.shape[1]}"
        )

    df.columns = [
        "time",
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_5",
        "signal_6",
    ]

    features = {}

    # Time information
    time = df["time"].values

    if len(time) > 1:
        features["duration"] = time[-1] - time[0]
        features["num_samples"] = len(time)

    # Extract features from the six sensor channels
    for column in [
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_5",
        "signal_6",
    ]:

        values = df[column].dropna().values

        if len(values) == 0:
            continue

        prefix = column

        features[f"{prefix}_mean"] = np.mean(values)
        features[f"{prefix}_std"] = np.std(values)
        features[f"{prefix}_min"] = np.min(values)
        features[f"{prefix}_max"] = np.max(values)
        features[f"{prefix}_median"] = np.median(values)
        features[f"{prefix}_range"] = np.max(values) - np.min(values)

        # Root Mean Square
        features[f"{prefix}_rms"] = np.sqrt(
            np.mean(values ** 2)
        )

        # Signal energy
        features[f"{prefix}_energy"] = np.sum(values ** 2)

        # Interquartile range
        q75, q25 = np.percentile(values, [75, 25])
        features[f"{prefix}_iqr"] = q75 - q25

    return features


# ============================================================
# Parse filename
# ============================================================

def parse_filename(filename):
    """
    Example:
        001_CrossArms_LeftWrist.txt

    Returns:
        patient_id = 001
        task = CrossArms
        wrist = LeftWrist
    """

    stem = filename.stem

    parts = stem.split("_")

    if len(parts) < 3:
        return None

    patient_id = parts[0]
    wrist = parts[-1]
    task = "_".join(parts[1:-1])

    return patient_id, task, wrist


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PADS PARKINSON'S SENSOR FEATURE EXTRACTION")
    print("=" * 70)

    print("\nLoading patient labels...")

    labels = pd.read_csv(LABEL_FILE)

    # Keep only Healthy and Parkinson's
    labels = labels[labels["label"].isin([0, 1])].copy()

    labels["id"] = labels["id"].astype(str).str.zfill(3)

    label_map = dict(
        zip(labels["id"], labels["label"])
    )

    print(f"Healthy + Parkinson patients: {len(labels)}")

    print("\nSearching sensor recordings...")

    files = list(TIMESERIES_DIR.glob("*.txt"))

    print(f"Sensor recordings found: {len(files)}")

    results = []

    for index, file_path in enumerate(files, start=1):

        parsed = parse_filename(file_path)

        if parsed is None:
            continue

        patient_id, task, wrist = parsed

        # Ignore patients that are not Healthy/Parkinson
        if patient_id not in label_map:
            continue

        try:

            features = extract_features(file_path)

            features["patient_id"] = patient_id
            features["task"] = task
            features["wrist"] = wrist
            features["label"] = label_map[patient_id]

            results.append(features)

        except Exception as error:

            print(
                f"\nERROR processing {file_path.name}: {error}"
            )

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(files)} files..."
            )

    print("\nCreating feature dataset...")

    result_df = pd.DataFrame(results)

    # Put identifiers first
    identifier_columns = [
        "patient_id",
        "task",
        "wrist",
        "label",
    ]

    other_columns = [
        column
        for column in result_df.columns
        if column not in identifier_columns
    ]

    result_df = result_df[
        identifier_columns + other_columns
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"\nOutput file:")
    print(OUTPUT_FILE.resolve())

    print(
        f"\nRows generated: {len(result_df):,}"
    )

    print(
        f"Columns generated: {len(result_df.columns):,}"
    )

    print("\nClass distribution:")

    print(
        result_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nPreview:")

    print(
        result_df.head().to_string()
    )


if __name__ == "__main__":
    main()