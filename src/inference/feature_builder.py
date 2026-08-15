from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# SENSOR FEATURE EXTRACTION
# ============================================================

SENSOR_COLUMNS = [
    "signal_1",
    "signal_2",
    "signal_3",
    "signal_4",
    "signal_5",
    "signal_6",
]


def extract_features(file_path):
    """
    Extract the same recording-level features used during training.

    Expected PADS recording format:
        time, signal_1, ..., signal_6
    """

    df = pd.read_csv(file_path, header=None)

    if df.shape[1] != 7:
        raise ValueError(
            f"Expected 7 columns in {file_path.name}, "
            f"but found {df.shape[1]}."
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

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    time = df["time"].dropna().values

    if len(time) > 1:
        features["duration"] = time[-1] - time[0]
        features["num_samples"] = len(time)

    # --------------------------------------------------------
    # Sensor features
    # --------------------------------------------------------

    for column in SENSOR_COLUMNS:

        values = df[column].dropna().values

        if len(values) == 0:
            continue

        prefix = column

        features[f"{prefix}_mean"] = np.mean(values)
        features[f"{prefix}_std"] = np.std(values)
        features[f"{prefix}_min"] = np.min(values)
        features[f"{prefix}_max"] = np.max(values)
        features[f"{prefix}_median"] = np.median(values)
        features[f"{prefix}_range"] = (
            np.max(values) - np.min(values)
        )

        features[f"{prefix}_rms"] = np.sqrt(
            np.mean(values ** 2)
        )

        features[f"{prefix}_energy"] = np.sum(
            values ** 2
        )

        q75, q25 = np.percentile(
            values,
            [75, 25]
        )

        features[f"{prefix}_iqr"] = q75 - q25

    return features


# ============================================================
# BUILD NEW USER FEATURES
# ============================================================

def build_patient_features(recording_files):
    """
    Convert multiple smartwatch recordings into the exact
    112-feature format expected by the trained model.

    Parameters
    ----------
    recording_files:
        List of paths to .txt smartwatch recordings.

    Returns
    -------
    pandas.DataFrame
        One row containing 112 model features.
    """

    if not recording_files:
        raise ValueError(
            "No smartwatch recordings were provided."
        )

    recording_features = []

    for file_path in recording_files:

        features = extract_features(file_path)

        if features:
            recording_features.append(features)

    if not recording_features:
        raise ValueError(
            "Could not extract features from the uploaded recordings."
        )

    recording_df = pd.DataFrame(
        recording_features
    )

    # --------------------------------------------------------
    # Match training pipeline
    #
    # Training did:
    #
    # groupby(patient_id).mean()
    # groupby(patient_id).std()
    #
    # For a new patient, all uploaded recordings belong
    # to one patient.
    # --------------------------------------------------------

    patient_mean = (
        recording_df.mean()
        .add_suffix("_mean")
    )

    patient_std = (
        recording_df.std()
        .add_suffix("_std")
    )

    patient_features = pd.concat(
        [
            patient_mean,
            patient_std,
        ]
    )

    # Convert Series -> one-row DataFrame
    patient_features = pd.DataFrame(
        [patient_features]
    )

    return patient_features


# ============================================================
# ZIP SUPPORT
# ============================================================

def build_features_from_zip(zip_path):
    """
    Extract a ZIP containing smartwatch .txt recordings
    and generate the 112 patient-level features.

    Returns
    -------
    patient_features : pandas.DataFrame
    recording_count : int
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            # Security: reject path traversal
            for member in zip_ref.namelist():

                member_path = Path(member)

                if member_path.is_absolute():
                    raise ValueError(
                        "ZIP contains an invalid absolute path."
                    )

                if ".." in member_path.parts:
                    raise ValueError(
                        "ZIP contains an invalid path."
                    )

            zip_ref.extractall(temp_dir)

        # Find smartwatch recordings
        recording_files = list(
            temp_dir.rglob("*.txt")
        )

        if not recording_files:
            raise ValueError(
                "No .txt smartwatch recordings were found "
                "inside the ZIP file."
            )

        patient_features = build_patient_features(
            recording_files
        )

        return (
            patient_features,
            len(recording_files),
        )