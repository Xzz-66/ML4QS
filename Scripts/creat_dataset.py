import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dask.dataframe as dd


# ============================================================
# PATH SETUP
# ============================================================

sys.path.append(str(Path(__file__).parent.parent))

# from Python3Code.Chapter2.CreateDataset import CreateDataset


# ============================================================
# CONFIGURATION
# ============================================================

base_data_dir = Path("Datasets/Abdullah")

# FIXED:
# - Consistent naming
# - Removed plural mismatches
experiment_types = ["Pushup", "Squat", "Pullup"]

# FIXED:
# - Explicit separation between:
#   sampling rate
#   window duration
#   window samples
sampling_rate = 50  # Hz

window_sec = 2  # seconds

window_samples = sampling_rate * window_sec  # 100 samples

overlap = 0.5

# FIXED:
# - Step size for overlapping sliding windows
step_samples = int(window_samples * (1 - overlap))


# ============================================================
# LABEL MAPPING
# ============================================================

# FIXED:
# - Cleaner and safer than chained if-statements
class_map = {
    "Pushup": 0,
    "Pullup": 1,
    "Squat": 2
}


# ============================================================
# SENSOR FILE CONFIGURATION
# ============================================================

# FIXED:
# - Added missing comma
files_to_process = [
    (
        "Gyroscope.csv",
        "Time (s)",
        ["X (rad/s)", "Y (rad/s)", "Z (rad/s)"],
        "gyro_"
    ),

    (
        "Linear accelerometer.csv",
        "Time (s)",
        ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
        "linacc_"
    ),

    (
        "Accelerometer.csv",
        "Time (s)",
        ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"],
        "accel_"
    ),

    (
        "Orientation.csv",
        "Time (s)",
        ["Yaw (°)", "Pitch (°)", "Roll (°)"],
        "orient_"
    ),

    (
        "Barometer.csv",
        "Time (s)",
        ["X (hPa)"],
        "pressure_"
    )
]


# ============================================================
# FIND AVAILABLE SESSIONS
# ============================================================

def get_experiment_sessions(session_type):
    """
    Find all available sessions for an exercise type.
    """

    exp_dir = base_data_dir / session_type

    if not exp_dir.exists():
        return []

    sessions = []

    for item in exp_dir.iterdir():

        if item.is_dir() and item.name.startswith("session_"):

            try:
                session_num = int(item.name.split("_")[1])
                sessions.append(session_num)

            except (IndexError, ValueError):
                continue

    return sorted(sessions)


# ============================================================
# LOAD AND ALIGN SENSOR DATA
# ============================================================

def load_sensor_data(file_path, time_col, value_cols, prefix):
    """
    Load sensor CSV and prepare dataframe.
    """

    df = pd.read_csv(file_path)

    # FIXED:
    # - Sort by time
    # - Ensures proper temporal alignment
    df = df.sort_values(time_col)

    # Keep only required columns
    keep_cols = [time_col] + value_cols

    df = df[keep_cols]

    # Rename feature columns
    rename_map = {
        col: f"{prefix}{col}"
        for col in value_cols
    }

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_window_features(window_df, feature_columns):
    """
    Extract statistical features from a sliding window.
    """

    features = {}

    for col in feature_columns:

        signal = window_df[col].values

        # ====================================================
        # TIME DOMAIN FEATURES
        # ====================================================

        features[f"{col}_mean"] = np.mean(signal)

        features[f"{col}_std"] = np.std(signal)

        features[f"{col}_min"] = np.min(signal)

        features[f"{col}_max"] = np.max(signal)

        features[f"{col}_rms"] = np.sqrt(np.mean(signal ** 2))

        features[f"{col}_energy"] = np.sum(signal ** 2)

        # ====================================================
        # FREQUENCY DOMAIN FEATURE
        # ====================================================

        fft_vals = np.abs(np.fft.rfft(signal))

        features[f"{col}_fft_mean"] = np.mean(fft_vals)

        features[f"{col}_fft_max"] = np.max(fft_vals)

    return features


# ============================================================
# MAIN SESSION PROCESSING
# ============================================================

def process_session(session_type, session_num):

    input_dir = (
        base_data_dir
        / session_type
        / f"session_{session_num}"
    )

    output_dir = (
        base_data_dir
        / "processed_data"
        / "combined_data"
        / f"{session_type}_session_{session_num}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "feature_windows.csv"

    print("\n================================================")
    print(f"Processing {session_type} session {session_num}")
    print("================================================")

    # ========================================================
    # VERIFY FILES EXIST
    # ========================================================

    for file, _, _, _ in files_to_process:

        if not (input_dir / file).exists():

            print(f"Missing file: {file}")

            return False

    try:

        sensor_dfs = []

        # ====================================================
        # LOAD ALL SENSOR FILES
        # ====================================================

        for file, time_col, value_cols, prefix in files_to_process:

            print(f"\nLoading {file}")

            file_path = input_dir / file

            df = load_sensor_data(
                file_path,
                time_col,
                value_cols,
                prefix
            )

            sensor_dfs.append(df)

        # ====================================================
        # MERGE SENSOR STREAMS
        # ====================================================

        print("\nMerging sensor streams...")

        combined_df = sensor_dfs[0]

        for df in sensor_dfs[1:]:

            combined_df = pd.merge_asof(
                combined_df.sort_values("Time (s)"),
                df.sort_values("Time (s)"),
                on="Time (s)",
                direction="nearest"
            )

        # ====================================================
        # REMOVE MISSING VALUES
        # ====================================================

        combined_df = combined_df.dropna()

        print(f"Combined samples: {len(combined_df)}")

        # ====================================================
        # FEATURE EXTRACTION USING SLIDING WINDOWS
        # ====================================================

        print("\nExtracting sliding window features...")

        feature_columns = [
            col for col in combined_df.columns
            if col != "Time (s)"
        ]

        feature_rows = []

        # FIXED:
        # - TRUE overlapping sliding windows
        # - Instead of simple groupby averaging
        for start_idx in range(
            0,
            len(combined_df) - window_samples,
            step_samples
        ):

            end_idx = start_idx + window_samples

            window = combined_df.iloc[start_idx:end_idx]

            features = extract_window_features(
                window,
                feature_columns
            )

            # Add metadata
            features["exercise_class"] = class_map[session_type]

            features["exercise_type"] = session_type

            features["session"] = session_num

            features["window_start_time"] = window["Time (s)"].iloc[0]

            feature_rows.append(features)

        # ====================================================
        # FINAL DATAFRAME
        # ====================================================

        feature_df = pd.DataFrame(feature_rows)

        print("\nFinal feature dataset:")
        print(feature_df.head())

        print(f"\nTotal windows: {len(feature_df)}")

        # ====================================================
        # SAVE
        # ====================================================

        feature_df.to_csv(output_path, index=False)

        print(f"\nSaved to:\n{output_path}")

        return True

    except Exception as e:

        print(f"\nERROR:\n{str(e)}")

        return False


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

for session_type in experiment_types:

    session_numbers = get_experiment_sessions(session_type)

    if not session_numbers:

        print(f"\nNo sessions found for {session_type}")

        continue

    print(
        f"\nFound {len(session_numbers)} sessions for "
        f"{session_type}: {session_numbers}"
    )

    for session_num in session_numbers:

        success = process_session(
            session_type,
            session_num
        )

        if success:

            print(
                f"\nSUCCESS: "
                f"{session_type} session {session_num}"
            )

        else:

            print(
                f"\nFAILED: "
                f"{session_type} session {session_num}"
            )

print("\n================================================")
print("PROCESSING COMPLETE")
print("================================================")