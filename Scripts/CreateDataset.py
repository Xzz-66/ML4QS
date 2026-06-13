import os 
import sys
import pandas as pd
import dask.dataframe as dd
from pathlib import Path


# sys.path.append(str(Path(__file__).parent.parent))
# from Python3Code.Chapter2.CreateDataset import CreateDataset
BASE_DIR = Path(__file__).resolve().parent.parent

base_data_dir = BASE_DIR / "Datasets" / "Subjects"
print(base_data_dir.resolve())

experiment_type = ['Pushup', 'Squat','Pullup']
sampling_rate = 50 #Hz
window_sec = 2

class_map = {
    "Pushup": 0,
    "Pullup": 1,
    "Squat": 2
}

# rename_map = {
#     "Abdullah": "subj01",
#     "Momo":   "subj02",
# }


files_to_process = [
    ('Gyroscope.csv', 'Time (s)', ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'], 'gyro_'),
    ('Linear accelerometer.csv', 'Time (s)', ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'], 'linacc_'),
    ('Accelerometer.csv', 'Time (s)', ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'], 'accel_'),
    ('Orientation.csv', 'Time (s)', ["w","x","y","z","Direct (°)","Yaw (°)","Pitch (°)","Roll (°)"], 'orient_'),
    ('Barometer.csv','Time (s)', ["X (hPa)"], 'pressure_')
]


def get_subjects():

    subjects = []

    for item in base_data_dir.iterdir():

        subjects.append(item.name)

    return sorted(subjects)

def get_experiment_sessions(subject, session_type):
    '''find all exercise sessions'''
    exp_dir = base_data_dir / subject / session_type
    if not exp_dir.exists():
        return []
    
    sessions = []
    for item in exp_dir.iterdir():
        if item.is_dir() and item.name.startswith('session_'):
            try:
                session_num = int(item.name.split('_')[1])
                sessions.append(session_num)
            except (IndexError, ValueError):
                continue
    return sorted(sessions)



def process_sessions(subject, session_type, session_num):
    """Process data for a specific session type and number using Dask"""
   
    input_dir = base_data_dir / subject /session_type / f'session_{session_num}'
    output_dir = base_data_dir / subject /'processed_data' / 'combined_data' / f'{session_type}_session_{session_num}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'combined_sensors.csv'

    print(f"\n=== Processing {session_type} experiment {session_num} ===")
    print(f"Input directory: {input_dir}")

        # Verify all required files exist
    for file, _, _, _ in files_to_process:
        if not (input_dir / file).exists():
            print(f"File {file} not found, skipping experiment")
            return False
        

    try:
        # Initialize empty DataFrame to hold combined data
        combined_df = None

        for file, time_col, value_cols, prefix in files_to_process:
            print(f"\nProcessing {file}...")
            file_path = input_dir / file

            # Load raw data with Dask
            raw_data = dd.read_csv(file_path)
            print(f"Raw data points: {raw_data.shape[0].compute()}")

            # Convert relative timestamps to time bins
            raw_data['time_bin'] = (raw_data[time_col] // window_sec) * window_sec

            # Aggregate values within each time window
            aggregated = raw_data.groupby('time_bin')[value_cols].mean()

            # Add prefix to column names
            aggregated.columns = [f"{prefix}{col}" for col in aggregated.columns]

            # Merge with combined DataFrame
            if combined_df is None:
                combined_df = aggregated
            else:
                combined_df = dd.merge(combined_df, aggregated, how='outer', left_index=True, right_index=True)

        # Add labels

        combined_df["exercise_class"] = (
            class_map[session_type]
        )

        combined_df["exercise_type"] = session_type

        combined_df["session"] = session_num

        # Compute results and convert to Pandas for final operations
        print("\nComputing final results...")
        combined_pd = combined_df.compute()

        combined_pd = combined_pd.sort_index()
        # combined_pd = combined_pd.ffill()

        # Save results
        print("\n=== Final Dataset ===")
        print(f"Total rows: {len(combined_pd)}")
        print(f"Time range: {combined_pd.index.min()} to {combined_pd.index.max()} seconds")
        print(combined_pd.head())

        combined_pd.to_csv(output_path)
        print(f"\nSaved to {output_path}")
        return True

    except Exception as e:
        print(f"Error processing experiment: {str(e)}")
        return False



subjects = get_subjects()

# print(f'\nProcessing {subject}')

for subject in subjects:
    for session in experiment_type:
        # Get all available experiment numbers for this type
        experiment_numbers = get_experiment_sessions(subject, session)

        print(f"{session} : {experiment_numbers}")

        if not experiment_numbers:
            print(f"\nNo experiments found for type: {session}")
            continue

        print(f"\nFound {len(experiment_numbers)} experiments for {session}: {experiment_numbers}")

        for exp_num in experiment_numbers:
            print(f"\n{'=' * 50}")
            print(f"Starting processing for {session} experiment {exp_num}")
            print(f"{'=' * 50}")

            success = process_sessions(subject, session, exp_num)

            if success:
                print(f"SUCCESS: Processed {session} experiment {exp_num}")
            else:
                print(f"FAILED: Processing {session} experiment {exp_num}")

print("\n=== Processing complete ===")
        

