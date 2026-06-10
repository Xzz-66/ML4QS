import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  
base_data_dir = BASE_DIR / "Datasets" 


def get_csv_paths():
    '''Function to retrun all paths of subjects' dataset '''
    

    subjects_path = {}

    for subject_path in base_data_dir.iterdir():
        if not subject_path.is_dir():
            continue

        processed_path = subject_path / "processed_data/combined_data"

        # print(processed_path)

        if not processed_path.exists():
            continue

        sessions_path = []

        for session_path in processed_path.iterdir():
            if not session_path.is_dir():
                continue

            csv_files = list(session_path.glob("*.csv"))
            sessions_path.extend(csv_files)

        subjects_path[subject_path.name] = sessions_path

    return subjects_path


def get_subjects():

    subjects = []

    for item in base_data_dir.iterdir():

        subjects.append(item.name)

    return sorted(subjects)


def load_datasets():
    csv_paths = get_csv_paths()

    subject_dataframes = {}
    for subject, files in csv_paths.items():

        dfs = []

        for file in files:

            df = pd.read_csv(file)
            df.dropna(inplace= True)   # drop na from aggregation


            dfs.append(df)

        # Combine all sessions for this subject
        combined_df = pd.concat(
            dfs,
            ignore_index=True
        )

        subject_dataframes[subject] = combined_df

    return subject_dataframes

def scale_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def perform_PCA(df, num_components = 0.9):
    pca = PCA(n_components=num_components)
    x = pca.fit_transform(df)

    print(f"\nExplained variance Ratio : {pca.explained_variance_ratio_}")
    print(f"\nSum of Variance ration: {pca.explained_variance_ratio_.sum()}")

    return x
    
def perform_lof(df):
    lof = LocalOutlierFactor(
        n_neighbors=5,
        contamination=0.05
    )

    lof_labels = lof.fit_predict(df)
    lof_scores = (lof.negative_outlier_factor_)

    # outlier_mask = (df[])

    return lof_labels, lof_scores



def detect_outliers(df):
    excluded_cols = ["exercise_type", "exercise_class", "session"]

    sensor_cols = [col for col in df.columns if col not in excluded_cols]
    results = []

    for exercise in df["exercise_type"].unique():
        print(f"Processing...: {exercise}")

        df_ = df[df["exercise_type"] == exercise].copy()
        # print(df_["exercise_type"].unique())

        X = df_[sensor_cols]
        X_scaled = scale_data(X)

        X_pca = perform_PCA(X_scaled)
        X_vis = perform_PCA(X_scaled, num_components=2)

        lof_l, lof_sc = perform_lof(X_pca)

        df_["LOF_Label"] = lof_l
        df_["LOF_Score"] = lof_sc


        df_["PC1"] = X_vis[:,0]
        df_["PC2"] = X_vis[:,1]

        outlier_mask = (df_["LOF_Label"] == -1)
        df_.loc[outlier_mask,sensor_cols] = np.nan
        df_[sensor_cols] = df_[sensor_cols].interpolate().ffill().bfill()

        results.append(df_)
    
    final_data = pd.concat(results, ignore_index= True)

   
    return final_data


def visualize_PCA(df):
    sensor_to_plot = "accel_Y (m/s^2)"
    for exercise in df["exercise_type"].unique():

        subset = df[df["exercise_type"] == exercise]
        normal = subset[subset["LOF_Label"] == 1]
        outliers = subset[subset["LOF_Label"] == -1]

        plt.figure(figsize=(10,8))
        plt.scatter(normal["PC1"], normal["PC2"], alpha=0.6, label="Normal")
        plt.scatter(outliers["PC1"], outliers["PC2"], alpha=1.0, label="Outlier")
        plt.title(f"{exercise}: PCA + LOF")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.show()

        if exercise == 'Pullup':
            plt.figure(figsize=(14,5))
            plt.plot(subset[sensor_to_plot].values, color='lightgrey', label='Raw signal')
            plt.scatter(outliers.index, outliers[sensor_to_plot], color='red', label='Outliers')        
            plt.title(f"{exercise} - {sensor_to_plot}")
            plt.xlabel("Time")
            plt.ylabel(sensor_to_plot)
            plt.legend()
            plt.show()


def table_missing_data(df):
    '''
    Dataframe of missing data from each variable
    
    :param df: Description
    '''
    missing_count = df.isna().sum(axis=0)
    missing_percent = missing_count / len(df) * 100

    missing_data = pd.DataFrame({
        'NaN Count': missing_count,
        'Percentage [%]': missing_percent
    }).sort_values(by='NaN Count', ascending=False)
    missing_data.index.name = 'Column Name'

    # missing_data = missing_data[missing_data['NaN Count'] > 0]
    return pd.DataFrame(missing_data)

        

def main():
    df = load_datasets()

    subjects = get_subjects()

    outlier_dfs = {}
    for num_subject, subject in enumerate(subjects, start= 1):

        outlier_dfs[f"subj{num_subject}"] = df[subject]
        outlier_dfs[f"subj{num_subject}"] = detect_outliers(outlier_dfs[f"subj{num_subject}"])
    

    visualize_PCA(outlier_dfs['subj1'])

    df_combined = pd.concat(outlier_dfs)

    output_dir = base_data_dir
    output_path = output_dir / 'DataSetPerSubjectClean.csv'
    df_combined.to_csv(output_path)
    print(f"\nSaved to {output_path}")

    # number and percentage of outliers
    print(table_missing_data(df_combined.loc["subj2"]))



if __name__ == '__main__':
    main()




