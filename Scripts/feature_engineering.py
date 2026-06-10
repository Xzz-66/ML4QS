import dask.dataframe as dd
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import entropy
from pathlib import Path
import warnings
import dask
import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from OutlierDetection import scale_data, perform_lof, perform_PCA
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_DATA_DIR = BASE_DIR / "Datasets" 

SAMPLE_RATE = 50
WINDOW_SIZES = [3.0]  # Window sizes in seconds TODO: compare different window sizes for performance
FREQ_WINDOW_SIZE = '5s'  # Window size for frequency features
MAX_DECIMALS = 4  # Maximum decimal places to keep
NPERSG = 50


def compute_frequency_features(data, fs=SAMPLE_RATE, nperseg=NPERSG):
    if len(data) < nperseg:
        return {k: np.nan for k in ['x_max_f', 'psd_entropy', 'spec_energy']}

    try:
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)

        if psd.sum() == 0:
            return {k: np.nan for k in ['x_max_f', 'psd_entropy', 'spec_energy']}

        psd_norm = psd / psd.sum()

        return {
            'x_max_f': round(freqs[np.argmax(psd)], MAX_DECIMALS),
            'psd_entropy': round(entropy(psd_norm), MAX_DECIMALS),
            'spec_energy': round(
                psd[(freqs >= 0.5) & (freqs <= 10)].sum(),
                MAX_DECIMALS
            )
        }

    except Exception:
        return {k: np.nan for k in ['x_max_f', 'psd_entropy', 'spec_energy']}



def compute_frequency_domain_features(
    df,
    value_cols,
    window_size=FREQ_WINDOW_SIZE,
    prefix='fd_'
):
    """Compute frequency domain features using pandas."""

    window_samples = int(
        pd.Timedelta(window_size).total_seconds() * SAMPLE_RATE
    )

    feature_dfs = []

    for col in value_cols:
        output_cols = [
            f'{prefix}{col}_{feat}'
            for feat in ['x_max_f', 'psd_entropy', 'spec_energy']
        ]

        result = pd.DataFrame(
            index=df.index,
            columns=output_cols,
            dtype='float64'
        )

        series = df[col]

        for i in range(len(series)):
            start = max(0, i - window_samples + 1)
            window = series.iloc[start:i + 1].values

            if len(window) < NPERSG or np.isnan(window).any():
                result.iloc[i] = [np.nan] * 3
                continue

            features = compute_frequency_features(window)

            result.iloc[i] = [
                features['x_max_f'],
                features['psd_entropy'],
                features['spec_energy']
            ]

        feature_dfs.append(result)

    return pd.concat(feature_dfs, axis=1)



def compute_time_domain_features(ddf, value_columns, window_sizes=WINDOW_SIZES, prefix=''):
    """Time domain features using pure Dask operations"""
    features_list = []

    for col in value_columns:
        for window_size in window_sizes:
            window_samples = int(window_size * SAMPLE_RATE)
            w_str = str(window_size).replace(".", "_") + "s"

            roller = ddf[col].rolling(window_samples, min_periods=1)

            features = pd.concat([
                roller.mean().rename(f'{prefix}{col}_mean_{w_str}'),
                roller.std().rename(f'{prefix}{col}_std_{w_str}'),
                roller.max().rename(f'{prefix}{col}_max_{w_str}'),
                roller.min().rename(f'{prefix}{col}_min_{w_str}'),
                roller.var().rename(f'{prefix}{col}_var_{w_str}'),
                roller.kurt().rename(f'{prefix}{col}_kurt_{w_str}'),
                roller.skew().rename(f'{prefix}{col}_skew_{w_str}'),

                (roller.max() - roller.min()).rename(f'{prefix}{col}_range_{w_str}'),
                roller.apply(lambda x: np.sqrt((x ** 2).mean()), raw=True).rename(f'{prefix}{col}_rms_{w_str}')
            ], axis=1)

            features_list.append(features)

    return pd.concat(features_list, axis=1)


# Vector magnitude (combined acceleration and guro)
def compute_vector_magnitude(ddf):

    # Compute vector magnitude
    ddf['accel_magnitude'] = np.sqrt(
        ddf['accel_x'] ** 2 +
        ddf['accel_y'] ** 2 +
        ddf['accel_z'] ** 2
    )

    ddf["gyro_magnitude"] = np.sqrt(
        ddf["gyro_x"]**2 +
        ddf["gyro_y"]**2 +
        ddf["gyro_z"]**2
    )
    return ddf




def process_combined_data():
    """Process the combined dataset with robust feature engineering"""
    input_path = BASE_DATA_DIR / 'DataSetPerSubjectClean.csv'

        # Read and clean data
    df_ = pd.read_csv(input_path, dtype={
        'session': 'str',
        'exercise_type': 'str',
        'time_bin': 'float64',
        'exercise_class': 'int64'
    }).rename(columns={
        'accel_X (m/s^2)': 'accel_x',
        'accel_Y (m/s^2)': 'accel_y',
        'accel_Z (m/s^2)': 'accel_z',
        'gyro_X (rad/s)': 'gyro_x',
        'gyro_Y (rad/s)': 'gyro_y',
        'gyro_Z (rad/s)': 'gyro_z',
        'linacc_X (m/s^2)': 'linacc_X',
        'linacc_Y (m/s^2)': 'linacc_Y',
        'linacc_Z (m/s^2)': 'linacc_Z',
    })

    cols_to_drop = ['Unnamed: 1', 'LOF_Label', 'LOF_Score', 'PC1', 'PC2']

    df_.drop(columns= [col for col in cols_to_drop if col in df_.columns], inplace= True)

    df_.rename(columns={'Unnamed: 0' : 'subject_id'}, inplace= True)
    # Set index
    # if 'subject_id' in df_.columns:
    #     df_ = df_.set_index('subject_id')


    df_ = compute_vector_magnitude(df_)

    excluded_cols = ["subject_id", "time_bin", "unique_exer_id", "exercise_type", "exercise_class", "session"]

    sensor_cols = [col for col in df_.columns if col not in excluded_cols]


    results = []
    for (subj, exercise, session), group_df in df_.groupby(['subject_id', 'exercise_type', 'session']):
        print(f"Processing {subj} - {exercise}")
        
        time_feats = compute_time_domain_features(group_df, sensor_cols)
        freq_feats = compute_frequency_domain_features(group_df, sensor_cols)

        # combine features
        feats = pd.concat([time_feats, freq_feats], axis = 1)

        feats['subject_id']    = subj
        feats["exercise_type"] = exercise
        feats["session"] = session
        feats["exercise_class"] = group_df["exercise_class"].iloc[0]

        feature_cols = [col for col in feats.columns if col not in ['subject_id', 'exercise_type', 'session', 'exercise_class']]

        X = feats[feature_cols]

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how= 'all')

        # Remove correlated features
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X = X.drop(columns = to_drop)

        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index = X.index)

        if len(X) < 20:
            feats["LOF_Label"] = 1
            feats["LOF_Score"] = np.nan

            results.append(feats)
            continue


        X_scaled = scale_data(X)
        X_pca = perform_PCA(X_scaled)
        X_vis = perform_PCA(X_scaled, num_components=2)

        feats["PC1"] = X_vis[:,0]
        feats["PC2"] = X_vis[:,1]

        labels, scores = perform_lof(X_pca)

        feats["LOF_Label"] = labels
        feats["LOF_Score"] = scores

        feats = feats[feats["LOF_Label"] == 1].copy()


        results.append(feats)

        

    features_df = pd.concat(results, ignore_index=True)  

    return features_df
    

    
# def clean_engineered_data(df):
#     drop_cols = ["subject_id", "session", "exercise_type"]

#     X = df.drop(columns = drop_cols + ["exercise_class"])

#     imputer = SimpleImputer(strategy="median")
#     X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#     X_imputed = pd.DataFrame(imputer.fit_transform(X), columns= X.columns, index = X.index)

    

#     X_scaled = scale_data(X)
#     X_pca = perform_PCA(X_scaled)
#     X_vis = perform_PCA(X_scaled, num_components=2)

#     df["PC1"] = X_vis[:,0]
#     df["PC2"] = X_vis[:,1]

#     labels, scores = perform_lof(X_pca)

#     df["LOF_Label"] = labels
#     df["LOF_Score"] = scores

#     clean_mask = df["LOF_Label"] == 1
#     cleaned_feat_df = (df.loc[clean_mask].copy())

#     cleaned_indices = cleaned_feat_df.index
#     X_clean = X_imputed.loc[cleaned_indices]

#     # X_clean = cleaned_feat_df[X.columns]
#     # X_clean = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
#     y_clean = cleaned_feat_df["exercise_class"]

#     print(X_clean.isna().sum().sum())

#     return X_clean, y_clean


def feature_importance(df, top_k = 50):
    output_dir = BASE_DATA_DIR / 'engineered_data'
    output_path1 = output_dir / 'cleaned_engineered_data.csv'
    output_path2 = output_dir / 'feature_importance.csv'


    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = ["subject_id", "exercise_type", "session", "PC1", "PC2", "LOF_Label", "LOF_Score"]

    X = df.drop(columns = drop_cols + ["exercise_class"])
    y = df["exercise_class"]


    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.dropna(axis=1, how= 'all')

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
        n_jobs=-1
    )

    rf.fit(X,y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values("importance", ascending=False)
    print(importance_df.head(30))

    selected_features = importance_df.head(top_k)["feature"].tolist()
    X_selected = X[selected_features]

    metadata = df[["subject_id", "exercise_type","session"]].reset_index(drop= True)


    
    finalDataset = pd.concat([metadata, X_selected.reset_index(drop = True), y.reset_index(drop=True)], axis= 1)
    finalDataset.to_csv(output_path1, index = False)
    print(f"\nSaved data to {output_path1}")

    importance_df.to_csv(output_path2, index= False)
    print(f"\nSaved important features to {output_path2}")

    return finalDataset, importance_df


def main():
    df = process_combined_data()

    # print(X_clean.isna().sum().sum())
    df_extracted_features, importance_feats_ = feature_importance(df)

    print(df_extracted_features.head(10))



if __name__ == "__main__":
    main()











    





