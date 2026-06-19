from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = BASE_DIR / "Datasets" 
PLOT_DIR   = BASE_DIR / "plots"
RANDOM_STATE = 42
tune = True


def load_data():

    input_path = BASE_DATA_DIR / "engineered_data" / "cleaned_engineered_data.csv"

    df = pd.read_csv(input_path)
    return df


def process_data(df):

    excluded_cols = ["subject_id", "exercise_type", "unique_exer_id", "session", "exercise_class"]
    X = df.drop(columns=excluded_cols)
    y = df["exercise_class"]

    groups = df["subject_id"]
    exercise_groups = df["unique_exer_id"]

    return X, y, groups, exercise_groups


def evaluate_model(y_test, y_pred, fold, all_classes = [0, 1, 2]):
    """Evaluate the model performance"""
    # y_pred = model.predict(X_test)
    # y_proba = model.predict_proba(X_test)

    print(f"\nClasses present in test fold {fold}: {sorted(set(y_test))}")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels= all_classes, zero_division = 0)
    recall = recall_score(y_test, y_pred, average='weighted', labels= all_classes, zero_division = 0)
    f1 = f1_score(y_test, y_pred, average= 'weighted', labels= all_classes, zero_division = 0)
    # roc_auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # print(f"ROC AUC: {roc_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels = all_classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels= all_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_classes,
                yticklabels=all_classes)
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    path = PLOT_DIR / "Confusion_Matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {path}")
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'roc_auc': roc_auc
    }



def rf_clf():

    df = load_data()

    X, y, groups, exercise_groups = process_data(df)

    LOGO = LeaveOneGroupOut()
    results = []
    all_y_true = []
    all_y_pred = []

    fold = 1
    for train_idx, test_idx in LOGO.split(X, y, groups):

        print(f'\n Fold {fold}')

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = exercise_groups.iloc[train_idx]

        # Create pipeline with standardization and Random Forest
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'))
        ])

        if tune:
            param_grid = {
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [5, 10, 15],
                'rf__min_samples_split': [2, 5],
                'rf__min_samples_leaf': [1, 2]
            }

            inner_cv = GroupKFold(n_splits=5)

            print("Performing grid search...")
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=inner_cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train, groups = train_groups)
            # Best model
            best_model = grid_search.best_estimator_
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        else:
            print("Using optimal hyperparameters")
            best_params = {
                'rf__max_depth': 100,
                'rf__min_samples_leaf': 1,
                'rf__min_samples_split': 2,
                'rf__n_estimators': 300
            }
            
            # Update pipeline with the best parameters
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        y_pred = best_model.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        # Evaluate on the test set
        metrics = evaluate_model(y_test, y_pred, fold)

        results.append(metrics)
        fold += 1 

        # # Save the model
        # model_dir = BASE_DATA_DIR / 'models'
        # model_dir.mkdir(exist_ok=True)
        # model_path = model_dir / 'random_forest_classifier.pkl'
        # joblib.dump(best_model, model_path)
        # print(f"\nModel saved to: {model_path}")

    evaluate_model(all_y_true, all_y_pred, fold)


    return best_model, results


if __name__ == "__main__":
    print("Starting Random Forest classification...")
    model, metrics = rf_clf()
