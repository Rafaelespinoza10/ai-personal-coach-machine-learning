import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame
    Args:
        file_path: str, path to the CSV file
    Returns:
        df: pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    return df


def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, scale: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Preprocess features: handle missing values, encode, scale
    Args:
        X_train: pd.DataFrame, training data
        X_test: pd.DataFrame, test data
        scale: bool, whether to scale the data
    Returns:
        X_train_processed: pd.DataFrame, training data processed
        X_test_processed: pd.DataFrame, test data processed
    """
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Identify column types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Handle missing values in numeric columns
    numeric_imputer = SimpleImputer(strategy='median')
    X_train_processed[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_test_processed[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

    # Handle missing values in categorical columns and encode
    if len(categorical_cols) > 0:
        # Impute missing values in categorical columns first
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_train_processed[categorical_cols] = categorical_imputer.fit_transform(X_train_processed[categorical_cols])
        X_test_processed[categorical_cols] = categorical_imputer.transform(X_test_processed[categorical_cols])
        
        # Encode categorical variables
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_train_categorical_encoded = ohe.fit_transform(X_train_processed[categorical_cols]) 
        X_test_categorical_encoded = ohe.transform(X_test_processed[categorical_cols])
        
        categorical_feature_names = ohe.get_feature_names_out(categorical_cols)

        X_train_categorical_df = pd.DataFrame(
            X_train_categorical_encoded,
            columns=categorical_feature_names,
            index=X_train_processed.index
        )

        X_test_categorical_df = pd.DataFrame(
            X_test_categorical_encoded,
            columns=categorical_feature_names,
            index=X_test_processed.index
        )

        X_train_processed = pd.concat([
            X_train_processed[numeric_cols].reset_index(drop=True),
            X_train_categorical_df.reset_index(drop=True),
        ], axis=1)
        X_test_processed = pd.concat([
            X_test_processed[numeric_cols].reset_index(drop=True),
            X_test_categorical_df.reset_index(drop=True),
        ], axis=1)
    else: 
        X_train_processed = X_train_processed[numeric_cols]
        X_test_processed = X_test_processed[numeric_cols]

    
    scaler = None
    if scale: 
        scaler = StandardScaler()
        X_train_processed = pd.DataFrame(
            scaler.fit_transform(X_train_processed),
            columns=X_train_processed.columns,
            index=X_train_processed.index
        )
        X_test_processed = pd.DataFrame(
            scaler.transform(X_test_processed),
            columns=X_test_processed.columns,
            index=X_test_processed.index
        )

    return X_train_processed, X_test_processed, scaler



def evaluate_models(model, X_train, X_test, y_train, y_test, model_name: str, cv_folds: int, random_state: int=42) :
    """
    Evaluate model and return metrics
    """

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    # Use 'binary' for binary classification, 'weighted' for multiclass
    is_binary = len(np.unique(y_test)) == 2
    avg_method = 'binary' if is_binary else 'weighted'
    
    test_precision = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average=avg_method, zero_division=0)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'y_test_pred': y_test_pred
    }


def find_best_model(models,results, metric:str='test_f1') -> str:
    """
    Find the best model based on test accuracy
    """
    best_model_name = max(results.items(), key=lambda x: x[1][metric])[0]
    best_model = models[best_model_name]
    best_metrics = results[best_model_name]

    return best_model_name, best_model, best_metrics


def save_model_artifacts(model, scaler, models_dir: Path) -> tuple[Path, Path | None]:
    """
    Save the trained model and scaler to disk
    
    Args:
        model: Trained model to save
        scaler: Scaler object (can be None)
        models_dir: Directory where to save the files
        
    Returns:
        tuple: (model_path, scaler_path or None)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Best model saved: {model_path}")
    
    # Save scaler (if available)
    scaler_path = None
    if scaler is not None:
        scaler_path = models_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved: {scaler_path}")
    
    return model_path, scaler_path


def save_training_results(results: dict, best_model_name: str, best_metrics: dict, 
                         config: dict, models_dir: Path) -> Path:
    """
    Save training results to JSON file and print summary
    
    Args:
        results: Dictionary with all model results (can be nested with 'all_features' and 'selected_features')
        best_model_name: Name of the best model
        best_metrics: Metrics of the best model
        config: Configuration dictionary
        models_dir: Directory where to save the results
        
    Returns:
        Path to the saved results file
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare results dictionary
    # Exclude non-numeric keys from metrics
    excluded_keys = {'y_test_pred', 'model_name'}
    
    def convert_to_serializable(metrics_dict):
        """Convert metrics dictionary to JSON-serializable format"""
        return {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in metrics_dict.items()
            if k not in excluded_keys
        }
    
    # Check if results is nested (has 'all_features' and 'selected_features')
    if 'all_features' in results and 'selected_features' in results:
        # Nested structure - save both
        results_to_save = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_model': best_model_name,
            'best_metrics': convert_to_serializable(best_metrics),
            'all_features_results': {
                name: convert_to_serializable(metrics)
                for name, metrics in results['all_features'].items()
            },
            'selected_features_results': {
                name: convert_to_serializable(metrics)
                for name, metrics in results['selected_features'].items()
            },
            'config': config
        }
    else:
        # Simple structure - original format
        results_to_save = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_model': best_model_name,
            'best_metrics': convert_to_serializable(best_metrics),
            'all_results': {
                name: convert_to_serializable(metrics)
                for name, metrics in results.items()
            },
            'config': config
        }
    
    # Save to JSON
    results_path = models_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Training results saved: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {best_metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {best_metrics['test_f1']:.4f}")
    print(f"\nFiles saved:")
    print(f"  - Results: {results_path}")
    print("="*70)
    
    return results_path



def select_features_by_importance(models_dict, X_train, X_test,
                                  importance_threshold=0.001, top_n=None):

    feature_importances = []

    for model in models_dict.values():
        if hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)

    if not feature_importances:
        return X_train, X_test, list(X_train.columns)

    importance_df = pd.DataFrame(
        np.mean(feature_importances, axis=0),
        index=X_train.columns,
        columns=['avg_importance']
    ).sort_values(by='avg_importance', ascending=False)

    if top_n:
        selected = importance_df.head(top_n).index
    else:
        selected = importance_df[
            importance_df['avg_importance'] >= importance_threshold
        ].index

    return X_train[selected], X_test[selected], list(selected)


def engineer_features(df):
    """
    Created engineered features that may improve model performance
    """
    df_eng = df.copy()
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()

    #Health composite features
     # HEALTH COMPOSITE FEATURES
    if all(c in numeric_cols for c in ['sleep_quality', 'physical_activity']):
        df_eng['health_stress_index'] = (
            df_eng['sleep_quality'] - df_eng['physical_activity'] / 10
        )
    
    if all(c in numeric_cols for c in ['sleep_quality', 'sleep_duration']):
        df_eng['sleep_efficiency'] = (
            df_eng['sleep_quality'] / (df_eng['sleep_duration'] + 1)
        )
    
    # PHYSIOLOGICAL RATIOS
    if all(c in numeric_cols for c in ['heart_rate', 'physical_activity']):
        df_eng['hr_activity_ratio'] = (
            df_eng['heart_rate'] / (df_eng['physical_activity'] + 1)
        )
    
    if all(c in numeric_cols for c in ['stress_level', 'physical_activity']):
        df_eng['stress_activity_balance'] = (
            df_eng['stress_level'] / (df_eng['physical_activity'] + 1)
        )
    
    # INTERACTION FEATURES
    if all(c in numeric_cols for c in ['sleep_quality', 'stress_level']):
        df_eng['sleep_stress_interaction'] = (
            df_eng['sleep_quality'] * df_eng['stress_level']
        )
    
    # PHYSIOLOGICAL STRESS SCORE
    if all(c in numeric_cols for c in ['cortisol_level', 'heart_rate']):
        df_eng['physiological_stress_score'] = (
            df_eng['cortisol_level'] * 10 + df_eng['heart_rate'] / 10
        )    
    return df_eng