"""
Train model - AI Personal Performance Coach Main Model
"""

# Standard library imports
import sys
from pathlib import Path

# Add src to path BEFORE any utils imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Third-party imports
import pandas as pd
import numpy as np
import warnings
import pickle
import json
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


# XGBoost (optional)
try: 
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available, will use it for training")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will not use it for training")

warnings.filterwarnings('ignore')

# Import from utils barrel file (AFTER sys.path is configured)
try:
    from utils import (
        BASE_DIR,
        FINAL_DATA_DIR,
        MODELS_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        CV_FOLDS,
        TARGET_COL,
        EXCLUDE_COLS,
        load_data,
        preprocess_data,
        evaluate_models, 
        find_best_model,
        save_model_artifacts,
        save_training_results,
        select_features_by_importance,
        engineer_features
    )
    USE_CONSTANTS = True
    print("Successfully imported from utils")
except ImportError as e:
    print(f"Error importing from utils: {e}")
    # Fallback: calculate paths dynamically
    BASE_DIR = PROJECT_ROOT
    FINAL_DATA_DIR = BASE_DIR / 'datasets' / 'final'
    MODELS_DIR = BASE_DIR / 'models'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TARGET_COL = 'stress_level_norm'
    EXCLUDE_COLS = ['dataset_source', TARGET_COL]
    
    def load_data(file_path):
        """Fallback load_data function"""
        return pd.read_csv(file_path)
    
    USE_CONSTANTS = False
    print("Using fallback configuration")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  BASE_DIR: {BASE_DIR}")
print(f"  FINAL_DATA_DIR: {FINAL_DATA_DIR}")
print(f"  MODELS_DIR: {MODELS_DIR}")
print(f"  USE_CONSTANTS: {USE_CONSTANTS}")

# Test load
data_path = FINAL_DATA_DIR / '01_unified_dataset.csv'
print(f"\nLoading data from: {data_path}")
df = load_data(data_path)

# Prepare features and target
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

y_original = df[TARGET_COL].copy()
X = df.drop(columns=EXCLUDE_COLS, errors='ignore')

# Convert to binary classification: Low (0) vs Medium+High (1)
# Low = 0, Medium = 1, High = 2 -> Low (0) vs Medium+High (1)
print("\n" + "="*70)
print("BINARY CLASSIFICATION: Low vs Medium+High")
print("="*70)
print(f"Original target distribution: {y_original.value_counts().sort_index().to_dict()}")

y = (y_original >= 1).astype(int)
print(f"Binary target distribution:")
print(f"  0 (Low): {(y == 0).sum()} samples ({(y == 0).mean()*100:.1f}%)")
print(f"  1 (Medium+High): {(y == 1).sum()} samples ({(y == 1).mean()*100:.1f}%)")

print(f"\nFeatures: {X.shape[1]} columns")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

X_train_fe = engineer_features(X_train)
X_test_fe = engineer_features(X_test)

# Train models
models = {}
models_data = {}

# Random Forest
print("\nTraining models...")
print("Training Random Forest...")

X_train_rf, X_test_rf, _ = preprocess_data(X_train_fe, X_test_fe, scale=False)
rf = RandomForestClassifier(
    n_estimators=200,          
    max_depth=8,               
    min_samples_split=20,      
    min_samples_leaf=10,       
    max_features='sqrt',      
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_rf, y_train)
models['RandomForest'] = rf
models_data['RandomForest'] = {
    'X_train': X_train_rf,
    'X_test': X_test_rf,
    'scaler': None
}
print("Random Forest trained successfully")

# Gradient Boosting
print("Training Gradient Boosting...")
X_train_gb, X_test_gb, _ = preprocess_data(X_train_fe, X_test_fe, scale=False)
gb = GradientBoostingClassifier(
    n_estimators=100,          
    learning_rate=0.05,      
    max_depth=3,             
    min_samples_split=20,      
    min_samples_leaf=10,     
    subsample=0.8,            
    max_features='sqrt',      
    random_state=RANDOM_STATE
)
gb.fit(X_train_gb, y_train)
models['GradientBoosting'] = gb
models_data['GradientBoosting'] = {
    'X_train': X_train_gb,
    'X_test': X_test_gb,
    'scaler': None
}
print("Gradient Boosting trained successfully")

# SVM
print("\nTraining SVM...")
X_train_svm, X_test_svm, scaler = preprocess_data(X_train_fe, X_test_fe, scale=True)
svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_svm, y_train)
models['SVM'] = svm
models_data['SVM'] = {
    'X_train': X_train_svm,
    'X_test': X_test_svm,
    'scaler': scaler
}
print("SVM trained successfully")

# Neural Network
print("\nTraining Neural Network...")
X_train_mlp, X_test_mlp, scaler = preprocess_data(X_train_fe, X_test_fe, scale=True)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1)
mlp.fit(X_train_mlp, y_train)
models['NeuralNetwork'] = mlp
models_data['NeuralNetwork'] = {
    'X_train': X_train_mlp,
    'X_test': X_test_mlp,
    'scaler': scaler
}
print("Neural Network trained successfully")


if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost...")
    X_train_xgb, X_test_xgb, scaler = preprocess_data(X_train_fe, X_test_fe, scale=True)
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_train_xgb, y_train)
    models['XGBoost'] = xgb_model
    models_data['XGBoost'] = {
        'X_train': X_train_xgb,
        'X_test': X_test_xgb,
        'scaler': scaler
    }
    print("XGBoost trained successfully")

print("\nAll models trained successfully")


print("\nEvaluating models with all features...")
results = {}
for name, model in models.items():
    data = models_data[name]
    results[name] = evaluate_models(
        model,
        data['X_train'],
        data['X_test'],
        y_train,
        y_test,
        name,
        CV_FOLDS,
        RANDOM_STATE
    )

# Display results
print("\n" + "-"*70)
print("MODEL PERFORMANCE SUMMARY")
print("-"*70)
print(f"\n{'Model':<20} {'Test Acc':<12} {'F1 Score':<12} {'CV Mean':<12} {'CV Std':<12}")
print("-"*70)

for name, metrics in results.items():
    print(f"{name:<20} {metrics['test_accuracy']:>11.4f}  {metrics['test_f1']:>11.4f}  "
          f"{metrics['cv_mean']:>11.4f}  {metrics['cv_std']:>11.4f}")

# find the best model
best_model_name, best_model, best_metrics = find_best_model(models, results, metric='test_f1')

print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name}")
print("="*70)
print(f"  Test Accuracy: {best_metrics['test_accuracy']:.4f}")
print(f"  Test F1 Score: {best_metrics['test_f1']:.4f}")
print(f"  CV Accuracy: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, best_metrics['y_test_pred'], target_names=['Low', 'Medium+High']))


print("\n" + "="*70)
print("STEP 7: SAVE MODEL AND ARTIFACTS")
print("="*70)

### Feature selection with top ten features ###
print("\n" + "="*70)
print("FEATURE SELECTION WITH TOP TEN FEATURES")
print("="*70)

models_for_selection = {
    'RandomForest': models['RandomForest'],
    'GradientBoosting': models['GradientBoosting'],
}

if XGBOOST_AVAILABLE:
    models_for_selection['XGBoost'] = models['XGBoost']

X_train_ref = models_data['RandomForest']['X_train']
X_test_ref = models_data['RandomForest']['X_test']

X_train_selected, X_test_selected, selected_features = select_features_by_importance(
    models_for_selection,
    X_train_ref,
    X_test_ref,
    top_n=10
)

print(f"\nSelected top 10 features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# Re-train models with selected features
models_selected = {}
models_data_selected = {}

# Random Forest with selected features
print("\nRe-training Random Forest with top 10 features...")
rf_selected = RandomForestClassifier(
    n_estimators=200,          
    max_depth=8,               
    min_samples_split=20,      
    min_samples_leaf=10,       
    max_features='sqrt',      
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_selected.fit(X_train_selected, y_train)
models_selected['RandomForest'] = rf_selected
models_data_selected['RandomForest'] = {
    'X_train': X_train_selected,
    'X_test': X_test_selected,
    'scaler': None
}

# Gradient Boosting with selected features
print("Re-training Gradient Boosting with top 10 features...")
gb_selected = GradientBoostingClassifier(
    n_estimators=100,          
    learning_rate=0.05,      
    max_depth=3,             
    min_samples_split=20,      
    min_samples_leaf=10,     
    subsample=0.8,            
    max_features='sqrt',      
    random_state=RANDOM_STATE
)
gb_selected.fit(X_train_selected, y_train)
models_selected['GradientBoosting'] = gb_selected
models_data_selected['GradientBoosting'] = {
    'X_train': X_train_selected,
    'X_test': X_test_selected,
    'scaler': None
}

# SVM with selected features (needs scaling)
print("Re-training SVM with top 10 features...")
X_train_svm_sel, X_test_svm_sel, scaler_svm_sel = preprocess_data(
    X_train_selected, 
    X_test_selected, 
    scale=True
)
svm_selected = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
svm_selected.fit(X_train_svm_sel, y_train)
models_selected['SVM'] = svm_selected
models_data_selected['SVM'] = {
    'X_train': X_train_svm_sel,
    'X_test': X_test_svm_sel,
    'scaler': scaler_svm_sel
}

# Neural Network with selected features (needs scaling)
print("Re-training Neural Network with top 10 features...")
X_train_nn_sel, X_test_nn_sel, scaler_nn_sel = preprocess_data(
    X_train_selected, 
    X_test_selected, 
    scale=True
)
mlp_selected = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1)
mlp_selected.fit(X_train_nn_sel, y_train)
models_selected['NeuralNetwork'] = mlp_selected
models_data_selected['NeuralNetwork'] = {
    'X_train': X_train_nn_sel,
    'X_test': X_test_nn_sel,
    'scaler': scaler_nn_sel
}

if XGBOOST_AVAILABLE:
    print("Re-training XGBoost with top 10 features...")
    X_train_xgb_sel, X_test_xgb_sel, scaler_xgb_sel = preprocess_data(
        X_train_selected, 
        X_test_selected, 
        scale=True
    )
    xgb_selected = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_selected.fit(X_train_xgb_sel, y_train)
    models_selected['XGBoost'] = xgb_selected
    models_data_selected['XGBoost'] = {
        'X_train': X_train_xgb_sel,
        'X_test': X_test_xgb_sel,
        'scaler': scaler_xgb_sel
    }
# Evaluate models with selected features
print("\nEvaluating models with top 10 features...")
results_selected = {}
for name, model in models_selected.items():
    data = models_data_selected[name]
    results_selected[name] = evaluate_models(
        model,
        data['X_train'],
        data['X_test'],
        y_train,
        y_test,
        name,
        CV_FOLDS,
        RANDOM_STATE
    )


# Compare results
print("\n" + "="*70)
print("COMPARISON: ALL FEATURES vs TOP 10 FEATURES")
print("="*70)
print(f"\n{'Model':<20} {'All Features F1':<18} {'Top 10 F1':<15} {'Improvement':<12}")
print("-"*70)

for name in results.keys():
    if name in results_selected:
        f1_all = results[name]['test_f1']
        f1_selected = results_selected[name]['test_f1']
        improvement = f1_selected - f1_all
        improvement_pct = (improvement / f1_all * 100) if f1_all > 0 else 0
        print(f"{name:<20} {f1_all:>17.4f}  {f1_selected:>14.4f}  {improvement:>+11.4f} ({improvement_pct:>+6.2f}%)")

# Choose best results (all features or selected)
print("\n" + "="*70)
print("FINAL MODEL SELECTION")
print("="*70)

# Find best model from all features
best_all_name, best_all_model, best_all_metrics = find_best_model(models, results, metric='test_f1')
# Find best model from selected features
best_selected_name, best_selected_model, best_selected_metrics = find_best_model(models_selected, results_selected, metric='test_f1')

# Compare and choose the best
if best_all_metrics['test_f1'] >= best_selected_metrics['test_f1']:
    best_model_name = best_all_name
    best_model = best_all_model
    best_metrics = best_all_metrics
    best_models_data = models_data
    use_selected = False
    print(f"Best model: {best_model_name} (with all features)")
    print(f"  Test F1 Score: {best_metrics['test_f1']:.4f}")
else:
    best_model_name = best_selected_name
    best_model = best_selected_model
    best_metrics = best_selected_metrics
    best_models_data = models_data_selected
    use_selected = True
    print(f"Best model: {best_model_name} (with top 10 features)")
    print(f"  Test F1 Score: {best_metrics['test_f1']:.4f}")
    print(f"  Selected features: {len(selected_features)}")

# Display final results
print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*70)
print(f"\n{'Model':<20} {'Test Acc':<12} {'F1 Score':<12} {'CV Mean':<12} {'CV Std':<12}")
print("-"*70)

final_results = results_selected if use_selected else results
for name, metrics in final_results.items():
    print(f"{name:<20} {metrics['test_accuracy']:>11.4f}  {metrics['test_f1']:>11.4f}  "
          f"{metrics['cv_mean']:>11.4f}  {metrics['cv_std']:>11.4f}")


print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name}")
print("="*70)
print(f"  Test Accuracy: {best_metrics['test_accuracy']:.4f}")
print(f"  Test F1 Score: {best_metrics['test_f1']:.4f}")
print(f"  CV Accuracy: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, best_metrics['y_test_pred'], target_names=['Low', 'Medium+High']))

# Get the scaler for the best model
best_model_scaler = best_models_data[best_model_name]['scaler']

print("\n" + "="*70)
print("STEP 7: SAVE MODEL AND ARTIFACTS")
print("="*70)

# Save model and scaler
model_path, scaler_path = save_model_artifacts(best_model, best_model_scaler, MODELS_DIR)

# Save preprocessors for inference (only when using all features; API stress service uses them)
if not use_selected:
    _, _, _, preprocessors = preprocess_data(X_train_fe, X_test_fe, scale=True, return_preprocessors=True)
    preprocessors['base_columns'] = list(X_train.columns)
    preprocessors['base_numeric'] = X_train.select_dtypes(include=[np.number]).columns.tolist()
    preprocessors['base_categorical'] = X_train.select_dtypes(include=['object']).columns.tolist()
    preprocessors_path = MODELS_DIR / 'preprocessors.pkl'
    with open(preprocessors_path, 'wb') as f:
        pickle.dump(preprocessors, f)
    print(f"  - Preprocessors: {preprocessors_path}")

# Save training results (include both all and selected results)
config = {
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
    'cv_folds': CV_FOLDS,
    'target': TARGET_COL,
    'classification_type': 'binary',
    'target_mapping': {
        '0': 'Low',
        '1': 'Medium+High'
    },
    'feature_selection': {
        'used': use_selected,
        'top_n_features': len(selected_features) if use_selected else None,
        'selected_features': selected_features if use_selected else None
    }
}


# Combine results for saving - need to modify save_training_results to handle this
# For now, save with the best results
results_to_save = {
    'all_features': results,
    'selected_features': results_selected
}

# Save training results (include both all and selected results)
results_path = save_training_results(results_to_save, best_model_name, best_metrics, config, MODELS_DIR)

# Print all saved files
print(f"\nAll files saved:")
print(f"  - Model: {model_path}")
if scaler_path:
    print(f"  - Scaler: {scaler_path}")
print(f"  - Results: {results_path}")
print(f"\nFeature Selection Summary:")
print(f"  - Used top 10 features: {use_selected}")
if use_selected:
    print(f"  - Selected features: {', '.join(selected_features[:5])}... ({len(selected_features)} total)")

