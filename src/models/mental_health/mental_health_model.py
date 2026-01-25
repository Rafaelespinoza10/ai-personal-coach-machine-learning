"""
train model - social media impact on mental health (multi output regression)
predicts multiple mental health indicators based on social media usage patterns
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
import numpy as np
import warnings
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from utils import (
        BASE_DIR,
        MODELS_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        CV_FOLDS,
        load_data,
        preprocess_data,
        save_model_artifacts,
        save_training_results,
        plot_predictions,	
        plot_feature_importance,
)
MENTAL_HEALTH_MODELS_DIR = MODELS_DIR / 'mental_health'
MENTAL_HEALTH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_DIR = BASE_DIR / 'datasets' / 'processed'

# Load data
data_path = PROCESSED_DATA_DIR / '08_cleaned_data_mental_health_social_media.csv'
df = load_data(data_path)

TARGET_COLS = [
    'depression_level',
    'sleep_issues',
    'easily_distracted',
    'worry_level',
    'concentration_difficulty',
]

EXCLUDE_COLS = [
    'timestamp',
    'age_group',
    'depression_level_risk',
    'worry_level_risk',
    'sleep_issues_risk',
    'easily_distracted_risk',
    'usage_time_ordinal',
    'hig_usage',
    'platforms_diversity',
] + TARGET_COLS

missing_targets = [col for col in TARGET_COLS if col not in df.columns]
if missing_targets:
    raise ValueError(f"Missing target columns: {missing_targets}")

# Prepare features and targets
X = df.drop(columns=EXCLUDE_COLS, errors='ignore')
y = df[TARGET_COLS].copy()

print(f"Features shape: {X.shape[1]} columns")
print(f"Targets shape: {y.shape[1]} variables")
print(f"Total samples: {len(df)}")

missing_values_features = X.isnull().sum()
if missing_values_features.sum() > 0:
    print("\nWARNING: Missing values found in features!")
else: 
    print("\nNo missing values found in features!")

missing_values_targets = y.isnull().sum()
if missing_values_targets.sum() > 0:
    print("\nWARNING: Missing values found in targets!")
else: 
    print("\nNo missing values found in targets!")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train models
models = {}
models_data = {}

print("\nTraining models...")

#Random Forest
print("Training Random Forest...")
X_train_rf, X_test_rf, scaler_rf = preprocess_data(X_train, X_test, scale=False)
rf = RandomForestRegressor(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=10,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_rf, y_train)
models['RandomForest'] = rf
models_data['RandomForest'] = {
    'X_train': X_train_rf,
    'X_test': X_test_rf,
    'scaler': None,
}
print("Random Forest training complete!")

# Gradient Boosting
print("Training Gradient Boosting...")
# Capture the 3rd return value (scaler) to avoid the previous unpacking error
X_train_gb, X_test_gb, scaler_gb = preprocess_data(X_train, X_test, scale=False)

# Define the base regressor
gb_base = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    random_state=RANDOM_STATE,
)

# Wrap it for Multi-Output
gb = MultiOutputRegressor(gb_base)
gb.fit(X_train_gb, y_train)

models['GradientBoosting'] = gb
models_data['GradientBoosting'] = {
    'X_train': X_train_gb,
    'X_test': X_test_gb,
    'scaler': None,
}
print("Gradient Boosting training complete!")


# SVM
print("Training SVM...")
X_train_svm, X_test_svm, scaler_svm = preprocess_data(X_train, X_test, scale=True)
from sklearn.multioutput import MultiOutputRegressor
svm = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
svm.fit(X_train_svm, y_train)
models['SVM'] = svm
models_data['SVM'] = {
    'X_train': X_train_svm,
    'X_test': X_test_svm,
    'scaler': scaler_svm,
}
print("SVM training complete!")

# Neural Network
print("Training Neural Network...")
X_train_nn, X_test_nn, scaler_nn = preprocess_data(X_train, X_test, scale=True)
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    alpha=0.01
)
mlp.fit(X_train_nn, y_train)
models['NeuralNetwork'] = mlp
models_data['NeuralNetwork'] = {
    'X_train': X_train_nn,
    'X_test': X_test_nn,
    'scaler': scaler_nn,
}
print("Neural Network training complete!")

# XGBoost (use return_preprocessors so we can save OHE/imputers for inference)
print("Training XGBoost...")
X_train_xgb, X_test_xgb, scaler_xgb, preprocessors = preprocess_data(
    X_train, X_test, scale=True, return_preprocessors=True
)
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    reg_alpha=0.5,
    reg_lambda=1.5,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train_xgb, y_train)

models['XGBoost'] = xgb_model
models_data['XGBoost'] = {
    'X_train': X_train_xgb,
    'X_test': X_test_xgb,
    'scaler': scaler_xgb,
}
print("XGBoost training complete!")

print("\nTraining complete! All models trained successfully.")

# Evaluate models
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name, cv_folds=5):
    """
    Evaluate multi-output regression model
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for each target
    results = {
        'model_name': model_name,
        'targets': {}
    }
    
    # Per-target metrics
    for idx, target_name in enumerate(TARGET_COLS):
        train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, idx], y_train_pred[:, idx]))
        test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, idx], y_test_pred[:, idx]))
        train_mae = mean_absolute_error(y_train.iloc[:, idx], y_train_pred[:, idx])
        test_mae = mean_absolute_error(y_test.iloc[:, idx], y_test_pred[:, idx])
        train_r2 = r2_score(y_train.iloc[:, idx], y_train_pred[:, idx])
        test_r2 = r2_score(y_test.iloc[:, idx], y_test_pred[:, idx])
        
        results['targets'][target_name] = {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        }
    
    # Average metrics across all targets
    results['avg_train_rmse'] = np.mean([r['train_rmse'] for r in results['targets'].values()])
    results['avg_test_rmse'] = np.mean([r['test_rmse'] for r in results['targets'].values()])
    results['avg_train_mae'] = np.mean([r['train_mae'] for r in results['targets'].values()])
    results['avg_test_mae'] = np.mean([r['test_mae'] for r in results['targets'].values()])
    results['avg_train_r2'] = np.mean([r['train_r2'] for r in results['targets'].values()])
    results['avg_test_r2'] = np.mean([r['test_r2'] for r in results['targets'].values()])
    
    # Cross-validation (using average R² across targets)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create temporary model for CV
        if hasattr(model, 'estimator') and isinstance(model, MultiOutputRegressor):
            # Handles MultiOutputRegressor (SVM, GradientBoosting)
            base_params = model.estimator.get_params()
            temp_model = MultiOutputRegressor(type(model.estimator)(**base_params))
        else:
            # Handles Native Models (RandomForest, NeuralNetwork, XGBoost)
            # Use sklearn's clone for a safe, clean copy
            from sklearn.base import clone
            temp_model = clone(model)

        temp_model.fit(X_cv_train, y_cv_train)
        y_cv_pred = temp_model.predict(X_cv_val)
        
        # Average R² across targets
        cv_r2 = np.mean([r2_score(y_cv_val.iloc[:, i], y_cv_pred[:, i]) for i in range(len(TARGET_COLS))])
        cv_scores.append(cv_r2)
    
    results['cv_mean_r2'] = float(np.mean(cv_scores))
    results['cv_std_r2'] = float(np.std(cv_scores))
    
    results['y_test_pred'] = y_test_pred.tolist()  # For later analysis
    
    return results

# Evaluate all models
results = {}
for name, model in models.items():
    data = models_data[name]
    print(f"\nEvaluating {name}...")
    results[name] = evaluate_regression_model(
        model,
        data['X_train'],
        data['X_test'],
        y_train,
        y_test,
        name,
        cv_folds=CV_FOLDS
    )
print("\nEvaluation complete! All models evaluated successfully.")

# display results
print(f"\n{'Model':<20} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12} {'CV R²':<12}")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['avg_test_r2']:>11.4f}  {metrics['avg_test_rmse']:>11.4f}  "
          f"{metrics['avg_test_mae']:>11.4f}  {metrics['cv_mean_r2']:>11.4f}")

# find best model
best_model_name = max(results.items(), key=lambda x: x[1]['avg_test_r2'])[0]
best_model = models[best_model_name]
best_metrics = results[best_model_name]
best_model_data = models_data[best_model_name]
print(f"\nBest model: {best_model_name}")
print(f"  Average Test R²: {best_metrics['avg_test_r2']:.4f}")
print(f"  Average Test RMSE: {best_metrics['avg_test_rmse']:.4f}")
print(f"  Average Test MAE: {best_metrics['avg_test_mae']:.4f}")
print(f"  CV R²: {best_metrics['cv_mean_r2']:.4f} (±{best_metrics['cv_std_r2']:.4f})")


# overfitting check
print(f"\n--- Overfitting Analysis: {best_model_name} ---")
train_r2 = best_metrics['avg_train_r2']
test_r2 = best_metrics['avg_test_r2']
gap = train_r2 - test_r2

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Gap: {gap:.4f}")

if gap > 0.15:
    print("WARNING: Possible overfitting detected. The gap is significant.")
elif gap < 0:
    print("NOTE: The model performs better on Test. This can happen on small datasets.")
else:
    print("STATUS: The model seems to be well balanced.")
# plot predictions
y_best_pred = np.array(best_metrics['y_test_pred'])
plot_predictions(y_test, y_best_pred, TARGET_COLS)

# plot feature importance
feature_names_transformed = models_data['XGBoost']['X_train'].columns.tolist()

print(f"\nVerificando dimensiones:")
print(f"- Nombres de columnas detectadas: {len(feature_names_transformed)}")
print(f"- Importancias del modelo: {len(best_model.feature_importances_)}")

# 2. Plot with the correct names
try:
    importance_df = plot_feature_importance(best_model, feature_names_transformed)
    
    # Show the top 10 in console
    print("\n--- Top 10 Variables más importantes ---")
    print(importance_df.head(10))
    
except Exception as e:
    print(f"Error al graficar importancia: {e}")

# selected features with based on feature importance (the last analysis showed that the model is overfitting)
selector = SelectFromModel(best_model, threshold='mean', prefit=True)
X_train_selected = selector.transform(models_data['XGBoost']['X_train'])
X_test_selected = selector.transform(models_data['XGBoost']['X_test'])
# get selected features
selected_features = X_train_xgb.columns[selector.get_support()].tolist()

print(f"Original features: {models_data['XGBoost']['X_train'].shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")

# re-train the model with the selected features
best_model_re_trained = xgb.XGBRegressor(
    **best_model.get_params()
)
best_model_re_trained.fit(
    X_train_selected, y_train,
    eval_set=[(X_test_selected, y_test)],
    verbose=False
)

# Evaluate the re-trained model to see if the overfitting is reduced
y_train_new = best_model_re_trained.predict(X_train_selected)
y_test_new = best_model_re_trained.predict(X_test_selected)

train_r2_new = r2_score(y_train, y_train_new)
test_r2_new = r2_score(y_test, y_test_new)
gap_new = train_r2_new - test_r2_new

print(f"\n--- Analysis after Feature Selection ---")
print(f"New Train R²: {train_r2_new:.4f}")
print(f"New Test R²: {test_r2_new:.4f}")
print(f"New Gap: {gap_new:.4f}")

if gap > 0.15:
    print("WARNING: Possible overfitting detected. The gap is significant.")
elif gap < 0:
    print("NOTE: The model performs better on Test. This can happen on small datasets.")
else:
    print("STATUS: The model seems to be well balanced.")

# save artifacts
print("\nSaving the model and artifacts...")
# save the final model
with open(MENTAL_HEALTH_MODELS_DIR / 'best_mental_health_model.pkl', 'wb') as f:
    pickle.dump(best_model_re_trained, f)
# save scaler
with open(MENTAL_HEALTH_MODELS_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler_xgb, f)
# save selected features
with open(MENTAL_HEALTH_MODELS_DIR / 'selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
# save preprocessors (OHE + imputers) for inference
with open(MENTAL_HEALTH_MODELS_DIR / 'preprocessors.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

# save metadata
model_config = {
    'model_name': 'XGBoost_Optimized',
    'features_count': len(selected_features),
    'selected_features': selected_features,
    'metrics': {'test_r2': float(test_r2), 'train_r2': float(train_r2)},
    'targets': TARGET_COLS,
    'training_date': datetime.now().isoformat()
}
with open(MENTAL_HEALTH_MODELS_DIR / 'model_config.json', 'w') as f:
    json.dump(model_config, f, indent=4)

print(f"All artifacts saved in: {MENTAL_HEALTH_MODELS_DIR}")

