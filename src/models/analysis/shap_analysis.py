"""
SHAP Analysis for SVM Model - AI Personal Coach
"""

import sys
from pathlib import Path


# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Import from utils
try:
    from utils import (
        BASE_DIR,
        FINAL_DATA_DIR,
        MODELS_DIR,
        RANDOM_STATE,
        TARGET_COL,
        EXCLUDE_COLS,
        load_data,
        preprocess_data,
        engineer_features
    )
except ImportError as e:
    print(f"Error importing from utils: {e}")
    BASE_DIR = PROJECT_ROOT
    FINAL_DATA_DIR = BASE_DIR / 'datasets' / 'final'
    MODELS_DIR = BASE_DIR / 'models'
    RANDOM_STATE = 42
    TARGET_COL = 'stress_level_norm'
    EXCLUDE_COLS = ['dataset_source', TARGET_COL]
warnings.filterwarnings('ignore')

# Create output directory for SHAP plots
SHAP_OUTPUT_DIR = MODELS_DIR / 'shap_analysis'
SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSHAP Analysis Configuration:")
print(f"  MODELS_DIR: {MODELS_DIR}")
print(f"  SHAP_OUTPUT_DIR: {SHAP_OUTPUT_DIR}")

# Load the best model and scaler
print("\n" + "="*70)
print("LOADING MODEL AND SCALER")
print("="*70)

model_path = MODELS_DIR / 'best_model.pkl'
scaler_path = MODELS_DIR / 'scaler.pkl'

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as f:
    svm_model = pickle.load(f)

print(f"Loaded model: {type(svm_model).__name__}")

scaler = None
if scaler_path.exists():
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler: {type(scaler).__name__}")
else: 
    print("Warning: Scaler file not found, assuming model doesn't need scaling")


print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

data_path = FINAL_DATA_DIR / '01_unified_dataset.csv'
df = load_data(data_path)

# Prepare features and target
y_original = df[TARGET_COL].copy()
X = df.drop(columns=EXCLUDE_COLS, errors='ignore')

# Convert to binary
y = (y_original >= 1).astype(int)

#Apply feature engineering
X = engineer_features(X)


print(f"Data shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Preprocess data (same as training)
print("\nPreprocessing data...")
X_processed, _, _ = preprocess_data(X, X, scale=(scaler is not None))


# Use a sample for SHAP (SHAP can be slow with large datasets)
# Use 100 samples for background and 50 for explanation
print("\n" + "="*70)
print("PREPARING DATA FOR SHAP")
print("="*70)

# Sample data for SHAP
np.random.seed(RANDOM_STATE)
n_background = min(100, len(X_processed))
n_explain = min(50, len(X_processed))

background_indices = np.random.choice(len(X_processed), n_background, replace=False)
explain_indices = np.random.choice(len(X_processed), n_explain, replace=False)

X_background = X_processed.iloc[background_indices]
X_explain = X_processed.iloc[explain_indices]


print(f"Background samples: {len(X_background)}")
print(f"Explanation samples: {len(X_explain)}")

# SHAP Analysis
if not SHAP_AVAILABLE:
    print("\nERROR: SHAP is not installed. Please install it with: pip install shap")
    sys.exit(1)

print("\n" + "="*70)
print("SHAP ANALYSIS")
print("="*70)

# For SVM with RBF kernel, use KernelExplainer
# Note: This can be slow, but provides accurate explanations
print("\nInitializing SHAP KernelExplainer...")
print("Note: This may take a few minutes for SVM with RBF kernel...")

# Create explainer
explainer = shap.KernelExplainer(
    svm_model.predict_proba,
    X_background,
    link='logit'  # For probability outputs
)



print("Computing SHAP values...")
# Compute SHAP values for the explanation set
shap_values = explainer.shap_values(X_explain, nsamples=100)

# For binary classification, shap_values is a list with one array
# Get the SHAP values for class 1 (Medium+High)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]  # Class 1 (Medium+High)
else:
    shap_values_class1 = shap_values

# Ensure shap_values_class1 is a 2D array (n_samples, n_features)
shap_values_class1 = np.array(shap_values_class1)
original_shape = shap_values_class1.shape

# Get expected shape from X_explain
expected_shape = (len(X_explain), len(X_explain.columns))

# Reshape if necessary
if shap_values_class1.ndim == 1:
    # If 1D, reshape to (1, n_features) - but this shouldn't happen
    print(f"Warning: SHAP values are 1D, reshaping...")
    shap_values_class1 = shap_values_class1.reshape(1, -1)
elif shap_values_class1.ndim > 2:
    # If more than 2D, flatten extra dimensions
    print(f"Warning: SHAP values have {shap_values_class1.ndim} dimensions, reshaping...")
    shap_values_class1 = shap_values_class1.reshape(shap_values_class1.shape[0], -1)

# Ensure shapes match
if shap_values_class1.shape != expected_shape:
    print(f"Warning: Shape mismatch detected!")
    print(f"  SHAP values shape: {shap_values_class1.shape}")
    print(f"  Expected shape: {expected_shape}")
    print(f"  Original shape: {original_shape}")
    
    # Try to fix: ensure we have the right number of samples
    if shap_values_class1.shape[0] != expected_shape[0]:
        print(f"  Adjusting number of samples from {shap_values_class1.shape[0]} to {expected_shape[0]}")
        shap_values_class1 = shap_values_class1[:expected_shape[0]]
    
    # Try to fix: ensure we have the right number of features
    if shap_values_class1.shape[1] != expected_shape[1]:
        print(f"  Adjusting number of features from {shap_values_class1.shape[1]} to {expected_shape[1]}")
        if shap_values_class1.shape[1] > expected_shape[1]:
            shap_values_class1 = shap_values_class1[:, :expected_shape[1]]
        else:
            # Pad with zeros if needed (shouldn't happen normally)
            padding = np.zeros((shap_values_class1.shape[0], expected_shape[1] - shap_values_class1.shape[1]))
            shap_values_class1 = np.hstack([shap_values_class1, padding])

print(f"Final SHAP values shape: {shap_values_class1.shape}")
print(f"X_explain shape: {X_explain.shape}")
assert shap_values_class1.shape == X_explain.shape, f"Shape mismatch: {shap_values_class1.shape} != {X_explain.shape}"

# Visualizations
print("\n" + "="*70)
print("GENERATING SHAP VISUALIZATIONS")
print("="*70)

# Convert X_explain to numpy array for SHAP plots
X_explain_array = X_explain.values if isinstance(X_explain, pd.DataFrame) else np.array(X_explain)
feature_names_list = X_explain.columns.tolist() if isinstance(X_explain, pd.DataFrame) else list(range(X_explain.shape[1]))

# 1. Summary Plot (Feature Importance)
print("\n1. Generating Summary Plot...")
try:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_class1,
        X_explain_array,
        feature_names=feature_names_list,
        show=False,
        plot_type="bar"
    )
    plt.title("SHAP Feature Importance - SVM Model (Class: Medium+High)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = SHAP_OUTPUT_DIR / 'shap_summary_bar.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {summary_path}")
except Exception as e:
    print(f"   Warning: Could not generate summary bar plot: {e}")
    print("   Skipping...")

# 2. Summary Plot (Dot Plot)
print("2. Generating Summary Dot Plot...")
try:
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_class1,
        X_explain_array,
        feature_names=feature_names_list,
        show=False
    )
    plt.title("SHAP Summary Plot - SVM Model (Class: Medium+High)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_dot_path = SHAP_OUTPUT_DIR / 'shap_summary_dot.png'
    plt.savefig(summary_dot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {summary_dot_path}")
except Exception as e:
    print(f"   Warning: Could not generate summary dot plot: {e}")
    print("   Skipping...")


# 3. Waterfall Plot for a single instance
print("3. Generating Waterfall Plot (single instance)...")
try:
    instance_idx = 0
    
    # Get expected value for class 1
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = float(explainer.expected_value[1])
    else:
        expected_value = float(explainer.expected_value)
    
    # Ensure we're using a single instance (1D array)
    shap_values_single = shap_values_class1[instance_idx].copy()
    if shap_values_single.ndim > 1:
        shap_values_single = shap_values_single.flatten()
    
    # Ensure it's a 1D numpy array
    shap_values_single = np.array(shap_values_single).flatten()
    
    # Get data for this instance
    instance_data = X_explain.iloc[instance_idx].values
    feature_names = X_explain.columns.tolist()
    
    # Create Explanation object with proper formatting
    explanation = shap.Explanation(
        values=shap_values_single,
        base_values=expected_value,
        data=instance_data,
        feature_names=feature_names
    )
    
    # Try to create waterfall plot
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall Plot - Instance {instance_idx} (Class: Medium+High)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    waterfall_path = SHAP_OUTPUT_DIR / 'shap_waterfall_instance0.png'
    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {waterfall_path}")
except Exception as e:
    print(f"   Warning: Could not generate waterfall plot: {e}")
    print("   Skipping waterfall plot...")


# 4. Feature Importance DataFrame
print("4. Generating Feature Importance DataFrame...")
# Ensure we have the right shape for calculations
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
mean_shap = shap_values_class1.mean(axis=0)
std_shap = shap_values_class1.std(axis=0)

# Ensure all are 1D arrays
mean_abs_shap = np.array(mean_abs_shap).flatten()
mean_shap = np.array(mean_shap).flatten()
std_shap = np.array(std_shap).flatten()

# Get feature names
feature_names = list(X_explain.columns)

# Ensure all arrays have the same length
min_len = min(len(feature_names), len(mean_abs_shap), len(mean_shap), len(std_shap))
feature_names = feature_names[:min_len]
mean_abs_shap = mean_abs_shap[:min_len]
mean_shap = mean_shap[:min_len]
std_shap = std_shap[:min_len]

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap': mean_shap,
    'std_shap': std_shap
}).sort_values('mean_abs_shap', ascending=False)

importance_path = SHAP_OUTPUT_DIR / 'shap_feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"   Saved: {importance_path}")

print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (by mean |SHAP value|)")
print("="*70)
print(feature_importance.head(10).to_string(index=False))


# 5. Partial Dependence Plot for top features
print("\n5. Generating Partial Dependence Plots for top 3 features...")
top_features = feature_importance.head(3)['feature'].tolist()

for i, feature in enumerate(top_features):
    if feature in X_explain.columns:
        print(f"   Plotting {feature}...")
        plt.figure(figsize=(10, 6))
        shap.partial_dependence_plot(
            feature,
            svm_model.predict_proba,
            X_background,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            show=False
        )
        plt.title(f"Partial Dependence Plot - {feature}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        pdp_path = SHAP_OUTPUT_DIR / f'shap_pdp_{feature.replace(" ", "_")}.png'
        plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {pdp_path}")

# 6. Force Plot for a few instances
print("\n6. Generating Force Plots (HTML) for 5 instances...")
for i in range(min(5, len(X_explain))):
    # Get expected value for class 1
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        exp_val = explainer.expected_value[1]
    else:
        exp_val = explainer.expected_value
    
    # Ensure shap values are 1D
    shap_vals = shap_values_class1[i]
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.flatten()
    
    force_plot = shap.force_plot(
        exp_val,
        shap_vals,
        X_explain.iloc[i].values,
        feature_names=X_explain.columns.tolist(),
        matplotlib=False,
        show=False
    )
    force_path = SHAP_OUTPUT_DIR / f'shap_force_plot_instance{i}.html'
    shap.save_html(str(force_path), force_plot)
    print(f"   Saved: {force_path}")

# Summary
print("\n" + "="*70)
print("SHAP ANALYSIS COMPLETE!")
print("="*70)
print(f"\nAll visualizations saved to: {SHAP_OUTPUT_DIR}")
print("\nGenerated files:")
print("  - shap_summary_bar.png: Feature importance bar plot")
print("  - shap_summary_dot.png: Feature importance dot plot")
print("  - shap_waterfall_instance0.png: Waterfall plot for one instance")
print("  - shap_feature_importance.csv: Feature importance table")
print("  - shap_pdp_*.png: Partial dependence plots for top features")
print("  - shap_force_plot_instance*.html: Interactive force plots")
print("\n" + "="*70)
