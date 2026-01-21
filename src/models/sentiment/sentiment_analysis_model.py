"""
Train Sentiment Analysis Model
"""

import sys 
import re
from pathlib import Path

# Add src to path BEFORE any utils imports
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

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


try: 
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available, will use it for training")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will not use it for training")

warnings.filterwarnings('ignore')

# Import from utils barrel file
try:
    from utils import (
        BASE_DIR,
        MODELS_DIR,
        RANDOM_STATE,
        TEST_SIZE,
        CV_FOLDS,
        load_data,
        evaluate_models,
        find_best_model,
        save_model_artifacts,
        save_training_results
    )
    USE_CONSTANTS = True
    print("Successfully imported from utils")
except ImportError as e:
    print(f"Error importing from utils: {e}")
    # Fallback configuration
    BASE_DIR = PROJECT_ROOT
    MODELS_DIR = BASE_DIR / 'models'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    def load_data(file_path):
        return pd.read_csv(file_path)

    USE_CONSTANTS = False
    print("Using fallback configuration for constants")


# Create directories
PROCESSED_DATA_DIR = BASE_DIR / 'datasets' / 'processed'
SENTIMENT_MODELS_DIR = MODELS_DIR / 'sentiment'
SENTIMENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Load data
data_path = PROCESSED_DATA_DIR / '07_cleaned_sentiment_data.csv'
df = load_data(data_path)
print(f"Data loaded successfully: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Prepare target and features 

TARGET_COL ='sentiment'
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

X_text = df['text'].fillna('').astype(str)
y = df[TARGET_COL]

print(f"\nTarget variable: {TARGET_COL}")
print(f"Target distribution:")
print(y.value_counts())
print(f"\nTarget distribution (percentages):")
print(y.value_counts(normalize=True) * 100)

# Check for missing values
print(f"\nMissing values in text: {X_text.isna().sum()}")
print(f"Missing values in target: {y.isna().sum()}")

if y.isna().sum() > 0:
    mask = ~y.isna()
    X_text = X_text[mask]
    y = y[mask]
    print(f"Removed {mask.sum()} rows with missing target")

# Text processing
def clean_text(text):
    """ Basic text cleaning """
    if pd.isna(text):
        return ""
    text = str(text)
    # remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

print("cleaning text...")
X_text_cleaned = X_text.apply(clean_text)

# remove empty text after cleaning
mask = X_text_cleaned.str.len() > 0
X_text_cleaned = X_text_cleaned[mask]
y = y[mask]

print(f"Texts after cleaning: {len(X_text_cleaned)}")
print(f"Average text length: {X_text_cleaned.str.len().mean():.1f} characters")
print(f"Average word count: {X_text_cleaned.str.split().str.len().mean():.1f} words")

# Check for duplicates and potential leakage
print("\n" + "="*70)
print("DATA QUALITY CHECKS")
print("="*70)
n_duplicates = X_text_cleaned.duplicated().sum()
print(f"Duplicate texts in dataset: {n_duplicates} ({n_duplicates/len(X_text_cleaned)*100:.2f}%)")

if n_duplicates > 0:
    print(" WARNING: Duplicates found! This can cause train/test leakage.")
    print("Checking if duplicates have same labels...")
    dup_mask = X_text_cleaned.duplicated(keep=False)
    if dup_mask.sum() > 0:
        dup_texts = X_text_cleaned[dup_mask]
        dup_labels = y[dup_mask]
        label_consistency = dup_texts.groupby(dup_texts).apply(
            lambda x: dup_labels.loc[x.index].nunique() == 1
        )
        inconsistent = (~label_consistency).sum()
        if inconsistent > 0:
            print(f" WARNING: {inconsistent} duplicate texts have different labels!")

# Train test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text_cleaned, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {len(X_train_text)} samples")
print(f"Test set: {len(X_test_text)} samples")

train_set = set(X_train_text)
test_set = set(X_test_text)
overlap = train_set.intersection(test_set)
print(f"\nText overlap between train and test: {len(overlap)} texts")
if len(overlap) > 0:
    print(f"WARNING: {len(overlap)} texts appear in both train and test sets!")
    print(f"   This is potential data leakage. Removing duplicates...")
    mask_test_no_overlap = ~X_test_text.isin(overlap)
    X_test_text = X_test_text[mask_test_no_overlap]
    y_test = y_test[mask_test_no_overlap]
    print(f"   Test set after removing overlap: {len(X_test_text)} samples")

print(f"\nTraining set distribution:")
print(y_train.value_counts())
print(f"\nTest set distribution:")
print(y_test.value_counts())

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

print("label encoding")
for label, encoded in label_mapping.items():
    print(f"Label: {label} -> Encoded: {encoded}")

y_train_original = y_train.copy()
y_test_original = y_test.copy()

# TF-IDF Vectorization 
print("Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,              
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words=None
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (test): {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

X_train_dense = X_train_tfidf.toarray()  # Only for GB, MLP, XGBoost
X_test_dense = X_test_tfidf.toarray()

# Training models
models = {}
models_data = {}

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,           
    max_depth=20,                
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_tfidf, y_train)  
models['RandomForest'] = rf_model
models_data['RandomForest'] = {
    'X_train': X_train_tfidf,
    'X_test': X_test_tfidf,
    'vectorizer': tfidf_vectorizer
}
print("Random Forest trained successfully")

# Gradient Boosting
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,             
    max_depth=4,                    
    learning_rate=0.08,             
    subsample=0.8,                  
    max_features='sqrt',            
    min_samples_split=10,           
    min_samples_leaf=5,             
    random_state=RANDOM_STATE,
    validation_fraction=0.1,        
    n_iter_no_change=10
)
gb_model.fit(X_train_dense, y_train_original)
models['GradientBoosting'] = gb_model
models_data['GradientBoosting'] = {
    'X_train': X_train_dense,
    'X_test': X_test_dense,
    'vectorizer': tfidf_vectorizer
}
print("Gradient Boosting trained successfully")

# LinearSVC - Better for text classification with TF-IDF
print("\nTraining LinearSVC (Better for TF-IDF text classification)...")
svm_model = LinearSVC(
    C=1.0,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    max_iter=2000,
    dual=False,  
    tol=1e-4
)
svm_model.fit(X_train_tfidf, y_train)  
models['LinearSVC'] = svm_model
models_data['LinearSVC'] = {
    'X_train': X_train_tfidf,
    'X_test': X_test_tfidf,
    'vectorizer': tfidf_vectorizer
}
print("LinearSVC trained successfully")

# Neural Network - Reduced complexity
print("\nTraining Neural Network (MLP)...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(50,),       
    alpha=0.01,                      
    learning_rate='adaptive',       
    max_iter=300,                    
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.15,        
    n_iter_no_change=15,             
    batch_size='auto',
    tol=1e-4
)
mlp_model.fit(X_train_dense, y_train_encoded)
models['MLP'] = mlp_model
models_data['MLP'] = {
    'X_train': X_train_dense,
    'X_test': X_test_dense,
    'vectorizer': tfidf_vectorizer,
    'label_encoder': label_encoder
}
print("Neural Network trained successfully")

# XGBoost
if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=80,             
        max_depth=4,                 
        learning_rate=0.08,          
        subsample=0.8,               
        colsample_bytree=0.8,        
        colsample_bylevel=0.8,       
        min_child_weight=5,          
        gamma=0.1,                   
        reg_alpha=0.1,               
        reg_lambda=1.0,              
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_model.fit(
        X_train_dense, 
        y_train_encoded,
        eval_set=[(X_test_dense, y_test_encoded)],  
        verbose=False
    )
    models['XGBoost'] = xgb_model
    models_data['XGBoost'] = {
        'X_train': X_train_dense,
        'X_test': X_test_dense,
        'vectorizer': tfidf_vectorizer,
        'label_encoder': label_encoder
    }
    print("XGBoost trained successfully")

print("\nAll models trained successfully")

# evaluate models

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    data = models_data[name]
    X_test = data['X_test']
    
    # Predictions
    y_pred_encoded = model.predict(X_test)
    
    # Decode predictions for models that use encoded labels
    if name in ['MLP', 'XGBoost']:
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred = y_pred_encoded
    
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test_original, y_pred)
    precision_weighted = precision_score(y_test_original, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test_original, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test_original, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test_original, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_original, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test_original, y_pred, average='macro', zero_division=0)
    
    print(f"  Running cross-validation for {name}...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  
    X_train_cv = data['X_train']
    
    if name in ['RandomForest', 'LinearSVC']:
        X_train_cv_for_cv = X_train_cv
    else:
        if hasattr(X_train_cv, 'toarray'):
            X_train_cv_for_cv = X_train_cv.toarray()
        else:
            X_train_cv_for_cv = X_train_cv
    
    y_train_cv = y_train_encoded if name in ['MLP', 'XGBoost'] else y_train_original
    
    if name == 'XGBoost' and XGBOOST_AVAILABLE:
        cv_model = xgb.XGBClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            colsample_bylevel=model.colsample_bylevel,
            min_child_weight=model.min_child_weight,
            gamma=model.gamma,
            reg_alpha=model.reg_alpha,
            reg_lambda=model.reg_lambda,
            random_state=model.random_state,
            n_jobs=1,  
            eval_metric=model.eval_metric
        )
    else:
        cv_model = model
    
    cv_scores_weighted = cross_val_score(cv_model, X_train_cv_for_cv, y_train_cv, cv=cv, scoring='f1_weighted', n_jobs=1)
    cv_scores_macro = cross_val_score(cv_model, X_train_cv_for_cv, y_train_cv, cv=cv, scoring='f1_macro', n_jobs=1)
    cv_mean_weighted = cv_scores_weighted.mean()
    cv_std_weighted = cv_scores_weighted.std()
    cv_mean_macro = cv_scores_macro.mean()
    cv_std_macro = cv_scores_macro.std()
    
    results[name] = {
        'test_accuracy': accuracy,
        'test_precision_weighted': precision_weighted,
        'test_precision_macro': precision_macro,
        'test_recall_weighted': recall_weighted,
        'test_recall_macro': recall_macro,
        'test_f1_weighted': f1_weighted,
        'test_f1_macro': f1_macro,
        'cv_mean_weighted': cv_mean_weighted,
        'cv_std_weighted': cv_std_weighted,
        'cv_mean_macro': cv_mean_macro,
        'cv_std_macro': cv_std_macro,
        'y_test_pred': y_pred,
        'y_test_pred_proba': y_pred_proba
    }
    print(f"  {name} evaluation completed: F1_weighted={f1_weighted:.4f}, F1_macro={f1_macro:.4f}")

# Calculate train metrics for each model
train_test_comparison = {}


for name, model in models.items():
    print(f"\nCalculating train metrics for {name}...")
    data = models_data[name]
    X_train = data['X_train']
    X_test = data['X_test']
    
    
    if name in ['RandomForest', 'LinearSVC']:
        X_train_for_pred = X_train  
        X_test_for_pred = X_test    
    else:
        if hasattr(X_train, 'toarray'):
            X_train_for_pred = X_train.toarray()
        else:
            X_train_for_pred = X_train
        
        if hasattr(X_test, 'toarray'):
            X_test_for_pred = X_test.toarray()
        else:
            X_test_for_pred = X_test
    
    y_train_pred_encoded = model.predict(X_train_for_pred)
    y_test_pred_encoded = model.predict(X_test_for_pred)
    
    # Decode if needed
    if name in ['MLP', 'XGBoost']:
        y_train_pred = label_encoder.inverse_transform(y_train_pred_encoded)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
    else:
        y_train_pred = y_train_pred_encoded
        y_test_pred = y_test_pred_encoded
    
    # Calculate metrics - Include both weighted and macro F1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    train_accuracy = accuracy_score(y_train_original, y_train_pred)
    train_precision_weighted = precision_score(y_train_original, y_train_pred, average='weighted', zero_division=0)
    train_precision_macro = precision_score(y_train_original, y_train_pred, average='macro', zero_division=0)
    train_recall_weighted = recall_score(y_train_original, y_train_pred, average='weighted', zero_division=0)
    train_recall_macro = recall_score(y_train_original, y_train_pred, average='macro', zero_division=0)
    train_f1_weighted = f1_score(y_train_original, y_train_pred, average='weighted', zero_division=0)
    train_f1_macro = f1_score(y_train_original, y_train_pred, average='macro', zero_division=0)
    
    test_accuracy = accuracy_score(y_test_original, y_test_pred)
    test_precision_weighted = precision_score(y_test_original, y_test_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_test_original, y_test_pred, average='macro', zero_division=0)
    test_recall_weighted = recall_score(y_test_original, y_test_pred, average='weighted', zero_division=0)
    test_recall_macro = recall_score(y_test_original, y_test_pred, average='macro', zero_division=0)
    test_f1_weighted = f1_score(y_test_original, y_test_pred, average='weighted', zero_division=0)
    test_f1_macro = f1_score(y_test_original, y_test_pred, average='macro', zero_division=0)
    
    train_test_comparison[name] = {
        'train_accuracy': train_accuracy,
        'train_precision_weighted': train_precision_weighted,
        'train_precision_macro': train_precision_macro,
        'train_recall_weighted': train_recall_weighted,
        'train_recall_macro': train_recall_macro,
        'train_f1_weighted': train_f1_weighted,
        'train_f1_macro': train_f1_macro,
        'test_accuracy': test_accuracy,
        'test_precision_weighted': test_precision_weighted,
        'test_precision_macro': test_precision_macro,
        'test_recall_weighted': test_recall_weighted,
        'test_recall_macro': test_recall_macro,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_macro': test_f1_macro,
        'overfitting_gap_weighted': train_f1_weighted - test_f1_weighted,  
        'overfitting_gap_macro': train_f1_macro - test_f1_macro  
    }

# Create visualization
print("\nGenerating overfitting analysis plots...")


# Create figure with subplots for each metric
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Train vs Test Performance Comparison (Overfitting Detection)', fontsize=16, fontweight='bold')

models_list = list(train_test_comparison.keys())
x_pos = np.arange(len(models_list))
width = 0.35

# Accuracy comparison
ax1 = axes[0, 0]
train_acc = [train_test_comparison[m]['train_accuracy'] for m in models_list]
test_acc = [train_test_comparison[m]['test_accuracy'] for m in models_list]
ax1.bar(x_pos - width/2, train_acc, width, label='Train', alpha=0.8, color='#3498db')
ax1.bar(x_pos + width/2, test_acc, width, label='Test', alpha=0.8, color='#e74c3c')
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy: Train vs Test')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_list, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Precision comparison
ax2 = axes[0, 1]
train_prec = [train_test_comparison[m]['train_precision_weighted'] for m in models_list]
test_prec = [train_test_comparison[m]['test_precision_weighted'] for m in models_list]
ax2.bar(x_pos - width/2, train_prec, width, label='Train', alpha=0.8, color='#3498db')
ax2.bar(x_pos + width/2, test_prec, width, label='Test', alpha=0.8, color='#e74c3c')
ax2.set_xlabel('Models')
ax2.set_ylabel('Precision')
ax2.set_title('Precision: Train vs Test')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_list, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

# Recall comparison
ax3 = axes[1, 0]
train_rec = [train_test_comparison[m]['train_recall_weighted'] for m in models_list]
test_rec = [train_test_comparison[m]['test_recall_weighted'] for m in models_list]
ax3.bar(x_pos - width/2, train_rec, width, label='Train', alpha=0.8, color='#3498db')
ax3.bar(x_pos + width/2, test_rec, width, label='Test', alpha=0.8, color='#e74c3c')
ax3.set_xlabel('Models')
ax3.set_ylabel('Recall')
ax3.set_title('Recall: Train vs Test')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models_list, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.05])

# F1 Score comparison (most important for overfitting detection) - Use weighted F1
ax4 = axes[1, 1]
train_f1 = [train_test_comparison[m]['train_f1_weighted'] for m in models_list]
test_f1 = [train_test_comparison[m]['test_f1_weighted'] for m in models_list]
bars1 = ax4.bar(x_pos - width/2, train_f1, width, label='Train', alpha=0.8, color='#3498db')
bars2 = ax4.bar(x_pos + width/2, test_f1, width, label='Test', alpha=0.8, color='#e74c3c')
ax4.set_xlabel('Models')
ax4.set_ylabel('F1 Score')
ax4.set_title('F1 Score: Train vs Test (Overfitting Indicator)')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models_list, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1.05])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save plot
plot_path = SENTIMENT_MODELS_DIR / 'train_test_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# Display plot (for console/IDE that supports it)
try:
    plt.show()
except:
    print("Note: Plot display not available in this environment. Check saved file.")

# Print detailed comparison table - Show both weighted and macro F1
print("OVERFITTING ANALYSIS - WEIGHTED F1 (accounts for class imbalance)")
print(f"{'Model':<20} {'Train F1':<12} {'Test F1':<12} {'Gap':<12} {'Status':<25}")
for name in models_list:
    comp = train_test_comparison[name]
    gap = comp['overfitting_gap_weighted']
    if gap > 0.15:
        status = "HIGH OVERFITTING"
    elif gap > 0.05:
        status = "MODERATE OVERFITTING"
    elif gap < -0.05:
        status = "UNDERFITTING"
    else:
        status = "GOOD FIT"
    
    print(f"{name:<20} {comp['train_f1_weighted']:>11.4f}  {comp['test_f1_weighted']:>11.4f}  "
          f"{gap:>+11.4f}  {status:<25}")

print("OVERFITTING ANALYSIS - MACRO F1 (treats all classes equally)")
print(f"{'Model':<20} {'Train F1':<12} {'Test F1':<12} {'Gap':<12} {'Status':<25}")
for name in models_list:
    comp = train_test_comparison[name]
    gap = comp['overfitting_gap_macro']
    if gap > 0.15:
        status = "HIGH OVERFITTING"
    elif gap > 0.05:
        status = "MODERATE OVERFITTING"
    elif gap < -0.05:
        status = "UNDERFITTING"
    else:
        status = "GOOD FIT"
    
    print(f"{name:<20} {comp['train_f1_macro']:>11.4f}  {comp['test_f1_macro']:>11.4f}  "
          f"{gap:>+11.4f}  {status:<25}")

print("\nINTERPRETATION:")
print("  • Gap > 0.15: HIGH OVERFITTING (model memorizing training data)")
print("  • Gap 0.05-0.15: MODERATE OVERFITTING (some overfitting)")
print("  • Gap -0.05 to 0.05: GOOD FIT (balanced performance)")
print("  • Gap < -0.05: UNDERFITTING (model too simple)")
print("\nNOTE: Weighted F1 accounts for class imbalance, Macro F1 treats all classes equally.")

# Display results
print("MODEL PERFORMANCE SUMMARY - WEIGHTED F1 (accounts for class imbalance)")
print(f"{'Model':<20} {'Test Acc':<12} {'Test F1':<12} {'CV Mean':<12} {'CV Std':<12}")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['test_accuracy']:>11.4f}  {metrics['test_f1_weighted']:>11.4f}  "
          f"{metrics['cv_mean_weighted']:>11.4f}  {metrics['cv_std_weighted']:>11.4f}")

print("MODEL PERFORMANCE SUMMARY - MACRO F1 (treats all classes equally)")
print(f"{'Model':<20} {'Test Acc':<12} {'Test F1':<12} {'CV Mean':<12} {'CV Std':<12}")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['test_accuracy']:>11.4f}  {metrics['test_f1_macro']:>11.4f}  "
          f"{metrics['cv_mean_macro']:>11.4f}  {metrics['cv_std_macro']:>11.4f}")

# Find best model (using weighted F1)
best_model_name, best_model, best_metrics = find_best_model(models, results, metric='test_f1_weighted')

print(f"BEST MODEL: {best_model_name}")
print(f"  Test Accuracy: {best_metrics['test_accuracy']:.4f}")
print(f"  Test Precision (weighted): {best_metrics['test_precision_weighted']:.4f}")
print(f"  Test Precision (macro): {best_metrics['test_precision_macro']:.4f}")
print(f"  Test Recall (weighted): {best_metrics['test_recall_weighted']:.4f}")
print(f"  Test Recall (macro): {best_metrics['test_recall_macro']:.4f}")
print(f"  Test F1 Score (weighted): {best_metrics['test_f1_weighted']:.4f}")
print(f"  Test F1 Score (macro): {best_metrics['test_f1_macro']:.4f}")
print(f"  CV F1 Score (weighted): {best_metrics['cv_mean_weighted']:.4f} (±{best_metrics['cv_std_weighted']:.4f})")
print(f"  CV F1 Score (macro): {best_metrics['cv_mean_macro']:.4f} (±{best_metrics['cv_std_macro']:.4f})")

# Classification report
print(f"\nDetailed Classification Report:")
target_names = sorted(y.unique())
print(classification_report(y_test_original, best_metrics['y_test_pred'], target_names=target_names))

# Confusion matrix
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test_original, best_metrics['y_test_pred'], labels=target_names)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df)

# saving the best model and artifacts for explanibility
print(f"Saving best model and artifacts for explanibility...")
best_model_path = SENTIMENT_MODELS_DIR / 'best_model.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model {best_model_name} saved to: {best_model_path}")

# saving the vectorizer for later use
vectorizer_path = SENTIMENT_MODELS_DIR / 'tfidf_vectorizer.pkl'
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"Vectorizer saved to: {vectorizer_path}")

# saving emotion labels (sorted as in model)
emotion_labels = sorted(y.unique())
labels_path = SENTIMENT_MODELS_DIR / 'emotion_labels.pkl'
with open(labels_path, 'wb') as f:
    pickle.dump(np.array(emotion_labels), f)
print(f" Emotion labels saved: {labels_path}")
print(f"   Labels: {', '.join(emotion_labels)}")

# saving label encoder (if the best model uses encoded labels)
if best_model_name in ['MLP', 'XGBoost']:
    encoder_path = SENTIMENT_MODELS_DIR / 'label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {encoder_path}")
else:
    print(" Best model does not use encoded labels, so no label encoder is saved.")

# Always save LinearSVC model for explainability (coef_ extraction)
if 'LinearSVC' in models:
    linearsvc_path = SENTIMENT_MODELS_DIR / 'linearsvc_model.pkl'
    with open(linearsvc_path, 'wb') as f:
        pickle.dump(models['LinearSVC'], f)
    print(f" LinearSVC model saved (for explainability): {linearsvc_path}")
else:
    print("  LinearSVC model not found, skipping explainability model save")

# save model metadata and configuration
model_config = {
    'best_model_name': best_model_name,
    'best_model_metrics': {
        'test_accuracy': float(best_metrics['test_accuracy']),
        'test_f1_weighted': float(best_metrics['test_f1_weighted']),
        'test_f1_macro': float(best_metrics['test_f1_macro']),
        'cv_mean_weighted': float(best_metrics['cv_mean_weighted']),
        'cv_std_weighted': float(best_metrics['cv_std_weighted']),
        'cv_mean_macro': float(best_metrics['cv_mean_macro']),
        'cv_std_macro': float(best_metrics['cv_std_macro'])
    },
    'emotion_labels': emotion_labels,
    'vectorizer_params': {
        'max_features': tfidf_vectorizer.max_features,
        'min_df': tfidf_vectorizer.min_df,
        'max_df': tfidf_vectorizer.max_df,
        'ngram_range': tfidf_vectorizer.ngram_range
    },
    'training_date': datetime.now().isoformat(),
    'random_state': RANDOM_STATE,
    'n_train_samples': len(X_train_text),
    'n_test_samples': len(X_test_text)
}
config_path = SENTIMENT_MODELS_DIR / 'model_config.json'
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)
print(f"Model configuration saved to: {config_path}")

print(f"Model training and evaluation completed successfully!")