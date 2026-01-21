"""
Sentiment Model Explainer - Emotion Detection Explanation
Extracts top words per emotion and generates understandable explanations for the AI Coach
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
import json
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# Import from utils
try:
    from utils import (
        BASE_DIR,
        MODELS_DIR,
        RANDOM_STATE,
        load_data
    )
except ImportError:
    BASE_DIR = PROJECT_ROOT
    MODELS_DIR = BASE_DIR / 'models'
    RANDOM_STATE = 42
    
    def load_data(file_path):
        return pd.read_csv(file_path)

SENTIMENT_MODELS_DIR = MODELS_DIR / 'sentiment'
SENTIMENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_top_words_per_emotion(model, vectorizer, emotion_labels, top_n=10):
    """
    Extract top N most important words for each emotion using LinearSVC coefficients
    
    Args:
        model: Trained LinearSVC model
        vectorizer: TfidfVectorizer used for training
        emotion_labels: List of emotion labels (in class order)
        top_n: Number of top words to extract
    
    Returns:
        dict: {emotion: [(word, coefficient), ...]}
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model must have coef_ attribute (LinearSVC)")
    
    # Get model coefficients (shape: [n_classes, n_features])
    coefficients = model.coef_
    
    # Get feature names (words/n-grams)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    top_words = {}
    
    for idx, emotion in enumerate(emotion_labels):
        # Coefficients for this emotion
        emotion_coefs = coefficients[idx]
        
        # Top words with highest positive coefficient (indicate this emotion)
        top_indices = np.argsort(emotion_coefs)[-top_n:][::-1]
        top_words[emotion] = [
            (feature_names[i], float(emotion_coefs[i]))
            for i in top_indices
        ]
    
    return top_words


def clean_text(text):
    """
    Clean text using the same function as in training
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def explain_prediction(text, model, vectorizer, emotion_labels, top_words_to_show=5):
    """
    Explain why a text was classified as a specific emotion
    
    Args:
        text: Text to explain
        model: Trained LinearSVC model
        vectorizer: TfidfVectorizer used
        emotion_labels: List of emotion labels
        top_words_to_show: Number of key words to show
    
    Returns:
        dict: {
            'predicted_emotion': str,
            'confidence': float,
            'explanation': str,
            'key_words': list,
            'all_emotion_scores': dict
        }
    """
    # Clean text
    text_cleaned = clean_text(text)
    
    if not text_cleaned:
        return {
            'predicted_emotion': 'unknown',
            'confidence': 0.0,
            'explanation': 'Empty text after cleaning',
            'key_words': [],
            'word_contributions': [],
            'all_emotion_scores': {}
        }
    
    # Vectorize text
    text_vectorized = vectorizer.transform([text_cleaned])
    
    # Predict
    prediction_idx = model.predict(text_vectorized)[0]
    
    # Convert index to label
    if isinstance(prediction_idx, (int, np.integer)):
        predicted_emotion = emotion_labels[prediction_idx]
    else:
        # If model returns label directly
        predicted_emotion = str(prediction_idx)
        prediction_idx = np.where(emotion_labels == predicted_emotion)[0][0]
    
    # Get decision scores for all emotions
    decision_scores = model.decision_function(text_vectorized)[0]
    
    # Convert to probabilities using softmax
    exp_scores = np.exp(decision_scores - np.max(decision_scores))  # For numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    confidence = probabilities[prediction_idx]
    
    # Get all probabilities per emotion
    all_emotion_scores = {
        str(emotion_labels[i]): float(probabilities[i])
        for i in range(len(emotion_labels))
    }
    
    # Get coefficients for predicted emotion
    coefficients = model.coef_[prediction_idx]
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get words present in text with highest weight
    text_vector = text_vectorized.toarray()[0]
    present_indices = np.where(text_vector > 0)[0]
    
    # Calculate contribution of each word
    word_contributions = []
    for idx in present_indices:
        word = feature_names[idx]
        contribution = coefficients[idx] * text_vector[idx]
        word_contributions.append((word, float(contribution), float(text_vector[idx])))
    
    # Sort by contribution
    word_contributions.sort(key=lambda x: x[1], reverse=True)
    top_contributing_words = word_contributions[:top_words_to_show]
    
    # Generate explanation
    key_words = [word for word, _, _ in top_contributing_words]
    explanation = f"Your emotion was detected as **{predicted_emotion.upper()}** mainly due to:\n"
    if key_words:
        explanation += "\n".join([f"- '{word}'" for word in key_words])
    else:
        explanation += "- (no significant keywords found)"
    
    return {
        'predicted_emotion': predicted_emotion,
        'confidence': float(confidence),
        'explanation': explanation,
        'key_words': key_words,
        'word_contributions': top_contributing_words,
        'all_emotion_scores': all_emotion_scores
    }


def generate_emotion_keywords_report(top_words_dict, output_path=None):
    """
    Generate a keyword report by emotion
    
    Args:
        top_words_dict: Dict of top words by emotion
        output_path: Path where to save the report (optional)
    
    Returns:
        str: Complete report
    """
    print("\n" + "="*80)
    print("KEY WORDS BY EMOTION (Top 10 most important words)")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("KEY WORDS BY EMOTION - SENTIMENT ANALYSIS MODEL")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("These are the words that most influence the detection of each emotion.")
    report_lines.append("Positive coefficient = word indicates that emotion")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    for emotion, words in top_words_dict.items():
        print(f"\n{emotion.upper()}:")
        print("-" * 60)
        report_lines.append(f"\n{emotion.upper()}:")
        report_lines.append("-" * 60)
        
        for i, (word, coef) in enumerate(words, 1):
            print(f"  {i:2d}. {word:30s} (coef: {coef:+.4f})")
            report_lines.append(f"  {i:2d}. {word:30s} (coef: {coef:+.4f})")
        
        report_lines.append("")
    
    # Save report if path specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to: {output_path}")
    
    return '\n'.join(report_lines)


def print_probability_bar(emotion, probability, max_width=50):
    """
    Print a visual probability bar
    """
    bar_width = int(probability * max_width)
    bar = '█' * bar_width + '░' * (max_width - bar_width)
    print(f"   {emotion:12s} [{bar}] {probability:.1%}")


def load_or_train_linearsvc():
    """
    Load LinearSVC if exists, or train it if not available
    
    Returns:
        tuple: (model, vectorizer, emotion_labels)
    """
    linearsvc_path = SENTIMENT_MODELS_DIR / 'linearsvc_model.pkl'
    vectorizer_path = SENTIMENT_MODELS_DIR / 'tfidf_vectorizer.pkl'
    labels_path = SENTIMENT_MODELS_DIR / 'emotion_labels.pkl'
    
    # Try to load LinearSVC
    if linearsvc_path.exists() and vectorizer_path.exists() and labels_path.exists():
        print("Loading LinearSVC and saved artifacts...")
        try:
            with open(linearsvc_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(labels_path, 'rb') as f:
                emotion_labels = pickle.load(f)
            print("Model loaded successfully")
            return model, vectorizer, emotion_labels
        except Exception as e:
            print(f"Error loading model: {e}")
            print("   Trying to train from scratch...")
    
    # If not exists, train it
    print("LinearSVC not found. Training from data...")
    
    # Load data
    PROCESSED_DATA_DIR = BASE_DIR / 'datasets' / 'processed'
    data_path = PROCESSED_DATA_DIR / '07_cleaned_sentiment_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = load_data(data_path)
    
    X_text = df['text'].fillna('').astype(str)
    y = df['sentiment']
    
    # Clean text
    X_text_cleaned = X_text.apply(clean_text)
    mask = X_text_cleaned.str.len() > 0
    X_text_cleaned = X_text_cleaned[mask]
    y = y[mask]
    
    print(f"Data loaded: {len(X_text_cleaned)} texts")
    print(f"Emotion distribution:")
    print(y.value_counts())
    
    # Vectorize and train
    from sklearn.model_selection import train_test_split
    
    X_train_text, _, y_train, _ = train_test_split(
        X_text_cleaned, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    emotion_labels = np.array(sorted(y.unique()))
    
    print(f"\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words=None
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    print(f"Vocabulary: {len(vectorizer.vocabulary_)} words")
    
    print(f"\nTraining LinearSVC...")
    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_iter=2000,
        dual=False,
        tol=1e-4
    )
    model.fit(X_train_tfidf, y_train)
    print("LinearSVC trained")
    
    # Save model
    print(f"\nSaving model and artifacts...")
    with open(linearsvc_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(labels_path, 'wb') as f:
        pickle.dump(emotion_labels, f)
    
    print("LinearSVC trained and saved")
    return model, vectorizer, emotion_labels


def test_interactive_mode(model, vectorizer, emotion_labels):
    """
    Interactive mode to test the model with custom texts
    """
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Test the model with your own texts")
    print("="*80)
    print("Type 'quit' or 'exit' to leave")
    print()
    
    test_results = []
    
    while True:
        try:
            text = input("\nEnter a text (or 'quit' to exit): ").strip()
            
            if text.lower() in ['quit', 'exit', 'salir', 'q']:
                break
            
            if not text:
                print("Please enter a valid text")
                continue
            
            # Explain prediction
            explanation = explain_prediction(text, model, vectorizer, emotion_labels, top_words_to_show=5)
            
            print(f"\nPredicted emotion: {explanation['predicted_emotion'].upper()}")
            print(f"Confidence: {explanation['confidence']:.1%}")
            
            print(f"\nExplanation:")
            print(explanation['explanation'])
            
            print(f"\nProbabilities by emotion:")
            for emotion, prob in sorted(explanation['all_emotion_scores'].items(), 
                                        key=lambda x: x[1], reverse=True):
                print_probability_bar(emotion, prob)
            
            # Save result
            test_results.append({
                'text': text,
                'prediction': explanation['predicted_emotion'],
                'confidence': explanation['confidence'],
                'timestamp': datetime.now().isoformat()
            })
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    # Save test results
    if test_results:
        results_path = SENTIMENT_MODELS_DIR / 'test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"\nTest results saved to: {results_path}")


def main():
    """
    Main function: generate model explanations
    """
    print("="*80)
    print("SENTIMENT MODEL EXPLAINER")
    print("="*80)
    
    try:
        # Load or train LinearSVC
        model, vectorizer, emotion_labels = load_or_train_linearsvc()
        
        print(f"\nDetected emotions: {', '.join(emotion_labels)}")
        
        # Generate keyword report
        print("\nExtracting key words by emotion...")
        top_words = get_top_words_per_emotion(model, vectorizer, emotion_labels, top_n=10)
        
        report_path = SENTIMENT_MODELS_DIR / 'emotion_keywords_report.txt'
        generate_emotion_keywords_report(top_words, output_path=report_path)
        
        # Prediction examples
        print("\n" + "="*80)
        print("PREDICTION EXPLANATION EXAMPLES")
        print("="*80)
        
        example_texts = [
               "Me siento muy solo y cansado de todo",
                "¡Esto es increíble! Estoy muy feliz y emocionado",
                "Tengo miedo y ansiedad por lo que viene",
                "Hoy me siento poderoso y lleno de confianza",
                "Estoy muy enojado por lo que pasó"

        ]
        
        for text in example_texts:
            print(f"\nText: \"{text}\"")
            print("-" * 60)
            explanation = explain_prediction(text, model, vectorizer, emotion_labels, top_words_to_show=5)
            print(f"Predicted emotion: {explanation['predicted_emotion'].upper()}")
            print(f"Confidence: {explanation['confidence']:.1%}")
            print(f"\nExplanation:")
            print(explanation['explanation'])
            print(f"\nProbabilities by emotion:")
            for emotion, prob in sorted(explanation['all_emotion_scores'].items(), 
                                        key=lambda x: x[1], reverse=True)[:3]:  # Top 3
                print_probability_bar(emotion, prob)
        
        # Save complete explainer for future use
        explainer_data = {
            'model': model,
            'vectorizer': vectorizer,
            'emotion_labels': emotion_labels,
            'top_words': top_words
        }
        
        explainer_path = SENTIMENT_MODELS_DIR / 'model_explainer.pkl'
        with open(explainer_path, 'wb') as f:
            pickle.dump(explainer_data, f)
        
        print(f"\nExplainer saved to: {explainer_path}")
        
        # Ask if want interactive mode
        print("\n" + "="*80)
        response = input("\nDo you want to test the model with your own texts? (y/n): ").strip().lower()
        if response in ['s', 'si', 'sí', 'yes', 'y']:
            test_interactive_mode(model, vectorizer, emotion_labels)
        
        print("\nAnalysis completed!")
        print("\nFor production use:")
        print("   from src.utils.sentiment_model_explainer import explain_prediction, load_or_train_linearsvc")
        print("   model, vectorizer, labels = load_or_train_linearsvc()")
        print("   explanation = explain_prediction('your text here', model, vectorizer, labels)")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
