import re
import numpy as np
import pandas as pd
from backend.helpers.stop_words_sentiment_helper import EMOTION_ROOTS, STRONG_KEYWORDS, spanish_stopwords, stemmer, EMOTION_STYLES

def clean_text(text: str) -> str:
    """Limpia el texto, elimina stopwords y aplica stemming."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    
    words = text.split()
    stemmed = [stemmer.stem(w) for w in words if w not in spanish_stopwords]
    
    return " ".join(stemmed).strip()

def analyze(model_server, text: str, top_words: int = 5):
    """Analiza la emoción usando el modelo cargado y refuerzo de raíces."""
    if model_server.sentiment is None:
        raise RuntimeError("Sentiment model not loaded")
    
    data = model_server.sentiment
    model = data["model"]
    vectorizer = data["vectorizer"]
    emotion_labels = data["emotion_labels"]
    
    if isinstance(emotion_labels, np.ndarray):
        emotion_labels = emotion_labels.tolist()

    cleaned = clean_text(text)
    if not cleaned:
        return format_for_human({
            "predicted_emotion": "unknown",
            "confidence": 0.0,
            "key_words": [],
            "all_emotions_scores": {},
            "text_processed": ""
        })

    text_lower = str(text).lower()

    X = vectorizer.transform([cleaned])
    dec_scores = model.decision_function(X)[0]
    probs = np.exp(dec_scores - np.max(dec_scores))
    probs /= probs.sum()
    
    pred_idx = np.argmax(probs)
    pred_emotion = emotion_labels[pred_idx]
    confidence = float(probs[pred_idx])
    all_scores = {str(emotion_labels[i]): float(probs[i]) for i in range(len(emotion_labels))}

 
    strong_match_emotion = None
    strong_match_count = 0
    
    for emotion, keywords in STRONG_KEYWORDS.items():
        if emotion not in emotion_labels:
            continue
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > strong_match_count:
            strong_match_count = matches
            strong_match_emotion = emotion
    
    if strong_match_emotion and (strong_match_count >= 2 or confidence < 0.30):
        pred_emotion = strong_match_emotion
        confidence = all_scores[strong_match_emotion]
    elif strong_match_emotion and strong_match_count >= 1 and confidence < 0.40:
        if all_scores.get(strong_match_emotion, 0) > 0.15:
            pred_emotion = strong_match_emotion
            confidence = all_scores[strong_match_emotion]
    elif confidence < 0.35:
        max_matches = 0
        detected_emotion = None
        
        for emotion, roots in EMOTION_ROOTS.items():
            if emotion not in emotion_labels:
                continue
            matches_cleaned = sum(1 for r in roots if r in cleaned)
            matches_original = sum(1 for r in roots if r in text_lower)
            total_matches = matches_cleaned + matches_original
            
            if total_matches > max_matches:
                max_matches = total_matches
                detected_emotion = emotion
        
        if detected_emotion and detected_emotion in emotion_labels:
            pred_emotion = detected_emotion
            confidence = all_scores[detected_emotion]

    feature_names = vectorizer.get_feature_names_out()
    text_vector = X.toarray()[0]
    present_indices = np.where(text_vector > 0)[0]
    
    target_idx = emotion_labels.index(pred_emotion)
    contributions = [
        (feature_names[i], model.coef_[target_idx][i] * text_vector[i]) 
        for i in present_indices
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)
    key_words_list = [c[0] for c in contributions[:top_words]]

    return format_for_human({
        "predicted_emotion": pred_emotion,
        "confidence": confidence,
        "key_words": key_words_list,
        "all_emotions_scores": all_scores,
        "text_processed": cleaned 
    })

def format_for_human(prediction_result: dict) -> dict:
    """Transform the technical response into something more human and friendly."""
    emotion = prediction_result["predicted_emotion"]
    confidence = prediction_result["confidence"]
    
    if emotion == "unknown":
        prediction_result["explanation"] = "No pude procesar el texto. ¿Puedes compartir algo más?"
        prediction_result["display_metadata"] = {
            "title": "Necesito más información",
            "color": "#9E9E9E",
            "icon": "help"
        }
        return prediction_result
    
    style = EMOTION_STYLES.get(emotion, {
        "title": "Análisis del momento",
        "title_low_conf": "Reflexión",
        "color": "#9E9E9E",
        "icon": "help",
        "messages": {
            "high": "He analizado lo que compartiste.",
            "medium": "Veo algunas señales en tus palabras.",
            "low": "Estoy procesando lo que me cuentas."
        }
    })
    
    if confidence >= 0.60:
        conf_level = "high"
    elif confidence >= 0.35:
        conf_level = "medium"
    else:
        conf_level = "low"
    
    if confidence < 0.25:
        display_title = style["title_low_conf"]
    else:
        display_title = style["title"]
    
    human_message = style["messages"][conf_level]
    
    all_scores = prediction_result.get("all_emotions_scores", {})
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_scores) > 1:
        second_emotion, second_score = sorted_scores[1]
        if second_score >= confidence - 0.15 and second_score > 0.20:
            second_style = EMOTION_STYLES.get(second_emotion, {})
            if second_style:
                emotion_translations = {
                    "scared": "preocupación",
                    "joyful": "alegría",
                    "sad": "tristeza",
                    "mad": "frustración",
                    "peaceful": "tranquilidad",
                    "powerful": "confianza"
                }
                emotion_spanish = emotion_translations.get(second_emotion, second_emotion)
                human_message += f" También noto algo de {emotion_spanish} en tus palabras."
    
    prediction_result["display_metadata"] = {
        "title": display_title,
        "color": style["color"],
        "icon": style["icon"]
    }
    
    prediction_result["explanation"] = human_message
    
    return prediction_result