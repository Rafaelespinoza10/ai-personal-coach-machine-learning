import re
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
stemmer = SnowballStemmer('spanish')
spanish_stopwords = set(stopwords.words('spanish'))

EMOTION_ROOTS = {
    "scared": ["mied", "tem", "ansied", "preocup", "nervi", "asust", "panic", "terror", "miedo", "ansiedad", "preocupado", "nervioso"],
    "sad": ["trist", "deprim", "melancol", "desanim", "desesper", "llor", "mal", "malo", "malos", "mala", "malas", "triste", "tristeza", "deprimido", "desanimado"],
    "joyful": ["feliz", "alegr", "content", "emocion", "entusiasm", "genial", "fantast", "bien", "bueno", "buenos", "buena", "buenas", "alegre", "contento"],
    "mad": ["enoj", "furi", "irrit", "molest", "rabi", "ira", "enojado", "furioso", "molesto"],
    "peaceful": ["tranquil", "calm", "seren", "relaj", "paz", "armoni", "tranquilo", "calma", "relajado"],
    "powerful": ["poder", "fuert", "capaz", "confi", "segur", "triunf", "poderoso", "fuerte", "confiado", "seguro"],
}

# Palabras completas que tienen peso fuerte (no solo raíces)
STRONG_KEYWORDS = {
    "sad": ["mal", "malo", "mala", "malos", "malas", "triste", "tristeza", "deprimido", "desanimado", "llorar", "lloro", "tristeza"],
    "scared": ["miedo", "ansiedad", "preocupado", "nervioso", "asustado", "pánico", "preocupación"],
    "joyful": ["feliz", "alegre", "contento", "bien", "bueno", "genial", "fantástico", "increible", "increíble", "maravilloso", "excelente", "perfecto", "perfecta", "emocionado", "entusiasmado"],
    "mad": ["enojado", "furioso", "molesto", "rabia", "ira", "enfadado"],
    "peaceful": ["tranquilo", "calma", "relajado", "paz", "sereno", "tranquila"],
    "powerful": ["poderoso", "fuerte", "confiado", "seguro", "triunfo", "capaz", "segura"]
}

# Diccionario de estilos y mensajes humanos por emoción
EMOTION_STYLES = {
    "scared": {
        "title": "Momento de calma",
        "title_low_conf": "Reflexión del momento",
        "color": "#7E57C2",
        "icon": "spa",
        "messages": {
            "high": "Veo que estás pasando por un momento de preocupación. Es normal sentir miedo o ansiedad a veces.",
            "medium": "Parece que hay algo que te está generando inquietud. ¿Quieres hablar de ello?",
            "low": "Detecto algunas señales de preocupación en tus palabras. ¿Cómo te sientes realmente?"
        }
    },
    "joyful": {
        "title": "¡Qué buena energía!",
        "title_low_conf": "Buen momento",
        "color": "#FFD54F",
        "icon": "wb_sunny",
        "messages": {
            "high": "¡Me encanta ver esta energía positiva! Sigue así.",
            "medium": "Parece que estás en un buen momento. Disfrútalo.",
            "low": "Veo algunos destellos de alegría en lo que compartes."
        }
    },
    "sad": {
        "title": "Estoy aquí contigo",
        "title_low_conf": "Un momento difícil",
        "color": "#64B5F6",
        "icon": "cloud",
        "messages": {
            "high": "Siento que estás pasando por un momento difícil. No estás solo en esto.",
            "medium": "Parece que hay algo que te está afectando. Es válido sentirse así.",
            "low": "Detecto algunas señales de tristeza. ¿Quieres compartir cómo te sientes?"
        }
    },
    "mad": {
        "title": "Entiendo tu frustración",
        "title_low_conf": "Un momento intenso",
        "color": "#EF5350",
        "icon": "whatshot",
        "messages": {
            "high": "Veo que hay algo que te está molestando mucho. Es válido sentir enojo.",
            "medium": "Parece que estás pasando por un momento de frustración. ¿Qué está pasando?",
            "low": "Detecto algunas señales de irritación. ¿Quieres hablar de ello?"
        }
    },
    "peaceful": {
        "title": "Qué tranquilidad",
        "title_low_conf": "Un momento sereno",
        "color": "#81C784",
        "icon": "self_improvement",
        "messages": {
            "high": "Me encanta sentir esta calma en tus palabras. Disfruta este momento.",
            "medium": "Parece que estás en un estado de tranquilidad. Eso es bueno.",
            "low": "Veo algunas señales de serenidad en lo que compartes."
        }
    },
    "powerful": {
        "title": "¡Sigue así!",
        "title_low_conf": "Buen momento",
        "color": "#FF9800",
        "icon": "trending_up",
        "messages": {
            "high": "¡Qué energía tan poderosa! Sientes que puedes con todo.",
            "medium": "Parece que estás en un momento de fuerza y determinación.",
            "low": "Veo algunas señales de confianza y poder en tus palabras."
        }
    }
}
