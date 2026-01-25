FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias de sistema necesarias para XGBoost
RUN apt-get update && apt-get install -y \
    gcc python3-dev \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Copiamos el código
COPY backend/ ./backend/
COPY src/ ./src/

# Creamos la carpeta de modelos vacía (se llenará desde S3)
RUN mkdir -p /app/models

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Script de inicio que descarga modelos y luego lanza uvicorn
CMD ["sh", "-c", "python backend/utils/download_models.py && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]