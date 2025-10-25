FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash audio_processor

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY src/ /app/src/
WORKDIR /app

# Создание директории для логов
RUN mkdir -p logs && chown -R audio_processor:audio_processor /app

# Переменные окружения
ENV PYTHONPATH=/app
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Переключение на пользователя
USER audio_processor

# Команда по умолчанию
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
