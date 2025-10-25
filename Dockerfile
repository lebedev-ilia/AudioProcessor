# ===== 1. Базовый образ =====
FROM python:3.12.3-slim

# ===== 2. Установка системных зависимостей =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    curl \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ===== 3. Создание непривилегированного пользователя =====
RUN useradd --create-home --shell /bin/bash audio_processor

# ===== 4. Копирование зависимостей и установка =====
WORKDIR /app
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ===== 5. Копирование исходного кода =====
COPY src/ /app/src/

# ===== 6. Настройка окружения =====
ENV PYTHONPATH=/app
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0
ENV TRANSFORMERS_CACHE=/app/.cache
ENV TORCH_HOME=/app/.cache/torch
ENV PYTHONUNBUFFERED=1

# ===== 7. Создание директорий и прав =====
RUN mkdir -p /app/logs /app/.cache \
    && chown -R audio_processor:audio_processor /app

# ===== 8. Healthcheck =====
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ===== 9. Переключение на безопасного пользователя =====
USER audio_processor

# ===== 10. Команда по умолчанию =====
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
