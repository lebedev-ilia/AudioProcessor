# AudioProcessor

Микросервис для извлечения аудио признаков из медиафайлов. Построен на FastAPI + Celery архитектуре с модульной системой extractors.

## 🎯 Возможности

### ✅ Реализованные Extractors
- **MFCC** - Mel-frequency cepstral coefficients (13 + delta)
- **Mel Spectrogram** - 64 мел-банда с статистическими признаками
- **Chroma** - 12 тональных классов для гармонического анализа
- **RMS/Loudness** - энергетические характеристики (RMS, LUFS)
- **VAD** - Voice Activity Detection с извлечением F0
- **CLAP** - семантические аудио эмбеддинги (512 dim)

### 🔄 Планируемые Extractors
- **ASR** - автоматическое распознавание речи (Whisper)
- **Sentiment** - анализ сентимента
- **NER** - извлечение именованных сущностей
- **Topic Modeling** - тематическое моделирование
- **Text Embeddings** - текстовые эмбеддинги

## 🏗 Архитектура

```
audio_processor/
├── src/
│   ├── main.py                    # FastAPI entrypoint
│   ├── celery_app.py             # Celery configuration
│   ├── config.py                 # Configuration settings
│   ├── core/
│   │   ├── base_extractor.py     # Base extractor interface
│   │   └── utils.py              # Utilities
│   ├── extractors/
│   │   ├── mfcc_extractor.py     # MFCC features
│   │   ├── mel_extractor.py      # Mel spectrogram
│   │   ├── chroma_extractor.py   # Chroma features
│   │   ├── loudness_extractor.py # RMS/Loudness
│   │   ├── vad_extractor.py      # Voice Activity Detection
│   │   └── openl3_extractor.py   # OpenL3 embeddings
│   ├── storage/
│   │   └── s3_client.py          # S3 client
│   ├── schemas/
│   │   └── models.py             # Pydantic models
│   ├── monitor/
│   │   └── metrics.py            # Prometheus metrics
│   └── tests/
├── k8s/                          # Kubernetes manifests
├── Dockerfile                    # Docker image
├── docker-compose.yml           # Development environment
└── requirements.txt             # Python dependencies
```

## 🚀 Быстрый старт

### Локальная разработка

1. **Клонирование и настройка**
```bash
git clone <repository-url>
cd AudioProcessor

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

2. **Запуск Redis (требуется для Celery)**
```bash
# macOS с Homebrew
brew services start redis

# Или через Docker
docker run -d -p 6379:6379 redis:alpine
```

3. **Запуск Celery Worker**
```bash
cd AudioProcessor
source .venv/bin/activate
PYTHONPATH=/path/to/AudioProcessor/src celery -A src.celery_app worker --loglevel=info --concurrency=1
```

4. **Запуск FastAPI сервера**
```bash
cd AudioProcessor
source .venv/bin/activate
PYTHONPATH=/path/to/AudioProcessor/src python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **Проверка статуса**
```bash
curl http://localhost:8000/health
```

### Docker Compose (рекомендуется)
```bash
docker-compose up -d
```

### API Endpoints

- `POST /process` - обработка аудио файла (rate limited: 10 req/min)
- `GET /task/{task_id}` - статус задачи
- `GET /extractors` - список доступных экстракторов
- `GET /health` - базовая проверка здоровья сервиса
- `GET /health/detailed` - детальная диагностика всех компонентов
- `GET /health/{check_name}` - проверка конкретного компонента
- `GET /metrics` - Prometheus метрики
- `GET /docs` - Swagger UI документация

### Пример запроса

```bash
# Обработка аудио файла
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test_video_123",
    "audio_uri": "test_audio.wav",
    "task_id": "task_456",
    "dataset": "test",
    "meta": {
      "test": true
    }
  }'

# Проверка статуса задачи
curl http://localhost:8000/task/task_456

# Список экстракторов
curl http://localhost:8000/extractors
```

## 🔧 Разработка

### Создание нового extractor

```python
# src/extractors/new_extractor.py
from core.base_extractor import BaseExtractor, ExtractorResult

class NewExtractor(BaseExtractor):
    name = "new_feature"
    version = "0.1.0"
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        try:
            # Ваша логика извлечения признаков
            features = self._extract_features(input_uri)
            
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=True,
                payload=features
            )
        except Exception as e:
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=False,
                error=str(e)
            )
```

### Тестирование

```bash
# Запуск тестов
pytest src/tests/ -v

# Проверка кода
flake8 src/
black src/
mypy src/
```

## ✅ Статус проекта

### 🎉 Полностью функциональные компоненты:
- ✅ **FastAPI сервер** - REST API с полной документацией
- ✅ **Celery Worker** - асинхронная обработка задач с retry логикой
- ✅ **6 Audio Extractors** - все экстракторы работают корректно
- ✅ **Redis интеграция** - очереди и результаты
- ✅ **S3 клиент** - с fallback на локальное сохранение
- ✅ **Мониторинг задач** - отслеживание прогресса в реальном времени
- ✅ **Обработка ошибок** - graceful degradation
- ✅ **Prometheus метрики** - комплексный мониторинг
- ✅ **Структурированное логирование** - JSON логи с correlation ID
- ✅ **Health checks** - проверка всех зависимостей
- ✅ **Rate limiting** - защита от перегрузки (10 req/min)
- ✅ **CORS и middleware** - настройка безопасности

### 🧪 Протестировано:
- ✅ Обработка аудио через API
- ✅ Все экстракторы (MFCC, Mel, Chroma, Loudness, VAD, CLAP)
- ✅ Создание манифестов
- ✅ Мониторинг задач
- ✅ Health checks (Redis, S3, MasterML, Celery, System)
- ✅ Prometheus метрики
- ✅ Структурированное логирование
- ✅ Rate limiting
- ✅ Retry логика Celery
- ✅ Progress tracking

## 📊 Мониторинг

- **API документация**: `http://localhost:8000/docs`
- **Prometheus метрики**: `http://localhost:8000/metrics`
- **Flower (Celery)**: `http://localhost:5555`
- **Grafana**: `http://localhost:3000` (admin/admin)

## 🐳 Развертывание

### Docker

```bash
# Сборка образа
docker build -t audio-processor:latest .

# Запуск
docker run -p 8000:8000 audio-processor:latest
```

### Kubernetes

```bash
# Развертывание
kubectl apply -f k8s/

# Проверка статуса
kubectl get pods -n ml-service
```

## 📚 Документация

- [Архитектура](docs/architecture.md)
- [Руководство разработчика](docs/developer/AudioProcessor_Development_Checklist.md)
- [API документация](http://localhost:8000/docs)

## 🤝 Вклад в проект

1. Fork репозитория
2. Создать feature branch
3. Внести изменения
4. Добавить тесты
5. Создать Pull Request

## 📄 Лицензия

MIT License

## 🆘 Поддержка

- Создать Issue в GitHub
- Проверить [troubleshooting guide](docs/troubleshooting.md)
- Обратиться к команде разработки
