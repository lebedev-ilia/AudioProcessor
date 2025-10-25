# AudioProcessor

Микросервис для извлечения аудио признаков из медиафайлов. Построен на FastAPI + Celery архитектуре с модульной системой extractors.

## 🎯 Возможности

### CORE Extractors (MVP)
- **MFCC** - Mel-frequency cepstral coefficients (13 + delta)
- **Mel Spectrogram** - 64 мел-банда
- **Chroma** - 12 тональных классов
- **RMS/Loudness** - энергетические характеристики
- **VAD** - Voice Activity Detection
- **OpenL3** - семантические эмбеддинги (512 dim)

### ADVANCED Extractors
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
cd audio_processor

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

2. **Запуск через Docker Compose**
```bash
docker-compose up -d
```

3. **Проверка статуса**
```bash
curl http://localhost:8000/health
```

### API Endpoints

- `POST /process` - обработка аудио файла
- `GET /health` - проверка здоровья сервиса
- `GET /metrics` - Prometheus метрики

### Пример запроса

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test_video_123",
    "audio_uri": "s3://bucket/audio.wav",
    "dataset": "test"
  }'
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

## 📊 Мониторинг

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
