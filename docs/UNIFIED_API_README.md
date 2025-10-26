# Unified Audio Processing API

Новый unified API объединяет традиционную обработку AudioProcessor с per-segment pipeline в единый endpoint. Теперь вы можете получить как агрегированные фичи, так и per-segment последовательности в одном запросе!

## 🎯 Основные возможности

### 1. **Три режима обработки**
- `aggregates_only` - только агрегированные фичи (традиционный AudioProcessor)
- `segments_only` - только per-segment последовательности
- `both` - и агрегированные фичи, и последовательности

### 2. **Гибкая конфигурация**
- Выбор extractors
- Настройка параметров сегментации
- Настройка PCA сжатия
- Выбор стратегий выбора сегментов

### 3. **Batch обработка**
- Обработка множества видео в одном запросе
- Оптимизированное использование ресурсов

## 🚀 Быстрый старт

### Запуск сервера
```bash
cd AudioProcessor
python -m src.main
```

### Базовый запрос (только агрегированные фичи)
```bash
curl -X POST "http://localhost:8000/unified/process" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "my_video",
    "audio_uri": "s3://bucket/audio.wav",
    "processing_mode": "aggregates_only",
    "extractor_names": ["clap_extractor", "loudness_extractor", "vad_extractor"]
  }'
```

### Полная обработка (агрегаты + сегменты)
```bash
curl -X POST "http://localhost:8000/unified/process" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "my_video",
    "audio_uri": "s3://bucket/audio.wav",
    "processing_mode": "both",
    "segment_config": {
      "segment_len": 3.0,
      "hop": 1.5,
      "max_seq_len": 128,
      "k_start": 16,
      "k_end": 16
    },
    "extractor_names": ["clap_extractor", "loudness_extractor", "vad_extractor", "advanced_embeddings"]
  }'
```

## 📋 API Endpoints

### POST `/unified/process`
Основной endpoint для обработки одного видео/аудио файла.

**Параметры запроса:**
```json
{
  "video_id": "string",                    // Обязательно: ID видео
  "audio_uri": "string",                   // URI аудио файла (взаимоисключающий с video_uri)
  "video_uri": "string",                   // URI видео файла (взаимоисключающий с audio_uri)
  "processing_mode": "aggregates_only",    // Режим обработки
  "segment_config": {                      // Конфигурация сегментов (если не aggregates_only)
    "segment_len": 3.0,
    "hop": 1.5,
    "max_seq_len": 128,
    "k_start": 16,
    "k_end": 16,
    "pca_dims": {
      "clap": 128,
      "wav2vec": 64,
      "yamnet": 128
    }
  },
  "extractor_names": ["clap_extractor"],   // Список extractors (опционально)
  "output_dir": "string",                  // Директория вывода (опционально)
  "dataset": "default",                    // Название датасета
  "meta": {}                               // Дополнительные метаданные
}
```

**Ответ:**
```json
{
  "accepted": true,
  "celery_task_id": "task-123",
  "message": "Unified audio processing request accepted",
  "processing_mode": "both"
}
```

### POST `/unified/batch`
Batch обработка множества видео.

**Параметры запроса:**
```json
{
  "videos": [
    {
      "video_id": "video_001",
      "audio_uri": "s3://bucket/audio1.wav"
    },
    {
      "video_id": "video_002",
      "video_uri": "s3://bucket/video2.mp4"
    }
  ],
  "processing_mode": "both",
  "segment_config": {
    "segment_len": 3.0,
    "max_seq_len": 128
  },
  "extractor_names": ["clap_extractor", "loudness_extractor"]
}
```

### GET `/unified/task/{task_id}`
Получение статуса задачи.

**Ответ:**
```json
{
  "task_id": "task-123",
  "status": "completed",
  "progress": 100.0,
  "result": {
    "video_id": "my_video",
    "success": true,
    "aggregates_extracted": true,
    "segments_extracted": true,
    "manifest_path": "/output/my_video_manifest.json",
    "segment_files": {
      "features_file": "/output/my_video_features.npy",
      "mask_file": "/output/my_video_mask.npy",
      "meta_file": "/output/my_video_meta.json"
    },
    "num_segments": 20,
    "num_selected_segments": 16,
    "feature_shape": [16, 256],
    "processing_time": 45.2
  }
}
```

### GET `/unified/config`
Получение дефолтной конфигурации.

### GET `/unified/examples`
Получение примеров запросов.

## 🔧 Режимы обработки

### 1. `aggregates_only`
Традиционная обработка AudioProcessor - извлекает только агрегированные фичи.

**Результат:**
- `manifest.json` с результатами extractors
- Никаких per-segment файлов

**Использование:**
```json
{
  "processing_mode": "aggregates_only",
  "extractor_names": ["clap_extractor", "loudness_extractor", "vad_extractor"]
}
```

### 2. `segments_only`
Извлекает только per-segment последовательности (без manifest).

**Результат:**
- `features.npy` - матрица фичей (max_seq_len, feature_dim)
- `mask.npy` - attention mask (max_seq_len,)
- `meta.json` - метаданные сегментов

**Использование:**
```json
{
  "processing_mode": "segments_only",
  "segment_config": {
    "segment_len": 3.0,
    "hop": 1.5,
    "max_seq_len": 128
  }
}
```

### 3. `both`
Извлекает и агрегированные фичи, и per-segment последовательности.

**Результат:**
- `manifest.json` - агрегированные фичи
- `features.npy`, `mask.npy`, `meta.json` - per-segment данные

**Использование:**
```json
{
  "processing_mode": "both",
  "segment_config": {
    "segment_len": 3.0,
    "max_seq_len": 128,
    "pca_dims": {"clap": 128, "wav2vec": 64}
  }
}
```

## ⚙️ Конфигурация сегментов

### Основные параметры
```json
{
  "segment_len": 3.0,        // Длина сегмента в секундах
  "hop": 1.5,                // Hop между сегментами в секундах
  "max_seq_len": 128,        // Максимум сегментов на видео
  "k_start": 16,             // Сегментов в начале (сохраняются всегда)
  "k_end": 16                // Сегментов в конце (сохраняются всегда)
}
```

### PCA сжатие
```json
{
  "pca_dims": {
    "clap": 128,             // CLAP: 512 → 128
    "wav2vec": 64,           // wav2vec: 768 → 64
    "yamnet": 128            // YAMNet: 1024 → 128
  }
}
```

### Стратегии выбора сегментов
```json
{
  "importance_weights": {
    "rms": 0.6,              // Вес RMS для важности
    "voiced_fraction": 0.4   // Вес voiced_fraction для важности
  }
}
```

## 📊 Доступные Extractors

### Основные extractors
- `clap_extractor` - CLAP семантические эмбеддинги
- `loudness_extractor` - RMS энергия и громкость
- `vad_extractor` - Voice Activity Detection и pitch
- `spectral_extractor` - Спектральные характеристики
- `tempo_extractor` - Темпо и onset детекция
- `onset_extractor` - Onset детекция
- `quality_extractor` - Качество аудио
- `emotion_recognition_extractor` - Распознавание эмоций

### Продвинутые extractors
- `advanced_embeddings` - wav2vec, YAMNet, VGGish
- `asr_extractor` - Автоматическое распознавание речи
- `source_separation_extractor` - Разделение источников
- `speaker_diarization_extractor` - Диаризация спикеров

## 🐍 Python клиент

### Базовое использование
```python
import requests
import time

def process_video_unified(video_id: str, audio_uri: str, processing_mode: str = "both"):
    # Отправить запрос
    response = requests.post("http://localhost:8000/unified/process", json={
        "video_id": video_id,
        "audio_uri": audio_uri,
        "processing_mode": processing_mode,
        "segment_config": {
            "segment_len": 3.0,
            "max_seq_len": 128
        }
    })
    
    if response.status_code == 200:
        task_id = response.json()["celery_task_id"]
        
        # Ждать завершения
        while True:
            status_response = requests.get(f"http://localhost:8000/unified/task/{task_id}")
            status = status_response.json()
            
            if status["status"] == "completed":
                return status["result"]
            elif status["status"] == "failed":
                raise Exception(f"Task failed: {status.get('error')}")
            
            time.sleep(5)

# Использование
result = process_video_unified("my_video", "s3://bucket/audio.wav", "both")
print(f"Создано сегментов: {result['num_selected_segments']}")
print(f"Форма фичей: {result['feature_shape']}")
```

### Batch обработка
```python
def process_batch_unified(videos: list, processing_mode: str = "both"):
    response = requests.post("http://localhost:8000/unified/batch", json={
        "videos": videos,
        "processing_mode": processing_mode,
        "segment_config": {
            "segment_len": 3.0,
            "max_seq_len": 64
        }
    })
    
    if response.status_code == 200:
        task_id = response.json()["celery_task_id"]
        
        # Ждать завершения batch
        while True:
            status_response = requests.get(f"http://localhost:8000/unified/task/{task_id}")
            status = status_response.json()
            
            if status["status"] == "completed":
                return status["result"]
            
            time.sleep(10)

# Использование
videos = [
    {"video_id": "video_001", "audio_uri": "s3://bucket/audio1.wav"},
    {"video_id": "video_002", "audio_uri": "s3://bucket/audio2.wav"}
]

batch_result = process_batch_unified(videos, "both")
print(f"Обработано: {batch_result['successful']}/{batch_result['total_videos']}")
```

## 📁 Структура результатов

### Aggregates only
```
output_dir/
└── video_123_manifest.json    # Результаты extractors
```

### Segments only
```
output_dir/
├── video_123_features.npy     # (max_seq_len, feature_dim)
├── video_123_mask.npy         # (max_seq_len,) - attention mask
└── video_123_meta.json        # Метаданные сегментов
```

### Both
```
output_dir/
├── video_123_manifest.json    # Агрегированные фичи
├── video_123_features.npy     # Per-segment фичи
├── video_123_mask.npy         # Attention mask
└── video_123_meta.json        # Метаданные сегментов
```

## 🔍 Загрузка результатов

### Загрузка per-segment данных
```python
import numpy as np
import json

# Загрузить фичи
features = np.load("video_123_features.npy")  # (max_seq_len, feature_dim)
mask = np.load("video_123_mask.npy")          # (max_seq_len,)

# Загрузить метаданные
with open("video_123_meta.json", "r") as f:
    meta = json.load(f)

print(f"Фичи: {features.shape}")
print(f"Валидных сегментов: {mask.sum()}")
print(f"Сегментов создано: {meta['num_segments']}")
```

### Использование в PyTorch
```python
import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, video_ids: list, output_dir: str):
        self.video_ids = video_ids
        self.output_dir = output_dir
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Загрузить данные
        features = np.load(f"{self.output_dir}/{video_id}_features.npy")
        mask = np.load(f"{self.output_dir}/{video_id}_mask.npy")
        
        return {
            "features": torch.FloatTensor(features),
            "attention_mask": torch.LongTensor(mask),
            "video_id": video_id
        }

# Создать dataset
dataset = SegmentDataset(["video_001", "video_002"], "output_dir")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Использовать в обучении
for batch in dataloader:
    features = batch["features"]      # (batch_size, max_seq_len, feature_dim)
    masks = batch["attention_mask"]   # (batch_size, max_seq_len)
    # ... обучение модели
```

## 🚀 Демонстрация

Запустить демонстрационный скрипт:
```bash
python demo_unified_api.py
```

Скрипт покажет:
- Получение конфигурации API
- Примеры запросов
- Демонстрацию всех режимов обработки

## 🔧 Troubleshooting

### Частые проблемы

1. **API недоступен**
   ```
   ❌ API недоступен
   ```
   Решение: Убедитесь, что сервер запущен на http://localhost:8000

2. **Файл не найден**
   ```
   ❌ Task failed: File not found
   ```
   Решение: Проверьте пути к аудио/видео файлам

3. **Недостаточно памяти**
   ```
   ❌ Task failed: MemoryError
   ```
   Решение: Уменьшите max_seq_len или batch размер

4. **Extractor недоступен**
   ```
   ❌ Task failed: Extractor not found
   ```
   Решение: Проверьте список доступных extractors через `/unified/config`

### Отладка
```bash
# Проверить статус API
curl http://localhost:8000/health

# Получить конфигурацию
curl http://localhost:8000/unified/config

# Получить примеры
curl http://localhost:8000/unified/examples

# Проверить статус задачи
curl http://localhost:8000/unified/task/{task_id}
```

## 📈 Производительность

### Рекомендации
1. **Batch обработка**: Используйте batch endpoint для множества видео
2. **Выбор extractors**: Используйте только нужные extractors
3. **PCA сжатие**: Включайте PCA для больших эмбеддингов
4. **max_seq_len**: Оптимизируйте под вашу модель

### Мониторинг
```python
# Проверить время обработки
result = process_video_unified("video", "audio.wav")
print(f"Время обработки: {result['processing_time']:.2f}s")

# Проверить использование ресурсов
import psutil
print(f"Память: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
```

## 🎯 Интеграция с AudioTransformer

После обработки через unified API, данные готовы для AudioTransformer:

```python
# Загрузить обработанные данные
features = np.load("video_123_features.npy")
mask = np.load("video_123_mask.npy")

# Подготовить для Transformer
transformer_input = {
    "input_ids": features,           # или другой формат входа
    "attention_mask": mask,
    "video_id": "video_123"
}

# Использовать в модели
output = transformer_model(**transformer_input)
```

## 🔄 Миграция с старого API

### Старый способ (2 запроса)
```python
# Шаг 1: AudioProcessor
response1 = requests.post("http://localhost:8000/process", json={
    "video_id": "video_123",
    "audio_uri": "s3://bucket/audio.wav"
})

# Шаг 2: Segment Pipeline
response2 = requests.post("http://localhost:8000/segment/process", json={
    "manifest_path": "manifest.json"
})
```

### Новый способ (1 запрос)
```python
# Unified API
response = requests.post("http://localhost:8000/unified/process", json={
    "video_id": "video_123",
    "audio_uri": "s3://bucket/audio.wav",
    "processing_mode": "both"
})
```

## 🎉 Заключение

Unified API объединяет лучшее из двух миров:
- **Гибкость** традиционного AudioProcessor
- **Эффективность** per-segment pipeline
- **Простота** единого API endpoint

Теперь вы можете получить как агрегированные фичи, так и готовые для Transformer последовательности в одном запросе! 🚀
