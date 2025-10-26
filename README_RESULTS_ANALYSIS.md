# 📊 Анализ результатов аудио-обработки

Этот документ описывает доступные инструменты и команды для анализа результатов обработки аудио/видео файлов.

## 🔧 Доступные инструменты

### 1. `analyze_manifest.py` - Анализ manifest файла

Основной инструмент для анализа файла `manifest_test_video_local.json`.

#### Команды:

```bash
# Показать общую сводку по всем экстракторам
python analyze_manifest.py summary

# Показать список всех экстракторов
python analyze_manifest.py list

# Показать детальную информацию по конкретному экстрактору
python analyze_manifest.py show <extractor_name>

# Показать все экстракторы с детальной информацией
python analyze_manifest.py all
```

#### Примеры использования:

```bash
# Общая сводка
python analyze_manifest.py summary

# Анализ конкретного экстрактора
python analyze_manifest.py show pitch
python analyze_manifest.py show mfcc_extractor
python analyze_manifest.py show vad_extractor

# Полный анализ всех экстракторов
python analyze_manifest.py all
```

### 2. `view_results.py` - Анализ результатов тестирования

Инструмент для анализа файлов результатов тестирования (формат `full_extraction_results_*.json`).

#### Команды:

```bash
# Показать общую сводку
python view_results.py summary

# Показать список всех экстракторов
python view_results.py list

# Показать детальную информацию по конкретному экстрактору
python view_results.py show <extractor_name>

# Показать все экстракторы
python view_results.py all
```

## 📊 Доступные экстракторы

В системе доступны следующие экстракторы:

| Экстрактор | Описание | Количество признаков |
|------------|----------|---------------------|
| `mfcc_extractor` | MFCC коэффициенты | 56 |
| `mel_extractor` | Mel-спектрограммы | 263 |
| `chroma_extractor` | Хроматические признаки | 59 |
| `loudness_extractor` | Громкость и RMS | 36 |
| `vad_extractor` | Voice Activity Detection | 23 |
| `clap_extractor` | CLAP эмбеддинги | 520 |
| `asr` | Автоматическое распознавание речи | 15 |
| `pitch` | Анализ высоты тона | 40 |
| `spectral` | Спектральные характеристики | 41 |
| `tempo` | Темп и ритм | 26 |
| `quality` | Качество аудио | 38 |
| `onset` | Детекция начала звуков | 39 |
| `speaker_diarization` | Диаризация спикеров | 8 |
| `voice_quality` | Качество голоса | 27 |
| `emotion_recognition` | Распознавание эмоций | 7 |
| `phoneme_analysis` | Анализ фонем | 14 |
| `advanced_spectral` | Расширенные спектральные признаки | 75 |
| `music_analysis` | Анализ музыки | 47 |
| `source_separation` | Разделение источников | 16 |
| `sound_event_detection` | Детекция звуковых событий | 27 |
| `rhythmic_analysis` | Ритмический анализ | 27 |
| `advanced_embeddings` | Продвинутые эмбеддинги | 24 |

## 🔍 Типы данных

### Скалярные признаки
- Числовые значения (float, int)
- Строки (str)
- Булевы значения (bool)
- Null значения

### Массивные признаки
- Временные ряды (например, `f0_array`, `rms_array`)
- Эмбеддинги (например, `clap_embedding`, `yamnet_embeddings`)
- Временные метки (например, `beat_times`, `onset_times`)
- Классификационные результаты

## 📈 Самые большие массивы данных

1. **`loudness_extractor.rms_array`** - 2,469 элементов
2. **`vad_extractor.f0_array`** - 1,235 элементов
3. **`vad_extractor.voiced_flag_array`** - 1,235 элементов
4. **`music_analysis.chord_sequence`** - 1,235 элементов
5. **`advanced_embeddings.yamnet_embeddings`** - 1,024 элементов

## 🛠️ Полезные команды для анализа

### Быстрая проверка JSON
```bash
# Проверить валидность JSON
python -m json.tool manifest_test_video_local.json > /dev/null && echo "JSON is valid"

# Подсчитать строки в файле
wc -l manifest_test_video_local.json

# Найти конкретные значения
grep -c "null" manifest_test_video_local.json
```

### Анализ структуры данных
```bash
# Показать все ключи в JSON
python -c "import json; data=json.load(open('manifest_test_video_local.json')); print(list(data.keys()))"

# Показать все экстракторы
python -c "import json; data=json.load(open('manifest_test_video_local.json')); [print(ext['name']) for ext in data['extractors']]"

# Подсчитать общее количество признаков
python -c "import json; data=json.load(open('manifest_test_video_local.json')); total=sum(len(ext['payload']) for ext in data['extractors']); print(f'Total features: {total}')"
```

### Анализ массивов
```bash
# Найти все массивы
python -c "
import json
with open('manifest_test_video_local.json', 'r') as f:
    data = json.load(f)
for ext in data['extractors']:
    for key, value in ext['payload'].items():
        if isinstance(value, list) and len(value) > 0:
            print(f'{ext[\"name\"]}.{key}: {len(value)} items')
"

# Анализ конкретного массива
python -c "
import json
with open('manifest_test_video_local.json', 'r') as f:
    data = json.load(f)
# Замените 'extractor_name' и 'array_key' на нужные значения
for ext in data['extractors']:
    if ext['name'] == 'vad_extractor':
        array = ext['payload'].get('f0_array', [])
        non_null = sum(1 for x in array if x is not None)
        print(f'f0_array: {len(array)} total, {non_null} non-null ({non_null/len(array)*100:.1f}%)')
        break
"
```

## 🚨 Решение проблем

### Проблема с NaN значениями
Если в JSON файле встречаются значения `NaN`, они делают файл невалидным. Для исправления:

```bash
# Заменить все NaN на null
sed -i 's/NaN/null/g' manifest_test_video_local.json

# Или использовать Python
python -c "
import json
with open('manifest_test_video_local.json', 'r') as f:
    content = f.read().replace('NaN', 'null')
with open('manifest_test_video_local.json', 'w') as f:
    f.write(content)
"
```

### Проблема с большими файлами
Для файлов больше 20,000 строк:

```bash
# Показать только первые N строк
head -n 100 manifest_test_video_local.json

# Показать только последние N строк
tail -n 100 manifest_test_video_local.json

# Поиск конкретного экстрактора
grep -A 10 -B 2 '"name": "pitch"' manifest_test_video_local.json
```

## 📝 Примеры вывода

### Общая сводка
```
📊 MANIFEST SUMMARY
============================================================
🎬 Video ID: test_video_local
📅 Timestamp: 2025-10-26T02:46:50.295180Z
📊 Dataset: default
🆔 Task ID: None
🔢 Total extractors: 22
✅ Successful: 22
❌ Failed: 0
📈 Success rate: 100.0%
```

### Детали экстрактора
```
🔍 PITCH
============================================================
Status: ✅ SUCCESS
Version: 1.0.0

📊 Scalar Features (40):
  • f0_mean_pyin: 80.879070
  • f0_std_pyin: 36.688432
  • f0_min_pyin: 50.000000
  • f0_max_pyin: 247.655272
  ...
```

## 🔗 Связанные файлы

- `manifest_test_video_local.json` - Основной файл с результатами
- `analyze_manifest.py` - Скрипт для анализа manifest файла
- `view_results.py` - Скрипт для анализа результатов тестирования
- `test_with_full_results.py` - Скрипт для генерации полных результатов
