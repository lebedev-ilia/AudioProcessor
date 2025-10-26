#!/usr/bin/env python3
"""
Полное тестирование UnifiedAudioProcessor без Celery.

Этот скрипт тестирует все возможности unified processor:
1. Только агрегированные фичи
2. Только per-segment последовательности  
3. И то, и другое
4. Batch обработка
"""

import os
import sys
import json
import numpy as np
import tempfile
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.unified_processor import UnifiedAudioProcessor
from src.segment_config import create_config, get_default_config
from src.schemas.unified_models import ProcessingMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_audio_file(duration: float = 10.0, sample_rate: int = 22050) -> str:
    """Создать тестовый аудио файл."""
    import librosa
    import soundfile as sf
    
    # Создать тестовый аудио сигнал
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Создать сложный сигнал с разными частотами
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.1 * np.sin(2 * np.pi * 1320 * t) + # E6
        0.1 * np.random.randn(len(t))         # Шум
    )
    
    # Нормализовать
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Сохранить во временный файл
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, signal, sample_rate)
    temp_file.close()
    
    logger.info(f"Создан тестовый аудио файл: {temp_file.name} ({duration}s)")
    return temp_file.name


def test_aggregates_only():
    """Тест 1: Только агрегированные фичи."""
    logger.info("=" * 60)
    logger.info("ТЕСТ 1: Только агрегированные фичи")
    logger.info("=" * 60)
    
    # Создать тестовый аудио файл
    audio_file = create_test_audio_file(duration=15.0)
    
    try:
        # Создать processor
        processor = UnifiedAudioProcessor()
        
        # Обработать только агрегированные фичи (все extractors)
        result = processor.process_audio(
            input_uri=audio_file,
            video_id="test_aggregates_001",
            aggregates_only=True,
            extractor_names=None,  # Использовать все доступные extractors
            output_dir="test_output/aggregates_only"
        )
        
        # Проверить результат
        if result["success"]:
            logger.info("✅ Тест 1 ПРОЙДЕН")
            logger.info(f"   Агрегированные фичи извлечены: {result['aggregates_extracted']}")
            logger.info(f"   Per-segment фичи: {result['segments_extracted']}")
            logger.info(f"   Manifest: {result['manifest_path']}")
            logger.info(f"   Extractors: {len(result['extractor_results'])}")
            
            # Проверить manifest файл
            if os.path.exists(result['manifest_path']):
                with open(result['manifest_path'], 'r') as f:
                    manifest = json.load(f)
                logger.info(f"   Manifest содержит {len(manifest['extractors'])} extractors")
            
            return True
        else:
            logger.error(f"❌ Тест 1 ПРОВАЛЕН: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Тест 1 ПРОВАЛЕН с исключением: {e}")
        return False
    finally:
        # Очистить временный файл
        if os.path.exists(audio_file):
            os.unlink(audio_file)


def test_segments_only():
    """Тест 2: Только per-segment последовательности."""
    logger.info("=" * 60)
    logger.info("ТЕСТ 2: Только per-segment последовательности")
    logger.info("=" * 60)
    
    # Создать тестовый аудио файл
    audio_file = create_test_audio_file(duration=20.0)
    
    try:
        # Создать конфигурацию для сегментов
        config = create_config(
            segment_len=3.0,
            hop=1.5,
            max_seq_len=16,  # Меньше для теста
            k_start=4,
            k_end=4,
            pca_dims={"clap": 64, "wav2vec": 32}
        )
        
        # Создать processor
        processor = UnifiedAudioProcessor(config)
        
        # Обработать только per-segment фичи (ограниченный набор extractors автоматически)
        result = processor.process_audio(
            input_uri=audio_file,
            video_id="test_segments_001",
            aggregates_only=False,
            segment_config={
                "segment_len": 3.0,
                "hop": 1.5,
                "max_seq_len": 16
            },
            extractor_names=None,  # UnifiedAudioProcessor автоматически выберет правильные extractors
            output_dir="test_output/segments_only"
        )
        
        # Проверить результат
        if result["success"]:
            logger.info("✅ Тест 2 ПРОЙДЕН")
            logger.info(f"   Агрегированные фичи: {result['aggregates_extracted']}")
            logger.info(f"   Per-segment фичи извлечены: {result['segments_extracted']}")
            logger.info(f"   Сегментов создано: {result['num_segments']}")
            logger.info(f"   Сегментов выбрано: {result['num_selected_segments']}")
            logger.info(f"   Форма фичей: {result['feature_shape']}")
            logger.info(f"   Файлы: {list(result['segment_files'].keys())}")
            
            # Проверить файлы сегментов
            for file_type, file_path in result['segment_files'].items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    logger.info(f"   {file_type}: {file_size:.1f} KB")
                    
                    # Проверить содержимое features файла
                    if file_type == "features_file":
                        features = np.load(file_path)
                        logger.info(f"   Features shape: {features.shape}")
                        logger.info(f"   Features dtype: {features.dtype}")
                        logger.info(f"   Features range: [{features.min():.3f}, {features.max():.3f}]")
            
            return True
        else:
            logger.error(f"❌ Тест 2 ПРОВАЛЕН: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Тест 2 ПРОВАЛЕН с исключением: {e}")
        return False
    finally:
        # Очистить временный файл
        if os.path.exists(audio_file):
            os.unlink(audio_file)


def test_both():
    """Тест 3: И агрегированные фичи, и per-segment последовательности."""
    logger.info("=" * 60)
    logger.info("ТЕСТ 3: И агрегированные фичи, и per-segment последовательности")
    logger.info("=" * 60)
    
    # Создать тестовый аудио файл
    audio_file = create_test_audio_file(duration=25.0)
    
    try:
        # Создать конфигурацию
        config = create_config(
            segment_len=3.0,
            hop=1.5,
            max_seq_len=32,
            k_start=8,
            k_end=8,
            pca_dims={"clap": 128, "wav2vec": 64, "yamnet": 128}
        )
        
        # Создать processor
        processor = UnifiedAudioProcessor(config)
        
        # Обработать и то, и другое (все extractors для агрегатов + ограниченные для сегментов автоматически)
        result = processor.process_audio(
            input_uri=audio_file,
            video_id="test_both_001",
            aggregates_only=False,
            segment_config={
                "segment_len": 3.0,
                "hop": 1.5,
                "max_seq_len": 32,
                "k_start": 8,
                "k_end": 8,
                "importance_weights": {
                    "rms": 0.7,
                    "voiced_fraction": 0.3
                }
            },
            extractor_names=None,  # UnifiedAudioProcessor автоматически выберет правильные extractors
            output_dir="test_output/both"
        )
        
        # Проверить результат
        if result["success"]:
            logger.info("✅ Тест 3 ПРОЙДЕН")
            logger.info(f"   Агрегированные фичи: {result['aggregates_extracted']}")
            logger.info(f"   Per-segment фичи: {result['segments_extracted']}")
            logger.info(f"   Manifest: {result['manifest_path']}")
            logger.info(f"   Сегментов: {result['num_selected_segments']}")
            logger.info(f"   Форма фичей: {result['feature_shape']}")
            
            # Проверить оба типа файлов
            if result['manifest_path'] and os.path.exists(result['manifest_path']):
                with open(result['manifest_path'], 'r') as f:
                    manifest = json.load(f)
                logger.info(f"   Manifest extractors: {len(manifest['extractors'])}")
            
            segment_files = result.get('segment_files', {})
            for file_type, file_path in segment_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024
                    logger.info(f"   {file_type}: {file_size:.1f} KB")
            
            return True
        else:
            logger.error(f"❌ Тест 3 ПРОВАЛЕН: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Тест 3 ПРОВАЛЕН с исключением: {e}")
        return False
    finally:
        # Очистить временный файл
        if os.path.exists(audio_file):
            os.unlink(audio_file)


def test_batch_processing():
    """Тест 4: Batch обработка."""
    logger.info("=" * 60)
    logger.info("ТЕСТ 4: Batch обработка")
    logger.info("=" * 60)
    
    # Создать несколько тестовых аудио файлов
    audio_files = []
    try:
        for i in range(3):
            audio_file = create_test_audio_file(duration=10.0 + i * 5)  # Разные длительности
            audio_files.append(audio_file)
        
        # Подготовить данные для batch
        video_data = []
        for i, audio_file in enumerate(audio_files):
            video_data.append({
                "video_id": f"batch_test_{i:03d}",
                "input_uri": audio_file
            })
        
        # Создать processor
        processor = UnifiedAudioProcessor()
        
        # Обработать batch (ограниченные extractors для сегментов автоматически)
        result = processor.process_batch(
            video_data=video_data,
            aggregates_only=False,
            segment_config={
                "segment_len": 3.0,
                "hop": 1.5,
                "max_seq_len": 16,
                "k_start": 4,
                "k_end": 4
            },
            extractor_names=None,  # UnifiedAudioProcessor автоматически выберет правильные extractors
            output_dir="test_output/batch"
        )
        
        # Проверить результат
        logger.info(f"📊 Batch обработка завершена:")
        logger.info(f"   Всего видео: {result['total_videos']}")
        logger.info(f"   Успешно: {result['successful']}")
        logger.info(f"   Ошибок: {result['failed']}")
        
        if result['successful'] > 0:
            logger.info("✅ Тест 4 ПРОЙДЕН")
            
            # Показать детали по каждому видео
            for video_result in result['results']:
                if video_result['success']:
                    logger.info(f"   ✅ {video_result['video_id']}: {video_result.get('num_selected_segments', 0)} сегментов")
                else:
                    logger.info(f"   ❌ {video_result['video_id']}: {video_result.get('error')}")
            
            return True
        else:
            logger.error("❌ Тест 4 ПРОВАЛЕН: Нет успешно обработанных видео")
            return False
            
    except Exception as e:
        logger.error(f"❌ Тест 4 ПРОВАЛЕН с исключением: {e}")
        return False
    finally:
        # Очистить временные файлы
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                os.unlink(audio_file)


def test_data_loading():
    """Тест 5: Загрузка и валидация данных."""
    logger.info("=" * 60)
    logger.info("ТЕСТ 5: Загрузка и валидация данных")
    logger.info("=" * 60)
    
    try:
        # Проверить, есть ли созданные файлы
        test_dirs = ["test_output/segments_only", "test_output/both"]
        found_files = False
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.endswith('_features.npy'):
                        found_files = True
                        
                        # Загрузить данные
                        features_path = os.path.join(test_dir, file)
                        mask_path = features_path.replace('_features.npy', '_mask.npy')
                        meta_path = features_path.replace('_features.npy', '_meta.json')
                        
                        if os.path.exists(mask_path) and os.path.exists(meta_path):
                            # Загрузить файлы
                            features = np.load(features_path)
                            mask = np.load(mask_path)
                            
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)
                            
                            logger.info(f"📁 Загружены данные из {file}:")
                            logger.info(f"   Features: {features.shape}, dtype: {features.dtype}")
                            logger.info(f"   Mask: {mask.shape}, dtype: {mask.dtype}")
                            logger.info(f"   Валидных сегментов: {mask.sum()}")
                            logger.info(f"   Метаданные: {meta.get('num_segments')} сегментов")
                            
                            # Валидация
                            assert features.shape[0] == mask.shape[0], "Features и mask должны иметь одинаковую длину"
                            assert features.shape[0] == meta.get('num_selected_segments'), "Features и meta должны совпадать"
                            assert mask.dtype == np.uint8, "Mask должен быть uint8"
                            assert features.dtype == np.float32, "Features должен быть float32"
                            
                            # Проверить статистики
                            valid_features = features[mask == 1]
                            if len(valid_features) > 0:
                                logger.info(f"   Статистики фичей:")
                                logger.info(f"     Среднее: {valid_features.mean():.4f}")
                                logger.info(f"     Стд: {valid_features.std():.4f}")
                                logger.info(f"     Мин: {valid_features.min():.4f}")
                                logger.info(f"     Макс: {valid_features.max():.4f}")
                            
                            logger.info("✅ Валидация данных прошла успешно")
                            return True
        
        if not found_files:
            logger.warning("⚠️ Не найдены файлы для тестирования загрузки")
            return False
            
    except Exception as e:
        logger.error(f"❌ Тест 5 ПРОВАЛЕН с исключением: {e}")
        return False


def test_configuration():
    """Тест 6: Конфигурация и параметры."""
    logger.info("=" * 60)
    logger.info("ТЕСТ 6: Конфигурация и параметры")
    logger.info("=" * 60)
    
    try:
        # Тест дефолтной конфигурации
        default_config = get_default_config()
        logger.info(f"📋 Дефолтная конфигурация:")
        logger.info(f"   segment_len: {default_config.segment_len}")
        logger.info(f"   max_seq_len: {default_config.max_seq_len}")
        logger.info(f"   PCA dims: {default_config.pca_dims}")
        
        # Тест кастомной конфигурации
        custom_config = create_config(
            segment_len=5.0,
            hop=2.0,
            max_seq_len=64,
            k_start=8,
            k_end=8,
            pca_dims={"clap": 64, "wav2vec": 32}
        )
        
        logger.info(f"📋 Кастомная конфигурация:")
        logger.info(f"   segment_len: {custom_config.segment_len}")
        logger.info(f"   hop: {custom_config.hop}")
        logger.info(f"   max_seq_len: {custom_config.max_seq_len}")
        logger.info(f"   k_start: {custom_config.k_start}")
        logger.info(f"   k_end: {custom_config.k_end}")
        logger.info(f"   PCA dims: {custom_config.pca_dims}")
        
        # Тест feature mapping
        feature_mapping = custom_config.get_feature_mapping()
        logger.info(f"📋 Feature mapping: {len(feature_mapping)} mappings")
        
        # Тест array fields
        array_fields = custom_config.get_array_fields()
        logger.info(f"📋 Array fields: {len(array_fields)} fields")
        
        # Тест scalar fields
        scalar_fields = custom_config.get_scalar_fields()
        logger.info(f"📋 Scalar fields: {len(scalar_fields)} fields")
        
        logger.info("✅ Тест 6 ПРОЙДЕН")
        return True
        
    except Exception as e:
        logger.error(f"❌ Тест 6 ПРОВАЛЕН с исключением: {e}")
        return False


def main():
    """Главная функция тестирования."""
    logger.info("🚀 Запуск полного тестирования UnifiedAudioProcessor")
    logger.info("=" * 60)
    
    # Создать директории для тестов
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_output/aggregates_only", exist_ok=True)
    os.makedirs("test_output/segments_only", exist_ok=True)
    os.makedirs("test_output/both", exist_ok=True)
    os.makedirs("test_output/batch", exist_ok=True)
    
    # Запустить тесты
    tests = [
        ("Конфигурация", test_configuration),
        ("Только агрегированные фичи", test_aggregates_only),
        ("Только per-segment последовательности", test_segments_only),
        ("И то, и другое", test_both),
        ("Batch обработка", test_batch_processing),
        ("Загрузка данных", test_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Запуск теста: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ Тест '{test_name}' упал с исключением: {e}")
            results.append((test_name, False))
    
    # Показать итоги
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        logger.info(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        logger.info(f"⚠️ {total - passed} тестов провалено")
    
    # Показать созданные файлы
    logger.info(f"\n📁 Созданные файлы:")
    for root, dirs, files in os.walk("test_output"):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            logger.info(f"   {file_path} ({file_size:.1f} KB)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
