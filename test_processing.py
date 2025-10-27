#!/usr/bin/env python3
"""
Тест обработки видеофайлов с 22 экстракторами
"""

# Подавляем предупреждения
import warnings

# Подавляем предупреждения PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="torch.meshgrid.*indexing.*")

# Подавляем предупреждения TensorFlow/Keras
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")

# Подавляем предупреждения transformers
warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

import asyncio
import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from unified_processor import AsyncUnifiedAudioProcessor
from extractors import discover_extractors

async def test_processing():
    """Тестируем обработку видеофайлов"""
    
    # Пути к тестовым файлам
    test_files = [
        "src/tests/test_videos/-69HDT6DZEM.mp4",
        "src/tests/test_videos/-JuF2ivdnAg.mp4", 
        "src/tests/test_videos/-niwQ0xGEGk.mp4"
    ]
    
    # Проверяем, что файлы существуют
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"❌ Файл не найден: {file_path}")
            return
        else:
            print(f"✅ Файл найден: {file_path}")
    
    # Получаем все экстракторы
    extractors = discover_extractors()
    print(f"\n🔍 Найдено экстракторов: {len(extractors)}")
    
    # Выводим список экстракторов
    for i, extractor in enumerate(extractors, 1):
        print(f"  {i:2d}. {extractor.__class__.__name__}")
    
    # Создаем процессор
    processor = AsyncUnifiedAudioProcessor()
    
    print(f"\n🚀 Начинаем обработку {len(test_files)} файлов...")
    
    # Обрабатываем каждый файл
    for i, file_path in enumerate(test_files, 1):
        print(f"\n📁 Обработка файла {i}/{len(test_files)}: {os.path.basename(file_path)}")
        
        try:
            # Обрабатываем файл
            result = await processor.process_audio_async(
                input_uri=file_path,
                video_id=f"test_{i}",
                aggregates_only=True
            )
            
            if result.get('success', False):
                print(f"✅ Файл {os.path.basename(file_path)} обработан успешно")
                print(f"   📊 Время обработки: {result.get('processing_time', 0):.2f}с")
                print(f"   📁 Манифест: {result.get('manifest_path', 'N/A')}")
                
                if result.get('aggregates_extracted', False):
                    print(f"   ✅ Агрегированные признаки извлечены")
                if result.get('segments_extracted', False):
                    print(f"   ✅ Сегментные признаки извлечены")
            else:
                print(f"❌ Ошибка обработки файла {os.path.basename(file_path)}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Ошибка при обработке {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(test_processing())
