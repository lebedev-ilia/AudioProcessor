"""
Универсальные утилиты для загрузки и обработки аудио.
Исправляет проблемы с многоканальным аудио и обеспечивает правильную форму для GPU-оптимизированных экстракторов.
"""

import soundfile as sf
import numpy as np
import torch
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)

def load_audio_mono(path: str, target_sr: int = None) -> Tuple[torch.Tensor, int]:
    """
    Загружает аудио файл и конвертирует в моно-волну с правильной формой для GPU экстракторов.
    
    Args:
        path: Путь к аудио файлу
        target_sr: Целевая частота дискретизации (если None, оставляет исходную)
        
    Returns:
        Tuple[torch.Tensor, int]: (audio_tensor, sample_rate)
        audio_tensor имеет форму [1, 1, n_samples] для совместимости с GPU MFCC
    """
    try:
        # Загружаем аудио с сохранением исходной частоты
        y, sr = sf.read(path, always_2d=True)  # (n_samples, n_channels)
        logger.debug(f"Загружено аудио: {y.shape}, sr={sr}")
        
        # Конвертируем в моно если нужно
        if y.ndim == 2:
            if y.shape[1] > 1:
                # Стерео -> моно через усреднение
                y = np.mean(y, axis=1)
                logger.debug("Конвертировано из стерео в моно")
            else:
                y = y[:, 0]
        
        # Ресемплинг если нужно
        if target_sr is not None and sr != target_sr:
            import librosa
            y = librosa.resample(
                y, 
                orig_sr=sr, 
                target_sr=target_sr,
                res_type='kaiser_fast'
            )
            sr = target_sr
            logger.debug(f"Ресемплинг: {sr} -> {target_sr}")
        
        # Нормализация и приведение к float32
        y = y.astype('float32')
        max_val = max(1e-8, np.max(np.abs(y)))
        y = y / max_val
        
        # Преобразование в torch tensor с правильной формой для GPU MFCC: [1, 1, N]
        audio_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
        
        logger.debug(f"Финальная форма аудио: {audio_tensor.shape}")
        return audio_tensor, sr
        
    except Exception as e:
        logger.error(f"Ошибка загрузки аудио {path}: {str(e)}")
        raise

def ensure_mono_tensor(audio: Union[np.ndarray, torch.Tensor], 
                      sample_rate: int = None) -> Tuple[torch.Tensor, int]:
    """
    Обеспечивает, что аудио имеет правильную форму для GPU экстракторов.
    
    Args:
        audio: Аудио данные (numpy array или torch tensor)
        sample_rate: Частота дискретизации
        
    Returns:
        Tuple[torch.Tensor, int]: (audio_tensor, sample_rate)
    """
    try:
        # Конвертируем в numpy если нужно
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        
        # Обрабатываем многомерные массивы
        if audio_np.ndim > 1:
            if audio_np.shape[0] == 1 and audio_np.shape[1] > 1:
                # Форма [1, N] -> [N]
                audio_np = audio_np[0]
            elif audio_np.shape[1] == 1 and audio_np.shape[0] > 1:
                # Форма [N, 1] -> [N]
                audio_np = audio_np[:, 0]
            elif audio_np.shape[0] > 1 and audio_np.shape[1] > 1:
                # Стерео [N, 2] -> моно [N]
                audio_np = np.mean(audio_np, axis=1)
                logger.warning("Обнаружено стерео аудио, конвертировано в моно")
        
        # Нормализация
        audio_np = audio_np.astype('float32')
        max_val = max(1e-8, np.max(np.abs(audio_np)))
        audio_np = audio_np / max_val
        
        # Преобразование в torch tensor с формой [1, 1, N]
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        
        return audio_tensor, sample_rate or 22050
        
    except Exception as e:
        logger.error(f"Ошибка обработки аудио: {str(e)}")
        raise

def validate_audio_shape(audio: torch.Tensor, expected_channels: int = 1) -> bool:
    """
    Проверяет, что аудио имеет правильную форму для экстракторов.
    
    Args:
        audio: Аудио тензор
        expected_channels: Ожидаемое количество каналов
        
    Returns:
        bool: True если форма корректна
    """
    if audio.dim() != 3:
        logger.error(f"Ожидается 3D тензор, получен {audio.dim()}D: {audio.shape}")
        return False
    
    if audio.shape[1] != expected_channels:
        logger.error(f"Ожидается {expected_channels} канал(ов), получено {audio.shape[1]}: {audio.shape}")
        return False
    
    return True
