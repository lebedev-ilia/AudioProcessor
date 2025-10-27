"""
Скрипт для подавления предупреждений PyTorch, Keras и других библиотек.
Импортируйте этот модуль в начале вашего основного скрипта.
"""

import warnings
import os

# Подавляем предупреждения PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="torch.meshgrid.*indexing.*")

# Подавляем предупреждения TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Убирает INFO сообщения TensorFlow
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")

# Подавляем предупреждения transformers
warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

# Подавляем предупреждения oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Предупреждения подавлены")
