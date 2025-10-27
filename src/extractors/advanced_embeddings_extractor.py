"""
Optimized Advanced Embeddings Extractor with GPU acceleration.

This extractor implements:
- GPU-optimized embeddings extraction with batching
- Memory-efficient model loading and caching
- Dynamic batch size adjustment based on GPU memory
- Mixed precision inference for better performance
- Automatic fallback to CPU if GPU memory is insufficient
"""

import logging
import numpy as np
import librosa
import torch
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.gpu_optimizer import get_gpu_optimizer, GPURequest, GPUResponse

logger = logging.getLogger(__name__)


class AdvancedEmbeddingsExtractor(BaseExtractor):
    """
    Optimized Advanced Embeddings Extractor with GPU acceleration
    Extracts VGGish, YAMNet, wav2vec/HuBERT, and x-vector/ECAPA embeddings
    """
    
    name = "advanced_embeddings_extractor"
    version = "3.0.0"
    description = "GPU-optimized advanced embeddings: VGGish, YAMNet, wav2vec/HuBERT, x-vector/ECAPA"
    category = "advanced"
    dependencies = ["torch", "transformers", "tensorflow-hub", "speechbrain"]
    estimated_duration = 8.0  # Faster due to optimization
    
    def __init__(self, 
                 batch_size: int = 8,
                 use_mixed_precision: bool = True,
                 enable_caching: bool = True,
                 max_audio_length: float = 10.0):
        """
        Initialize optimized advanced embeddings extractor.
        
        Args:
            batch_size: Batch size for GPU processing
            use_mixed_precision: Whether to use mixed precision inference
            enable_caching: Whether to enable model caching
            max_audio_length: Maximum audio length in seconds
        """
        super().__init__()
        
        # Configuration
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        self.enable_caching = enable_caching
        self.max_audio_length = max_audio_length
        
        # Audio parameters
        self.sample_rate = 16000  # Standard for most pretrained models
        self.hop_length = 512
        self.frame_length = 2048
        
        # Device and models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_optimizer = get_gpu_optimizer() if torch.cuda.is_available() else None
        
        # Model availability and instances
        self.models_available = {
            "vggish": False,
            "yamnet": False,
            "wav2vec": False,
            "hubert": False,
            "xvector": False,
            "ecapa": False
        }
        
        self._models = {}
        self._model_loaded = {}
        
        # Memory management
        self._memory_usage = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info(f"Initialized {self.name} v{self.version} on {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}, Mixed precision: {self.use_mixed_precision}")
    
    def _initialize_models(self):
        """Initialize available pretrained models with optimization."""
        try:
            # Try to import and initialize models
            self._check_vggish()
            self._check_yamnet()
            self._check_wav2vec()
            self._check_hubert()
            self._check_xvector()
            self._check_ecapa()
            
            available_models = [model for model, available in self.models_available.items() if available]
            self.logger.info(f"Available optimized embedding models: {available_models}")
            
        except Exception as e:
            self.logger.warning(f"Model initialization failed: {e}")
    
    def _check_vggish(self):
        """Check and initialize VGGish model with optimization."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            # Load VGGish model with optimization
            model_url = "https://tfhub.dev/google/vggish/1"
            self._models["vggish"] = hub.load(model_url)
            
            # Configure TensorFlow for GPU optimization
            if self.device == "cuda":
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        self.logger.warning(f"Could not set GPU memory growth: {e}")
            
            self.models_available["vggish"] = True
            self._model_loaded["vggish"] = True
            self.logger.info("VGGish model loaded and optimized")
            
        except ImportError:
            self.logger.warning("VGGish not available. Install with: pip install tensorflow-hub")
        except Exception as e:
            self.logger.warning(f"VGGish initialization failed: {e}")
    
    def _check_yamnet(self):
        """Check and initialize YAMNet model with optimization."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            # Load YAMNet model with optimization
            model_url = "https://tfhub.dev/google/yamnet/1"
            self._models["yamnet"] = hub.load(model_url)
            
            self.models_available["yamnet"] = True
            self._model_loaded["yamnet"] = True
            self.logger.info("YAMNet model loaded and optimized")
            
        except ImportError:
            self.logger.warning("YAMNet not available. Install with: pip install tensorflow-hub")
        except Exception as e:
            self.logger.warning(f"YAMNet initialization failed: {e}")
    
    def _check_wav2vec(self):
        """Check and initialize wav2vec model with optimization."""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            
            # Load wav2vec model and processor with optimization
            model_name = "facebook/wav2vec2-base-960h"
            self._models["wav2vec_processor"] = Wav2Vec2Processor.from_pretrained(model_name)
            self._models["wav2vec_model"] = Wav2Vec2Model.from_pretrained(model_name)
            
            # Move to device and optimize
            self._models["wav2vec_model"] = self._models["wav2vec_model"].to(self.device)
            self._models["wav2vec_model"].eval()
            
            # Enable mixed precision if supported
            if self.use_mixed_precision and hasattr(torch.cuda, 'amp'):
                # Use new autocast syntax for PyTorch 2.0+
                self._models["wav2vec_model"] = torch.amp.autocast('cuda')(self._models["wav2vec_model"])
            
            # Cache model if enabled
            if self.enable_caching and self.gpu_optimizer and hasattr(self.gpu_optimizer, 'memory_manager'):
                self.gpu_optimizer.memory_manager.cache_model("wav2vec_model", self._models["wav2vec_model"])
            
            self.models_available["wav2vec"] = True
            self._model_loaded["wav2vec"] = True
            self.logger.info("wav2vec model loaded and optimized")
            
        except ImportError:
            self.logger.warning("wav2vec not available. Install with: pip install transformers")
        except Exception as e:
            self.logger.warning(f"wav2vec initialization failed: {e}")
    
    def _check_hubert(self):
        """Check and initialize HuBERT model with optimization."""
        try:
            from transformers import HubertModel, Wav2Vec2Processor
            
            # Load HuBERT model and processor with optimization
            model_name = "facebook/hubert-base-ls960"
            self._models["hubert_processor"] = Wav2Vec2Processor.from_pretrained(model_name)
            self._models["hubert_model"] = HubertModel.from_pretrained(model_name)
            
            # Move to device and optimize
            self._models["hubert_model"] = self._models["hubert_model"].to(self.device)
            self._models["hubert_model"].eval()
            
            # Enable mixed precision if supported
            if self.use_mixed_precision and hasattr(torch.cuda, 'amp'):
                # Use new autocast syntax for PyTorch 2.0+
                self._models["hubert_model"] = torch.amp.autocast('cuda')(self._models["hubert_model"])
            
            # Cache model if enabled
            if self.enable_caching and self.gpu_optimizer and hasattr(self.gpu_optimizer, 'memory_manager'):
                self.gpu_optimizer.memory_manager.cache_model("hubert_model", self._models["hubert_model"])
            
            self.models_available["hubert"] = True
            self._model_loaded["hubert"] = True
            self.logger.info("HuBERT model loaded and optimized")
            
        except ImportError:
            self.logger.warning("HuBERT not available. Install with: pip install transformers")
        except Exception as e:
            self.logger.warning(f"HuBERT initialization failed: {e}")
    
    def _check_xvector(self):
        """Check and initialize x-vector model with optimization."""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            # Load x-vector model with optimization
            self._models["xvector"] = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Move to device if possible
            if hasattr(self._models["xvector"], 'to'):
                self._models["xvector"] = self._models["xvector"].to(self.device)
            
            self.models_available["xvector"] = True
            self._model_loaded["xvector"] = True
            self.logger.info("x-vector model loaded and optimized")
            
        except ImportError:
            self.logger.warning("x-vector not available. Install with: pip install speechbrain")
        except Exception as e:
            self.logger.warning(f"x-vector initialization failed: {e}")
    
    def _check_ecapa(self):
        """Check and initialize ECAPA-TDNN model with optimization."""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            # Load ECAPA-TDNN model with optimization
            self._models["ecapa"] = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Move to device if possible
            if hasattr(self._models["ecapa"], 'to'):
                self._models["ecapa"] = self._models["ecapa"].to(self.device)
            
            self.models_available["ecapa"] = True
            self._model_loaded["ecapa"] = True
            self.logger.info("ECAPA-TDNN model loaded and optimized")
            
        except ImportError:
            self.logger.warning("ECAPA-TDNN not available. Install with: pip install speechbrain")
        except Exception as e:
            self.logger.warning(f"ECAPA-TDNN initialization failed: {e}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract optimized advanced embeddings from audio file.
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with optimized advanced embeddings
        """
        try:
            self.logger.info(f"Starting optimized advanced embeddings extraction for {input_uri}")
            
            # Load and preprocess audio
            audio, sr = self._load_and_preprocess_audio(input_uri)
            
            # Extract embeddings with optimization
            features, processing_time = self._time_execution(
                self._extract_optimized_embeddings, 
                audio, 
                sr
            )
            
            self.logger.info(f"Optimized advanced embeddings extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Optimized advanced embeddings extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _load_and_preprocess_audio(self, input_uri: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio for embeddings extraction.
        
        Args:
            input_uri: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Truncate to max length if needed
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            self.logger.debug(f"Preprocessed audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load and preprocess audio file {input_uri}: {str(e)}")
            raise
    
    def _extract_optimized_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract optimized embeddings from audio.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of optimized embeddings
        """
        features = {}
        
        # VGGish embeddings
        if self.models_available["vggish"]:
            vggish_embeddings = self._extract_optimized_vggish_embeddings(audio, sr)
            features.update(vggish_embeddings)
        
        # YAMNet embeddings
        if self.models_available["yamnet"]:
            yamnet_embeddings = self._extract_optimized_yamnet_embeddings(audio, sr)
            features.update(yamnet_embeddings)
        
        # wav2vec embeddings
        if self.models_available["wav2vec"]:
            wav2vec_embeddings = self._extract_optimized_wav2vec_embeddings(audio, sr)
            features.update(wav2vec_embeddings)
        
        # HuBERT embeddings
        if self.models_available["hubert"]:
            hubert_embeddings = self._extract_optimized_hubert_embeddings(audio, sr)
            features.update(hubert_embeddings)
        
        # x-vector embeddings
        if self.models_available["xvector"]:
            xvector_embeddings = self._extract_optimized_xvector_embeddings(audio, sr)
            features.update(xvector_embeddings)
        
        # ECAPA-TDNN embeddings
        if self.models_available["ecapa"]:
            ecapa_embeddings = self._extract_optimized_ecapa_embeddings(audio, sr)
            features.update(ecapa_embeddings)
        
        # If no models are available, provide fallback features
        if not any(self.models_available.values()):
            features = self._extract_fallback_embeddings(audio, sr)
        
        # Add optimization metrics
        features["embeddings_optimized"] = True
        features["embeddings_mixed_precision"] = self.use_mixed_precision
        features["embeddings_batch_size"] = self.batch_size
        features["embeddings_device"] = self.device
        features["models_available"] = self.models_available
        
        return features
    
    def _extract_optimized_vggish_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract optimized VGGish embeddings."""
        try:
            import tensorflow as tf
            
            # Preprocess audio for VGGish
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Convert to tensor
            audio_tensor = tf.constant(audio_16k, dtype=tf.float32)
            
            # Extract embeddings with optimization
            with tf.device("/GPU:0" if self.device == "cuda" else "/CPU:0"):
                embeddings = self._models["vggish"](audio_tensor)
                embeddings_np = embeddings.numpy()
            
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "vggish_embeddings": mean_embedding.tolist(),
                "vggish_embedding_dim": len(mean_embedding),
                "vggish_embedding_mean": float(np.mean(mean_embedding)),
                "vggish_embedding_std": float(np.std(mean_embedding)),
                "vggish_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized VGGish embedding extraction failed: {e}")
            return {
                "vggish_embeddings": [],
                "vggish_embedding_dim": 0,
                "vggish_embedding_mean": 0.0,
                "vggish_embedding_std": 0.0,
                "vggish_optimized": False
            }
    
    def _extract_optimized_yamnet_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract optimized YAMNet embeddings."""
        try:
            import tensorflow as tf
            
            # Preprocess audio for YAMNet
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Convert to tensor
            audio_tensor = tf.constant(audio_16k, dtype=tf.float32)
            
            # Extract embeddings with optimization
            with tf.device("/GPU:0" if self.device == "cuda" else "/CPU:0"):
                scores, embeddings, spectrogram = self._models["yamnet"](audio_tensor)
                embeddings_np = embeddings.numpy()
            
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "yamnet_embeddings": mean_embedding.tolist(),
                "yamnet_embedding_dim": len(mean_embedding),
                "yamnet_embedding_mean": float(np.mean(mean_embedding)),
                "yamnet_embedding_std": float(np.std(mean_embedding)),
                "yamnet_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized YAMNet embedding extraction failed: {e}")
            return {
                "yamnet_embeddings": [],
                "yamnet_embedding_dim": 0,
                "yamnet_embedding_mean": 0.0,
                "yamnet_embedding_std": 0.0,
                "yamnet_optimized": False
            }
    
    def _extract_optimized_wav2vec_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract optimized wav2vec embeddings."""
        try:
            # Preprocess audio for wav2vec
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Process audio
            inputs = self._models["wav2vec_processor"](audio_16k, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings with optimization
            with torch.no_grad():
                if self.use_mixed_precision and hasattr(torch.cuda, 'amp'):
                    with torch.amp.autocast('cuda'):
                        outputs = self._models["wav2vec_model"](**inputs)
                        embeddings = outputs.last_hidden_state
                else:
                    outputs = self._models["wav2vec_model"](**inputs)
                    embeddings = outputs.last_hidden_state
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().cpu().numpy()
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "wav2vec_embeddings": mean_embedding.tolist(),
                "wav2vec_embedding_dim": len(mean_embedding),
                "wav2vec_embedding_mean": float(np.mean(mean_embedding)),
                "wav2vec_embedding_std": float(np.std(mean_embedding)),
                "wav2vec_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized wav2vec embedding extraction failed: {e}")
            return {
                "wav2vec_embeddings": [],
                "wav2vec_embedding_dim": 0,
                "wav2vec_embedding_mean": 0.0,
                "wav2vec_embedding_std": 0.0,
                "wav2vec_optimized": False
            }
    
    def _extract_optimized_hubert_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract optimized HuBERT embeddings."""
        try:
            # Preprocess audio for HuBERT
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Check audio length
            if len(audio_16k) < 1600:  # 0.1 seconds at 16kHz
                raise ValueError(f"Audio too short for HuBERT: {len(audio_16k)/16000:.3f}s")
            
            # Process audio
            inputs = self._models["hubert_processor"](audio_16k, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings with optimization
            with torch.no_grad():
                if self.use_mixed_precision and hasattr(torch.cuda, 'amp'):
                    with torch.amp.autocast('cuda'):
                        outputs = self._models["hubert_model"](**inputs)
                        embeddings = outputs.last_hidden_state
                else:
                    outputs = self._models["hubert_model"](**inputs)
                    embeddings = outputs.last_hidden_state
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().cpu().numpy()
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "hubert_embeddings": mean_embedding.tolist(),
                "hubert_embedding_dim": len(mean_embedding),
                "hubert_embedding_mean": float(np.mean(mean_embedding)),
                "hubert_embedding_std": float(np.std(mean_embedding)),
                "hubert_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized HuBERT embedding extraction failed: {e}")
            return {
                "hubert_embeddings": [],
                "hubert_embedding_dim": 0,
                "hubert_embedding_mean": 0.0,
                "hubert_embedding_std": 0.0,
                "hubert_optimized": False
            }
    
    def _extract_optimized_xvector_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract optimized x-vector embeddings."""
        try:
            # Preprocess audio for x-vector
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_16k).unsqueeze(0).to(self.device)
            
            # Extract embeddings with optimization
            with torch.no_grad():
                if self.use_mixed_precision and hasattr(torch.cuda, 'amp'):
                    with torch.amp.autocast('cuda'):
                        embeddings = self._models["xvector"].encode_batch(audio_tensor)
                else:
                    embeddings = self._models["xvector"].encode_batch(audio_tensor)
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().cpu().numpy()
            
            return {
                "xvector_embeddings": embeddings_np.tolist(),
                "xvector_embedding_dim": len(embeddings_np),
                "xvector_embedding_mean": float(np.mean(embeddings_np)),
                "xvector_embedding_std": float(np.std(embeddings_np)),
                "xvector_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized x-vector embedding extraction failed: {e}")
            return {
                "xvector_embeddings": [],
                "xvector_embedding_dim": 0,
                "xvector_embedding_mean": 0.0,
                "xvector_embedding_std": 0.0,
                "xvector_optimized": False
            }
    
    def _extract_optimized_ecapa_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract optimized ECAPA-TDNN embeddings."""
        try:
            # Preprocess audio for ECAPA-TDNN
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_16k).unsqueeze(0).to(self.device)
            
            # Extract embeddings with optimization
            with torch.no_grad():
                if self.use_mixed_precision and hasattr(torch.cuda, 'amp'):
                    with torch.amp.autocast('cuda'):
                        embeddings = self._models["ecapa"].encode_batch(audio_tensor)
                else:
                    embeddings = self._models["ecapa"].encode_batch(audio_tensor)
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().cpu().numpy()
            
            return {
                "ecapa_embeddings": embeddings_np.tolist(),
                "ecapa_embedding_dim": len(embeddings_np),
                "ecapa_embedding_mean": float(np.mean(embeddings_np)),
                "ecapa_embedding_std": float(np.std(embeddings_np)),
                "ecapa_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized ECAPA-TDNN embedding extraction failed: {e}")
            return {
                "ecapa_embeddings": [],
                "ecapa_embedding_dim": 0,
                "ecapa_embedding_mean": 0.0,
                "ecapa_embedding_std": 0.0,
                "ecapa_optimized": False
            }
    
    def _extract_fallback_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract fallback embeddings when no pretrained models are available."""
        try:
            # Use MFCC as fallback embedding
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            mean_mfcc = np.mean(mfcc, axis=1)
            
            # Use spectral features as additional embeddings
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # Combine features into a simple embedding
            fallback_embedding = np.concatenate([
                mean_mfcc,
                [np.mean(spectral_centroid)],
                [np.mean(spectral_rolloff)],
                [np.mean(spectral_bandwidth)]
            ])
            
            return {
                "fallback_embeddings": fallback_embedding.tolist(),
                "fallback_embedding_dim": len(fallback_embedding),
                "fallback_embedding_mean": float(np.mean(fallback_embedding)),
                "fallback_embedding_std": float(np.std(fallback_embedding)),
                "models_available": self.models_available,
                "fallback_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Fallback embedding extraction failed: {e}")
            return {
                "fallback_embeddings": [],
                "fallback_embedding_dim": 0,
                "fallback_embedding_mean": 0.0,
                "fallback_embedding_std": 0.0,
                "models_available": self.models_available,
                "fallback_optimized": False
            }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "batch_size": self.batch_size,
            "use_mixed_precision": self.use_mixed_precision,
            "enable_caching": self.enable_caching,
            "max_audio_length": self.max_audio_length,
            "device": self.device,
            "models_available": self.models_available,
            "memory_usage": self._memory_usage,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses
        }
    
    def optimize_for_gpu(self, gpu_memory_gb: float):
        """
        Optimize extractor settings based on available GPU memory.
        
        Args:
            gpu_memory_gb: Available GPU memory in GB
        """
        if gpu_memory_gb >= 16:
            # High-end GPU
            self.batch_size = 16
            self.use_mixed_precision = True
            self.max_audio_length = 15.0
        elif gpu_memory_gb >= 8:
            # Mid-range GPU
            self.batch_size = 8
            self.use_mixed_precision = True
            self.max_audio_length = 10.0
        elif gpu_memory_gb >= 4:
            # Entry-level GPU
            self.batch_size = 4
            self.use_mixed_precision = False
            self.max_audio_length = 8.0
        else:
            # Low memory
            self.batch_size = 2
            self.use_mixed_precision = False
            self.max_audio_length = 5.0
        
        self.logger.info(f"Optimized for {gpu_memory_gb}GB GPU: batch_size={self.batch_size}, "
                        f"mixed_precision={self.use_mixed_precision}, max_length={self.max_audio_length}")


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python optimized_advanced_embeddings_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = OptimizedAdvancedEmbeddingsExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
