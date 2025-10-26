"""
Advanced Embeddings Extractor for pretrained audio embeddings
Extracts VGGish, YAMNet, wav2vec/HuBERT, and x-vector/ECAPA embeddings
"""

import numpy as np
import librosa
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class AdvancedEmbeddingsExtractor(BaseExtractor):
    """
    Advanced Embeddings Extractor for pretrained audio embeddings
    Extracts VGGish, YAMNet, wav2vec/HuBERT, and x-vector/ECAPA embeddings
    """
    
    name = "advanced_embeddings"
    version = "1.0.0"
    description = "Advanced embeddings: VGGish, YAMNet, wav2vec/HuBERT, x-vector/ECAPA"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000  # Standard for most pretrained models
        self.hop_length = 512
        self.frame_length = 2048
        
        # Model availability flags
        self.models_available = {
            "vggish": False,
            "yamnet": False,
            "wav2vec": False,
            "hubert": False,
            "xvector": False,
            "ecapa": False
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available pretrained models"""
        try:
            # Try to import and initialize models
            self._check_vggish()
            self._check_yamnet()
            self._check_wav2vec()
            self._check_hubert()
            self._check_xvector()
            self._check_ecapa()
            
            available_models = [model for model, available in self.models_available.items() if available]
            # self.logger.info(f"Available embedding models: {available_models}")
            
        except Exception as e:
            self.logger.warning(f"Model initialization failed: {e}")
    
    def _check_vggish(self):
        """Check if VGGish model is available"""
        try:
            # VGGish is typically available through tensorflow-hub
            import tensorflow_hub as hub
            self.models_available["vggish"] = True
            # self.logger.info("VGGish model available")
        except ImportError:
            # self.logger.warning("VGGish not available. Install with: pip install tensorflow-hub")
            pass
    
    def _check_yamnet(self):
        """Check if YAMNet model is available"""
        try:
            # YAMNet is typically available through tensorflow-hub
            import tensorflow_hub as hub
            self.models_available["yamnet"] = True
            # self.logger.info("YAMNet model available")
        except ImportError:
            # self.logger.warning("YAMNet not available. Install with: pip install tensorflow-hub")
            pass
    
    def _check_wav2vec(self):
        """Check if wav2vec model is available"""
        try:
            # wav2vec is available through transformers
            from transformers import Wav2Vec2Model
            self.models_available["wav2vec"] = True
            # self.logger.info("wav2vec model available")
        except ImportError:
            # self.logger.warning("wav2vec not available. Install with: pip install transformers")
            pass
    
    def _check_hubert(self):
        """Check if HuBERT model is available"""
        try:
            # HuBERT is available through transformers
            from transformers import HubertModel
            self.models_available["hubert"] = True
            # self.logger.info("HuBERT model available")
        except ImportError:
            # self.logger.warning("HuBERT not available. Install with: pip install transformers")
            pass
    
    def _check_xvector(self):
        """Check if x-vector model is available"""
        try:
            # x-vector is available through speechbrain
            import speechbrain
            self.models_available["xvector"] = True
            # self.logger.info("x-vector model available")
        except ImportError:
            # self.logger.warning("x-vector not available. Install with: pip install speechbrain")
            pass
    
    def _check_ecapa(self):
        """Check if ECAPA-TDNN model is available"""
        try:
            # ECAPA-TDNN is available through speechbrain
            import speechbrain
            self.models_available["ecapa"] = True
            # self.logger.info("ECAPA-TDNN model available")
        except ImportError:
            # self.logger.warning("ECAPA-TDNN not available. Install with: pip install speechbrain")
            pass
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract advanced embeddings from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with advanced embeddings
        """
        try:
            self.logger.info(f"Starting advanced embeddings extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract advanced embeddings with timing
            features, processing_time = self._time_execution(self._extract_advanced_embeddings, audio, sr)
            
            self.logger.info(f"Advanced embeddings extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Advanced embeddings extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_advanced_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract advanced embeddings from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of advanced embeddings
        """
        features = {}
        
        # VGGish embeddings
        if self.models_available["vggish"]:
            vggish_embeddings = self._extract_vggish_embeddings(audio, sr)
            features.update(vggish_embeddings)
        
        # YAMNet embeddings
        if self.models_available["yamnet"]:
            yamnet_embeddings = self._extract_yamnet_embeddings(audio, sr)
            features.update(yamnet_embeddings)
        
        # wav2vec embeddings
        if self.models_available["wav2vec"]:
            wav2vec_embeddings = self._extract_wav2vec_embeddings(audio, sr)
            features.update(wav2vec_embeddings)
        
        # HuBERT embeddings
        if self.models_available["hubert"]:
            hubert_embeddings = self._extract_hubert_embeddings(audio, sr)
            features.update(hubert_embeddings)
        
        # x-vector embeddings
        if self.models_available["xvector"]:
            xvector_embeddings = self._extract_xvector_embeddings(audio, sr)
            features.update(xvector_embeddings)
        
        # ECAPA-TDNN embeddings
        if self.models_available["ecapa"]:
            ecapa_embeddings = self._extract_ecapa_embeddings(audio, sr)
            features.update(ecapa_embeddings)
        
        # If no models are available, provide fallback features
        if not any(self.models_available.values()):
            features = self._extract_fallback_embeddings(audio, sr)
        
        return features
    
    def _extract_vggish_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract VGGish embeddings"""
        try:
            import tensorflow_hub as hub
            
            # Load VGGish model
            model = hub.load("https://tfhub.dev/google/vggish/1")
            
            # Preprocess audio for VGGish (expects 16kHz, mono)
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # VGGish expects audio in specific format
            # Convert to the format expected by VGGish
            audio_tensor = tf.constant(audio_16k, dtype=tf.float32)
            
            # Extract embeddings
            embeddings = model(audio_tensor)
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.numpy()
            
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "vggish_embeddings": mean_embedding.tolist(),
                "vggish_embedding_dim": len(mean_embedding),
                "vggish_embedding_mean": float(np.mean(mean_embedding)),
                "vggish_embedding_std": float(np.std(mean_embedding))
            }
            
        except Exception as e:
            self.logger.warning(f"VGGish embedding extraction failed: {e}")
            return {
                "vggish_embeddings": [],
                "vggish_embedding_dim": 0,
                "vggish_embedding_mean": 0.0,
                "vggish_embedding_std": 0.0
            }
    
    def _extract_yamnet_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract YAMNet embeddings"""
        try:
            import tensorflow_hub as hub
            
            # Load YAMNet model
            model = hub.load("https://tfhub.dev/google/yamnet/1")
            
            # Preprocess audio for YAMNet (expects 16kHz, mono)
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # YAMNet expects audio in specific format
            audio_tensor = tf.constant(audio_16k, dtype=tf.float32)
            
            # Extract embeddings
            scores, embeddings, spectrogram = model(audio_tensor)
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.numpy()
            
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "yamnet_embeddings": mean_embedding.tolist(),
                "yamnet_embedding_dim": len(mean_embedding),
                "yamnet_embedding_mean": float(np.mean(mean_embedding)),
                "yamnet_embedding_std": float(np.std(mean_embedding))
            }
            
        except Exception as e:
            self.logger.warning(f"YAMNet embedding extraction failed: {e}")
            return {
                "yamnet_embeddings": [],
                "yamnet_embedding_dim": 0,
                "yamnet_embedding_mean": 0.0,
                "yamnet_embedding_std": 0.0
            }
    
    def _extract_wav2vec_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract wav2vec embeddings"""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            import torch
            
            # Load wav2vec model and processor
            model_name = "facebook/wav2vec2-base-960h"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2Model.from_pretrained(model_name)
            
            # Preprocess audio for wav2vec
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Process audio
            inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().numpy()
            
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "wav2vec_embeddings": mean_embedding.tolist(),
                "wav2vec_embedding_dim": len(mean_embedding),
                "wav2vec_embedding_mean": float(np.mean(mean_embedding)),
                "wav2vec_embedding_std": float(np.std(mean_embedding))
            }
            
        except Exception as e:
            self.logger.warning(f"wav2vec embedding extraction failed: {e}")
            return {
                "wav2vec_embeddings": [],
                "wav2vec_embedding_dim": 0,
                "wav2vec_embedding_mean": 0.0,
                "wav2vec_embedding_std": 0.0
            }
    
    def _extract_hubert_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract HuBERT embeddings"""
        try:
            from transformers import HubertModel, Wav2Vec2Processor
            import torch
            
            # Load HuBERT model and processor
            model_name = "facebook/hubert-base-ls960"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = HubertModel.from_pretrained(model_name)
            
            # Preprocess audio for HuBERT
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Check audio length - HuBERT needs at least 0.1 seconds
            if len(audio_16k) < 1600:  # 0.1 seconds at 16kHz
                raise ValueError(f"Audio too short for HuBERT: {len(audio_16k)/16000:.3f}s (minimum 0.1s required)")
            
            # Process audio
            inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            
            # Check if inputs are valid
            if inputs is None:
                raise ValueError("HuBERT processor returned None")
            
            if not hasattr(inputs, 'input_values') or inputs.input_values is None:
                raise ValueError("HuBERT processor returned invalid input_values")
            
            if inputs.input_values.shape[1] == 0:
                raise ValueError("HuBERT processor returned empty input_values")
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().numpy()
            
            # Calculate mean embedding
            mean_embedding = np.mean(embeddings_np, axis=0)
            
            return {
                "hubert_embeddings": mean_embedding.tolist(),
                "hubert_embedding_dim": len(mean_embedding),
                "hubert_embedding_mean": float(np.mean(mean_embedding)),
                "hubert_embedding_std": float(np.std(mean_embedding))
            }
            
        except Exception as e:
            # self.logger.warning(f"HuBERT embedding extraction failed: {e}")
            return {
                "hubert_embeddings": [],
                "hubert_embedding_dim": 0,
                "hubert_embedding_mean": 0.0,
                "hubert_embedding_std": 0.0
            }
    
    def _extract_xvector_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract x-vector embeddings"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            import torch
            
            # Load x-vector model
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Preprocess audio for x-vector
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_16k).unsqueeze(0)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = model.encode_batch(audio_tensor)
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().numpy()
            
            return {
                "xvector_embeddings": embeddings_np.tolist(),
                "xvector_embedding_dim": len(embeddings_np),
                "xvector_embedding_mean": float(np.mean(embeddings_np)),
                "xvector_embedding_std": float(np.std(embeddings_np))
            }
            
        except Exception as e:
            self.logger.warning(f"x-vector embedding extraction failed: {e}")
            return {
                "xvector_embeddings": [],
                "xvector_embedding_dim": 0,
                "xvector_embedding_mean": 0.0,
                "xvector_embedding_std": 0.0
            }
    
    def _extract_ecapa_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract ECAPA-TDNN embeddings"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            import torch
            
            # Load ECAPA-TDNN model
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Preprocess audio for ECAPA-TDNN
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_16k).unsqueeze(0)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = model.encode_batch(audio_tensor)
            
            # Convert to numpy and calculate statistics
            embeddings_np = embeddings.squeeze().numpy()
            
            return {
                "ecapa_embeddings": embeddings_np.tolist(),
                "ecapa_embedding_dim": len(embeddings_np),
                "ecapa_embedding_mean": float(np.mean(embeddings_np)),
                "ecapa_embedding_std": float(np.std(embeddings_np))
            }
            
        except Exception as e:
            self.logger.warning(f"ECAPA-TDNN embedding extraction failed: {e}")
            return {
                "ecapa_embeddings": [],
                "ecapa_embedding_dim": 0,
                "ecapa_embedding_mean": 0.0,
                "ecapa_embedding_std": 0.0
            }
    
    def _extract_fallback_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract fallback embeddings when no pretrained models are available"""
        try:
            # Use MFCC as fallback embedding
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Calculate mean MFCC as embedding
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
                "models_available": self.models_available
            }
            
        except Exception as e:
            self.logger.warning(f"Fallback embedding extraction failed: {e}")
            return {
                "fallback_embeddings": [],
                "fallback_embedding_dim": 0,
                "fallback_embedding_mean": 0.0,
                "fallback_embedding_std": 0.0,
                "models_available": self.models_available
            }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_embeddings_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = AdvancedEmbeddingsExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
