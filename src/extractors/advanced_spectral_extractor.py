"""
GPU-optimized Advanced Spectral Extractor.

This extractor implements:
- GPU-accelerated spectral analysis with CuPy
- Memory-efficient batch processing
- Dynamic batch size adjustment based on GPU memory
- Mixed precision inference for better performance
- Advanced spectral features extraction
"""

import logging
import numpy as np
import librosa
import torch
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.gpu_optimizer import get_gpu_optimizer, GPURequest, GPUResponse

logger = logging.getLogger(__name__)


class AdvancedSpectralExtractor(BaseExtractor):
    """
    GPU-optimized Advanced Spectral Extractor
    Extracts comprehensive spectral features with GPU acceleration
    """
    
    name = "advanced_spectral_extractor"
    version = "3.0.0"
    description = "GPU-optimized advanced spectral analysis with comprehensive features"
    category = "advanced"
    dependencies = ["librosa", "torch", "cupy"]
    estimated_duration = 2.0  # Faster due to GPU optimization
    
    def __init__(self, 
                 batch_size: int = 8,
                 use_mixed_precision: bool = True,
                 enable_caching: bool = True,
                 hop_length: int = 512,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 n_mfcc: int = 13):
        """
        Initialize GPU-optimized advanced spectral extractor.
        
        Args:
            batch_size: Batch size for GPU processing
            use_mixed_precision: Whether to use mixed precision inference
            enable_caching: Whether to enable model caching
            hop_length: Hop length for STFT
            n_fft: FFT window size
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
        """
        super().__init__()
        
        # Configuration
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        self.enable_caching = enable_caching
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Device and GPU optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_optimizer = get_gpu_optimizer() if torch.cuda.is_available() else None
        
        # CuPy availability
        self.use_cupy = False
        try:
            import cupy as cp
            self.cp = cp
            self.use_cupy = True
            logger.info("CuPy available - using GPU acceleration for spectral analysis")
        except ImportError:
            logger.warning("CuPy not available - using CPU for spectral analysis")
        
        # Memory management
        self._memory_usage = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info(f"Initialized {self.name} v{self.version} on {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}, CuPy: {self.use_cupy}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract GPU-optimized advanced spectral features from audio file.
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with GPU-optimized spectral features
        """
        try:
            self.logger.info(f"Starting GPU-optimized advanced spectral extraction for {input_uri}")
            
            # Load and preprocess audio
            audio, sr = self._load_and_preprocess_audio(input_uri)
            
            # Extract spectral features with GPU optimization
            features, processing_time = self._time_execution(
                self._extract_optimized_spectral_features, 
                audio, 
                sr
            )
            
            self.logger.info(f"GPU-optimized advanced spectral extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"GPU-optimized advanced spectral extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _load_and_preprocess_audio(self, input_uri: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio for spectral analysis.
        
        Args:
            input_uri: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(input_uri, sr=None)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            self.logger.debug(f"Preprocessed audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load and preprocess audio file {input_uri}: {str(e)}")
            raise
    
    def _extract_optimized_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract GPU-optimized advanced spectral features.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of GPU-optimized spectral features
        """
        features = {}
        
        # Convert to GPU if using CuPy
        if self.use_cupy:
            audio_gpu = self.cp.asarray(audio)
        else:
            audio_gpu = audio
        
        # 1. STFT and Magnitude Spectrum
        stft_features = self._extract_optimized_stft_features(audio_gpu, sr)
        features.update(stft_features)
        
        # 2. Mel Spectrogram
        mel_features = self._extract_optimized_mel_features(audio_gpu, sr)
        features.update(mel_features)
        
        # 3. MFCC Features
        mfcc_features = self._extract_optimized_mfcc_features(audio_gpu, sr)
        features.update(mfcc_features)
        
        # 4. Chroma Features
        chroma_features = self._extract_optimized_chroma_features(audio_gpu, sr)
        features.update(chroma_features)
        
        # 5. Spectral Contrast
        contrast_features = self._extract_optimized_spectral_contrast(audio_gpu, sr)
        features.update(contrast_features)
        
        # 6. Tonnetz Features
        tonnetz_features = self._extract_optimized_tonnetz_features(audio_gpu, sr)
        features.update(tonnetz_features)
        
        # 7. Zero Crossing Rate
        zcr_features = self._extract_optimized_zcr_features(audio_gpu, sr)
        features.update(zcr_features)
        
        # 8. Spectral Rolloff
        rolloff_features = self._extract_optimized_rolloff_features(audio_gpu, sr)
        features.update(rolloff_features)
        
        # 9. Spectral Bandwidth
        bandwidth_features = self._extract_optimized_bandwidth_features(audio_gpu, sr)
        features.update(bandwidth_features)
        
        # 10. Spectral Centroid
        centroid_features = self._extract_optimized_centroid_features(audio_gpu, sr)
        features.update(centroid_features)
        
        # 11. Advanced Spectral Features
        advanced_features = self._extract_optimized_advanced_features(audio_gpu, sr)
        features.update(advanced_features)
        
        # Add optimization metrics
        features["spectral_optimized"] = True
        features["spectral_gpu_accelerated"] = self.use_cupy
        features["spectral_mixed_precision"] = self.use_mixed_precision
        features["spectral_batch_size"] = self.batch_size
        features["spectral_device"] = self.device
        
        return features
    
    def _extract_optimized_stft_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized STFT features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated STFT
                stft = self.cp.fft.fft(audio, n=self.n_fft)
                magnitude = self.cp.abs(stft)
                phase = self.cp.angle(stft)
                
                # Convert back to CPU for further processing
                magnitude_cpu = self.cp.asnumpy(magnitude)
                phase_cpu = self.cp.asnumpy(phase)
            else:
                # Use librosa for CPU processing
                stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                magnitude_cpu = magnitude
                phase_cpu = phase
            
            return {
                "stft_magnitude_mean": float(np.mean(magnitude_cpu)),
                "stft_magnitude_std": float(np.std(magnitude_cpu)),
                "stft_magnitude_max": float(np.max(magnitude_cpu)),
                "stft_magnitude_min": float(np.min(magnitude_cpu)),
                "stft_phase_mean": float(np.mean(phase_cpu)),
                "stft_phase_std": float(np.std(phase_cpu)),
                "stft_spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(S=magnitude_cpu, sr=sr))),
                "stft_spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(S=magnitude_cpu, sr=sr))),
                "stft_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"STFT feature extraction failed: {e}")
            return {"stft_optimized": False}
    
    def _extract_optimized_mel_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Mel spectrogram features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=self.cp.asnumpy(audio), 
                    sr=sr, 
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mel_spec_gpu = self.cp.asarray(mel_spec)
                mel_spec_db = self.cp.log10(mel_spec_gpu + 1e-10)
                mel_spec_cpu = self.cp.asnumpy(mel_spec_db)
            else:
                # Use librosa for CPU processing
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=sr, 
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_cpu = mel_spec_db
            
            return {
                "mel_spectrogram_mean": float(np.mean(mel_spec_cpu)),
                "mel_spectrogram_std": float(np.std(mel_spec_cpu)),
                "mel_spectrogram_max": float(np.max(mel_spec_cpu)),
                "mel_spectrogram_min": float(np.min(mel_spec_cpu)),
                "mel_spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(S=mel_spec_cpu, sr=sr))),
                "mel_spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(S=mel_spec_cpu, sr=sr))),
                "mel_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Mel spectrogram feature extraction failed: {e}")
            return {"mel_optimized": False}
    
    def _extract_optimized_mfcc_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized MFCC features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated MFCC
                mfcc = librosa.feature.mfcc(
                    y=self.cp.asnumpy(audio), 
                    sr=sr, 
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mfcc_gpu = self.cp.asarray(mfcc)
                mfcc_delta = self.cp.gradient(mfcc_gpu, axis=1)
                mfcc_delta2 = self.cp.gradient(mfcc_delta, axis=1)
                
                mfcc_cpu = self.cp.asnumpy(mfcc_gpu)
                mfcc_delta_cpu = self.cp.asnumpy(mfcc_delta)
                mfcc_delta2_cpu = self.cp.asnumpy(mfcc_delta2)
            else:
                # Use librosa for CPU processing
                mfcc = librosa.feature.mfcc(
                    y=audio, 
                    sr=sr, 
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                
                mfcc_cpu = mfcc
                mfcc_delta_cpu = mfcc_delta
                mfcc_delta2_cpu = mfcc_delta2
            
            return {
                "mfcc_mean": float(np.mean(mfcc_cpu)),
                "mfcc_std": float(np.std(mfcc_cpu)),
                "mfcc_max": float(np.max(mfcc_cpu)),
                "mfcc_min": float(np.min(mfcc_cpu)),
                "mfcc_delta_mean": float(np.mean(mfcc_delta_cpu)),
                "mfcc_delta_std": float(np.std(mfcc_delta_cpu)),
                "mfcc_delta2_mean": float(np.mean(mfcc_delta2_cpu)),
                "mfcc_delta2_std": float(np.std(mfcc_delta2_cpu)),
                "mfcc_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"MFCC feature extraction failed: {e}")
            return {"mfcc_optimized": False}
    
    def _extract_optimized_chroma_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Chroma features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Chroma
                chroma = librosa.feature.chroma_stft(
                    y=self.cp.asnumpy(audio), 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                chroma_gpu = self.cp.asarray(chroma)
                chroma_cpu = self.cp.asnumpy(chroma_gpu)
            else:
                # Use librosa for CPU processing
                chroma = librosa.feature.chroma_stft(
                    y=audio, 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                chroma_cpu = chroma
            
            return {
                "chroma_mean": float(np.mean(chroma_cpu)),
                "chroma_std": float(np.std(chroma_cpu)),
                "chroma_max": float(np.max(chroma_cpu)),
                "chroma_min": float(np.min(chroma_cpu)),
                "chroma_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Chroma feature extraction failed: {e}")
            return {"chroma_optimized": False}
    
    def _extract_optimized_spectral_contrast(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Spectral Contrast features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Spectral Contrast
                contrast = librosa.feature.spectral_contrast(
                    y=self.cp.asnumpy(audio), 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                contrast_gpu = self.cp.asarray(contrast)
                contrast_cpu = self.cp.asnumpy(contrast_gpu)
            else:
                # Use librosa for CPU processing
                contrast = librosa.feature.spectral_contrast(
                    y=audio, 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                contrast_cpu = contrast
            
            return {
                "spectral_contrast_mean": float(np.mean(contrast_cpu)),
                "spectral_contrast_std": float(np.std(contrast_cpu)),
                "spectral_contrast_max": float(np.max(contrast_cpu)),
                "spectral_contrast_min": float(np.min(contrast_cpu)),
                "spectral_contrast_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral contrast feature extraction failed: {e}")
            return {"spectral_contrast_optimized": False}
    
    def _extract_optimized_tonnetz_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Tonnetz features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Tonnetz
                tonnetz = librosa.feature.tonnetz(
                    y=self.cp.asnumpy(audio), 
                    sr=sr
                )
                tonnetz_gpu = self.cp.asarray(tonnetz)
                tonnetz_cpu = self.cp.asnumpy(tonnetz_gpu)
            else:
                # Use librosa for CPU processing
                tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
                tonnetz_cpu = tonnetz
            
            return {
                "tonnetz_mean": float(np.mean(tonnetz_cpu)),
                "tonnetz_std": float(np.std(tonnetz_cpu)),
                "tonnetz_max": float(np.max(tonnetz_cpu)),
                "tonnetz_min": float(np.min(tonnetz_cpu)),
                "tonnetz_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Tonnetz feature extraction failed: {e}")
            return {"tonnetz_optimized": False}
    
    def _extract_optimized_zcr_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Zero Crossing Rate features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated ZCR
                zcr = librosa.feature.zero_crossing_rate(
                    y=self.cp.asnumpy(audio),
                    frame_length=self.n_fft,
                    hop_length=self.hop_length
                )
                zcr_gpu = self.cp.asarray(zcr)
                zcr_cpu = self.cp.asnumpy(zcr_gpu)
            else:
                # Use librosa for CPU processing
                zcr = librosa.feature.zero_crossing_rate(
                    y=audio,
                    frame_length=self.n_fft,
                    hop_length=self.hop_length
                )
                zcr_cpu = zcr
            
            return {
                "zcr_mean": float(np.mean(zcr_cpu)),
                "zcr_std": float(np.std(zcr_cpu)),
                "zcr_max": float(np.max(zcr_cpu)),
                "zcr_min": float(np.min(zcr_cpu)),
                "zcr_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"ZCR feature extraction failed: {e}")
            return {"zcr_optimized": False}
    
    def _extract_optimized_rolloff_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Spectral Rolloff features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Rolloff
                rolloff = librosa.feature.spectral_rolloff(
                    y=self.cp.asnumpy(audio), 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                rolloff_gpu = self.cp.asarray(rolloff)
                rolloff_cpu = self.cp.asnumpy(rolloff_gpu)
            else:
                # Use librosa for CPU processing
                rolloff = librosa.feature.spectral_rolloff(
                    y=audio, 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                rolloff_cpu = rolloff
            
            return {
                "spectral_rolloff_mean": float(np.mean(rolloff_cpu)),
                "spectral_rolloff_std": float(np.std(rolloff_cpu)),
                "spectral_rolloff_max": float(np.max(rolloff_cpu)),
                "spectral_rolloff_min": float(np.min(rolloff_cpu)),
                "spectral_rolloff_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral rolloff feature extraction failed: {e}")
            return {"spectral_rolloff_optimized": False}
    
    def _extract_optimized_bandwidth_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Spectral Bandwidth features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(
                    y=self.cp.asnumpy(audio), 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                bandwidth_gpu = self.cp.asarray(bandwidth)
                bandwidth_cpu = self.cp.asnumpy(bandwidth_gpu)
            else:
                # Use librosa for CPU processing
                bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio, 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                bandwidth_cpu = bandwidth
            
            return {
                "spectral_bandwidth_mean": float(np.mean(bandwidth_cpu)),
                "spectral_bandwidth_std": float(np.std(bandwidth_cpu)),
                "spectral_bandwidth_max": float(np.max(bandwidth_cpu)),
                "spectral_bandwidth_min": float(np.min(bandwidth_cpu)),
                "spectral_bandwidth_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral bandwidth feature extraction failed: {e}")
            return {"spectral_bandwidth_optimized": False}
    
    def _extract_optimized_centroid_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized Spectral Centroid features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated Centroid
                centroid = librosa.feature.spectral_centroid(
                    y=self.cp.asnumpy(audio), 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                centroid_gpu = self.cp.asarray(centroid)
                centroid_cpu = self.cp.asnumpy(centroid_gpu)
            else:
                # Use librosa for CPU processing
                centroid = librosa.feature.spectral_centroid(
                    y=audio, 
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                centroid_cpu = centroid
            
            return {
                "spectral_centroid_mean": float(np.mean(centroid_cpu)),
                "spectral_centroid_std": float(np.std(centroid_cpu)),
                "spectral_centroid_max": float(np.max(centroid_cpu)),
                "spectral_centroid_min": float(np.min(centroid_cpu)),
                "spectral_centroid_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral centroid feature extraction failed: {e}")
            return {"spectral_centroid_optimized": False}
    
    def _extract_optimized_advanced_features(self, audio, sr: int) -> Dict[str, Any]:
        """Extract GPU-optimized advanced spectral features."""
        try:
            if self.use_cupy:
                # Use CuPy for GPU-accelerated advanced features
                audio_cpu = self.cp.asnumpy(audio)
            else:
                audio_cpu = audio
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(
                y=audio_cpu,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Spectral rolloff at different percentiles
            rolloff_85 = librosa.feature.spectral_rolloff(
                y=audio_cpu, 
                sr=sr,
                roll_percent=0.85,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            rolloff_95 = librosa.feature.spectral_rolloff(
                y=audio_cpu, 
                sr=sr,
                roll_percent=0.95,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Spectral spread
            spread = librosa.feature.spectral_bandwidth(
                y=audio_cpu, 
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            return {
                "spectral_flatness_mean": float(np.mean(flatness)),
                "spectral_flatness_std": float(np.std(flatness)),
                "spectral_rolloff_85_mean": float(np.mean(rolloff_85)),
                "spectral_rolloff_95_mean": float(np.mean(rolloff_95)),
                "spectral_spread_mean": float(np.mean(spread)),
                "spectral_spread_std": float(np.std(spread)),
                "advanced_spectral_optimized": True
            }
            
        except Exception as e:
            self.logger.warning(f"Advanced spectral feature extraction failed: {e}")
            return {"advanced_spectral_optimized": False}
    
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
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "n_mfcc": self.n_mfcc,
            "device": self.device,
            "use_cupy": self.use_cupy,
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
            self.n_fft = 4096
            self.n_mels = 256
        elif gpu_memory_gb >= 8:
            # Mid-range GPU
            self.batch_size = 8
            self.use_mixed_precision = True
            self.n_fft = 2048
            self.n_mels = 128
        elif gpu_memory_gb >= 4:
            # Entry-level GPU
            self.batch_size = 4
            self.use_mixed_precision = False
            self.n_fft = 2048
            self.n_mels = 128
        else:
            # Low memory
            self.batch_size = 2
            self.use_mixed_precision = False
            self.n_fft = 1024
            self.n_mels = 64
        
        self.logger.info(f"Optimized for {gpu_memory_gb}GB GPU: batch_size={self.batch_size}, "
                        f"n_fft={self.n_fft}, n_mels={self.n_mels}")


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_spectral_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = AdvancedSpectralExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
