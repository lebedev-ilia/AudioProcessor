"""
GPU-accelerated audio processing utilities.

This module provides GPU-accelerated replacements for librosa functions
using PyTorch and torchaudio for maximum performance.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUAudioProcessor:
    """GPU-accelerated audio processing utilities."""
    
    def __init__(self, device: str = "cuda", sample_rate: int = 22050, 
                 enable_tensor_cores: bool = True, enable_mixed_precision: bool = True):
        """
        Initialize GPU audio processor.
        
        Args:
            device: Device to use for processing ("cuda" or "cpu")
            sample_rate: Default sample rate for audio processing
            enable_tensor_cores: Enable Tensor Core optimizations
            enable_mixed_precision: Enable mixed precision processing
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.sample_rate = sample_rate
        self.enable_tensor_cores = enable_tensor_cores and self.device != "cpu"
        self.enable_mixed_precision = enable_mixed_precision and self.device != "cpu"
        
        # Enable GPU optimizations
        self._enable_gpu_optimizations()
        
        # Pre-initialize transforms for better performance
        self._init_transforms()
        
        logger.info(f"GPU Audio Processor initialized on device: {self.device}")
        logger.info(f"Tensor Cores: {self.enable_tensor_cores}, Mixed Precision: {self.enable_mixed_precision}")
    
    def _enable_gpu_optimizations(self):
        """Enable GPU optimizations for maximum performance."""
        if not torch.cuda.is_available():
            return
        
        try:
            # Enable Tensor Core optimizations
            if self.enable_tensor_cores:
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("Tensor Core optimizations enabled")
            
            # Enable cuDNN optimizations
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'deterministic'):
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for performance
            
            # Set optimal memory allocation strategy
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            
            logger.info("GPU optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to apply some GPU optimizations: {e}")
    
    def _init_transforms(self):
        """Initialize commonly used transforms."""
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=None
        ).to(self.device)
        
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "f_min": 0.0,
                "f_max": None
            }
        ).to(self.device)
        
        self.spectrogram_transform = T.Spectrogram(
            n_fft=2048,
            hop_length=512,
            power=2.0
        ).to(self.device)
    
    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and convert to tensor.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None for original)
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if target_sr is not None and sample_rate != target_sr:
                resampler = T.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
                sample_rate = target_sr
            
            # Move to device immediately after loading
            waveform = waveform.to(self.device)
            
            return waveform.squeeze(0), sample_rate  # Remove channel dimension
            
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def stft(self, waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Complex STFT tensor
        """
        return torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        )
    
    def magnitude_spectrogram(self, waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """
        Compute magnitude spectrogram.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Magnitude spectrogram tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        return torch.abs(stft)
    
    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram with optional mixed precision.
        
        Args:
            waveform: Input audio tensor
            
        Returns:
            Mel spectrogram tensor
        """
        if self.enable_mixed_precision and waveform.dtype == torch.float32:
            # Use mixed precision for better performance
            with torch.cuda.amp.autocast(enabled=True):
                return self.mel_transform(waveform)
        else:
            return self.mel_transform(waveform)
    
    def mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC features with optional mixed precision.
        
        Args:
            waveform: Input audio tensor
            
        Returns:
            MFCC tensor
        """
        if self.enable_mixed_precision and waveform.dtype == torch.float32:
            # Use mixed precision for better performance
            with torch.cuda.amp.autocast(enabled=True):
                return self.mfcc_transform(waveform)
        else:
            return self.mfcc_transform(waveform)
    
    def spectral_centroid(self, waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral centroid.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Spectral centroid tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Frequency bins
        freqs = torch.linspace(0, self.sample_rate // 2, magnitude.shape[0], device=self.device)
        freqs = freqs.unsqueeze(1)
        
        # Spectral centroid
        centroid = torch.sum(freqs * magnitude, dim=0) / (torch.sum(magnitude, dim=0) + 1e-10)
        
        return centroid
    
    def spectral_bandwidth(self, waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral bandwidth.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Spectral bandwidth tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Frequency bins
        freqs = torch.linspace(0, self.sample_rate // 2, magnitude.shape[0], device=self.device)
        freqs = freqs.unsqueeze(1)
        
        # Spectral centroid
        centroid = self.spectral_centroid(waveform, n_fft, hop_length)
        
        # Spectral bandwidth
        bandwidth = torch.sqrt(
            torch.sum(((freqs - centroid.unsqueeze(0)) ** 2) * magnitude, dim=0) / 
            (torch.sum(magnitude, dim=0) + 1e-10)
        )
        
        return bandwidth
    
    def spectral_rolloff(self, waveform: torch.Tensor, roll_percent: float = 0.85, 
                        n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral rolloff.
        
        Args:
            waveform: Input audio tensor
            roll_percent: Rolloff percentage (0.85 = 85%)
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Spectral rolloff tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Frequency bins
        freqs = torch.linspace(0, self.sample_rate // 2, magnitude.shape[0], device=self.device)
        freqs = freqs.unsqueeze(1)
        
        # Cumulative sum
        cumsum = torch.cumsum(magnitude, dim=0)
        total_energy = cumsum[-1, :]
        
        # Find rolloff frequency
        threshold = roll_percent * total_energy
        rolloff_indices = torch.searchsorted(cumsum.T, threshold.unsqueeze(0)).T
        rolloff_indices = torch.clamp(rolloff_indices, 0, magnitude.shape[0] - 1)
        
        rolloff_freqs = freqs[rolloff_indices.squeeze(0), 0]
        
        return rolloff_freqs
    
    def zero_crossing_rate(self, waveform: torch.Tensor, frame_length: int = 2048, 
                          hop_length: int = 512) -> torch.Tensor:
        """
        Compute zero crossing rate.
        
        Args:
            waveform: Input audio tensor
            frame_length: Frame length
            hop_length: Hop length
            
        Returns:
            Zero crossing rate tensor
        """
        # Pad waveform
        padded = torch.nn.functional.pad(waveform, (frame_length // 2, frame_length // 2))
        
        # Create frames
        frames = padded.unfold(0, frame_length, hop_length)
        
        # Compute zero crossings
        diff = torch.diff(frames, dim=1)
        sign_changes = torch.diff(torch.sign(diff), dim=1)
        zcr = torch.sum(torch.abs(sign_changes), dim=1) / (2 * frame_length)
        
        return zcr
    
    def spectral_flatness(self, waveform: torch.Tensor, n_fft: int = 2048, 
                         hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral flatness (Wiener entropy).
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Spectral flatness tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Avoid log(0)
        magnitude = torch.clamp(magnitude, min=1e-10)
        
        # Geometric mean
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude), dim=0))
        
        # Arithmetic mean
        arithmetic_mean = torch.mean(magnitude, dim=0)
        
        # Spectral flatness
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        return flatness
    
    def spectral_contrast(self, waveform: torch.Tensor, n_fft: int = 2048, 
                         hop_length: int = 512, n_bands: int = 6) -> torch.Tensor:
        """
        Compute spectral contrast.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            n_bands: Number of frequency bands
            
        Returns:
            Spectral contrast tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Create frequency bands
        freqs = torch.linspace(0, self.sample_rate // 2, magnitude.shape[0], device=self.device)
        band_edges = torch.linspace(0, self.sample_rate // 2, n_bands + 1, device=self.device)
        
        contrast = []
        for i in range(n_bands):
            # Find frequency indices for this band
            start_idx = torch.searchsorted(freqs, band_edges[i])
            end_idx = torch.searchsorted(freqs, band_edges[i + 1])
            
            if start_idx < end_idx:
                band_magnitude = magnitude[start_idx:end_idx, :]
                band_max = torch.max(band_magnitude, dim=0)[0]
                band_min = torch.min(band_magnitude, dim=0)[0]
                band_contrast = band_max - band_min
            else:
                band_contrast = torch.zeros(magnitude.shape[1], device=self.device)
            
            contrast.append(band_contrast)
        
        return torch.stack(contrast, dim=0)
    
    def spectral_flux(self, waveform: torch.Tensor, n_fft: int = 2048, 
                     hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral flux.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Spectral flux tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Compute differences between consecutive frames
        diff = torch.diff(magnitude, dim=1)
        flux = torch.sum(diff ** 2, dim=0)
        
        return flux
    
    def spectral_entropy(self, waveform: torch.Tensor, n_fft: int = 2048, 
                        hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral entropy.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Spectral entropy tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Normalize to get probability distribution
        magnitude_norm = magnitude / (torch.sum(magnitude, dim=0, keepdim=True) + 1e-10)
        
        # Compute entropy: -sum(p * log2(p))
        entropy = -torch.sum(magnitude_norm * torch.log2(magnitude_norm + 1e-10), dim=0)
        
        return entropy
    
    def chroma_stft(self, waveform: torch.Tensor, n_fft: int = 2048, 
                   hop_length: int = 512, n_chroma: int = 12) -> torch.Tensor:
        """
        Compute chroma features from STFT.
        
        Args:
            waveform: Input audio tensor
            n_fft: FFT window size
            hop_length: Hop length
            n_chroma: Number of chroma bins
            
        Returns:
            Chroma tensor
        """
        stft = self.stft(waveform, n_fft, hop_length)
        magnitude = torch.abs(stft)
        
        # Create chroma filter bank
        freqs = torch.linspace(0, self.sample_rate // 2, magnitude.shape[0], device=self.device)
        
        # Create chroma mapping
        chroma_map = torch.zeros(n_chroma, magnitude.shape[0], device=self.device)
        
        for i in range(n_chroma):
            # Map frequencies to chroma bins
            chroma_freqs = 440.0 * (2.0 ** ((torch.arange(12, device=self.device) + i) / 12.0))
            
            for j, freq in enumerate(chroma_freqs):
                if freq <= self.sample_rate // 2:
                    # Find closest frequency bin
                    freq_idx = torch.argmin(torch.abs(freqs - freq))
                    chroma_map[i, freq_idx] = 1.0
        
        # Apply chroma filter
        chroma = torch.matmul(chroma_map, magnitude)
        
        return chroma
    
    def compute_statistics(self, tensor: torch.Tensor, prefix: str = "") -> Dict[str, float]:
        """
        Compute statistical features from tensor.
        
        Args:
            tensor: Input tensor
            prefix: Prefix for feature names
            
        Returns:
            Dictionary of statistical features
        """
        stats = {}
        
        if prefix:
            prefix = f"{prefix}_"
        
        # Basic statistics
        stats[f"{prefix}mean"] = float(torch.mean(tensor).item())
        stats[f"{prefix}std"] = float(torch.std(tensor).item())
        stats[f"{prefix}min"] = float(torch.min(tensor).item())
        stats[f"{prefix}max"] = float(torch.max(tensor).item())
        stats[f"{prefix}median"] = float(torch.median(tensor).item())
        
        # Percentiles
        stats[f"{prefix}p25"] = float(torch.quantile(tensor, 0.25).item())
        stats[f"{prefix}p75"] = float(torch.quantile(tensor, 0.75).item())
        stats[f"{prefix}p90"] = float(torch.quantile(tensor, 0.90).item())
        
        # Coefficient of variation
        mean_val = torch.mean(tensor)
        std_val = torch.std(tensor)
        stats[f"{prefix}cv"] = float((std_val / (mean_val + 1e-10)).item())
        
        return stats
    
    def compute_skewness(self, tensor: torch.Tensor) -> float:
        """Compute skewness of tensor."""
        if tensor.numel() < 3:
            return 0.0
        
        mean_val = torch.mean(tensor)
        std_val = torch.std(tensor)
        
        if std_val == 0:
            return 0.0
        
        skewness = torch.mean(((tensor - mean_val) / std_val) ** 3)
        return float(skewness.item())
    
    def compute_kurtosis(self, tensor: torch.Tensor) -> float:
        """Compute kurtosis of tensor."""
        if tensor.numel() < 4:
            return 0.0
        
        mean_val = torch.mean(tensor)
        std_val = torch.std(tensor)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = torch.mean(((tensor - mean_val) / std_val) ** 4) - 3
        return float(kurtosis.item())


# Global instance for easy access
_gpu_audio_processor = None

def get_gpu_audio_processor(device: str = "cuda", sample_rate: int = 22050) -> GPUAudioProcessor:
    """Get global GPU audio processor instance."""
    global _gpu_audio_processor
    
    if _gpu_audio_processor is None:
        _gpu_audio_processor = GPUAudioProcessor(device, sample_rate)
    
    return _gpu_audio_processor
