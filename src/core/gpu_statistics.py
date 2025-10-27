"""
GPU-accelerated statistical functions.

This module provides GPU-accelerated replacements for common statistical operations
using PyTorch and CuPy for maximum performance.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging

# Try to import CuPy for additional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUStatistics:
    """GPU-accelerated statistical functions."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU statistics processor.
        
        Args:
            device: Device to use for processing ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.cupy_available = CUPY_AVAILABLE and self.device == "cuda"
        
        logger.info(f"GPU Statistics initialized on device: {self.device}")
        logger.info(f"CuPy available: {self.cupy_available}")
    
    def to_tensor(self, data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """Convert data to PyTorch tensor on the specified device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        else:
            return torch.tensor(data).to(self.device)
    
    def to_cupy(self, data: Union[np.ndarray, List, torch.Tensor]) -> 'cp.ndarray':
        """Convert data to CuPy array (if available)."""
        if not self.cupy_available:
            raise RuntimeError("CuPy not available")
        
        if isinstance(data, cp.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return cp.asarray(data.cpu().numpy())
        else:
            return cp.asarray(data)
    
    def mean(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute mean using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.mean(tensor, dim=axis)
        return result.item() if result.numel() == 1 else result
    
    def std(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute standard deviation using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.std(tensor, dim=axis)
        return result.item() if result.numel() == 1 else result
    
    def var(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute variance using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.var(tensor, dim=axis)
        return result.item() if result.numel() == 1 else result
    
    def min(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute minimum using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.min(tensor, dim=axis)[0] if axis is not None else torch.min(tensor)
        return result.item() if result.numel() == 1 else result
    
    def max(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute maximum using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.max(tensor, dim=axis)[0] if axis is not None else torch.max(tensor)
        return result.item() if result.numel() == 1 else result
    
    def median(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute median using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.median(tensor, dim=axis)[0] if axis is not None else torch.median(tensor)
        return result.item() if result.numel() == 1 else result
    
    def percentile(self, data: Union[np.ndarray, List, torch.Tensor], q: float, axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute percentile using GPU acceleration."""
        tensor = self.to_tensor(data)
        result = torch.quantile(tensor, q / 100.0, dim=axis)
        return result.item() if result.numel() == 1 else result
    
    def skewness(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute skewness using GPU acceleration."""
        tensor = self.to_tensor(data)
        
        if axis is not None:
            # Compute skewness along specified axis
            mean_val = torch.mean(tensor, dim=axis, keepdim=True)
            std_val = torch.std(tensor, dim=axis, keepdim=True)
            
            if std_val.numel() == 1 and std_val.item() == 0:
                return torch.zeros_like(mean_val.squeeze(axis))
            
            normalized = (tensor - mean_val) / (std_val + 1e-10)
            skewness = torch.mean(normalized ** 3, dim=axis)
        else:
            # Compute skewness for entire tensor
            mean_val = torch.mean(tensor)
            std_val = torch.std(tensor)
            
            if std_val == 0:
                return 0.0
            
            normalized = (tensor - mean_val) / (std_val + 1e-10)
            skewness = torch.mean(normalized ** 3)
        
        return skewness.item() if skewness.numel() == 1 else skewness
    
    def kurtosis(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute kurtosis using GPU acceleration."""
        tensor = self.to_tensor(data)
        
        if axis is not None:
            # Compute kurtosis along specified axis
            mean_val = torch.mean(tensor, dim=axis, keepdim=True)
            std_val = torch.std(tensor, dim=axis, keepdim=True)
            
            if std_val.numel() == 1 and std_val.item() == 0:
                return torch.zeros_like(mean_val.squeeze(axis))
            
            normalized = (tensor - mean_val) / (std_val + 1e-10)
            kurtosis = torch.mean(normalized ** 4, dim=axis) - 3
        else:
            # Compute kurtosis for entire tensor
            mean_val = torch.mean(tensor)
            std_val = torch.std(tensor)
            
            if std_val == 0:
                return 0.0
            
            normalized = (tensor - mean_val) / (std_val + 1e-10)
            kurtosis = torch.mean(normalized ** 4) - 3
        
        return kurtosis.item() if kurtosis.numel() == 1 else kurtosis
    
    def correlation(self, x: Union[np.ndarray, List, torch.Tensor], y: Union[np.ndarray, List, torch.Tensor]) -> float:
        """Compute correlation coefficient using GPU acceleration."""
        x_tensor = self.to_tensor(x)
        y_tensor = self.to_tensor(y)
        
        # Ensure same shape
        if x_tensor.shape != y_tensor.shape:
            raise ValueError("Input arrays must have the same shape")
        
        # Compute correlation
        x_centered = x_tensor - torch.mean(x_tensor)
        y_centered = y_tensor - torch.mean(y_tensor)
        
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation.item()
    
    def correlation_matrix(self, data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """Compute correlation matrix using GPU acceleration."""
        tensor = self.to_tensor(data)
        
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        # Center the data
        centered = tensor - torch.mean(tensor, dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.mm(centered.T, centered) / (tensor.shape[0] - 1)
        
        # Compute standard deviations
        std_devs = torch.sqrt(torch.diag(cov))
        
        # Compute correlation matrix
        correlation = cov / torch.outer(std_devs, std_devs)
        
        return correlation
    
    def entropy(self, data: Union[np.ndarray, List, torch.Tensor], axis: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Compute entropy using GPU acceleration."""
        tensor = self.to_tensor(data)
        
        # Ensure non-negative values
        tensor = torch.abs(tensor) + 1e-10
        
        if axis is not None:
            # Normalize along specified axis
            tensor_norm = tensor / torch.sum(tensor, dim=axis, keepdim=True)
            # Compute entropy
            entropy = -torch.sum(tensor_norm * torch.log2(tensor_norm + 1e-10), dim=axis)
        else:
            # Normalize entire tensor
            tensor_norm = tensor / torch.sum(tensor)
            # Compute entropy
            entropy = -torch.sum(tensor_norm * torch.log2(tensor_norm + 1e-10))
        
        return entropy.item() if entropy.numel() == 1 else entropy
    
    def mutual_information(self, x: Union[np.ndarray, List, torch.Tensor], y: Union[np.ndarray, List, torch.Tensor], bins: int = 10) -> float:
        """Compute mutual information using GPU acceleration."""
        x_tensor = self.to_tensor(x)
        y_tensor = self.to_tensor(y)
        
        # Ensure same shape
        if x_tensor.shape != y_tensor.shape:
            raise ValueError("Input arrays must have the same shape")
        
        # Create bins
        x_min, x_max = torch.min(x_tensor), torch.max(x_tensor)
        y_min, y_max = torch.min(y_tensor), torch.max(y_tensor)
        
        x_bins = torch.linspace(x_min, x_max, bins + 1, device=self.device)
        y_bins = torch.linspace(y_min, y_max, bins + 1, device=self.device)
        
        # Digitize data
        x_digitized = torch.bucketize(x_tensor, x_bins) - 1
        y_digitized = torch.bucketize(y_tensor, y_bins) - 1
        
        # Compute joint histogram
        joint_hist = torch.zeros(bins, bins, device=self.device)
        for i in range(bins):
            for j in range(bins):
                joint_hist[i, j] = torch.sum((x_digitized == i) & (y_digitized == j))
        
        # Normalize
        joint_hist = joint_hist / torch.sum(joint_hist)
        
        # Compute marginal distributions
        p_x = torch.sum(joint_hist, dim=1)
        p_y = torch.sum(joint_hist, dim=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint_hist[i, j] * torch.log2(joint_hist[i, j] / (p_x[i] * p_y[j]))
        
        return mi.item()
    
    def compute_all_statistics(self, data: Union[np.ndarray, List, torch.Tensor], prefix: str = "") -> Dict[str, Any]:
        """Compute all basic statistics for data."""
        tensor = self.to_tensor(data)
        
        if prefix:
            prefix = f"{prefix}_"
        
        stats = {}
        
        # Basic statistics
        stats[f"{prefix}mean"] = self.mean(tensor)
        stats[f"{prefix}std"] = self.std(tensor)
        stats[f"{prefix}var"] = self.var(tensor)
        stats[f"{prefix}min"] = self.min(tensor)
        stats[f"{prefix}max"] = self.max(tensor)
        stats[f"{prefix}median"] = self.median(tensor)
        
        # Percentiles
        stats[f"{prefix}p25"] = self.percentile(tensor, 25)
        stats[f"{prefix}p75"] = self.percentile(tensor, 75)
        stats[f"{prefix}p90"] = self.percentile(tensor, 90)
        stats[f"{prefix}p95"] = self.percentile(tensor, 95)
        stats[f"{prefix}p99"] = self.percentile(tensor, 99)
        
        # Higher moments
        stats[f"{prefix}skewness"] = self.skewness(tensor)
        stats[f"{prefix}kurtosis"] = self.kurtosis(tensor)
        
        # Additional features
        mean_val = self.mean(tensor)
        std_val = self.std(tensor)
        stats[f"{prefix}cv"] = std_val / (abs(mean_val) + 1e-10)  # Coefficient of variation
        stats[f"{prefix}range"] = self.max(tensor) - self.min(tensor)
        stats[f"{prefix}iqr"] = self.percentile(tensor, 75) - self.percentile(tensor, 25)
        
        # Entropy
        stats[f"{prefix}entropy"] = self.entropy(tensor)
        
        return stats
    
    def compute_rolling_statistics(self, data: Union[np.ndarray, List, torch.Tensor], window_size: int, 
                                 statistics: List[str] = None) -> Dict[str, torch.Tensor]:
        """Compute rolling statistics using GPU acceleration."""
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max']
        
        tensor = self.to_tensor(data)
        
        if tensor.dim() > 1:
            raise ValueError("Rolling statistics only supported for 1D data")
        
        results = {}
        
        for stat in statistics:
            if stat == 'mean':
                results[stat] = self._rolling_mean(tensor, window_size)
            elif stat == 'std':
                results[stat] = self._rolling_std(tensor, window_size)
            elif stat == 'min':
                results[stat] = self._rolling_min(tensor, window_size)
            elif stat == 'max':
                results[stat] = self._rolling_max(tensor, window_size)
            else:
                logger.warning(f"Unknown statistic: {stat}")
        
        return results
    
    def _rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Compute rolling mean."""
        return torch.nn.functional.avg_pool1d(
            tensor.unsqueeze(0).unsqueeze(0), 
            kernel_size=window_size, 
            stride=1, 
            padding=window_size//2
        ).squeeze(0).squeeze(0)
    
    def _rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Compute rolling standard deviation."""
        rolling_mean = self._rolling_mean(tensor, window_size)
        
        # Pad tensor for proper rolling computation
        padded = torch.nn.functional.pad(tensor, (window_size//2, window_size//2), mode='replicate')
        
        # Compute rolling variance
        rolling_var = torch.nn.functional.avg_pool1d(
            (padded - rolling_mean) ** 2, 
            kernel_size=window_size, 
            stride=1, 
            padding=0
        )
        
        return torch.sqrt(rolling_var)
    
    def _rolling_min(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Compute rolling minimum."""
        return torch.nn.functional.max_pool1d(
            -tensor.unsqueeze(0).unsqueeze(0), 
            kernel_size=window_size, 
            stride=1, 
            padding=window_size//2
        ).squeeze(0).squeeze(0) * -1
    
    def _rolling_max(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Compute rolling maximum."""
        return torch.nn.functional.max_pool1d(
            tensor.unsqueeze(0).unsqueeze(0), 
            kernel_size=window_size, 
            stride=1, 
            padding=window_size//2
        ).squeeze(0).squeeze(0)


# Global instance for easy access
_gpu_statistics = None

def get_gpu_statistics(device: str = "cuda") -> GPUStatistics:
    """Get global GPU statistics instance."""
    global _gpu_statistics
    
    if _gpu_statistics is None:
        _gpu_statistics = GPUStatistics(device)
    
    return _gpu_statistics
