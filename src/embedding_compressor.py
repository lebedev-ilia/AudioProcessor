"""
Embedding compression module for reducing dimensionality of large embeddings.

This module provides PCA-based compression for high-dimensional embeddings
like CLAP, wav2vec, and YAMNet to make them suitable for transformer models.
"""

import numpy as np
import joblib
import os
from typing import Dict, Any, List, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from .segment_config import SegmentConfig

logger = logging.getLogger(__name__)


class EmbeddingCompressor:
    """Compressor for high-dimensional embeddings using PCA."""
    
    def __init__(self, config: SegmentConfig):
        """
        Initialize embedding compressor.
        
        Args:
            config: Configuration containing PCA dimensions
        """
        self.config = config
        self.pca_dims = config.pca_dims
        self.artifacts_dir = config.artifacts_dir
        
        # PCA models for different embedding types
        self.pca_models = {}
        self.scalers = {}
        
        # Ensure artifacts directory exists
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        logger.info(f"Initialized EmbeddingCompressor with PCA dims: {self.pca_dims}")
    
    def fit_pca_models(
        self, 
        embedding_data: Dict[str, List[np.ndarray]],
        save_models: bool = True
    ) -> Dict[str, PCA]:
        """
        Fit PCA models on training data.
        
        Args:
            embedding_data: Dictionary with embedding type -> list of arrays
            save_models: Whether to save fitted models to disk
            
        Returns:
            Dictionary of fitted PCA models
        """
        fitted_models = {}
        
        for embedding_type, embedding_list in embedding_data.items():
            if embedding_type not in self.pca_dims:
                logger.warning(f"No PCA dimension specified for {embedding_type}, skipping")
                continue
            
            if not embedding_list:
                logger.warning(f"No data provided for {embedding_type}, skipping")
                continue
            
            target_dim = self.pca_dims[embedding_type]
            logger.info(f"Fitting PCA for {embedding_type}: {len(embedding_list)} samples -> {target_dim} dims")
            
            try:
                # Stack all embeddings
                X = np.vstack(embedding_list).astype(np.float32)
                original_dim = X.shape[1]
                
                logger.info(f"Stacked embeddings shape: {X.shape}")
                
                # Fit scaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit PCA
                pca = PCA(
                    n_components=min(target_dim, original_dim),
                    svd_solver='auto',
                    random_state=42
                )
                X_transformed = pca.fit_transform(X_scaled)
                
                # Store models
                self.pca_models[embedding_type] = pca
                self.scalers[embedding_type] = scaler
                fitted_models[embedding_type] = pca
                
                # Log explained variance
                explained_var = np.sum(pca.explained_variance_ratio_)
                logger.info(f"PCA for {embedding_type}: {explained_var:.3f} variance explained")
                
                # Save models if requested
                if save_models:
                    self._save_models(embedding_type, pca, scaler)
                
            except Exception as e:
                logger.error(f"Failed to fit PCA for {embedding_type}: {e}")
                continue
        
        logger.info(f"Fitted {len(fitted_models)} PCA models")
        return fitted_models
    
    def load_pca_models(self) -> Dict[str, bool]:
        """
        Load pre-trained PCA models from disk.
        
        Returns:
            Dictionary indicating which models were successfully loaded
        """
        loaded_models = {}
        
        for embedding_type in self.pca_dims.keys():
            try:
                pca_path = os.path.join(self.artifacts_dir, f"pca_{embedding_type}.joblib")
                scaler_path = os.path.join(self.artifacts_dir, f"scaler_{embedding_type}.joblib")
                
                if os.path.exists(pca_path) and os.path.exists(scaler_path):
                    pca = joblib.load(pca_path)
                    scaler = joblib.load(scaler_path)
                    
                    self.pca_models[embedding_type] = pca
                    self.scalers[embedding_type] = scaler
                    loaded_models[embedding_type] = True
                    
                    logger.info(f"Loaded PCA model for {embedding_type}")
                else:
                    loaded_models[embedding_type] = False
                    logger.warning(f"PCA model files not found for {embedding_type}")
                    
            except Exception as e:
                logger.error(f"Failed to load PCA model for {embedding_type}: {e}")
                loaded_models[embedding_type] = False
        
        return loaded_models
    
    def compress_embedding(
        self, 
        embedding: Union[np.ndarray, List[float]], 
        embedding_type: str
    ) -> Optional[np.ndarray]:
        """
        Compress a single embedding using fitted PCA model.
        
        Args:
            embedding: Input embedding (array or list)
            embedding_type: Type of embedding (clap, wav2vec, yamnet)
            
        Returns:
            Compressed embedding or None if compression fails
        """
        if embedding_type not in self.pca_models:
            logger.warning(f"No PCA model available for {embedding_type}")
            return None
        
        if embedding is None:
            return None
        
        try:
            # Convert to numpy array
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # Ensure 2D array
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Get models
            pca = self.pca_models[embedding_type]
            scaler = self.scalers[embedding_type]
            
            # Scale and transform
            embedding_scaled = scaler.transform(embedding)
            embedding_compressed = pca.transform(embedding_scaled)
            
            # Return as 1D array
            return embedding_compressed.flatten()
            
        except Exception as e:
            logger.error(f"Failed to compress {embedding_type} embedding: {e}")
            return None
    
    def compress_embeddings_batch(
        self, 
        embeddings: List[Union[np.ndarray, List[float]]], 
        embedding_type: str
    ) -> List[Optional[np.ndarray]]:
        """
        Compress a batch of embeddings.
        
        Args:
            embeddings: List of input embeddings
            embedding_type: Type of embedding
            
        Returns:
            List of compressed embeddings (None for failed compressions)
        """
        compressed = []
        
        for embedding in embeddings:
            compressed_emb = self.compress_embedding(embedding, embedding_type)
            compressed.append(compressed_emb)
        
        return compressed
    
    def _save_models(self, embedding_type: str, pca: PCA, scaler: StandardScaler):
        """Save PCA model and scaler to disk."""
        try:
            pca_path = os.path.join(self.artifacts_dir, f"pca_{embedding_type}.joblib")
            scaler_path = os.path.join(self.artifacts_dir, f"scaler_{embedding_type}.joblib")
            
            joblib.dump(pca, pca_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Saved PCA model for {embedding_type} to {pca_path}")
            
        except Exception as e:
            logger.error(f"Failed to save PCA model for {embedding_type}: {e}")
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about fitted PCA models.
        
        Returns:
            Dictionary with model information
        """
        info = {}
        
        for embedding_type, pca in self.pca_models.items():
            info[embedding_type] = {
                "n_components": pca.n_components_,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
                "mean": pca.mean_.tolist() if hasattr(pca, 'mean_') else None,
                "components_shape": pca.components_.shape
            }
        
        return info


def collect_embeddings_from_extractors(
    extractor_results: List[Dict[str, Any]],
    embedding_types: List[str] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Collect embeddings from extractor results for PCA training.
    
    Args:
        extractor_results: List of extractor result dictionaries
        embedding_types: List of embedding types to collect (None for all)
        
    Returns:
        Dictionary mapping embedding type to list of embedding arrays
    """
    if embedding_types is None:
        embedding_types = ['clap', 'wav2vec', 'yamnet']
    
    collected_embeddings = {emb_type: [] for emb_type in embedding_types}
    
    for result in extractor_results:
        # Extract CLAP embeddings
        if 'clap' in embedding_types:
            clap_result = result.get('clap_extractor')
            if clap_result and clap_result.get('success'):
                payload = clap_result.get('payload', {})
                clap_embedding = payload.get('clap_embedding')
                if clap_embedding is not None:
                    collected_embeddings['clap'].append(np.array(clap_embedding))
        
        # Extract advanced embeddings
        if any(emb_type in ['wav2vec', 'yamnet'] for emb_type in embedding_types):
            adv_result = result.get('advanced_embeddings')
            if adv_result and adv_result.get('success'):
                payload = adv_result.get('payload', {})
                
                if 'wav2vec' in embedding_types:
                    wav2vec_embeddings = payload.get('wav2vec_embeddings')
                    if wav2vec_embeddings is not None:
                        wav2vec_array = np.array(wav2vec_embeddings)
                        if wav2vec_array.ndim == 2:
                            # Multiple embeddings - take mean
                            collected_embeddings['wav2vec'].append(np.mean(wav2vec_array, axis=0))
                        else:
                            collected_embeddings['wav2vec'].append(wav2vec_array)
                
                if 'yamnet' in embedding_types:
                    yamnet_embeddings = payload.get('yamnet_embeddings')
                    if yamnet_embeddings is not None:
                        yamnet_array = np.array(yamnet_embeddings)
                        if yamnet_array.ndim == 2:
                            # Multiple embeddings - take mean
                            collected_embeddings['yamnet'].append(np.mean(yamnet_array, axis=0))
                        else:
                            collected_embeddings['yamnet'].append(yamnet_array)
    
    # Log collection results
    for emb_type, embeddings in collected_embeddings.items():
        logger.info(f"Collected {len(embeddings)} {emb_type} embeddings")
    
    return collected_embeddings


def compress_segment_embeddings(
    segment_features: List[Dict[str, Any]],
    compressor: EmbeddingCompressor
) -> List[Dict[str, Any]]:
    """
    Compress embeddings in segment features.
    
    Args:
        segment_features: List of segment feature dictionaries
        compressor: Fitted embedding compressor
        
    Returns:
        List of segment features with compressed embeddings
    """
    compressed_features = []
    
    for features in segment_features:
        compressed_feat = features.copy()
        
        # Compress CLAP embeddings
        if 'clap_mean' in features and features['clap_mean'] is not None:
            # For now, use mean CLAP embedding (in future, could use per-segment CLAP)
            clap_compressed = compressor.compress_embedding(
                features['clap_mean'], 'clap'
            )
            if clap_compressed is not None:
                compressed_feat['clap_pca'] = clap_compressed.tolist()
            else:
                compressed_feat['clap_pca'] = None
        
        # Compress wav2vec embeddings
        if 'wav2vec_mean' in features and features['wav2vec_mean'] is not None:
            wav2vec_compressed = compressor.compress_embedding(
                features['wav2vec_mean'], 'wav2vec'
            )
            if wav2vec_compressed is not None:
                compressed_feat['wav2vec_pca'] = wav2vec_compressed.tolist()
            else:
                compressed_feat['wav2vec_pca'] = None
        
        # Compress yamnet embeddings
        if 'yamnet_mean' in features and features['yamnet_mean'] is not None:
            yamnet_compressed = compressor.compress_embedding(
                features['yamnet_mean'], 'yamnet'
            )
            if yamnet_compressed is not None:
                compressed_feat['yamnet_pca'] = yamnet_compressed.tolist()
            else:
                compressed_feat['yamnet_pca'] = None
        
        compressed_features.append(compressed_feat)
    
    logger.info(f"Compressed embeddings for {len(compressed_features)} segments")
    return compressed_features
