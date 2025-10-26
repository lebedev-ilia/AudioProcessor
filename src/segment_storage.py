"""
Storage module for per-segment audio features.

This module provides functionality for saving and loading per-video
segment features in various formats (npy+json, TFRecord, LMDB).
"""

import os
import json
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
from .segment_config import SegmentConfig
from .segment_selector import create_attention_mask, pad_segments_to_max_len, extract_features_from_segment

logger = logging.getLogger(__name__)


class SegmentStorage:
    """Storage handler for per-segment features."""
    
    def __init__(self, config: SegmentConfig):
        """
        Initialize segment storage.
        
        Args:
            config: Configuration containing storage parameters
        """
        self.config = config
        self.output_dir = config.output_dir
        self.storage_format = config.storage_format
        self.dtype = config.dtype
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized SegmentStorage: format={self.storage_format}, "
                   f"output_dir={self.output_dir}")
    
    def save_per_video(
        self,
        video_id: str,
        selected_segments: List[Dict[str, Any]],
        feature_vectors: np.ndarray,
        attention_mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save per-video segment features.
        
        Args:
            video_id: Video identifier
            selected_segments: List of selected segment features
            feature_vectors: Feature matrix (num_segments, feature_dim)
            attention_mask: Attention mask (num_segments,)
            metadata: Additional metadata
            
        Returns:
            Dictionary with file paths
        """
        if self.storage_format == "npy+json":
            return self._save_npy_json(video_id, selected_segments, feature_vectors, attention_mask, metadata)
        elif self.storage_format == "tfrecord":
            return self._save_tfrecord(video_id, selected_segments, feature_vectors, attention_mask, metadata)
        elif self.storage_format == "lmdb":
            return self._save_lmdb(video_id, selected_segments, feature_vectors, attention_mask, metadata)
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
    
    def _save_npy_json(
        self,
        video_id: str,
        selected_segments: List[Dict[str, Any]],
        feature_vectors: np.ndarray,
        attention_mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Save features in npy+json format."""
        try:
            # Save feature vectors
            features_file = os.path.join(self.output_dir, f"{video_id}_features.npy")
            np.save(features_file, feature_vectors.astype(self.dtype))
            
            # Save attention mask
            mask_file = os.path.join(self.output_dir, f"{video_id}_mask.npy")
            np.save(mask_file, attention_mask.astype(np.uint8))
            
            # Create metadata
            meta = {
                "video_id": video_id,
                "num_segments": feature_vectors.shape[0],
                "feature_dim": feature_vectors.shape[1],
                "segment_times": [(s.get('segment_start', 0), s.get('segment_end', 0)) for s in selected_segments],
                "segment_indices": [s.get('segment_index', i) for i, s in enumerate(selected_segments)],
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "storage_format": self.storage_format,
                "dtype": self.dtype
            }
            
            # Add additional metadata
            if metadata:
                meta.update(metadata)
            
            # Save metadata
            meta_file = os.path.join(self.output_dir, f"{video_id}_meta.json")
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)
            
            logger.info(f"Saved {video_id} features: {feature_vectors.shape}")
            
            return {
                "features_file": features_file,
                "mask_file": mask_file,
                "meta_file": meta_file
            }
            
        except Exception as e:
            logger.error(f"Failed to save {video_id} in npy+json format: {e}")
            raise
    
    def _save_tfrecord(
        self,
        video_id: str,
        selected_segments: List[Dict[str, Any]],
        feature_vectors: np.ndarray,
        attention_mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Save features in TFRecord format."""
        try:
            import tensorflow as tf
            
            # Create TFRecord file
            tfrecord_file = os.path.join(self.output_dir, f"{video_id}.tfrecord")
            
            with tf.io.TFRecordWriter(tfrecord_file) as writer:
                # Create example
                example = self._create_tfrecord_example(
                    video_id, selected_segments, feature_vectors, attention_mask, metadata
                )
                writer.write(example.SerializeToString())
            
            logger.info(f"Saved {video_id} features in TFRecord format")
            
            return {"tfrecord_file": tfrecord_file}
            
        except ImportError:
            logger.error("TensorFlow not available for TFRecord format")
            raise
        except Exception as e:
            logger.error(f"Failed to save {video_id} in TFRecord format: {e}")
            raise
    
    def _save_lmdb(
        self,
        video_id: str,
        selected_segments: List[Dict[str, Any]],
        feature_vectors: np.ndarray,
        attention_mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Save features in LMDB format."""
        try:
            import lmdb
            
            # Create LMDB file
            lmdb_file = os.path.join(self.output_dir, f"{video_id}.lmdb")
            
            # Create environment
            env = lmdb.open(lmdb_file, map_size=1024**4)  # 1GB map size
            
            with env.begin(write=True) as txn:
                # Save feature vectors
                features_key = f"{video_id}_features".encode()
                features_bytes = feature_vectors.astype(self.dtype).tobytes()
                txn.put(features_key, features_bytes)
                
                # Save attention mask
                mask_key = f"{video_id}_mask".encode()
                mask_bytes = attention_mask.astype(np.uint8).tobytes()
                txn.put(mask_key, mask_bytes)
                
                # Save metadata
                meta_key = f"{video_id}_meta".encode()
                meta = {
                    "video_id": video_id,
                    "num_segments": feature_vectors.shape[0],
                    "feature_dim": feature_vectors.shape[1],
                    "segment_times": [(s.get('segment_start', 0), s.get('segment_end', 0)) for s in selected_segments],
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "storage_format": self.storage_format,
                    "dtype": self.dtype
                }
                if metadata:
                    meta.update(metadata)
                
                meta_bytes = json.dumps(meta).encode()
                txn.put(meta_key, meta_bytes)
            
            env.close()
            
            logger.info(f"Saved {video_id} features in LMDB format")
            
            return {"lmdb_file": lmdb_file}
            
        except ImportError:
            logger.error("LMDB not available")
            raise
        except Exception as e:
            logger.error(f"Failed to save {video_id} in LMDB format: {e}")
            raise
    
    def _create_tfrecord_example(
        self,
        video_id: str,
        selected_segments: List[Dict[str, Any]],
        feature_vectors: np.ndarray,
        attention_mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'tf.train.Example':
        """Create TFRecord example."""
        import tensorflow as tf
        
        # Create feature dictionary
        feature_dict = {
            'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id.encode()])),
            'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_vectors.tobytes()])),
            'attention_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[attention_mask.tobytes()])),
            'num_segments': tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_vectors.shape[0]])),
            'feature_dim': tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_vectors.shape[1]]))
        }
        
        # Add segment times
        segment_times = [(s.get('segment_start', 0), s.get('segment_end', 0)) for s in selected_segments]
        feature_dict['segment_times'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[t for pair in segment_times for t in pair])
        )
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    feature_dict[f'meta_{key}'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value.encode()])
                    )
                elif isinstance(value, (int, float)):
                    feature_dict[f'meta_{key}'] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(value)])
                    )
        
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    def load_per_video(self, video_id: str) -> Dict[str, Any]:
        """
        Load per-video segment features.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Dictionary with loaded data
        """
        if self.storage_format == "npy+json":
            return self._load_npy_json(video_id)
        elif self.storage_format == "tfrecord":
            return self._load_tfrecord(video_id)
        elif self.storage_format == "lmdb":
            return self._load_lmdb(video_id)
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
    
    def _load_npy_json(self, video_id: str) -> Dict[str, Any]:
        """Load features from npy+json format."""
        try:
            # Load feature vectors
            features_file = os.path.join(self.output_dir, f"{video_id}_features.npy")
            feature_vectors = np.load(features_file)
            
            # Load attention mask
            mask_file = os.path.join(self.output_dir, f"{video_id}_mask.npy")
            attention_mask = np.load(mask_file)
            
            # Load metadata
            meta_file = os.path.join(self.output_dir, f"{video_id}_meta.json")
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            return {
                "feature_vectors": feature_vectors,
                "attention_mask": attention_mask,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to load {video_id} from npy+json format: {e}")
            raise
    
    def _load_tfrecord(self, video_id: str) -> Dict[str, Any]:
        """Load features from TFRecord format."""
        try:
            import tensorflow as tf
            
            tfrecord_file = os.path.join(self.output_dir, f"{video_id}.tfrecord")
            
            # Parse TFRecord
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            
            for raw_record in raw_dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Extract features
                features = example.features.feature
                
                # Parse feature vectors
                features_bytes = features['features'].bytes_list.value[0]
                feature_vectors = np.frombuffer(features_bytes, dtype=self.dtype)
                
                # Parse attention mask
                mask_bytes = features['attention_mask'].bytes_list.value[0]
                attention_mask = np.frombuffer(mask_bytes, dtype=np.uint8)
                
                # Parse metadata
                metadata = {
                    'video_id': features['video_id'].bytes_list.value[0].decode(),
                    'num_segments': features['num_segments'].int64_list.value[0],
                    'feature_dim': features['feature_dim'].int64_list.value[0]
                }
                
                return {
                    "feature_vectors": feature_vectors,
                    "attention_mask": attention_mask,
                    "metadata": metadata
                }
            
            raise ValueError(f"No data found for {video_id}")
            
        except ImportError:
            logger.error("TensorFlow not available for TFRecord format")
            raise
        except Exception as e:
            logger.error(f"Failed to load {video_id} from TFRecord format: {e}")
            raise
    
    def _load_lmdb(self, video_id: str) -> Dict[str, Any]:
        """Load features from LMDB format."""
        try:
            import lmdb
            
            lmdb_file = os.path.join(self.output_dir, f"{video_id}.lmdb")
            
            env = lmdb.open(lmdb_file, readonly=True)
            
            with env.begin() as txn:
                # Load feature vectors
                features_key = f"{video_id}_features".encode()
                features_bytes = txn.get(features_key)
                if features_bytes is None:
                    raise ValueError(f"No features found for {video_id}")
                
                feature_vectors = np.frombuffer(features_bytes, dtype=self.dtype)
                
                # Load attention mask
                mask_key = f"{video_id}_mask".encode()
                mask_bytes = txn.get(mask_key)
                if mask_bytes is None:
                    raise ValueError(f"No mask found for {video_id}")
                
                attention_mask = np.frombuffer(mask_bytes, dtype=np.uint8)
                
                # Load metadata
                meta_key = f"{video_id}_meta".encode()
                meta_bytes = txn.get(meta_key)
                if meta_bytes is None:
                    raise ValueError(f"No metadata found for {video_id}")
                
                metadata = json.loads(meta_bytes.decode())
            
            env.close()
            
            return {
                "feature_vectors": feature_vectors,
                "attention_mask": attention_mask,
                "metadata": metadata
            }
            
        except ImportError:
            logger.error("LMDB not available")
            raise
        except Exception as e:
            logger.error(f"Failed to load {video_id} from LMDB format: {e}")
            raise
    
    def list_videos(self) -> List[str]:
        """
        List all video IDs in the storage.
        
        Returns:
            List of video IDs
        """
        video_ids = set()
        
        if self.storage_format == "npy+json":
            for filename in os.listdir(self.output_dir):
                if filename.endswith('_features.npy'):
                    video_id = filename.replace('_features.npy', '')
                    video_ids.add(video_id)
        elif self.storage_format == "tfrecord":
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.tfrecord'):
                    video_id = filename.replace('.tfrecord', '')
                    video_ids.add(video_id)
        elif self.storage_format == "lmdb":
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.lmdb'):
                    video_id = filename.replace('.lmdb', '')
                    video_ids.add(video_id)
        
        return sorted(list(video_ids))
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        video_ids = self.list_videos()
        
        total_size = 0
        for video_id in video_ids:
            if self.storage_format == "npy+json":
                features_file = os.path.join(self.output_dir, f"{video_id}_features.npy")
                mask_file = os.path.join(self.output_dir, f"{video_id}_mask.npy")
                meta_file = os.path.join(self.output_dir, f"{video_id}_meta.json")
                
                for file_path in [features_file, mask_file, meta_file]:
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            else:
                # For other formats, estimate size
                file_pattern = f"{video_id}.{self.storage_format}"
                file_path = os.path.join(self.output_dir, file_pattern)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        return {
            "num_videos": len(video_ids),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_format": self.storage_format,
            "output_dir": self.output_dir
        }


def save_artifacts(
    config: SegmentConfig,
    pca_models: Dict[str, Any],
    scalers: Dict[str, Any]
) -> Dict[str, str]:
    """
    Save processing artifacts (PCA models, scalers, config).
    
    Args:
        config: Configuration object
        pca_models: Dictionary of PCA models
        scalers: Dictionary of scalers
        
    Returns:
        Dictionary with artifact file paths
    """
    artifacts_dir = config.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    
    artifact_paths = {}
    
    try:
        # Save PCA models
        for model_name, model in pca_models.items():
            model_path = os.path.join(artifacts_dir, f"pca_{model_name}.joblib")
            joblib.dump(model, model_path)
            artifact_paths[f"pca_{model_name}"] = model_path
        
        # Save scalers
        for scaler_name, scaler in scalers.items():
            scaler_path = os.path.join(artifacts_dir, f"scaler_{scaler_name}.joblib")
            joblib.dump(scaler, scaler_path)
            artifact_paths[f"scaler_{scaler_name}"] = scaler_path
        
        # Save configuration
        config_path = os.path.join(artifacts_dir, "segment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        artifact_paths["config"] = config_path
        
        logger.info(f"Saved {len(artifact_paths)} artifacts to {artifacts_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save artifacts: {e}")
        raise
    
    return artifact_paths
