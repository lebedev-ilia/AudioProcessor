"""
Base extractor interface for AudioProcessor.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import time
import logging
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for all audio feature extractors."""
    
    # These should be overridden in subclasses
    name: str = "base_extractor"
    version: str = "0.1.0"
    description: str = "Base extractor"
    category: str = "core"
    dependencies: list = []
    estimated_duration: float = 1.0
    
    def __init__(self):
        """Initialize the extractor."""
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abstractmethod
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Run the extractor on the given input.
        
        Args:
            input_uri: URI to the input audio file
            tmp_path: Path to temporary directory for processing
            
        Returns:
            ExtractorResult with extracted features or error information
        """
        pass
    
    def _create_result(
        self, 
        success: bool, 
        payload: Dict[str, Any] = None, 
        error: str = None,
        processing_time: float = None
    ) -> ExtractorResult:
        """
        Create an ExtractorResult object.
        
        Args:
            success: Whether extraction was successful
            payload: Extracted features data
            error: Error message if failed
            processing_time: Processing time in seconds
            
        Returns:
            ExtractorResult object
        """
        return ExtractorResult(
            name=self.name,
            version=self.version,
            success=success,
            payload=payload,
            error=error,
            processing_time=processing_time
        )
    
    def _time_execution(self, func, *args, **kwargs):
        """
        Time the execution of a function.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    
    def _validate_input(self, input_uri: str) -> bool:
        """
        Validate input URI.
        
        Args:
            input_uri: URI to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not input_uri:
            self.logger.error("Input URI is empty")
            return False
        
        # Accept both S3 URIs and local file paths for testing
        if not (input_uri.startswith("s3://") or input_uri.startswith("/") or input_uri.endswith(('.wav', '.mp3', '.flac'))):
            self.logger.error(f"Invalid URI format: {input_uri}")
            return False
        
        return True
    
    def _log_extraction_start(self, input_uri: str):
        """Log the start of extraction."""
        self.logger.info(f"Starting {self.name} extraction for {input_uri}")
    
    def _log_extraction_success(self, input_uri: str, processing_time: float):
        """Log successful extraction."""
        self.logger.info(
            f"Successfully completed {self.name} extraction for {input_uri} "
            f"in {processing_time:.2f}s"
        )
    
    def _log_extraction_error(self, input_uri: str, error: str, processing_time: float):
        """Log extraction error."""
        self.logger.error(
            f"Failed {self.name} extraction for {input_uri} "
            f"after {processing_time:.2f}s: {error}"
        )
    
    def get_info(self) -> Dict[str, str]:
        """
        Get extractor information.
        
        Returns:
            Dictionary with extractor metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description
        }
