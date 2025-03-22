"""
Abstract interfaces for the data ingestion module.
Defines the contract for data readers, processors, and chunkers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, BinaryIO, Optional, Iterator
from pathlib import Path


class DataSourceReader(ABC):
    """Abstract interface for reading data from various sources."""
    
    @abstractmethod
    def read(self, source_path: str) -> bytes:
        """
        Read binary data from a source.
        
        Args:
            source_path: Path or URI to the data source
            
        Returns:
            Raw binary data from the source
        """
        pass
    
    @abstractmethod
    def get_metadata(self, source_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the data source.
        
        Args:
            source_path: Path or URI to the data source
            
        Returns:
            Dictionary containing metadata
        """
        pass


class DataProcessor(ABC):
    """Abstract interface for processing raw data into text."""
    
    @abstractmethod
    def process(self, data: bytes, metadata: Dict[str, Any]) -> str:
        """
        Process binary data into text.
        
        Args:
            data: Raw binary data
            metadata: Metadata about the data source
            
        Returns:
            Processed text
        """
        pass


class DataChunker(ABC):
    """Abstract interface for chunking processed text data."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks suitable for vector DB ingestion.
        
        Args:
            text: Processed text to chunk
            metadata: Metadata about the data source
            
        Returns:
            List of chunks with their metadata
        """
        pass


class IngestionPipeline(ABC):
    """Abstract interface for the complete ingestion pipeline."""
    
    @abstractmethod
    def ingest(self, source_path: str) -> List[Dict[str, Any]]:
        """
        Run the complete ingestion pipeline on a data source.
        
        Args:
            source_path: Path or URI to the data source
            
        Returns:
            List of processed chunks ready for storage
        """
        pass 