"""
Factory classes for creating data ingestion components.
"""
import os
import logging
from typing import Dict, Type, Optional

from .interfaces import DataSourceReader, DataProcessor, DataChunker
from .readers import (
    PDFReader, 
    ExcelReader, 
    PowerPointReader, 
    AudioReader, 
    TextReader, 
    SharePointReader,
    OneDriveReader
)
from .processors import (
    PDFProcessor,
    ExcelProcessor,
    PowerPointProcessor,
    AudioProcessor,
    TextProcessor,
    SharePointProcessor,
    OneDriveProcessor
)
from .chunkers import (
    TextChunker,
    TableChunker,
    SlideDeckChunker
)

logger = logging.getLogger(__name__)


class ReaderFactory:
    """Factory for creating appropriate data source readers."""
    
    _instance = None
    
    # File extension to reader class mapping
    _readers: Dict[str, Type[DataSourceReader]] = {
        # Document formats
        '.pdf': PDFReader,
        '.txt': TextReader,
        '.md': TextReader,
        '.csv': ExcelReader,
        '.xlsx': ExcelReader,
        '.xls': ExcelReader,
        '.pptx': PowerPointReader,
        '.ppt': PowerPointReader,
        
        # Audio formats
        '.mp3': AudioReader,
        '.wav': AudioReader,
        '.ogg': AudioReader,
        '.m4a': AudioReader,
        
        # Cloud storage formats (use URI patterns)
        'sharepoint://': SharePointReader,
        'onedrive://': OneDriveReader,
    }
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ReaderFactory, cls).__new__(cls)
        return cls._instance
    
    def get_reader(self, source_path: str) -> DataSourceReader:
        """
        Get the appropriate reader for a data source.
        
        Args:
            source_path: Path or URI to the data source
            
        Returns:
            Appropriate reader instance
            
        Raises:
            ValueError: If no reader is available for the file type
        """
        # Handle URI schemes
        if '://' in source_path:
            scheme = source_path.split('://')[0] + '://'
            for uri_prefix, reader_class in self._readers.items():
                if uri_prefix == scheme:
                    return reader_class()
        
        # Handle file extensions
        ext = os.path.splitext(source_path)[1].lower()
        if ext in self._readers:
            return self._readers[ext]()
        
        # Default to text reader for unknown types
        logger.warning(f"No specific reader found for {ext}, using TextReader as fallback")
        return TextReader()


class ProcessorFactory:
    """Factory for creating appropriate data processors."""
    
    _instance = None
    
    # Reader type to processor mapping
    _processors = {
        PDFReader: PDFProcessor,
        ExcelReader: ExcelProcessor,
        PowerPointReader: PowerPointProcessor,
        AudioReader: AudioProcessor,
        TextReader: TextProcessor,
        SharePointReader: SharePointProcessor,
        OneDriveReader: OneDriveProcessor,
    }
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ProcessorFactory, cls).__new__(cls)
        return cls._instance
    
    def get_processor(self, reader: DataSourceReader) -> DataProcessor:
        """
        Get the appropriate processor for a reader.
        
        Args:
            reader: The reader instance used to read the data
            
        Returns:
            Appropriate processor instance
            
        Raises:
            ValueError: If no processor is available for the reader
        """
        reader_class = reader.__class__
        
        if reader_class in self._processors:
            return self._processors[reader_class]()
        
        # Default to text processor
        logger.warning(f"No specific processor found for {reader_class}, using TextProcessor as fallback")
        return TextProcessor()


class ChunkerFactory:
    """Factory for creating appropriate chunkers."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ChunkerFactory, cls).__new__(cls)
        return cls._instance
    
    def get_chunker(self, metadata: Dict) -> DataChunker:
        """
        Get the appropriate chunker based on content metadata.
        
        Args:
            metadata: Metadata about the content
            
        Returns:
            Appropriate chunker instance
        """
        content_type = metadata.get('content_type', 'text')
        
        if content_type == 'table' or content_type == 'spreadsheet':
            return TableChunker()
        elif content_type == 'slides' or content_type == 'presentation':
            return SlideDeckChunker()
        else:
            return TextChunker() 