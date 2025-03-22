"""
Complete data ingestion pipeline implementation.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from .interfaces import IngestionPipeline
from .factory import ReaderFactory, ProcessorFactory, ChunkerFactory

logger = logging.getLogger(__name__)


class StandardIngestionPipeline(IngestionPipeline):
    """Standard implementation of the data ingestion pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the pipeline.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize factories (singletons)
        self.reader_factory = ReaderFactory()
        self.processor_factory = ProcessorFactory()
        self.chunker_factory = ChunkerFactory()
    
    def ingest(self, source_path: str) -> List[Dict[str, Any]]:
        """
        Run the complete ingestion pipeline on a data source.
        
        Args:
            source_path: Path or URI to the data source
            
        Returns:
            List of processed chunks ready for storage
        """
        logger.info(f"Starting ingestion pipeline for: {source_path}")
        start_time = time.time()
        
        try:
            # Step 1: Get appropriate reader for the source
            reader = self.reader_factory.get_reader(source_path)
            logger.info(f"Using reader: {reader.__class__.__name__}")
            
            # Step 2: Read the data
            data = reader.read(source_path)
            logger.info(f"Read {len(data)} bytes from source")
            
            # Step 3: Extract metadata
            metadata = reader.get_metadata(source_path)
            logger.info(f"Extracted metadata: {metadata.get('content_type', 'unknown')} content type")
            
            # Add ingestion time to metadata
            metadata['ingestion_time'] = datetime.now().isoformat()
            
            # Step 4: Get appropriate processor for the reader
            processor = self.processor_factory.get_processor(reader)
            logger.info(f"Using processor: {processor.__class__.__name__}")
            
            # Step 5: Process the data into text
            text = processor.process(data, metadata)
            logger.info(f"Processed data into {len(text)} characters of text")
            
            # Step 6: Get appropriate chunker based on content type
            chunker = self.chunker_factory.get_chunker(metadata)
            logger.info(f"Using chunker: {chunker.__class__.__name__}")
            
            # Configure chunker
            if hasattr(chunker, 'chunk_size'):
                chunker.chunk_size = self.chunk_size
            if hasattr(chunker, 'chunk_overlap'):
                chunker.chunk_overlap = self.chunk_overlap
            
            # Step 7: Chunk the text
            chunks = chunker.chunk(text, metadata)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 8: Return the chunks
            end_time = time.time()
            logger.info(f"Completed ingestion in {end_time - start_time:.2f} seconds")
            return chunks
            
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
            return []


class BatchIngestionPipeline:
    """Pipeline for ingesting multiple sources in batch."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the batch pipeline.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.pipeline = StandardIngestionPipeline(chunk_size, chunk_overlap)
    
    def ingest_batch(self, source_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ingest multiple sources in batch.
        
        Args:
            source_paths: List of paths or URIs to data sources
            
        Returns:
            Dictionary mapping source paths to their chunks
        """
        results = {}
        
        for source_path in source_paths:
            try:
                chunks = self.pipeline.ingest(source_path)
                results[source_path] = chunks
            except Exception as e:
                logger.error(f"Error ingesting {source_path}: {str(e)}")
                results[source_path] = []
        
        return results


# Singleton instance for easy access
_default_pipeline = None

def get_pipeline(chunk_size: int = 1000, chunk_overlap: int = 200) -> IngestionPipeline:
    """
    Get or create the default pipeline singleton.
    
    Args:
        chunk_size: Target size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Configured ingestion pipeline
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = StandardIngestionPipeline(chunk_size, chunk_overlap)
    
    # Update configuration if different from current
    if (hasattr(_default_pipeline, 'chunk_size') and 
        (_default_pipeline.chunk_size != chunk_size or 
         _default_pipeline.chunk_overlap != chunk_overlap)):
        _default_pipeline.chunk_size = chunk_size
        _default_pipeline.chunk_overlap = chunk_overlap
    
    return _default_pipeline 