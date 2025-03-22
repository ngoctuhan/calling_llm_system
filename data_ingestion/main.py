"""
Main entry point for the data ingestion module.
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

from .pipeline import get_pipeline, BatchIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def ingest_file(file_path: str, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Ingest a single file and return chunks.
    
    Args:
        file_path: Path to the file to ingest
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunks with their metadata
    """
    # Get the pipeline with specified configuration
    pipeline = get_pipeline(chunk_size, chunk_overlap)
    
    # Ingest the file
    return pipeline.ingest(file_path)


def ingest_directory(directory_path: str, 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 200,
                    file_extensions: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ingest all files in a directory and return chunks.
    
    Args:
        directory_path: Path to the directory
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        file_extensions: List of file extensions to include (e.g. ['.pdf', '.txt'])
        
    Returns:
        Dictionary mapping file paths to their chunks
    """
    # Initialize the batch pipeline
    batch_pipeline = BatchIngestionPipeline(chunk_size, chunk_overlap)
    
    # Get all files in the directory
    directory = Path(directory_path)
    files_to_process = []
    
    for file_path in directory.glob('**/*'):
        if file_path.is_file():
            # Filter by extension if specified
            if file_extensions and file_path.suffix.lower() not in file_extensions:
                continue
            
            files_to_process.append(str(file_path))
    
    logger.info(f"Found {len(files_to_process)} files to process in {directory_path}")
    
    # Ingest all files
    return batch_pipeline.ingest_batch(files_to_process)


def ingest_cloud_storage(uri_list: List[str],
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ingest files from cloud storage sources.
    
    Args:
        uri_list: List of cloud storage URIs (sharepoint://, onedrive://)
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Dictionary mapping URIs to their chunks
    """
    # Initialize the batch pipeline
    batch_pipeline = BatchIngestionPipeline(chunk_size, chunk_overlap)
    
    # Ingest all URIs
    return batch_pipeline.ingest_batch(uri_list)


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_file: str):
    """
    Save chunks to a JSONL file for later ingestion into a vector database.
    
    Args:
        chunks: List of chunks to save
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='Data Ingestion Tool')
    
    # Main arguments
    parser.add_argument('--source', required=True, help='Source file, directory, or URI to ingest')
    parser.add_argument('--output', required=True, help='Output file to save chunks to')
    
    # Configuration options
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of chunks in characters')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Overlap between chunks in characters')
    parser.add_argument('--extensions', help='Comma-separated list of file extensions to include (e.g. .pdf,.txt)')
    
    # Source type
    parser.add_argument('--source-type', choices=['file', 'directory', 'cloud'], 
                        help='Type of source (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Process file extensions
    file_extensions = None
    if args.extensions:
        file_extensions = [ext.strip() for ext in args.extensions.split(',')]
        # Add dots if missing
        file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    
    # Auto-detect source type if not specified
    source_type = args.source_type
    if not source_type:
        if args.source.startswith(('sharepoint://', 'onedrive://')):
            source_type = 'cloud'
        elif os.path.isdir(args.source):
            source_type = 'directory'
        else:
            source_type = 'file'
    
    logger.info(f"Ingesting {source_type} from {args.source}")
    
    # Ingest based on source type
    if source_type == 'file':
        chunks = ingest_file(args.source, args.chunk_size, args.chunk_overlap)
        all_chunks = chunks
    elif source_type == 'directory':
        result = ingest_directory(args.source, args.chunk_size, args.chunk_overlap, file_extensions)
        all_chunks = [chunk for chunks in result.values() for chunk in chunks]
    elif source_type == 'cloud':
        result = ingest_cloud_storage([args.source], args.chunk_size, args.chunk_overlap)
        all_chunks = [chunk for chunks in result.values() for chunk in chunks]
    
    # Save chunks to output file
    save_chunks_to_jsonl(all_chunks, args.output)
    logger.info(f"Saved {len(all_chunks)} chunks to {args.output}")


if __name__ == '__main__':
    main() 