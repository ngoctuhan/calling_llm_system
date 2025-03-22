"""
Example script demonstrating usage of the data_ingestion module.
"""
import os
import json
import logging
from pathlib import Path

# Import data_ingestion components
from data_ingestion.pipeline import get_pipeline, BatchIngestionPipeline
from data_ingestion.factory import ReaderFactory, ProcessorFactory, ChunkerFactory
from data_ingestion.readers import PDFReader, ExcelReader, AudioReader
from data_ingestion.processors import PDFProcessor, ExcelProcessor, AudioProcessor
from data_ingestion.chunkers import TextChunker, TableChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_single_file():
    """Example of ingesting a single file."""
    # Example file path (replace with an actual file)
    file_path = "sample_data/document.pdf"
    
    logger.info(f"Ingesting single file: {file_path}")
    
    # Get the default pipeline
    pipeline = get_pipeline(chunk_size=1000, chunk_overlap=200)
    
    # Ingest the file
    chunks = pipeline.ingest(file_path)
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Print the first chunk
    if chunks:
        logger.info(f"First chunk text: {chunks[0]['text'][:100]}...")
        logger.info(f"First chunk metadata: {chunks[0]['metadata']}")


def example_directory():
    """Example of ingesting a directory of files."""
    # Example directory path (replace with an actual directory)
    directory_path = "sample_data/"
    
    logger.info(f"Ingesting directory: {directory_path}")
    
    # Initialize a batch pipeline
    batch_pipeline = BatchIngestionPipeline(chunk_size=1000, chunk_overlap=200)
    
    # Find all PDF, Excel, and text files in the directory
    files_to_process = []
    extensions = ['.pdf', '.xlsx', '.csv', '.txt']
    
    for ext in extensions:
        for file_path in Path(directory_path).glob(f'**/*{ext}'):
            files_to_process.append(str(file_path))
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Ingest all files
    results = batch_pipeline.ingest_batch(files_to_process)
    
    # Count total chunks
    total_chunks = sum(len(chunks) for chunks in results.values())
    logger.info(f"Created {total_chunks} chunks from {len(results)} files")
    
    # Save chunks to a JSON file
    output_file = "sample_data/chunks.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for source_path, chunks in results.items():
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
    
    logger.info(f"Saved all chunks to {output_file}")


def example_cloud_storage():
    """Example of ingesting files from cloud storage."""
    # Example SharePoint and OneDrive URIs
    sharepoint_uri = "sharepoint://mysite/Documents/report.pdf"
    onedrive_uri = "onedrive://MyDocuments/presentation.pptx"
    
    logger.info("Ingesting files from cloud storage")
    
    # Initialize a batch pipeline
    batch_pipeline = BatchIngestionPipeline(chunk_size=1000, chunk_overlap=200)
    
    # Ingest the cloud files
    results = batch_pipeline.ingest_batch([sharepoint_uri, onedrive_uri])
    
    # Count total chunks
    total_chunks = sum(len(chunks) for chunks in results.values())
    logger.info(f"Created {total_chunks} chunks from {len(results)} files")


def example_custom_pipeline():
    """Example of creating a custom pipeline."""
    # Create custom components
    pdf_reader = PDFReader()
    pdf_processor = PDFProcessor()
    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    # Example file path (replace with an actual file)
    file_path = "sample_data/document.pdf"
    
    logger.info(f"Custom pipeline for: {file_path}")
    
    # Manual pipeline
    try:
        # Read the data
        data = pdf_reader.read(file_path)
        
        # Get metadata
        metadata = pdf_reader.get_metadata(file_path)
        
        # Process the data
        text = pdf_processor.process(data, metadata)
        
        # Chunk the text
        chunks = text_chunker.chunk(text, metadata)
        
        logger.info(f"Created {len(chunks)} chunks with custom pipeline")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")


def main():
    """Run the examples."""
    logger.info("Starting data ingestion examples")
    
    # Create sample_data directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # Check if sample files exist
    pdf_path = "sample_data/document.pdf"
    if not os.path.exists(pdf_path):
        logger.warning(f"Sample file {pdf_path} not found. Some examples may fail.")
        logger.info("Please create sample files in sample_data/ for full example execution.")
    
    # Run examples (comment out as needed)
    logger.info("=" * 50)
    example_single_file()
    
    logger.info("=" * 50)
    example_directory()
    
    logger.info("=" * 50)
    example_cloud_storage()
    
    logger.info("=" * 50)
    example_custom_pipeline()
    
    logger.info("=" * 50)
    logger.info("Examples completed")


if __name__ == "__main__":
    main() 