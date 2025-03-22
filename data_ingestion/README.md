# Data Ingestion Module for Q&A System

This module handles data ingestion from various sources, converting them to text, chunking for vector DB storage, and processing specialized data formats.

## Features

- **Multi-format data ingestion**: PDF, Excel, PowerPoint, audio files, text, and more
- **Cloud storage support**: SharePoint and OneDrive integration
- **Audio processing**: Speech-to-text conversion using Google Cloud services
- **Smart chunking**: Format-specific chunking strategies for text, tables, and presentations
- **Metadata preservation**: Rich metadata for each chunk to enable advanced searching
- **Batch processing**: Process multiple files or entire directories efficiently
- **Command-line interface**: Easy-to-use CLI for integration into pipelines
- **Tools Engine integration**: Leverages the gRPC-based tools_engine for enhanced processing

## Architecture

The module follows a clean architecture using multiple design patterns:

- **Factory Pattern**: Creates appropriate readers, processors, and chunkers based on data type
- **Strategy Pattern**: Different processing strategies for various file formats
- **Singleton Pattern**: Single instances of factories and pipelines
- **Proxy Pattern**: Delegation to specialized processors for different formats

## Components

- **Readers**: Extract raw data from different sources
- **Processors**: Convert raw data to text (using tools_engine where appropriate)
- **Chunkers**: Split text into appropriate chunks for vector DB storage
- **Pipeline**: Orchestrates the complete ingestion workflow

## Integration with Tools Engine

The data_ingestion module integrates with the tools_engine microservice for advanced processing capabilities:

- **OCR Processing**: Extraction of text from images and scanned PDFs
- **Table Extraction**: Intelligent extraction of tabular data from documents
- **Speech-to-Text**: Conversion of audio files to text transcripts
- **Data Transformations**: Conversion between different data formats

The integration is handled seamlessly within the processors:

```python
from data_ingestion.processors.pdf_processor import PDFProcessor
from data_ingestion.processors.audio_processor import AudioProcessor

# The processors automatically use tools_engine when available
pdf_processor = PDFProcessor()
audio_processor = AudioProcessor()

# Process files with enhanced capabilities
pdf_text = pdf_processor.process("file.pdf")  # Uses OCR for scanned pages
audio_text = audio_processor.process("recording.mp3")  # Uses speech-to-text
```

## Usage

### Basic usage:

```python
from data_ingestion.pipeline import get_pipeline

# Get the default pipeline
pipeline = get_pipeline(chunk_size=1000, chunk_overlap=200)

# Ingest a file
chunks = pipeline.ingest("path/to/file.pdf")

# Process the chunks (e.g., store in vector DB)
for chunk in chunks:
    text = chunk['text']
    metadata = chunk['metadata']
    # ... store in vector DB or graph DB
```

### Batch processing:

```python
from data_ingestion.pipeline import BatchIngestionPipeline

# Create a batch pipeline
batch_pipeline = BatchIngestionPipeline(chunk_size=1000, chunk_overlap=200)

# Collect file paths
file_paths = ["file1.pdf", "file2.xlsx", "file3.mp3"]

# Process all files
results = batch_pipeline.ingest_batch(file_paths)

# results is a dictionary mapping file paths to their chunks
```

### Command-line usage:

```bash
# Process a single file
python -m data_ingestion.main --source path/to/file.pdf --output chunks.jsonl

# Process a directory
python -m data_ingestion.main --source path/to/directory --output chunks.jsonl --extensions pdf,txt,xlsx

# Process a cloud storage resource
python -m data_ingestion.main --source sharepoint://site/library/file.pptx --output chunks.jsonl
```

## Configuration

The module can be configured by adjusting the following parameters:

- **chunk_size**: Size of chunks in characters (default: 1000)
- **chunk_overlap**: Overlap between chunks in characters (default: 200)
- **tools_engine_url**: URL of the tools_engine gRPC service (default: "localhost:50051")
- **use_tools_engine**: Enable/disable tools_engine integration (default: True)

## Dependencies

See requirements.txt for a complete list of dependencies. The tools_engine service requires additional dependencies for gRPC communication and specialized processing. 