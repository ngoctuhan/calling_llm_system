"""
Chunkers that split text into manageable pieces for vector storage.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional
import uuid

from .interfaces import DataChunker

logger = logging.getLogger(__name__)


class BaseChunker(DataChunker):
    """Base class for chunking implementations."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Processed text to chunk
            metadata: Metadata about the data source
            
        Returns:
            List of chunks with their metadata
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_base_metadata(self, metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """
        Create base metadata for a chunk.
        
        Args:
            metadata: Source metadata
            chunk_index: Index of the current chunk
            
        Returns:
            Metadata dictionary for the chunk
        """
        # Create a new ID for the chunk
        chunk_id = str(uuid.uuid4())
        
        # Create a copy of the metadata with chunk-specific additions
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'ingestion_time': chunk_metadata.get('ingestion_time', '')
        })
        
        return chunk_metadata


class TextChunker(BaseChunker):
    """Chunker for general text documents."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Processed text to chunk
            metadata: Metadata about the data source
            
        Returns:
            List of chunks with their metadata
        """
        chunks = []
        
        # Skip empty text
        if not text.strip():
            return []
        
        # Split text by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds the chunk size and we already have content,
            # save the current chunk and start a new one
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                # Create metadata for this chunk
                chunk_metadata = self._get_base_metadata(metadata, current_chunk_index)
                
                # Add text range information
                start_idx = len("".join(paragraphs[:current_chunk_index]))
                end_idx = start_idx + len(current_chunk)
                chunk_metadata.update({
                    'text_range': {
                        'start': start_idx,
                        'end': end_idx
                    }
                })
                
                # Save the chunk
                chunks.append({
                    'text': current_chunk,
                    'metadata': chunk_metadata
                })
                
                # Start a new chunk with overlap if possible
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    # Get the last part of the previous chunk as overlap
                    current_chunk = current_chunk[-self.chunk_overlap:]
                else:
                    current_chunk = ""
                
                current_chunk_index += 1
            
            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_metadata = self._get_base_metadata(metadata, current_chunk_index)
            
            # Add text range information
            start_idx = len("".join(paragraphs[:current_chunk_index]))
            end_idx = start_idx + len(current_chunk)
            chunk_metadata.update({
                'text_range': {
                    'start': start_idx,
                    'end': end_idx
                }
            })
            
            chunks.append({
                'text': current_chunk,
                'metadata': chunk_metadata
            })
        
        return chunks


class TableChunker(BaseChunker):
    """Chunker for tabular data."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split tabular text into chunks.
        
        Args:
            text: Processed table text (likely JSON)
            metadata: Metadata about the data source
            
        Returns:
            List of chunks with their metadata
        """
        chunks = []
        
        # Skip empty text
        if not text.strip():
            return []
        
        try:
            # Try to parse as JSON
            if text.startswith("CSV Data:") or text.startswith("Excel Data:"):
                # Extract the JSON part
                json_text = text.split("\n", 1)[1].strip()
                data = json.loads(json_text)
            else:
                data = json.loads(text)
            
            # For CSV data (single table)
            if isinstance(data, list):
                # Chunk by groups of rows
                rows_per_chunk = max(1, self.chunk_size // 200)  # Approximate size
                
                for i in range(0, len(data), rows_per_chunk):
                    chunk_rows = data[i:i+rows_per_chunk]
                    
                    # Create chunk metadata
                    chunk_metadata = self._get_base_metadata(metadata, i // rows_per_chunk)
                    chunk_metadata.update({
                        'table_range': {
                            'start_row': i,
                            'end_row': min(i + rows_per_chunk - 1, len(data) - 1),
                            'total_rows': len(data)
                        }
                    })
                    
                    chunks.append({
                        'text': json.dumps(chunk_rows, indent=2),
                        'metadata': chunk_metadata
                    })
            
            # For Excel data (multiple sheets)
            elif isinstance(data, dict):
                sheet_index = 0
                for sheet_name, sheet_data in data.items():
                    # Parse sheet data
                    if isinstance(sheet_data, str):
                        sheet_rows = json.loads(sheet_data)
                    else:
                        sheet_rows = sheet_data
                    
                    # Skip empty sheets
                    if not sheet_rows:
                        continue
                    
                    # Chunk by groups of rows
                    rows_per_chunk = max(1, self.chunk_size // 200)  # Approximate size
                    
                    for i in range(0, len(sheet_rows), rows_per_chunk):
                        chunk_rows = sheet_rows[i:i+rows_per_chunk]
                        
                        # Create chunk metadata
                        chunk_index = (sheet_index * 1000) + (i // rows_per_chunk)
                        chunk_metadata = self._get_base_metadata(metadata, chunk_index)
                        chunk_metadata.update({
                            'sheet_name': sheet_name,
                            'table_range': {
                                'start_row': i,
                                'end_row': min(i + rows_per_chunk - 1, len(sheet_rows) - 1),
                                'total_rows': len(sheet_rows)
                            }
                        })
                        
                        chunks.append({
                            'text': json.dumps(chunk_rows, indent=2),
                            'metadata': chunk_metadata
                        })
                    
                    sheet_index += 1
        except json.JSONDecodeError:
            # Fallback to text chunking
            logger.warning("Could not parse table data as JSON, falling back to text chunking")
            text_chunker = TextChunker(self.chunk_size, self.chunk_overlap)
            return text_chunker.chunk(text, metadata)
        except Exception as e:
            logger.error(f"Error chunking table data: {str(e)}")
            return []
        
        return chunks


class SlideDeckChunker(BaseChunker):
    """Chunker for presentation slide decks."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split presentation text into chunks by slides.
        
        Args:
            text: Processed presentation text
            metadata: Metadata about the data source
            
        Returns:
            List of chunks with their metadata
        """
        chunks = []
        
        # Skip empty text
        if not text.strip():
            return []
        
        # Check if the text starts with the PowerPoint header
        if text.startswith("PowerPoint Presentation:"):
            # Try to split by slides
            slide_pattern = re.compile(r'Slide (\d+):\n(.*?)(?=Slide \d+:|$)', re.DOTALL)
            slides = slide_pattern.findall(text)
            
            for slide_num, slide_content in slides:
                # Create chunk metadata
                chunk_metadata = self._get_base_metadata(metadata, int(slide_num) - 1)
                chunk_metadata.update({
                    'slide_number': int(slide_num),
                    'content_type': 'slide'
                })
                
                chunks.append({
                    'text': slide_content.strip(),
                    'metadata': chunk_metadata
                })
        else:
            # Try to parse as JSON
            try:
                # Look for JSON in the text
                json_text = text
                if text.startswith("Presentation Data:"):
                    json_text = text.split("\n", 1)[1].strip()
                
                data = json.loads(json_text)
                
                # Extract slides data
                slides_data = data.get('slides', [])
                
                for slide in slides_data:
                    slide_num = slide.get('slide_number', 0)
                    slide_content = slide.get('content', '')
                    
                    # Create chunk metadata
                    chunk_metadata = self._get_base_metadata(metadata, slide_num - 1)
                    chunk_metadata.update({
                        'slide_number': slide_num,
                        'content_type': 'slide'
                    })
                    
                    chunks.append({
                        'text': slide_content,
                        'metadata': chunk_metadata
                    })
            except json.JSONDecodeError:
                # Fallback to text chunking
                logger.warning("Could not parse slide data as JSON, falling back to text chunking")
                text_chunker = TextChunker(self.chunk_size, self.chunk_overlap)
                return text_chunker.chunk(text, metadata)
            except Exception as e:
                logger.error(f"Error chunking slide data: {str(e)}")
                return []
        
        return chunks 