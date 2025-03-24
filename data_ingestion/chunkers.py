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


from typing import Dict, Any, List
import re
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextChunker:
    """Advanced text chunker with configurable parameters for optimal text splitting."""
    
    def __init__(self, 
                 min_chunk_size: int = 500, 
                 max_chunk_size: int = 1500, 
                 chunk_overlap: int = 100,
                 split_by_semantic_units: bool = True):
        """
        Initialize the text chunker with customizable parameters.
        
        Args:
            min_chunk_size: Minimum size of chunks in characters
            max_chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            split_by_semantic_units: Whether to split by semantic units (paragraphs, sentences)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by_semantic_units = split_by_semantic_units
        
    def _get_chunk_metadata(self, metadata: Dict[str, Any], chunk_index: int, 
                           start_idx: int, end_idx: int) -> Dict[str, Any]:
        """
        Create metadata for a chunk.
        
        Args:
            metadata: Source metadata
            chunk_index: Index of the current chunk
            start_idx: Start index of chunk in original text
            end_idx: End index of chunk in original text
            
        Returns:
            Metadata dictionary for the chunk
        """
        # Create a new ID for the chunk
        chunk_id = str(uuid.uuid4())
        
        # Create a copy of the metadata with chunk-specific additions
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'text_range_start': start_idx,
            'text_range_end': end_idx,
            'chunk_size': end_idx - start_idx
        })
        
        return chunk_metadata
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata about the data source
            
        Returns:
            List of chunks with their metadata
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        
        # Skip empty text
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Clean text: normalize whitespace 
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if self.split_by_semantic_units:
            # Split text by paragraphs first
            paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
            
            current_chunk = ""
            current_chunk_index = 0
            total_processed = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # Split paragraph into sentences for finer control
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # Check if adding this sentence would exceed max size
                    potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
                    
                    # If we're above min size and adding would exceed max, save the chunk
                    if (len(current_chunk) >= self.min_chunk_size and 
                        len(potential_chunk) > self.max_chunk_size):
                        
                        # Calculate positions in original text
                        start_idx = total_processed - len(current_chunk)
                        end_idx = total_processed
                        
                        # Create metadata
                        chunk_metadata = self._get_chunk_metadata(
                            metadata, current_chunk_index, start_idx, end_idx
                        )
                        
                        # Add chunk
                        chunks.append({
                            'text': current_chunk,
                            'metadata': chunk_metadata
                        })
                        
                        # Handle overlap for next chunk
                        if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                            # Try to find sentence boundary in overlap region
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            sentence_boundaries = list(re.finditer(r'(?<=[.!?])\s+', overlap_text))
                            
                            if sentence_boundaries:
                                # Start new chunk from last sentence boundary in overlap
                                last_boundary = sentence_boundaries[-1].end()
                                current_chunk = overlap_text[last_boundary:]
                            else:
                                # No sentence boundary found, try to find word boundary
                                word_boundary = list(re.finditer(r'\s+', overlap_text))
                                if word_boundary:
                                    # Start from a word boundary to avoid cutting words
                                    first_boundary = word_boundary[0].end()
                                    current_chunk = overlap_text[first_boundary:]
                                else:
                                    # If no word boundary either, use whole overlap
                                    current_chunk = overlap_text
                        else:
                            current_chunk = ""
                            
                        current_chunk_index += 1
                    
                    # Add current sentence to chunk
                    if current_chunk and not sentence.startswith(" "):
                        current_chunk += " "
                    current_chunk += sentence
                
                # Track position in original text
                total_processed += len(paragraph) + 2  # +2 for paragraph breaks
                
                # Add paragraph break if needed and not at the end
                if paragraphs.index(paragraph) < len(paragraphs) - 1 and current_chunk:
                    current_chunk += "\n\n"
        else:
            # Simple character-based chunking without respecting semantic units
            chunks = self._chunk_by_chars(text, metadata)
            
        # Don't forget the last chunk
        if current_chunk:
            start_idx = total_processed - len(current_chunk)
            end_idx = total_processed
            
            chunk_metadata = self._get_chunk_metadata(
                metadata, current_chunk_index, start_idx, end_idx
            )
            
            chunks.append({
                'text': current_chunk,
                'metadata': chunk_metadata
            })
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _chunk_by_chars(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple character-based chunking when semantic units aren't needed."""
        chunks = []
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position for this chunk
            end = min(start + self.max_chunk_size, len(text))
            
            # If we're not at the end and we're above min size, try to find a good break point
            if end < len(text) and end - start >= self.min_chunk_size:
                # Look for a period, question mark, or exclamation point followed by a space
                match = re.search(r'[.!?]\s+', text[end-100:end])
                if match:
                    end = end - 100 + match.end()
                else:
                    # If no sentence boundary, look for word boundary (space)
                    space_pos = text.rfind(' ', end - 50, end)
                    if space_pos > start:
                        end = space_pos + 1  # Include the space to ensure we start at word boundary
            
            # Extract chunk and add to results
            chunk_text = text[start:end].strip()
            
            chunk_metadata = self._get_chunk_metadata(
                metadata, chunk_index, start, end
            )
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
            
            # Move start position for next chunk, accounting for overlap
            if self.chunk_overlap < end - start:
                # Calculate potential new start position with overlap
                potential_start = end - self.chunk_overlap
                
                # Make sure we don't cut in the middle of a word
                # Find the next word boundary (space) after potential_start
                next_space = text.find(' ', potential_start)
                if next_space != -1 and next_space < end:
                    start = next_space + 1  # Start after the space
                else:
                    # If no space found, just use the calculated position
                    start = potential_start
            else:
                start = start + 1
            chunk_index += 1
        
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


