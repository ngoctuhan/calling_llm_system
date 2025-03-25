"""
Chunkers that split text into manageable pieces for vector storage.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional, Callable
import uuid
from dataclasses import dataclass, field
from .interfaces import DataChunker
from tiktoken import get_encoding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


@dataclass(frozen=True)
class TextChunk:
    """
    Represents a chunk of text with associated metadata.
    
    Attributes:
        text: The text content of the chunk
        source_indices: List of source document indices this chunk belongs to
        token_count: Number of tokens in the chunk (if tokenized)
        metadata: Additional metadata associated with the chunk
    """
    text: str
    source_indices: List[int] = field(default_factory=list)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        """Unique identifier for the chunk."""
        return self.metadata.get('chunk_id', str(uuid.uuid4()))
    
    @property
    def chunk_index(self) -> int:
        """Index of the chunk in the sequence."""
        return self.metadata.get('chunk_index', 0)
    
    @property
    def start_index(self) -> int:
        """Start index in original text."""
        return self.metadata.get('text_range_start', 0)
    
    @property
    def end_index(self) -> int:
        """End index in original text."""
        return self.metadata.get('text_range_end', 0)
    
    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)
    
    def __str__(self) -> str:
        """String representation of the chunk."""
        return f"TextChunk(id={self.chunk_id}, len={len(self.text)}, tokens={self.token_count})"


class TextChunker:
    """Advanced text chunker with configurable parameters for optimal text splitting."""
    
    def __init__(self, 
                 min_chunk_size: int = 500, 
                 max_chunk_size: int = 1500, 
                 chunk_overlap: int = 100,
                 split_by_semantic_units: bool = True,
                 length_function: Callable[[str], int] = len,
                 strip_whitespace: bool = True):
        """
        Initialize the text chunker with customizable parameters.
        
        Args:
            min_chunk_size: Minimum size of chunks in characters/tokens
            max_chunk_size: Maximum size of chunks in characters/tokens
            chunk_overlap: Number of characters/tokens to overlap between chunks
            split_by_semantic_units: Whether to split by semantic units (paragraphs, sentences)
            length_function: Function to measure text length (default is character count)
            strip_whitespace: Whether to strip whitespace from chunk text
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by_semantic_units = split_by_semantic_units
        self.length_function = length_function
        self.strip_whitespace = strip_whitespace
        
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
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata about the data source
            
        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        
        # Skip empty text
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Clean text: normalize whitespace if needed
        if self.strip_whitespace:
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
                    if (self.length_function(current_chunk) >= self.min_chunk_size and 
                        self.length_function(potential_chunk) > self.max_chunk_size):
                        
                        # Calculate positions in original text
                        start_idx = total_processed - len(current_chunk)
                        end_idx = total_processed
                        
                        # Create metadata
                        chunk_metadata = self._get_chunk_metadata(
                            metadata, current_chunk_index, start_idx, end_idx
                        )
                        
                        # Add chunk
                        chunks.append(TextChunk(
                            text=current_chunk,
                            source_indices=[0],  # Default source index
                            token_count=self.length_function(current_chunk),
                            metadata=chunk_metadata
                        ))
                        
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
            # Simple character/token-based chunking without respecting semantic units
            chunks = self._chunk_by_tokens(text, metadata)
            
        # Don't forget the last chunk
        if current_chunk and "current_chunk" in locals():
            start_idx = total_processed - len(current_chunk)
            end_idx = total_processed
            
            chunk_metadata = self._get_chunk_metadata(
                metadata, current_chunk_index, start_idx, end_idx
            )
            
            chunks.append(TextChunk(
                text=current_chunk,
                source_indices=[0],  # Default source index
                token_count=self.length_function(current_chunk),
                metadata=chunk_metadata
            ))
        
        logger.info(f"Created {len(chunks)} chunks from text of length {self.length_function(text)}")
        return chunks
    
    def _chunk_by_tokens(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Simple token-based chunking when semantic units aren't needed."""
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
            chunk_text = text[start:end].strip() if self.strip_whitespace else text[start:end]
            
            chunk_metadata = self._get_chunk_metadata(
                metadata, chunk_index, start, end
            )
            
            chunks.append(TextChunk(
                text=chunk_text,
                source_indices=[0],  # Default source index
                token_count=self.length_function(chunk_text),
                metadata=chunk_metadata
            ))
            
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
        
    def chunk_multiple_texts(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[TextChunk]:
        """
        Split multiple texts into chunks with source tracking.
        
        Args:
            texts: List of texts to chunk
            metadata: Optional list of metadata for each text
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        
        if metadata is None:
            metadata = [{} for _ in texts]
        elif len(metadata) != len(texts):
            raise ValueError("Length of metadata list must match length of texts list")
        
        for idx, (text, meta) in enumerate(zip(texts, metadata)):
            # Add source index to metadata
            meta = meta.copy()
            meta['source_index'] = idx
            
            # Chunk the text
            chunks = self.chunk(text, meta)
            
            # Update source indices
            for chunk in chunks:
                chunk.source_indices.append(idx)
                
            all_chunks.extend(chunks)
            
        return all_chunks


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class for chunking text by tokens."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""


class TokenTextChunker:
    """
    Token-based text chunker that splits text by token count,
    similar to the example code's TokenTextSplitter.
    """
    
    def __init__(
        self,
        tokenizer_fn: Optional[Callable[[str], List[int]]] = None,
        detokenizer_fn: Optional[Callable[[List[int]], str]] = None,
        tokens_per_chunk: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the token chunker.
        
        Args:
            tokenizer_fn: Function to convert text into tokens
            detokenizer_fn: Function to convert tokens back to text
            tokens_per_chunk: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        # Default simple tokenizer splits by space if none provided
        self.tokenizer_fn = tokenizer_fn or (lambda text: text.split())
        # Default detokenizer joins with spaces if none provided
        self.detokenizer_fn = detokenizer_fn or (lambda tokens: " ".join(tokens))
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        
        # Create tokenizer data class
        self.tokenizer = Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            decode=self.detokenizer_fn if isinstance(self.detokenizer_fn(["test"]), str) else lambda t: self.detokenizer_fn(t),
            encode=self.tokenizer_fn if isinstance(self.tokenizer_fn("test"), list) else lambda t: self.tokenizer_fn(t)
        )
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata about the data source
            
        Returns:
            List of TextChunk objects with token count information
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        
        # Skip empty text
        if not text or not text.strip():
            logger.warning("Empty text provided for token chunking")
            return []
        
        # Encode text into tokens
        try:
            input_ids = self.tokenizer.encode(text)
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []
        
        # Split into chunks based on token count
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(input_ids):
            # Determine end position for this chunk
            end_idx = min(start_idx + self.tokenizer.tokens_per_chunk, len(input_ids))
            
            # Get token ids for this chunk
            chunk_ids = input_ids[start_idx:end_idx]
            
            # Decode tokens back to text
            try:
                chunk_text = self.tokenizer.decode(chunk_ids)
            except Exception as e:
                logger.error(f"Error decoding tokens: {e}")
                chunk_text = ""
            
            if chunk_text:
                # Create metadata for this chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_index': chunk_index,
                    'token_start': start_idx,
                    'token_end': end_idx,
                    'token_count': len(chunk_ids)
                })
                
                # Create TextChunk object
                chunks.append(TextChunk(
                    text=chunk_text,
                    source_indices=[0],  # Default source index
                    token_count=len(chunk_ids),
                    metadata=chunk_metadata
                ))
            
            # Move to next chunk, with overlap
            start_idx += self.tokenizer.tokens_per_chunk - self.tokenizer.chunk_overlap
            chunk_index += 1
            
            # Avoid infinite loops if overlap is too large
            if self.tokenizer.chunk_overlap >= self.tokenizer.tokens_per_chunk:
                start_idx += 1
        
        logger.info(f"Created {len(chunks)} token-based chunks from text with {len(input_ids)} tokens")
        return chunks
    
    def chunk_multiple_texts(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[TextChunk]:
        """
        Split multiple texts into chunks with source tracking based on token count.
        
        Args:
            texts: List of texts to chunk
            metadata: Optional list of metadata for each text
            
        Returns:
            List of TextChunk objects
        """
        # Similar implementation to TextChunker.chunk_multiple_texts but using token-based chunking
        all_chunks = []
        
        if metadata is None:
            metadata = [{} for _ in texts]
        elif len(metadata) != len(texts):
            raise ValueError("Length of metadata list must match length of texts list")
        
        # Process each text
        for idx, (text, meta) in enumerate(zip(texts, metadata)):
            # Add source index to metadata
            meta = meta.copy()
            meta['source_index'] = idx
            
            # Chunk by tokens
            chunks = self.chunk(text, meta)
            
            # Update source indices
            for chunk in chunks:
                chunk.source_indices.append(idx)
                
            all_chunks.extend(chunks)
            
        return all_chunks


def chunk_with_tokenizer(text: str, tokenizer: Tokenizer) -> List[str]:
    """
    Split a single text into chunks using the provided tokenizer.
    Similar to split_single_text_on_tokens from the example code.
    
    Args:
        text: Text to chunk
        tokenizer: Tokenizer to use for encoding/decoding
        
    Returns:
        List of text chunks
    """
    result = []
    
    # Encode the text into tokens
    input_ids = tokenizer.encode(text)
    
    # Chunk by token count
    start_idx = 0
    while start_idx < len(input_ids):
        # Calculate end index for current chunk
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        
        # Get tokens for current chunk
        chunk_ids = input_ids[start_idx:cur_idx]
        
        # Decode tokens to text
        chunk_text = tokenizer.decode(list(chunk_ids))
        result.append(chunk_text)
        
        # Move to next position, considering overlap
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    
    return result


def chunk_multiple_texts_with_tokenizer(
    texts: List[str], 
    tokenizer: Tokenizer
) -> List[TextChunk]:
    """
    Split multiple texts and return chunks with metadata using the tokenizer.
    Similar to split_multiple_texts_on_tokens from the example code.
    
    Args:
        texts: List of texts to chunk
        tokenizer: Tokenizer to use for encoding/decoding
        
    Returns:
        List of TextChunk objects with metadata
    """
    result = []
    mapped_ids = []

    # Encode all texts
    for source_doc_idx, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        mapped_ids.append((source_doc_idx, encoded))

    # Flatten all tokens with their source document index
    input_ids = [
        (source_doc_idx, token_id) for source_doc_idx, ids in mapped_ids for token_id in ids
    ]

    # Chunk by token count
    start_idx = 0
    while start_idx < len(input_ids):
        # Calculate end index for current chunk
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        
        # Get tokens for current chunk
        chunk_ids = input_ids[start_idx:cur_idx]
        
        # Extract token ids and decode to text
        chunk_text = tokenizer.decode([token_id for _, token_id in chunk_ids])
        
        # Track which documents this chunk came from
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        
        # Create TextChunk object
        result.append(TextChunk(
            text=chunk_text,
            source_indices=doc_indices,
            token_count=len(chunk_ids),
            metadata={
                'chunk_id': str(uuid.uuid4()),
                'token_start': start_idx,
                'token_end': cur_idx,
                'token_count': len(chunk_ids)
            }
        ))
        
        # Move to next position, considering overlap
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    
    return result


