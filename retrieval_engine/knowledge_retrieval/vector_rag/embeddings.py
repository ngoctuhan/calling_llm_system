import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import hashlib
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query text into a vector"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of documents into vectors"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        pass
    
    def get_provider_name(self) -> str:
        """Return the name of the embedding provider"""
        return self.__class__.__name__


class HuggingFaceEmbeddings(EmbeddingProvider):
    """Sentence Transformer embedding provider using Hugging Face models"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Hugging Face embedding provider
        
        Args:
            model_name: The name of the model to use from Hugging Face
            cache_dir: Directory to cache models
            model_kwargs: Additional keyword arguments for model initialization
            encode_kwargs: Additional keyword arguments for encoding
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.model_kwargs = model_kwargs or {}
            self.encode_kwargs = encode_kwargs or {}
            
            # Set default encoding parameters
            self.encode_kwargs.setdefault("normalize_embeddings", True)
            self.encode_kwargs.setdefault("batch_size", 32)
            self.encode_kwargs.setdefault("convert_to_numpy", True)
            
            if cache_dir:
                self.model_kwargs["cache_folder"] = cache_dir
            
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name, **self.model_kwargs)
            logger.info(f"Model loaded, embedding dimension: {self.dimension}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Please install with 'pip install sentence-transformers'")
            raise ImportError("sentence-transformers is required. Install with 'pip install sentence-transformers'")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query text
        
        Args:
            text: The query text to embed
            
        Returns:
            np.ndarray: The query embedding
        """
        return self.model.encode(text, **self.encode_kwargs)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of documents
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            np.ndarray: The document embeddings
        """
        return self.model.encode(texts, **self.encode_kwargs)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider using their API"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 16,
        request_timeout: Optional[float] = None,
    ):
        """
        Initialize the OpenAI embedding provider
        
        Args:
            model_name: The OpenAI embedding model name
            api_key: The OpenAI API key (if None, will read from env var OPENAI_API_KEY)
            dimensions: Output dimensions of embeddings (if supported by model)
            batch_size: The batch size for document embeddings
            request_timeout: Timeout for API requests
        """
        try:
            from openai import OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            self.model_name = model_name
            self.batch_size = batch_size
            self.dimensions = dimensions
            self.request_timeout = request_timeout
            
            self.client = OpenAI(api_key=self.api_key)
            
            # Model dimensions mapping
            self.model_dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            
            # Check if model exists in our mapping
            if self.model_name not in self.model_dimensions:
                logger.warning(f"Model {model_name} not found in dimensions mapping. Using default (1536).")
                self._embedding_dim = 1536
            else:
                self._embedding_dim = self.model_dimensions[self.model_name]
                
            # Override with user-specified dimensions if provided
            if self.dimensions:
                self._embedding_dim = self.dimensions
                
            logger.info(f"Initialized OpenAI embeddings with model {model_name}, dimension: {self._embedding_dim}")
                
        except ImportError:
            logger.error("openai package not installed. Please install with 'pip install openai'")
            raise ImportError("openai is required. Install with 'pip install openai'")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query text using OpenAI API
        
        Args:
            text: The query text to embed
            
        Returns:
            np.ndarray: The query embedding
        """
        # Handle empty or whitespace-only text
        if not text or text.isspace():
            return np.zeros(self.dimension)
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimensions,
                timeout=self.request_timeout
            )
            
            return np.array(response.data[0].embedding, dtype=np.float32).tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.dimension).tolist()
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of documents using OpenAI API
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            np.ndarray: The document embeddings
        """
        if not texts:
            return np.array([])
        
        # Remove empty or whitespace-only texts
        texts = [text for text in texts if text and not text.isspace()]
        
        if not texts:
            return np.array([])
            
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    dimensions=self.dimensions,
                    timeout=self.request_timeout
                )
                
                # Sort by index to ensure correct order
                sorted_embeddings = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [embedding.embedding for embedding in sorted_embeddings]
                all_embeddings.extend(batch_embeddings)
                
                # Respect API rate limits with a small delay
                if i + self.batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error generating batch embeddings with OpenAI: {str(e)}")
                # Add zero embeddings for this batch as fallback
                all_embeddings.extend([np.zeros(self.dimension) for _ in range(len(batch_texts))])
        
        return np.array(all_embeddings, dtype=np.float32).tolist()
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        return self._embedding_dim


class GoogleAIEmbeddings(EmbeddingProvider):
    """Google AI embedding provider for PaLM and Gemini models"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        task_type: str = "RETRIEVAL_QUERY",
        batch_size: int = 5,
    ):
        """
        Initialize the Google AI embedding provider
        
        Args:
            model_name: The Google embedding model name
            api_key: The Google API key (if None, will read from env var GOOGLE_API_KEY)
            task_type: Task type for the embeddings (RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT)
            batch_size: The batch size for document embeddings
        """
        try:
            from google import genai
            
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            
            if not self.api_key:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            
            self.model_name = model_name
            self.batch_size = batch_size
            self.task_type = task_type
            
            # Configure the Google API
            self.client = genai.Client(api_key=self.api_key)
            
            # Model dimensions mapping
            self.model_dimensions = {
                "text-embedding-004": 768,
                "text-embedding-005": 768,
                "textembedding-gecko-multilingual@001": 768
            }
            
            # Check if model exists in our mapping
            if self.model_name not in self.model_dimensions:
                logger.warning(f"Model {model_name} not found in dimensions mapping. Using default (768).")
                self._embedding_dim = 768
            else:
                self._embedding_dim = self.model_dimensions[self.model_name]
                
            logger.info(f"Initialized Google AI embeddings with model {model_name}, dimension: {self._embedding_dim}")
                
        except ImportError:
            logger.error("google-generativeai package not installed. Please install with 'pip install google-generativeai'")
            raise ImportError("google-generativeai is required. Install with 'pip install google-generativeai'")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query text using Google AI
        
        Args:
            text: The query text to embed
            
        Returns:
            np.ndarray: The query embedding
        """
        # Handle empty or whitespace-only text
        if not text or text.isspace():
            return np.zeros(self.dimension)
        
        try:
            embedding = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return np.array(embedding.embeddings[0].values)
            
        except Exception as e:
            print(f"Error generating embeddings with Google AI: {str(e)}")
            logger.error(f"Error generating embeddings with Google AI: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.dimension)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of documents using Google AI
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            np.ndarray: The document embeddings
        """
        if not texts:
            return np.array([])
        
        # Remove empty or whitespace-only texts
        texts = [text for text in texts if text and not text.isspace()]
        if not texts:
            return np.array([])
            
        all_embeddings = []
        batch_embeddings = []
        # Process in batches due to API limits
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            try:
                embedding = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts,
                )
                batch_embeddings.append([emb.values for emb in embedding.embeddings])
                
            except Exception as e:
                logger.error(f"Error generating embedding for document with Google AI: {str(e)}")
                # Add zero embedding as fallback
                batch_embeddings.append(np.zeros(self.dimension))
        
        # Flatten the batch embeddings
        all_embeddings.extend([item for sublist in batch_embeddings for item in sublist])
        return np.array(all_embeddings, dtype=np.float32).tolist()
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        return self._embedding_dim


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    Wrapper for any embedding provider that adds caching functionality
    to avoid redundant embedding calculations
    """
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_dir: str = ".vector_cache",
        namespace: Optional[str] = None,
    ):
        """
        Initialize the cached embedding provider
        
        Args:
            provider: The base embedding provider to wrap
            cache_dir: Directory to store cached embeddings
            namespace: Optional namespace for cache isolation
        """
        self.provider = provider
        self.cache_dir = cache_dir
        self.namespace = namespace or self.provider.get_provider_name()
        
        # Create cache directory if it doesn't exist
        cache_path = Path(cache_dir) / self.namespace
        cache_path.mkdir(parents=True, exist_ok=True)
        
        self.cache_path = str(cache_path)
        logger.info(f"Initialized cached embedding provider with cache at {self.cache_path}")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text
        
        Args:
            text: The text to generate a cache key for
            
        Returns:
            str: The cache key
        """
        # Create hash from text
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return text_hash
    
    def _get_cache_file_path(self, key: str) -> str:
        """
        Get the file path for a cache key
        
        Args:
            key: The cache key
            
        Returns:
            str: The cache file path
        """
        return os.path.join(self.cache_path, f"{key}.npy")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query text with caching
        
        Args:
            text: The query text to embed
            
        Returns:
            np.ndarray: The query embedding
        """
        # Generate cache key
        cache_key = self._get_cache_key(text)
        cache_file = self._get_cache_file_path(cache_key)
        
        # Check if embedding is cached
        if os.path.exists(cache_file):
            try:
                embedding = np.load(cache_file)
                logger.debug(f"Retrieved embedding from cache for key {cache_key}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        # If not cached or cache loading failed, compute embedding
        embedding = self.provider.embed_query(text)
        
        # Cache the embedding
        try:
            np.save(cache_file, embedding)
            logger.debug(f"Cached embedding for key {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {str(e)}")
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of documents with caching
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            np.ndarray: The document embeddings
        """
        if not texts:
            return np.array([])
        
        # Check which embeddings are cached and which need to be computed
        cached_embeddings = {}
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_file = self._get_cache_file_path(cache_key)
            
            if os.path.exists(cache_file):
                try:
                    embedding = np.load(cache_file)
                    cached_embeddings[i] = embedding
                    logger.debug(f"Retrieved embedding from cache for key {cache_key}")
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {str(e)}")
                    texts_to_embed.append(text)
                    text_indices.append(i)
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Compute embeddings for non-cached texts
        if texts_to_embed:
            computed_embeddings = self.provider.embed_documents(texts_to_embed)
            
            # Cache the computed embeddings
            for idx, text_idx in enumerate(text_indices):
                text = texts_to_embed[idx]
                embedding = computed_embeddings[idx]
                
                cache_key = self._get_cache_key(text)
                cache_file = self._get_cache_file_path(cache_key)
                
                try:
                    np.save(cache_file, embedding)
                    logger.debug(f"Cached embedding for key {cache_key}")
                except Exception as e:
                    logger.warning(f"Failed to cache embedding: {str(e)}")
                
                cached_embeddings[text_idx] = embedding
        
        # Arrange embeddings in original order
        embeddings = []
        for i in range(len(texts)):
            embeddings.append(cached_embeddings[i])
        
        return np.array(embeddings)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings"""
        return self.provider.dimension
    
    def get_provider_name(self) -> str:
        """Return the name of the embedding provider"""
        return f"Cached{self.provider.get_provider_name()}"


class EmbeddingFactory:
    """
    Factory class for creating embedding providers
    """
    
    @staticmethod
    def create_provider(
        provider_type: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        cache_dir: str = ".vector_cache",
        **kwargs
    ) -> EmbeddingProvider:
        """
        Create an embedding provider based on type
        
        Args:
            provider_type: Type of provider (huggingface, openai, google, etc.)
            model_name: Model name for the provider
            api_key: API key for the provider
            cache: Whether to use caching
            cache_dir: Directory for caching embeddings
            **kwargs: Additional arguments for specific providers
            
        Returns:
            EmbeddingProvider: The created embedding provider
        """
        provider_type = provider_type.lower()
        
        if provider_type == "huggingface":
            model = model_name or "all-MiniLM-L6-v2"
            provider = HuggingFaceEmbeddings(model_name=model, **kwargs)
            
        elif provider_type == "openai":
            model = model_name or "text-embedding-3-small"
            provider = OpenAIEmbeddings(model_name=model, api_key=api_key, **kwargs)
            
        elif provider_type == "google":
            model = model_name or "text-embedding-004"
            provider = GoogleAIEmbeddings(model_name=model, api_key=api_key, **kwargs)
            
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Wrap with caching if requested
        if cache:
            return CachedEmbeddingProvider(provider, cache_dir=cache_dir)
        
        return provider


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create embeddings using the factory
    try:
        # Create a Hugging Face embedding provider
        # hf_embeddings = EmbeddingFactory.create_provider(
        #     provider_type="huggingface",
        #     model_name="all-MiniLM-L6-v2",
        #     cache=True
        # )

        # open_ai_embeddings = EmbeddingFactory.create_provider(
        #     provider_type="openai",
        #     model_name="text-embedding-3-small",
        #     cache=True
        # )

        google_ai_embeddings = EmbeddingFactory.create_provider(
            provider_type="google",
            model_name="text-embedding-004",
            cache=True
        )
        
        # Test embeddings
        texts = [
            "Call centers handle customer service inquiries.",
            "Vector embeddings help with semantic search."
        ]
        
        query = "How do call centers use AI?"
        
        # Generate embeddings
        query_embedding = google_ai_embeddings.embed_query(query)
        document_embeddings = google_ai_embeddings.embed_documents(texts)
        
        # # Print shapes
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Document embeddings shape: {document_embeddings.shape}")
        
        # Calculate similarities
        similarities = document_embeddings @ query_embedding
        
        # Print results
        for i, (text, similarity) in enumerate(zip(texts, similarities)):
            print(f"Document {i+1}: {text}")
            print(f"Similarity: {similarity:.4f}")
            print()
            
    except Exception as e:
        logger.error(f"Error in example: {str(e)}") 