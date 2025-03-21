import json
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from llm_services import LLMProvider

# Set up logging
logger = logging.getLogger(__name__)

# Define the knowledge extraction prompt template as a constant
DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

class KnowledgeTriplet:
    """Knowledge triplet extracted from text"""
    
    def __init__(
        self,
        subject: str,
        predicate: str,
        object: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a knowledge triplet
        
        Args:
            subject: The subject entity of the triplet
            predicate: The predicate/relation of the triplet
            object: The object entity of the triplet
            metadata: Dictionary of metadata for the triplet
        """
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"
    

class GraphExtractor:
    """Graph extractor for knowledge graph creation"""
    
    def __init__(
        self,
        llm: LLMProvider,
        max_knowledge_triplets: int = 10,
        prompt_template: Optional[str] = None,
        batch_size: int = 5,
        max_concurrency: int = 10
    ):
        """
        Initialize a graph extractor
        
        Args:
            llm: LLMProvider instance for text generation
            max_knowledge_triplets: Maximum number of triplets to extract
            prompt_template: Optional custom prompt template
            batch_size: Number of texts to process in a single batch
            max_concurrency: Maximum number of concurrent LLM calls
        """
        if not llm:
            raise ValueError("LLM provider must be specified.")
            
        self.llm = llm
        self.max_knowledge_triplets = max_knowledge_triplets
        self.prompt_template = prompt_template or DEFAULT_KG_TRIPLET_EXTRACT_TMPL
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
    
    async def _parse_triplets(
        self, 
        response: str, 
        max_length: int = 128
    ) -> List[Tuple[str, str, str, float]]:
        """
        Parse triplets from the response text
        
        Args:
            response: Response text from LLM
            max_length: Maximum length of each token
            
        Returns:
            List of tuples (subject, predicate, object, confidence)
        """
        knowledge_strs = response.strip().split("\n")
        results = []
        
        for text in knowledge_strs:
            if "(" not in text or ")" not in text or text.index(")") < text.index("("):
                # skip empty lines and non-triplets
                continue
                
            triplet_part = text[text.index("(") + 1 : text.index(")")]
            tokens = triplet_part.split(",")
            
            if len(tokens) != 3:
                continue

            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue

            # Strip double quotes and Capitalize triplets for disambiguation
            subj, pred, obj = (
                entity.strip('"').capitalize() for entity in [subj, pred, obj]
            )

            # Default confidence is 1.0
            results.append((subj, pred, obj, 1.0))
            
        return results
    
    async def extract_triplets(
        self, 
        text: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> List[KnowledgeTriplet]:
        """
        Extract knowledge triplets from text or multiple texts
        
        Args:
            text: Single text string or list of text strings to extract knowledge from
            metadata: Optional metadata dictionary or list of metadata dictionaries
                     If text is a list, metadata should also be a list of same length
                     
        Returns:
            List of KnowledgeTriplet objects
        """
        # Handle single text case
        if isinstance(text, str):
            return await self._extract_from_single_text(text, metadata or {})
        
        # Handle batch text case
        elif isinstance(text, list):
            if not text:
                return []
                
            # Validate metadata if provided
            if metadata is not None:
                if not isinstance(metadata, list):
                    raise ValueError("When text is a list, metadata must also be a list")
                if len(metadata) != len(text):
                    raise ValueError("metadata list must have the same length as text list")
            else:
                metadata = [{} for _ in text]
                
            return await self._extract_from_text_batch(text, metadata)
        
        else:
            raise ValueError("text must be a string or a list of strings")
    
    async def _extract_from_single_text(
        self, 
        text: str,
        metadata: Dict[str, Any]
    ) -> List[KnowledgeTriplet]:
        """Extract triplets from a single text"""
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for extraction, returning empty list")
            return []
            
        try:
            # Format the prompt
            prompt = self.prompt_template.format(
                max_knowledge_triplets=self.max_knowledge_triplets,
                text=text
            )
            
            # Get response from LLM
            response = await self.llm.generate(prompt)
            
            # Parse triplets
            parsed_triplets = await self._parse_triplets(response)
            
            # Convert to KnowledgeTriplet objects with metadata
            triplets = []
            for subj, pred, obj, conf in parsed_triplets:
                # Create new metadata dict with confidence and original metadata
                triplet_metadata = {"confidence": conf}
                triplet_metadata.update(metadata)
                
                triplet = KnowledgeTriplet(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    metadata=triplet_metadata
                )
                triplets.append(triplet)
            
            logger.info(f"Extracted {len(triplets)} triplets from text")
            return triplets
            
        except Exception as e:
            logger.error(f"Error extracting triplets: {str(e)}")
            return []
    
    async def _extract_from_text_batch(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]]
    ) -> List[KnowledgeTriplet]:
        """Extract triplets from a batch of texts"""
        all_triplets = []
        
        # Process texts in batches for better performance
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_metadata = metadata_list[i:i+self.batch_size]
            
            # Extract triplets concurrently with limited concurrency
            sem = asyncio.Semaphore(self.max_concurrency)
            
            async def extract_with_semaphore(text, meta):
                async with sem:
                    return await self._extract_from_single_text(text, meta)
            
            # Run extraction tasks concurrently
            extraction_tasks = [
                extract_with_semaphore(text, meta)
                for text, meta in zip(batch_texts, batch_metadata)
            ]
            batch_results = await asyncio.gather(*extraction_tasks)
            
            # Flatten results
            for triplet_list in batch_results:
                all_triplets.extend(triplet_list)
                
            logger.info(f"Processed batch of {len(batch_texts)} texts, extracted {len(all_triplets)} triplets so far")
        
        return all_triplets