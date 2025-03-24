import json
import uuid
import re
from unidecode import unidecode
import traceback
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from llm_services import LLMProvider
from .embeddings import EmbeddingProvider

# Set up logging
logger = logging.getLogger(__name__)

DEFAULT_KG_TRIPLET_EXTRACT_TMPL ="""
Extract up to {max_knowledge_triplets} knowledge triplets from the provided text in the format: (subject, predicate, object, description).

Instructions:
- Extract ALL meaningful triplets (up to max_triplets limit)
- Subject/Object should be specific entities or concepts (people, places, organizations, ideas)
- Predicates should clearly express relationships between subject and object
- Include temporal, spatial, and causal relationships
- Maintain original context and meaning from the source text
- Avoid including stopwords in subject/predicate/object components
- The description should be a natural language sentence that explains the triplet relationship using language from the source text

Examples:

Input text: "OpenAI was founded in San Francisco in 2015 by Sam Altman and Elon Musk among others. The company focuses on ensuring artificial general intelligence benefits humanity."

Output triplets:
(OpenAI, founded_in, San Francisco, OpenAI was established in the city of San Francisco.)
(OpenAI, founded_in, 2015, OpenAI was created in the year 2015.)
(Sam Altman, co-founded, OpenAI, Sam Altman was one of the founding members of OpenAI.)
(Elon Musk, co-founded, OpenAI, Elon Musk was among the original founders of OpenAI.)
(OpenAI, focuses_on, AGI_benefits, OpenAI aims to ensure that artificial general intelligence benefits humanity.)

Input text: "The Great Barrier Reef, located off Australia's northeastern coast, is the world's largest coral reef system. It faces threats from climate change, including rising sea temperatures and ocean acidification."

Output triplets:
(Great Barrier Reef, located_off, Australia's northeastern coast, The Great Barrier Reef is situated along the northeastern coastline of Australia.)
(Great Barrier Reef, is, largest coral reef system, The Great Barrier Reef represents the world's largest coral reef ecosystem.)
(climate change, threatens, Great Barrier Reef, Climate change poses a significant threat to the Great Barrier Reef.)
(rising sea temperatures, threatens, Great Barrier Reef, Increasing ocean temperatures endanger the Great Barrier Reef ecosystem.)
(ocean acidification, threatens, Great Barrier Reef, The process of ocean acidification presents a danger to the Great Barrier Reef.)

Text: {text}
Triplets:
"""


class KnowledgeTriplet:
    """Knowledge triplet extracted from text"""
    
    def __init__(
        self,
        subject: str,
        predicate: str,
        object: str,
        description: str
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
        self.description = description

        
    def __repr__(self):
        return f"({self.subject}) --[{self.predicate}]--> ({self.object}): {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triplet to dictionary"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description
        }

class GraphExtractor:
    """Graph extractor for knowledge graph creation"""
    
    def __init__(
        self,
        llm: LLMProvider,
        max_knowledge_triplets: int = 30,
        prompt_template: Optional[str] = None,
        batch_size: int = 16,
        max_concurrency: int = 16,
        embedding_provider: EmbeddingProvider = None,
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
        self.embedding_provider = embedding_provider
    
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
            
            if len(tokens) != 4:
                continue

            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj, description = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue

            # Strip double quotes and Capitalize triplets for disambiguation
            subj, pred, obj, description = (
                entity.strip('"').capitalize() for entity in [subj, pred, obj, description]
            )

            # Default confidence is 1.0
            results.append((subj, pred, obj, description, 1.0))
            
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
            for subj, pred, obj, description,  conf in parsed_triplets:
                # Create new metadata dict with confidence and original metadata
                triplet_metadata = {"confidence": conf}
                triplet_metadata.update(metadata)
                
                triplet = KnowledgeTriplet(
                    subject=subj.capitalize(),
                    predicate=unidecode(pred).lower().replace(" ", "_"),
                    object=obj.capitalize(),
                    description=description
                )
                triplets.append(triplet)
            
            logger.info(f"Extracted {len(triplets)} triplets from text")
            return triplets
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error extracting triplets: {str(e)}")
            return []
    
    async def extract_embeddings_entities(
        self,
        triplets: List[KnowledgeTriplet],
    ) -> List[Dict[str, Any]]:
        """Extract embeddings for entities from text"""
        if isinstance(triplets, KnowledgeTriplet):
            triplets = [triplets]
            entities = []
            for triplet in triplets:
                entities.append(triplet.subject)
                entities.append(triplet.object)
        elif isinstance(triplets, list):
            entities = [triplet.subject for triplet in triplets]
            entities.extend([triplet.object for triplet in triplets])
        else:
            raise ValueError("triplets must be a KnowledgeTriplet or a list of KnowledgeTriplet")
        
        # Remove duplicates
        entities = list(set(entities))

        # Extract embeddings for entities
        embeddings = self.embedding_provider.embed_documents(entities)

        # Create a dictionary of entity embeddings
        entity_embeddings = {entity: embedding for entity, embedding in zip(entities, embeddings)}

        return entity_embeddings
    
    async def extract_relationship_embeddings(
        self,
        triplets: List[KnowledgeTriplet],
    ) -> Dict[str, List[float]]:
        """
        Extract embeddings for relationships from triplets
        
        Args:
            triplets: List of KnowledgeTriplet objects
            
        Returns:
            Dictionary mapping relationship keys to embedding vectors
                Format of key: "subject|predicate|object"
        """
        if not self.embedding_provider:
            raise ValueError("Embedding provider is required to extract relationship embeddings")
            
        if isinstance(triplets, KnowledgeTriplet):
            triplets = [triplets]
            
        # Create relationship texts to embed
        relationship_texts = []
        relationship_keys = []
        
        for triplet in triplets:
            # Format the relationship text to capture semantics
            rel_text = f"{triplet.subject} {triplet.predicate} {triplet.object}"
            if triplet.description:
                rel_text = triplet.description
            else:
                rel_text = f"{triplet.subject} {triplet.predicate} {triplet.object}"
                
            relationship_texts.append(rel_text)
            # Create a unique key for this relationship
            rel_key = f"{triplet.subject}|{triplet.predicate}|{triplet.object}"
            relationship_keys.append(rel_key)
            
        # Generate embeddings
        embeddings = self.embedding_provider.embed_documents(relationship_texts)
        
        # Create a dictionary of relationship embeddings
        relationship_embeddings = {key: embedding for key, embedding in zip(relationship_keys, embeddings)}
        
        return relationship_embeddings
        
    async def _extract_from_text_batch(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract triplets from a batch of texts
        
        Example usage with Neo4jConnection:
        ```python
        # Extract triplets from multiple texts
        extractor = GraphExtractor(llm_provider)
        texts = ["Albert Einstein was a German physicist.", "Marie Curie was a Polish physicist."]
        metadata_list = [
            {"document_id": "doc1", "source": "Wikipedia", "file_path": "/path/to/einstein.txt"},
            {"document_id": "doc2", "source": "Encyclopedia", "file_path": "/path/to/curie.txt"}
        ]
        
        # Extract triplets 
        results = await extractor.extract_triplets(texts, metadata_list)
        
        # Save to Neo4j
        neo4j_conn = Neo4jConnection()
        await neo4j_conn.connect()
        await neo4j_conn.setup_database()
        await neo4j_conn.add_extraction_results(results)
        ```
        """
        all_results = []
        total_triplets = 0
        
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
            
            # Create result objects for this batch
            for idx, triplets in enumerate(batch_results):
                batch_idx = i + idx
                result = {
                    "text": texts[batch_idx],
                    "metadata": metadata_list[batch_idx],
                    "triplets": triplets
                }
                all_results.append(result)
                total_triplets += len(triplets)
            
            logger.info(f"Processed batch of {len(batch_texts)} texts, extracted {total_triplets} triplets so far")
        
        return all_results
    
    def get_sample_mock_data(self):
        """Get sample mock data for testing"""
        return [
            {
                "text": "\nNguyễn Trãi (chữ Hán: 阮廌; sinh 1380 – mất 19 tháng 9 năm 1442), hiệu là Ức Trai (抑齋), là một nhà chính trị, nhà văn, nhà văn hóa lớn của dân tộc Việt Nam. \nÔng đã tham gia tích cực cuộc Khởi nghĩa Lam Sơn do Lê Lợi lãnh đạo chống lại sự xâm lược của nhà Minh (Trung Quốc) với Đại Việt. \nKhi cuộc khởi nghĩa thành công vào năm 1428, Nguyễn Trãi trở thành một trong những khai quốc công thần của triều đại quân chủ nhà Hậu Lê trong Lịch sử Việt Nam.[2]\n",
                "metadata": {
                "source": "Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i",
                "title": "Nguyễn Trãi",
                "author": "Wikipedia",
                "published_date": "2021-03-01",
                "content_type": "text/plain",
                "language": "vi",
                "format": "text",
                "encoding": "utf-8"
                },
                "triplets": [
                    "(Nguyễn trãi) --[Chữ hán]--> (阮廌)",
                    "(Nguyễn trãi) --[Sinh]--> (1380)",
                    "(Nguyễn trãi) --[Mất]--> (1442)",
                    "(Nguyễn trãi) --[Mất month]--> (9)",
                    "(Nguyễn trãi) --[Hiệu]--> (Ức trai)",
                    "(Nguyễn trãi) --[Is a]--> (Nhà chính trị)",
                    "(Nguyễn trãi) --[Is a]--> (Nhà văn)",
                    "(Nguyễn trãi) --[Is a]--> (Nhà văn hóa)",
                    "(Nguyễn trãi) --[Is of]--> (Dân tộc việt nam)",
                    "(Nguyễn trãi) --[Tham gia]--> (Khởi nghĩa lam sơn)",
                    "(Khởi nghĩa lam sơn) --[Lãnh đạo]--> (Lê lợi)",
                    "(Khởi nghĩa lam sơn) --[Chống lại]--> (Xâm lược nhà minh)",
                    "(Nhà minh) --[Is of]--> (Trung quốc)",
                    "(Nhà minh) --[Xâm lược]--> (Đại việt)",
                    "(Khởi nghĩa) --[Thành công]--> (1428)",
                    "(Nguyễn trãi) --[Trở thành]--> (Khai quốc công thần)",
                    "(Nguyễn trãi) --[Is of]--> (Triều đại quân chủ nhà hậu lê)",
                    "(Triều đại quân chủ nhà hậu lê) --[Is in]--> (Lịch sử việt nam)"
                ]
            },
            {
                "text": "\nHoàng Mậu Trung (hiệu là Ngọc Tự Hàn) sinh năm 1998 năm nay 27 tuổi học tại PTIT. Sinh ra trong một gia đình thuần nông tại Hà Lĩnh, Hà Trung, Thanh Hóa.\n",
                "metadata": {
                "source": "Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i",
                "title": "Nguyễn Trãi",
                "author": "Wikipedia",
                "published_date": "2021-03-01",
                "content_type": "text/plain",
                "language": "vi",
                "format": "text",
                "encoding": "utf-8"
                },
                "triplets": [
                    "(Hoàng mậu trung) --[Hiệu là]--> (Ngọc tự hàn)",
                    "(Hoàng mậu trung) --[Sinh năm]--> (1998)",
                    "(Hoàng mậu trung) --[Tuổi]--> (27)",
                    "(Hoàng mậu trung) --[Học tại]--> (Ptit)",
                    "(Hoàng mậu trung) --[Sinh ra trong]--> (Gia đình)",
                    "(Hoàng mậu trung) --[Sinh ra tại]--> (Hà lĩnh)",
                    "(Hà lĩnh) --[Thuộc]--> (Hà trung)",
                    "(Hà trung) --[Thuộc]--> (Thanh hóa)",
                    "(Gia đình) --[Thuần nông]--> (True)"
                ]
            }
            ]