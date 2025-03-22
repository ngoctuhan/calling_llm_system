import json
import uuid
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from llm_services import LLMProvider
from .embeddings import EmbeddingProvider

# Set up logging
logger = logging.getLogger(__name__)

# Define the knowledge extraction prompt template as a constant
DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). "
    "Ensure completeness and avoid omissions. Exclude stopwords.\n"
    "\n"
    "### Instructions:\n"
    "- Identify all possible knowledge triplets without omission.\n"
    "- Subjects and objects should be meaningful entities (e.g., people, places, organizations, concepts).\n"
    "- Predicates should express a clear relationship (e.g., 'is a', 'founded in', 'works at').\n"
    "- Extract temporal and spatial relationships if applicable.\n"
    "- Maintain contextual integrity when forming triplets.\n"
    "- Exclude stopwords and unnecessary words.\n"
    "\n"
    "---------------------\n"
    "### Example:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets:\n"
    "(Alice, is mother of, Bob)\n"
    "\n"
    "Text: Tesla was founded by Elon Musk in 2003 in the United States.\n"
    "Triplets:\n"
    "(Tesla, was founded by, Elon Musk)\n"
    "(Tesla, was founded in, 2003)\n"
    "(Tesla, was founded in, United States)\n"
    "\n"
    "Text: The Eiffel Tower is located in Paris and was completed in 1889.\n"
    "Triplets:\n"
    "(Eiffel Tower, is located in, Paris)\n"
    "(Eiffel Tower, was completed in, 1889)\n"
    "\n"
    "Text: Apple Inc. acquired Beats Electronics for $3 billion in 2014.\n"
    "Triplets:\n"
    "(Apple Inc., acquired, Beats Electronics)\n"
    "(Apple Inc., acquired for, $3 billion)\n"
    "(Apple Inc., acquired in, 2014)\n"
    "\n"
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
        
    def __repr__(self):
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triplet to dictionary"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }

class GraphExtractor:
    """Graph extractor for knowledge graph creation"""
    
    def __init__(
        self,
        llm: LLMProvider,
        max_knowledge_triplets: int = 20,
        prompt_template: Optional[str] = None,
        batch_size: int = 5,
        max_concurrency: int = 10,
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
                )
                triplets.append(triplet)
            
            logger.info(f"Extracted {len(triplets)} triplets from text")
            return triplets
            
        except Exception as e:
            logger.error(f"Error extracting triplets: {str(e)}")
            return []
    
    async def extract_embeddings_entities(
        self,
        triplets: Any|List[KnowledgeTriplet],
    ) -> List[Dict[str, Any]]:
        """Extract embeddings for entities from text"""
        if isinstance(triplets, KnowledgeTriplet):
            triplets = [triplets]
            entities = []
            for triplet in triplets:
                entities.append(triplet.subject)
                entities.append(triplet.object)
        elif isinstance(triplets, list):
            entities = self._extract_entities_from_text_triplets(triplets)
        else:
            raise ValueError("triplets must be a KnowledgeTriplet or a list of KnowledgeTriplet")
        
        # Remove duplicates
        entities = list(set(entities))
        print(entities)
        # Extract embeddings for entities
        embeddings = self.embedding_provider.embed_documents(entities)

        # Create a dictionary of entity embeddings
        entity_embeddings = {entity: embedding for entity, embedding in zip(entities, embeddings)}

        return entity_embeddings
        
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
    
    def _extract_entities_from_text_triplets(
        self,
        triplets: List[str]
    ) -> List[str]:
        """Extract entities from triplets"""
        # "(Nguyễn trãi) --[Chữ hán]--> (阮廌)"
        entities = []
        for triplet in triplets:
            pattern = r"\((.*?)\)\s*--\[(.*?)\]-->\s*\((.*?)\)"
            match = re.search(pattern, triplet)
            if match:
                entities.append(match.group(1))
                entities.append(match.group(3))
        return entities


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