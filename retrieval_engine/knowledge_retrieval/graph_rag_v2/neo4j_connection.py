import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import re
from neo4j import GraphDatabase, Driver
from .graph_extractor import KnowledgeTriplet
import unicodedata

logger = logging.getLogger(__name__)

class SimpleNeo4jConnection:
    """Simple Neo4j connection for RAG triplet storage and querying"""
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = "neo4j",
        embedding_dim: int = 1536  # Default for OpenAI embeddings
    ):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env variable)
            username: Neo4j username (defaults to NEO4J_USERNAME env variable)
            password: Neo4j password (defaults to NEO4J_PASSWORD env variable)
            database: Neo4j database name
            embedding_dim: Dimension of entity embeddings for vector search
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.database = database
        self.embedding_dim = embedding_dim
        self._driver = None
        self._initialized = False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def connect(self):
        """Establish connection to Neo4j"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Check if connection is successful
            try:
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
    
    def close(self):
        """Close the Neo4j connection"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def setup_database(self):
        """Setup database schema, indexes, and constraints"""
        if self._initialized:
            return
            
        self.connect()
        
        # Create constraints and indexes
        with self._driver.session(database=self.database) as session:
            # Ensure entity names are unique
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            
            # Create index for documents
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
            
            # Create vector index for entity embeddings
            try:
                # First check if the database supports vector indexes
                version_result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
                version = version_result.single()["version"]
                major_version = int(version.split('.')[0])
                
                if major_version >= 5:  # Neo4j 5.0+ supports vector indexes
                    session.run(
                        f"""
                        CREATE VECTOR INDEX entity_embedding IF NOT EXISTS 
                        FOR (n:Entity) ON (n.embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dim},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                        """
                    )
                    logger.info("Vector index created for entity embeddings")
                else:
                    logger.warning(f"Neo4j version {version} doesn't support vector indexes. Embeddings will use manual similarity calculation.")
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")
        
        self._initialized = True
        logger.info("Database setup complete")
    
    def add_extraction_triplet_results(self, extraction_results: List[Dict[str, Any]]):
        """
        Add extraction results to the graph
        
        Args:
            extraction_results: List of extraction result dictionaries 
                Each dictionary should contain 'text', 'metadata', and 'triplets' keys
        """
        if not extraction_results:
            return
            
        self.connect()
        
        # Process each extraction result
        for result in extraction_results:
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            triplets = result.get('triplets', [])
            
            if not triplets:
                continue
                
            # Create document node if text exists
            document_id = metadata.get('title', str(uuid.uuid4()))
            
            with self._driver.session(database=self.database) as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {id: $document_id})
                    SET d += $metadata
                    """,
                    document_id=document_id,
                    metadata=metadata
                )
                
                # Add all triplets
                for triplet in triplets:
                    # Xử lý khác nhau dựa vào loại dữ liệu của triplet
                    if isinstance(triplet, KnowledgeTriplet):
                        # Trực tiếp lấy các thuộc tính từ đối tượng KnowledgeTriplet
                        subject = triplet.subject.capitalize()
                        predicate = triplet.predicate.capitalize()
                        object_entity = triplet.object.capitalize()
                    elif isinstance(triplet, tuple) and len(triplet) == 2:
                        # Trích xuất từ chuỗi chứa format: "(subject) --[predicate]--> (object)"
                        subject_str, rest = triplet
                        predicate_obj = rest.split('-->')
                        if len(predicate_obj) != 2:
                            continue
                            
                        predicate = predicate_obj[0].strip('--[]')
                        object_str = predicate_obj[1].strip(' ()')
                        
                        # Chuẩn hóa các tên thực thể
                        subject = subject_str.strip(' ()').capitalize()
                        predicate = predicate.strip().capitalize()
                        object_entity = object_str.strip().capitalize()
                    elif isinstance(triplet, str):
                        # Add logic here 
                        pattern = r"\((.*?)\)\s*--\[(.*?)\]-->\s*\((.*?)\)"
                        match = re.search(pattern, triplet)
                        if match:
                            entity1, relation, entity2 = match.groups()
                            subject = entity1.strip('').capitalize()
                            predicate = relation.strip().capitalize()
                            object_entity = entity2.strip('').capitalize()
                        else:
                            logger.warning(f"Unsupported triplet format: {triplet}")
                            continue
                    else:
                        logger.warning(f"Unsupported triplet format: {triplet}")
                        continue
                    
                    # Use sanitized predicate as the relationship type
                    # Transliterate Vietnamese characters and remove special characters for valid relationship type
                    def sanitize_rel_type(text):
                        # Transliterate Unicode characters to ASCII (including Vietnamese)
                        normalized = unicodedata.normalize('NFKD', text)
                        # Remove diacritical marks and keep only ASCII chars
                        ascii_text = ''.join([c for c in normalized if not unicodedata.combining(c) and ord(c) < 128])
                        # Replace spaces and special chars with underscores
                        sanitized = re.sub(r'[^A-Za-z0-9_]', '_', ascii_text)
                        # Ensure it doesn't start with a number (Neo4j requirement)
                        if sanitized and sanitized[0].isdigit():
                            sanitized = 'REL_' + sanitized
                        # Make sure we have a valid relationship name
                        if not sanitized or sanitized.isspace():
                            sanitized = 'RELATION'
                        return sanitized.upper()
                    
                    rel_type = sanitize_rel_type(predicate)
                    
                    # Tạo triplet trong Neo4j (sử dụng tên thực thể làm ID)
                    query = f"""
                    // Tạo hoặc lấy entity subject
                    MERGE (s:Entity {{name: $subject}})
                    
                    // Tạo hoặc lấy entity object
                    MERGE (o:Entity {{name: $object}})
                    
                    // Tạo quan hệ giữa subject và object với predicate là loại quan hệ
                    // Vẫn giữ nguyên tên quan hệ gốc trong thuộc tính name
                    MERGE (s)-[r:{rel_type} {{name: $predicate}}]->(o)
                    
                    // Liên kết với document
                    WITH s, o
                    MATCH (d:Document {{id: $document_id}})
                    MERGE (s)-[:FROM_DOCUMENT]->(d)
                    MERGE (o)-[:FROM_DOCUMENT]->(d)
                    """
                    
                    session.run(
                        query,
                        subject=subject,
                        predicate=predicate,
                        object=object_entity,
                        document_id=document_id
                    )
            
            logger.info(f"Added {len(triplets)} triplets from document '{document_id}' to Neo4j")
    
    def add_entity_embeddings(self, entity_embeddings: Dict[str, List[float]]):
        """
        Add embeddings to entity nodes
        
        Args:
            entity_embeddings: Dictionary mapping entity names to embedding vectors
        """
        if not entity_embeddings:
            return
            
        self.connect()
        
        valid_embeddings = 0
        skipped_embeddings = 0
        
        with self._driver.session(database=self.database) as session:
            for entity, embedding in entity_embeddings.items():
                # Skip if entity name is empty or embedding is invalid
                if not entity or not embedding or not isinstance(embedding, list):
                    logger.warning(f"Skipping invalid embedding for entity: {entity}")
                    skipped_embeddings += 1
                    continue
                
                try:
                    session.run(
                        """
                        MATCH (e:Entity {name: $entity_name})
                        SET e.embedding = $embedding
                        """,
                        entity_name=entity,
                        embedding=embedding
                    )
                    valid_embeddings += 1
                except Exception as e:
                    logger.error(f"Error adding embedding for entity '{entity}': {str(e)}")
                    skipped_embeddings += 1
                
        logger.info(f"Added embeddings for {valid_embeddings} entities. Skipped {skipped_embeddings} invalid embeddings.")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None):
        """
        Execute a Cypher query
        
        Args:
            query: The Cypher query string
            parameters: Query parameters
            
        Returns:
            Result from Neo4j query
        """
        self.connect()
        parameters = parameters or {}
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, **parameters)
            return [record.data() for record in result]
    
    def query_knowledge_graph(
        self,
        subject: str = None,
        predicate: str = None,
        object_entity: str = None,
        limit: int = 100
    ):
        """
        Thực hiện truy vấn đơn giản vào knowledge graph
        
        Args:
            subject: Optional subject to filter by
            predicate: Optional predicate to filter by
            object_entity: Optional object to filter by
            limit: Maximum number of triplets to return
            
        Returns:
            List of triplets matching the query
        """
        self.connect()
        
        # Build query conditions
        conditions = []
        params = {}
        
        if subject:
            conditions.append("toLower(s.name) CONTAINS toLower($subject)")
            params["subject"] = subject
            
        if object_entity:
            conditions.append("toLower(o.name) CONTAINS toLower($object)")
            params["object"] = object_entity
        
        # Build WHERE clause
        where_clause = " AND ".join(conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        
        # Build base query
        base_query = f"""
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE type(r) <> 'FROM_DOCUMENT'
        {where_clause}
        """
        
        # Add predicate filter if provided
        if predicate:
            predicate_condition = "toLower(r.name) CONTAINS toLower($predicate)"
            params["predicate"] = predicate
            
            if where_clause:
                base_query = f"""
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE type(r) <> 'FROM_DOCUMENT' AND {where_clause[6:]} AND {predicate_condition}
                """
            else:
                base_query = f"""
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE type(r) <> 'FROM_DOCUMENT' AND {predicate_condition}
                """
        
        query = base_query + """
        OPTIONAL MATCH (s)-[:FROM_DOCUMENT]->(d:Document)
        RETURN DISTINCT
            s.name AS subject, 
            r.name AS predicate,
            type(r) AS predicate_type, 
            o.name AS object,
            d.id AS document_id,
            d.title AS document_title
        LIMIT $limit
        """
        params["limit"] = limit
        
        return self.execute_query(query, params)
    
    def run_vector_search(
        self, 
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ):
        """
        Perform vector search using Neo4j
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Maximum number of results
            similarity_threshold: Similarity threshold
            
        Returns:
            List of entities and their similarity scores
        """
        self.connect()
        
        # Try using vector index if available
        try:
            vector_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            CALL db.index.vector.queryNodes(
                'entity_embedding',
                $k,
                $query_embedding
            ) YIELD node, score
            WHERE score >= $threshold
            RETURN DISTINCT node.name AS entity, score
            ORDER BY entity, score DESC
            """
            
            params = {
                "k": limit,
                "query_embedding": query_embedding,
                "threshold": similarity_threshold
            }
            
            result = self.execute_query(vector_query, params)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Vector index search failed, falling back to FAISS: {e}")
        
        # If vector index isn't available or fails, fall back to FAISS calculation
        try:
            import numpy as np
            import faiss
            logger.info("Using FAISS for vector similarity search")
            
            # First retrieve all entities with embeddings
            entities_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN DISTINCT e.name AS entity, e.embedding AS embedding
            """
            
            entities_with_embeddings = self.execute_query(entities_query)
            
            if not entities_with_embeddings:
                logger.warning("No entities with embeddings found")
                return []
                
            # Extract entity names and embeddings
            entity_names = []
            embeddings = []
            
            for entity_data in entities_with_embeddings:
                entity = entity_data.get("entity")
                embedding = entity_data.get("embedding")
                
                if entity and embedding:
                    entity_names.append(entity)
                    embeddings.append(embedding)
            
            if not embeddings:
                logger.warning("No valid embeddings found")
                return []
                
            # Convert to numpy arrays
            embeddings_array = np.array(embeddings).astype('float32')
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            # Get embedding dimension from data
            d = embeddings_array.shape[1]
            
            # Build a flat (CPU-based) index
            index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
            
            # Normalize vectors to unit length (for cosine similarity)
            faiss.normalize_L2(embeddings_array)
            faiss.normalize_L2(query_embedding_array)
            
            # Add vectors to the index
            index.add(embeddings_array)
            
            # Search the index
            search_k = min(limit * 2, len(embeddings))  # Search for more results than needed to apply threshold
            distances, indices = index.search(query_embedding_array, search_k)
            
            # Convert inner product distances to cosine similarity scores
            # (FAISS returns inner product which is equivalent to cosine for normalized vectors)
            
            # Create results with entity names and scores
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # FAISS distances are inner products for normalized vectors: they range from -1 to 1
                # For cosine similarity, we want 1 for most similar, so we use the distance directly
                score = float(distance)
                
                if score >= similarity_threshold and idx < len(entity_names):
                    results.append({
                        "entity": entity_names[idx],
                        "score": score
                    })
                    
                    # Break early if we have enough results
                    if len(results) >= limit:
                        break
            
            # Sort results by entity name, then by score
            results = sorted(results, key=lambda x: (x["entity"], -x["score"]))
            
            return results
            
        except ImportError:
            logger.warning("FAISS not installed, falling back to basic numpy calculation")
            # Fall back to NumPy calculation if FAISS is not available
            
            # First retrieve all entities with embeddings
            entities_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN DISTINCT e.name AS entity, e.embedding AS embedding
            """
            
            entities_with_embeddings = self.execute_query(entities_query)
            
            # Calculate cosine similarity in Python
            from numpy import dot
            from numpy.linalg import norm
            
            def cosine_similarity(v1, v2):
                """Calculate cosine similarity between two vectors"""
                if not v1 or not v2:
                    return 0
                return dot(v1, v2) / (norm(v1) * norm(v2))
            
            # Calculate similarity for each entity
            results = []
            for entity_data in entities_with_embeddings:
                entity = entity_data.get("entity")
                embedding = entity_data.get("embedding")
                
                if entity and embedding:
                    score = cosine_similarity(query_embedding, embedding)
                    if score >= similarity_threshold:
                        results.append({"entity": entity, "score": score})
            
            # Sort by entity name, then by score
            results = sorted(results, key=lambda x: (x["entity"], -x["score"]))[:limit]
            
            return results 