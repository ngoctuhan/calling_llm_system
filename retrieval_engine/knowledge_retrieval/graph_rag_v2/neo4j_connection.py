import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import re
from neo4j import GraphDatabase, Driver
from .graph_extractor import KnowledgeTriplet
import unicodedata
from unidecode import unidecode
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
                    # Process differently based on triplet type
                    subject = None
                    predicate = None
                    object_entity = None
                    description = None
                    
                    if isinstance(triplet, KnowledgeTriplet):
                        # Directly get attributes from KnowledgeTriplet object
                        subject = triplet.subject
                        predicate = triplet.predicate
                        object_entity = triplet.object
                        description = triplet.description
                    else:
                        logger.warning(f"Unsupported triplet format: {triplet}")
                        continue
                    
                    # Create triplet in Neo4j with description
                    query = f"""
                    // Create or get subject entity
                    MERGE (s:Entity {{name: $subject}})
                    
                    // Create or get object entity
                    MERGE (o:Entity {{name: $object}})
                    
                    // Create relationship between subject and object with predicate as relationship type
                    // Include description in relationship properties
                    MERGE (s)-[r:{predicate} {{name: $predicate, description: $description}}]->(o)
                    
                    // Link to document
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
                        description=description,
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
    
    def add_relationship_embeddings(self, relationship_embeddings: Dict[str, List[float]]):
        """
        Add embeddings to relationship properties
        
        Args:
            relationship_embeddings: Dictionary mapping relationship keys to embedding vectors
                Format of key: "subject|predicate|object"
        """
        if not relationship_embeddings:
            return
            
        self.connect()
        
        valid_embeddings = 0
        skipped_embeddings = 0
        
        with self._driver.session(database=self.database) as session:
            for rel_key, embedding in relationship_embeddings.items():
                # Skip if relationship key is empty or embedding is invalid
                if not rel_key or not embedding or not isinstance(embedding, list):
                    logger.warning(f"Skipping invalid embedding for relationship: {rel_key}")
                    skipped_embeddings += 1
                    continue
                
                # Parse relationship key
                try:
                    subject, predicate, object_entity = rel_key.split('|')
                    # Create Cypher query to find and update the relationship
                    session.run(
                        """
                        MATCH (s:Entity {name: $subject})-[r]->(o:Entity {name: $object})
                        WHERE r.name = $predicate
                        SET r.embedding = $embedding
                        """,
                        subject=subject,
                        predicate=predicate,
                        object=object_entity,
                        embedding=embedding
                    )
                    valid_embeddings += 1
                except Exception as e:
                    logger.error(f"Error adding embedding for relationship '{rel_key}': {str(e)}")
                    skipped_embeddings += 1
                
        logger.info(f"Added embeddings for {valid_embeddings} relationships. Skipped {skipped_embeddings} invalid embeddings.")
    
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
        Perform simple query on knowledge graph
        
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
            r.description AS description,
            d.id AS document_id,
            d.title AS document_title
        LIMIT $limit
        """
        params["limit"] = limit
        
        return self.execute_query(query, params)
    
    def run_vector_search(
        self, 
        query_embedding: List[float],
        node_type: str = "entity",  # Options: "entity" or "relationship"
        limit: int = 10,
        similarity_threshold: float = 0.7
    ):
        """
        Perform vector search using Neo4j
        
        Args:
            query_embedding: Embedding vector of the query
            node_type: Type of nodes to search ("entity" or "relationship")
            limit: Maximum number of results
            similarity_threshold: Similarity threshold
            
        Returns:
            List of entities/relationships and their similarity scores
        """
        self.connect()
        
        # Try using vector index if available
        try:
            if node_type.lower() == "entity":
                # Search entities using direct vector index query
                vector_query = """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                CALL db.index.vector.queryNodes(
                    'entity_embedding',
                    $k,
                    $query_embedding
                ) YIELD node, score
                WHERE score >= $threshold
                RETURN DISTINCT node.name AS name, score, 'entity' AS type
                ORDER BY score DESC
                LIMIT $limit
                """
            else:
                # Search relationships using direct vector index query
                vector_query = """
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE r.embedding IS NOT NULL
                CALL db.index.vector.queryRelationships(
                    'relationship_embedding',
                    $k,
                    $query_embedding
                ) YIELD relationship, score
                WHERE score >= $threshold
                MATCH (s)-[relationship]->(o)
                RETURN DISTINCT s.name AS subject, relationship.name AS predicate, o.name AS object, 
                       score, 'relationship' AS type, relationship.description AS description
                ORDER BY score DESC
                LIMIT $limit
                """
            
            params = {
                "query_embedding": query_embedding,
                "threshold": similarity_threshold,
                "limit": limit,
                "k": limit * 2  # Request more results to ensure we get enough after filtering
            }
            
            result = self.execute_query(vector_query, params)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
        
        # Fall back to FAISS calculation (simplified for your case)
        try:
            import numpy as np
            import faiss
            logger.info(f"Using FAISS for {node_type} similarity search")
            
            if node_type.lower() == "entity":
                # Retrieve all entities with embeddings
                query = """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                RETURN e.name AS name, e.embedding AS embedding
                """
                items = self.execute_query(query)
                
                if not items:
                    return []
                
                # Extract names and embeddings
                names = [item["name"] for item in items]
                embeddings = [item["embedding"] for item in items]
                
            else:
                # Retrieve all relationships with embeddings
                query = """
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE r.embedding IS NOT NULL
                RETURN s.name AS subject, r.name AS predicate, o.name AS object, 
                       r.embedding AS embedding, r.description AS description
                """
                items = self.execute_query(query)
                
                if not items:
                    return []
                
                # For relationships, create composite keys
                names = []
                embeddings = []
                for item in items:
                    names.append({
                        "subject": item["subject"],
                        "predicate": item["predicate"],
                        "object": item["object"],
                        "description": item.get("description", "")
                    })
                    embeddings.append(item["embedding"])
            
            # Convert to numpy arrays
            embeddings_array = np.array(embeddings).astype('float32')
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            # Build FAISS index
            d = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(d)
            
            # Normalize vectors
            faiss.normalize_L2(embeddings_array)
            faiss.normalize_L2(query_embedding_array)
            
            # Add vectors to index
            index.add(embeddings_array)
            
            # Search
            search_k = min(limit, len(embeddings))
            distances, indices = index.search(query_embedding_array, search_k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                score = float(distance)
                
                if score >= similarity_threshold and idx < len(names):
                    if node_type.lower() == "entity":
                        results.append({
                            "name": names[idx],
                            "score": score,
                            "type": "entity"
                        })
                    else:
                        rel_info = names[idx]
                        results.append({
                            "subject": rel_info["subject"],
                            "predicate": rel_info["predicate"],
                            "object": rel_info["object"],
                            "description": rel_info["description"],
                            "score": score,
                            "type": "relationship"
                        })
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except ImportError:
            logger.warning("FAISS not installed, using basic similarity calculation")
            # Implement basic similarity calculation if needed
            return []
    
    def query_entity_relationships(self, entity_names: List[str], relationship_names: List[str], limit: int = 20):
        """
        Query relationships based on specified entities and relationships
        
        Args:
            entity_names: List of entity names to filter by
            relationship_names: List of relationship names to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching relationships
        """
        self.connect()
        
        query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE 
            (s.name IN $entity_names OR o.name IN $entity_names) 
            AND r.name IN $relationship_names
            AND type(r) <> 'FROM_DOCUMENT'
        RETURN DISTINCT
            s.name AS subject,
            r.name AS predicate,
            type(r) AS predicate_type,
            o.name AS object,
            r.description AS description,
            CASE 
                WHEN s.name IN $entity_names AND o.name IN $entity_names THEN 3.0
                ELSE 2.0
            END AS relevance_score
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        
        params = {
            "entity_names": entity_names,
            "relationship_names": relationship_names,
            "limit": limit
        }
        
        return self.execute_query(query, params) 