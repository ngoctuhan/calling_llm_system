import os
import uuid
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError

# Import our knowledge triplet class
from .graph_extractor import KnowledgeTriplet

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Asynchronous connection to Neo4j for graph storage and querying"""
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = "neo4j",
        embedding_dim: int = 1536,  # Default for OpenAI embeddings
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0,
        connection_acquisition_timeout: float = 60.0,
        max_transaction_retry_time: float = 30.0
    ):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env variable)
            username: Neo4j username (defaults to NEO4J_USERNAME env variable)
            password: Neo4j password (defaults to NEO4J_PASSWORD env variable)
            database: Neo4j database name
            embedding_dim: Dimension of entity embeddings for vector search
            max_connection_pool_size: Maximum number of connections in the connection pool
            connection_timeout: Connection timeout in seconds
            connection_acquisition_timeout: Timeout for acquiring a connection from the pool
            max_transaction_retry_time: Maximum time to retry failed transactions
        """
        
        self.uri = uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.database = database
        self.embedding_dim = embedding_dim
        
        # Connection settings
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_timeout = connection_timeout
        self.connection_acquisition_timeout = connection_acquisition_timeout
        self.max_transaction_retry_time = max_transaction_retry_time
        
        self._driver = None
        self._initialized = False
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        """Establish connection to Neo4j"""
        if self._driver is None:
            # Configure driver with optimized settings
            self._driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
                connection_timeout=self.connection_timeout,
                connection_acquisition_timeout=self.connection_acquisition_timeout,
                max_transaction_retry_time=self.max_transaction_retry_time
            )
            
            # Check if connection is successful
            try:
                await self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
    
    async def close(self):
        """Close the Neo4j connection"""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    async def setup_database(self):
        """Setup database schema, indexes, and constraints"""
        if self._initialized:
            return
            
        await self.connect()
        
        # Create constraints to ensure uniqueness
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Relation) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        
        # Add indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (r:RELATES) ON (r.name)",
            # Vector index for embedding search
            f"CREATE VECTOR INDEX IF NOT EXISTS entity_embedding IF EXISTS FOR (n:Entity) ON (n.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {self.embedding_dim}, `vector.similarity_function`: 'cosine'}}}}"
        ]
        
        # Execute constraints and indexes
        async with self._driver.session(database=self.database) as session:
            # Apply constraints
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Neo4jError as e:
                    logger.error(f"Error setting up database constraint: {e}")
                    raise
            
            # Apply indexes
            for index in indexes:
                try:
                    await session.run(index)
                except Neo4jError as e:
                    # Some older Neo4j versions might not support vector indexes
                    if "vector" in str(e).lower():
                        logger.warning(f"Vector index not supported in this Neo4j version: {e}")
                    else:
                        logger.error(f"Error setting up database index: {e}")
                        raise
        
        self._initialized = True
        logger.info("Database setup complete")
    
    async def _add_document_and_chunks(
        self, 
        document_id: str, 
        document_metadata: Dict[str, Any] = None,
        chunks_metadata: List[Dict[str, Any]] = None
    ):
        """Add document and chunks to the graph"""
        document_metadata = document_metadata or {}
        chunks_metadata = chunks_metadata or []
        
        # Skip if no chunks to add
        if not document_id:
            return
            
        async with self._driver.session(database=self.database) as session:
            # Create document node
            doc_params = {
                "document_id": document_id,
                **document_metadata
            }
            
            # Create document
            await session.run(
                """
                MERGE (d:Document {id: $document_id})
                SET d += $metadata
                """,
                document_id=document_id,
                metadata={k: v for k, v in doc_params.items() if k != "document_id"}
            )
            
            # Batch chunk creation for better performance
            if chunks_metadata:
                # Prepare parameters for batch operation
                chunk_params = []
                for chunk_meta in chunks_metadata:
                    chunk_id = chunk_meta.get("chunk_id", str(uuid.uuid4()))
                    chunk_params.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "metadata": {k: v for k, v in chunk_meta.items() if k != "chunk_id"}
                    })
                
                # Execute batch chunk creation
                await session.run(
                    """
                    UNWIND $chunks AS chunk
                    MERGE (c:Chunk {id: chunk.chunk_id})
                    SET c += chunk.metadata
                    WITH c, chunk
                    MATCH (d:Document {id: chunk.document_id})
                    MERGE (c)-[:PART_OF]->(d)
                    """,
                    chunks=chunk_params
                )
    
    async def add_triplets(
        self, 
        triplets: List[KnowledgeTriplet], 
        batch_size: int = 100
    ):
        """
        Add knowledge triplets to the graph in batches
        
        Args:
            triplets: List of KnowledgeTriplet objects
            batch_size: Number of triplets to add in a single batch
        """
        if not triplets:
            return
            
        await self.connect()
        start_time = time.time()
        
        # Group by document for efficiency
        document_chunks = defaultdict(list)
        document_ids = set()
        
        # Collect documents and chunks
        for triplet in triplets:
            if triplet.document_id:
                document_ids.add(triplet.document_id)
                
                if triplet.chunk_id:
                    chunk_meta = {
                        "chunk_id": triplet.chunk_id,
                        "page_number": triplet.page_number
                    }
                    
                    # Check if chunk already exists in the list
                    chunk_exists = any(
                        chunk.get("chunk_id") == triplet.chunk_id 
                        for chunk in document_chunks[triplet.document_id]
                    )
                    
                    if not chunk_exists:
                        document_chunks[triplet.document_id].append(chunk_meta)
        
        # Add documents and chunks first
        for doc_id in document_ids:
            await self._add_document_and_chunks(doc_id, chunks_metadata=document_chunks[doc_id])
        
        # Process triplets in batches
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            
            # Create parameters for bulk operation
            entity_params = []
            relation_params = []
            relation_doc_params = []
            relation_chunk_params = []
            relation_source_params = []
            relation_file_path_params = []
            
            # Collect all entities and relationships
            for triplet in batch:
                # Get metadata values
                triplet_id = triplet.triplet_id
                document_id = triplet.document_id
                chunk_id = triplet.chunk_id
                confidence = triplet.confidence
                source = triplet.source if hasattr(triplet, 'source') else None
                file_path = triplet.file_path if hasattr(triplet, 'file_path') else None
                
                # Subject entity parameters
                entity_params.append({
                    "name": triplet.subject,
                    "id": str(uuid.uuid4())
                })
                
                # Object entity parameters
                entity_params.append({
                    "name": triplet.object,
                    "id": str(uuid.uuid4())
                })
                
                # Relationship parameters
                relation_params.append({
                    "subject": triplet.subject,
                    "predicate": triplet.predicate,
                    "object": triplet.object,
                    "triplet_id": triplet_id,
                    "confidence": confidence
                })
                
                # Relationship to document
                if document_id:
                    relation_doc_params.append({
                        "triplet_id": triplet_id,
                        "document_id": document_id
                    })
                
                # Relationship to chunk
                if chunk_id:
                    relation_chunk_params.append({
                        "triplet_id": triplet_id,
                        "chunk_id": chunk_id
                    })
                
                # Source information
                if source:
                    relation_source_params.append({
                        "triplet_id": triplet_id,
                        "source": source
                    })
                
                # File path information
                if file_path:
                    relation_file_path_params.append({
                        "triplet_id": triplet_id,
                        "file_path": file_path
                    })
            
            async with self._driver.session(database=self.database) as session:
                # 1. Create or get all entities in bulk
                await session.run(
                    """
                    UNWIND $entities AS entity
                    MERGE (e:Entity {name: entity.name})
                    ON CREATE SET e.id = entity.id
                    """,
                    entities=entity_params
                )
                
                # 2. Create all relationships in bulk
                await session.run(
                    """
                    UNWIND $relations AS rel
                    MATCH (s:Entity {name: rel.subject})
                    MATCH (o:Entity {name: rel.object})
                    MERGE (s)-[r:RELATES {name: rel.predicate}]->(o)
                    SET r.id = rel.triplet_id,
                        r.confidence = rel.confidence
                    """,
                    relations=relation_params
                )
                
                # 3. Connect relationships to documents in bulk
                if relation_doc_params:
                    await session.run(
                        """
                        UNWIND $rel_docs AS rel_doc
                        MATCH ()-[r:RELATES {id: rel_doc.triplet_id}]->()
                        MATCH (d:Document {id: rel_doc.document_id})
                        MERGE (r)-[:EXTRACTED_FROM]->(d)
                        """,
                        rel_docs=relation_doc_params
                    )
                
                # 4. Connect relationships to chunks in bulk
                if relation_chunk_params:
                    await session.run(
                        """
                        UNWIND $rel_chunks AS rel_chunk
                        MATCH ()-[r:RELATES {id: rel_chunk.triplet_id}]->()
                        MATCH (c:Chunk {id: rel_chunk.chunk_id})
                        MERGE (r)-[:APPEARS_IN]->(c)
                        """,
                        rel_chunks=relation_chunk_params
                    )
                
                # 5. Set source information if available
                if relation_source_params:
                    await session.run(
                        """
                        UNWIND $rel_sources AS rel_source
                        MATCH ()-[r:RELATES {id: rel_source.triplet_id}]->()
                        SET r.source = rel_source.source
                        """,
                        rel_sources=relation_source_params
                    )
                
                # 6. Set file path information if available
                if relation_file_path_params:
                    await session.run(
                        """
                        UNWIND $rel_file_paths AS rel_file_path
                        MATCH ()-[r:RELATES {id: rel_file_path.triplet_id}]->()
                        SET r.file_path = rel_file_path.file_path
                        """,
                        rel_file_paths=relation_file_path_params
                    )
            
            logger.info(f"Added batch of {len(batch)} triplets to Neo4j")
        
        total_time = time.time() - start_time
        logger.info(f"Added {len(triplets)} triplets in {total_time:.2f}s ({len(triplets)/total_time:.1f} triplets/sec)")
    
    async def add_extraction_results(
        self,
        extraction_results: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        Add extraction results from GraphExtractor to the graph
        
        Args:
            extraction_results: List of extraction result dictionaries from GraphExtractor.extract_triplets
                Each dictionary should contain 'text', 'metadata', and 'triplets' keys
            batch_size: Number of triplets to add in a single batch
        """
        if not extraction_results:
            return
            
        # Flatten all triplets from all results
        all_triplets = []
        for result in extraction_results:
            if 'triplets' in result and result['triplets']:
                # Ensure each triplet has the necessary metadata
                for triplet in result['triplets']:
                    # Add missing properties from metadata if they don't exist
                    for key, value in result.get('metadata', {}).items():
                        if key not in triplet.metadata:
                            triplet.metadata[key] = value
                    
                    # Set additional properties on triplet object for easy access in add_triplets
                    triplet.triplet_id = triplet.metadata.get('triplet_id', str(uuid.uuid4()))
                    triplet.document_id = triplet.metadata.get('document_id')
                    triplet.chunk_id = triplet.metadata.get('chunk_id')
                    triplet.page_number = triplet.metadata.get('page_number')
                    triplet.confidence = triplet.metadata.get('confidence', 1.0)
                    triplet.source = triplet.metadata.get('source')
                    triplet.file_path = triplet.metadata.get('file_path')
                    
                all_triplets.extend(result['triplets'])
        
        # Use the existing add_triplets method to add all triplets
        await self.add_triplets(all_triplets, batch_size)
    
    async def add_embeddings(
        self,
        entity_embeddings: Dict[str, List[float]],
        batch_size: int = 100
    ):
        """
        Add embeddings to entity nodes
        
        Args:
            entity_embeddings: Dictionary mapping entity names to embedding vectors
            batch_size: Number of entities to update in a single batch
        """
        if not entity_embeddings:
            return
            
        await self.connect()
        start_time = time.time()
        
        # Process entities in batches
        entities = list(entity_embeddings.keys())
        
        for i in range(0, len(entities), batch_size):
            batch_entities = entities[i:i+batch_size]
            embedding_params = []
            
            # Prepare parameters for bulk operation
            for entity in batch_entities:
                embedding = entity_embeddings.get(entity)
                if embedding:
                    embedding_params.append({
                        "entity_name": entity,
                        "embedding": embedding
                    })
            
            # Execute bulk embedding update
            async with self._driver.session(database=self.database) as session:
                await session.run(
                    """
                    UNWIND $embeddings AS emb
                    MATCH (e:Entity {name: emb.entity_name})
                    SET e.embedding = emb.embedding
                    """,
                    embeddings=embedding_params
                )
            
            logger.info(f"Added embeddings for batch of {len(batch_entities)} entities")
        
        total_time = time.time() - start_time
        logger.info(f"Added {len(entities)} embeddings in {total_time:.2f}s ({len(entities)/total_time:.1f} entities/sec)")
    
    async def run_query(self, query: str, params: Dict[str, Any] = None):
        """
        Run a custom Cypher query against the database
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results
        """
        await self.connect()
        params = params or {}
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            records = await result.values()
            return records
            
    async def semantic_search(
        self, 
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filter_conditions: Dict[str, Any] = None
    ):
        """
        Perform semantic search using vector similarity
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            filter_conditions: Additional conditions to filter results
            
        Returns:
            List of entities and their similarity scores
        """
        await self.connect()
        
        # Construct filter clause if conditions are provided
        filter_clause = ""
        
        if filter_conditions:
            conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, str):
                    conditions.append(f"node.{field} = '{value}'")
                else:
                    conditions.append(f"node.{field} = {value}")
            
            if conditions:
                filter_clause = f"AND {' AND '.join(conditions)}"
        
        query = f"""
        CALL db.index.vector.queryNodes(
            'entity_embedding',
            $k,
            $query_embedding
        ) YIELD node, score
        WHERE score >= $threshold {filter_clause}
        RETURN node.name AS entity, score, node.id AS id
        ORDER BY score DESC
        """
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                query,
                k=limit,
                query_embedding=query_embedding,
                threshold=similarity_threshold
            )
            
            records = await result.values()
            return [{"entity": record[0], "score": record[1], "id": record[2]} for record in records]
            
    async def query_knowledge_graph(
        self,
        subject: str = None,
        predicate: str = None,
        object: str = None,
        limit: int = 100,
        exact_match: bool = False,
        include_metadata: bool = True
    ):
        """
        Query the knowledge graph by subject, predicate, or object
        
        Args:
            subject: Optional subject to filter by
            predicate: Optional predicate to filter by
            object: Optional object to filter by
            limit: Maximum number of triplets to return
            exact_match: Whether to require exact matches (vs. partial matches)
            include_metadata: Whether to include document and chunk metadata
            
        Returns:
            List of triplets matching the query
        """
        await self.connect()
        
        # Determine match type
        match_operator = "=" if exact_match else "CONTAINS"
        
        # Build query conditions
        conditions = []
        params = {}
        
        if subject:
            conditions.append(f"toLower(s.name) {match_operator} toLower($subject)")
            params["subject"] = subject
            
        if predicate:
            conditions.append(f"toLower(r.name) {match_operator} toLower($predicate)")
            params["predicate"] = predicate
            
        if object:
            conditions.append(f"toLower(o.name) {match_operator} toLower($object)")
            params["object"] = object
        
        # Build WHERE clause
        where_clause = " AND ".join(conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        
        # Additional fields for metadata
        metadata_fields = """
            , d.id AS document_id
            , collect(DISTINCT {
                chunk_id: c.id,
                page_number: c.page_number
            }) AS chunks
        """ if include_metadata else ""
        
        # Metadata joins
        metadata_joins = """
            OPTIONAL MATCH (r)-[:EXTRACTED_FROM]->(d:Document)
            OPTIONAL MATCH (r)-[:APPEARS_IN]->(c:Chunk)
        """ if include_metadata else ""
        
        # Build and execute query
        query = f"""
        MATCH (s:Entity)-[r:RELATES]->(o:Entity)
        {where_clause}
        {metadata_joins}
        RETURN 
            s.name AS subject, 
            r.name AS predicate, 
            o.name AS object, 
            r.id AS triplet_id, 
            r.confidence AS confidence
            {metadata_fields}
        GROUP BY s.name, r.name, o.name, r.id, r.confidence, d.id
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        
        params["limit"] = limit
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            records = await result.values()
            
            # Format results based on query type
            if include_metadata:
                return [
                    {
                        "subject": record[0], 
                        "predicate": record[1], 
                        "object": record[2],
                        "triplet_id": record[3],
                        "confidence": record[4],
                        "document_id": record[5],
                        "chunks": record[6]
                    } 
                    for record in records
                ]
            else:
                return [
                    {
                        "subject": record[0], 
                        "predicate": record[1], 
                        "object": record[2],
                        "triplet_id": record[3],
                        "confidence": record[4]
                    } 
                    for record in records
                ]
    
    async def get_entity_context(
        self,
        entity_name: str,
        limit: int = 5
    ):
        """
        Get document context for an entity
        
        Args:
            entity_name: Name of the entity
            limit: Maximum number of context chunks to return
            
        Returns:
            List of document contexts
        """
        await self.connect()
        
        query = """
        MATCH (e:Entity {name: $entity_name})-[:RELATES|EXTRACTED_FROM|APPEARS_IN*1..3]-(c:Chunk)-[:PART_OF]->(d:Document)
        RETURN 
            d.id AS document_id, 
            d.title AS document_title,
            c.id AS chunk_id,
            c.text AS chunk_text,
            c.page_number AS page_number
        ORDER BY c.page_number
        LIMIT $limit
        """
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                query,
                entity_name=entity_name,
                limit=limit
            )
            
            records = await result.values()
            return [
                {
                    "document_id": record[0],
                    "document_title": record[1],
                    "chunk_id": record[2],
                    "chunk_text": record[3],
                    "page_number": record[4]
                }
                for record in records
            ] 