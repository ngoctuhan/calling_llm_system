def run_vector_search(self, query_embedding, limit=10, similarity_threshold=0.7):
    """
    Perform vector search using Neo4j vector index
    
    Args:
        query_embedding: Embedding vector to search with
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of matching entities with similarity scores
    """
    self.connect()
    
    try:
        # Try using vector index
        with self._driver.session(database=self.database) as session:
            vector_query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            CALL db.index.vector.queryNodes(
                'entity_embedding',
                $k,
                $query_embedding
            ) YIELD node, score
            WHERE score >= $threshold
            RETURN node.name AS entity, score
            ORDER BY score DESC
            """
            
            params = {
                "k": limit,
                "query_embedding": query_embedding,
                "threshold": similarity_threshold
            }
            
            result = session.run(vector_query, params)
            entities = [{"entity": record["entity"], "score": record["score"]} for record in result]
            
            logger.info(f"Vector search found {len(entities)} results using vector index")
            return entities
            
    except Exception as e:
        logger.warning(f"Vector index search failed: {e}")
        
        # Fallback to manual similarity calculation in Python
        # NOTE: This avoids using gds.similarity.cosine which requires the Graph Data Science library
        logger.info("Falling back to manual similarity calculation in Python")
        
        # First retrieve all entities with embeddings
        with self._driver.session(database=self.database) as session:
            query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN e.name AS entity, e.embedding AS embedding
            """
            
            result = session.run(query)
            entities = [{"entity": record["entity"], "embedding": record["embedding"]} 
                       for record in result]
        
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
        for entity in entities:
            score = cosine_similarity(query_embedding, entity["embedding"])
            if score >= similarity_threshold:
                results.append({"entity": entity["entity"], "score": score})
        
        # Sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        
        logger.info(f"Fallback search found {len(results)} results using Python calculation")
        return results 