import os 
import asyncio
import logging
from llm_services import LLMProviderFactory
from .graph_extractor import GraphExtractor, KnowledgeTriplet
from .neo4j_connection import SimpleNeo4jConnection
from .graph_rag import GraphRAG
from retrieval_engine.knowledge_retrieval.graph_rag_v2.embeddings import create_embedding

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_graph_rag():
    """Test the enhanced GraphRAG with entity and relationship embeddings"""
    
    llm = LLMProviderFactory.create_provider(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    embedding_provider = create_embedding(
            provider_type="google",
            model_name="text-embedding-004",
            api_key=os.getenv("GOOGLE_API_KEY"),
            cache=False
        )
    
    extractor = GraphExtractor(
        llm=llm,
        max_knowledge_triplets=5,
        embedding_provider=embedding_provider
    )
    
    # Kết nối Neo4j
    neo4j_conn = SimpleNeo4jConnection(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j"
    )
    
    # Đảm bảo kết nối và thiết lập DB
    neo4j_conn.connect()
    neo4j_conn.setup_database()
    
    # Tạo ví dụ triplets với descriptions
    triplets = [
        KnowledgeTriplet(
            subject="OpenAI",
            predicate="founded_in",
            object="San Francisco",
            description="OpenAI was established in the city of San Francisco."
        ),
        KnowledgeTriplet(
            subject="OpenAI",
            predicate="founded_in",
            object="2015",
            description="OpenAI was created in the year 2015."
        ),
        KnowledgeTriplet(
            subject="Sam Altman",
            predicate="co_founded",
            object="OpenAI",
            description="Sam Altman was one of the founding members of OpenAI."
        ),
        KnowledgeTriplet(
            subject="Elon Musk",
            predicate="co_founded",
            object="OpenAI",
            description="Elon Musk was among the original founders of OpenAI."
        ),
        KnowledgeTriplet(
            subject="OpenAI",
            predicate="developed",
            object="GPT-4",
            description="OpenAI developed the GPT-4 large language model."
        ),
        KnowledgeTriplet(
            subject="GPT-4",
            predicate="is_a",
            object="language model",
            description="GPT-4 is a large language model for natural language processing."
        ),
        KnowledgeTriplet(
            subject="OpenAI",
            predicate="creates",
            object="AI products",
            description="OpenAI creates artificial intelligence products and research."
        )
    ]
    
    # Tạo extraction results format
    extraction_result = {
        "text": "Information about OpenAI and its founders",
        "metadata": {
            "title": "OpenAI Information",
            "source": "Test Data"
        },
        "triplets": triplets
    }
    
    # Thêm triplets vào Neo4j
    logger.info("Adding triplets to Neo4j...")
    neo4j_conn.add_extraction_triplet_results([extraction_result])
    
    # Tạo entity embeddings
    logger.info("Generating entity embeddings...")
    entity_embeddings = asyncio.run(extractor.extract_embeddings_entities(triplets))

    # Thêm entity embeddings vào Neo4j
    logger.info("Adding entity embeddings to Neo4j...")
    neo4j_conn.add_entity_embeddings(entity_embeddings)
    
    # Tạo relationship embeddings
    logger.info("Generating relationship embeddings...")
    relationship_embeddings = asyncio.run(extractor.extract_relationship_embeddings(triplets))
    for rel in relationship_embeddings.keys():
        print(rel)
        # print(len(relationship_embeddings[rel]))
    
    # Thêm relationship embeddings vào Neo4j
    logger.info("Adding relationship embeddings to Neo4j...")
    neo4j_conn.add_relationship_embeddings(relationship_embeddings)
    
    # Khởi tạo GraphRAG
    graph_rag = GraphRAG(
        graph_store=neo4j_conn,
        graph_extractor=extractor,
        embedding_provider=embedding_provider,
        similarity_threshold=0.7
    )
    
    # Thử nghiệm các truy vấn khác nhau
    test_queries = [
        "Who founded OpenAI?",
        "When was OpenAI founded?",
        "What has OpenAI developed?",
        "Tell me about GPT-4"
    ]
    
    for query in test_queries:
        logger.info(f"\n\nTesting query: {query}")
        
        # Lấy kết quả GraphRAG
        rag_results = graph_rag.retrieve(query, top_k=5)
        
        logger.info(f"Results for query: {query}")
        for i, r in enumerate(rag_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"Text: {r['text']}")
            logger.info(f"Subject: {r['triplet']['subject']}")
            logger.info(f"Predicate: {r['triplet']['predicate']}")
            logger.info(f"Object: {r['triplet']['object']}")
            logger.info(f"Description: {r['triplet']['description']}")
            logger.info(f"Source: {r['metadata']['source']}")
            logger.info(f"Confidence: {r['metadata']['confidence']}")
            logger.info("---")
    

if __name__ == "__main__":
    test_enhanced_graph_rag()