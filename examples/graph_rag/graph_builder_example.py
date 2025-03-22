#!/usr/bin/env python3
"""
GraphBuilder Example Script

This example demonstrates how to use GraphBuilder to construct a knowledge graph from text
and store it in Neo4j.
"""

import os
import sys
import asyncio
import logging
import traceback

# Import required modules
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphBuilder
from retrieval_engine.knowledge_retrieval.graph_rag_v2.graph_extractor import GraphExtractor
from retrieval_engine.knowledge_retrieval.graph_rag_v2.neo4j_connection import SimpleNeo4jConnection
from llm_services import LLMProviderFactory
from retrieval_engine.knowledge_retrieval.graph_rag_v2.embeddings import create_embedding

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample data
SAMPLE_DOCUMENTS = [
    {
        "text": """Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science.""",
        "document_id": "einstein_bio",
        "metadata": {
            "source": "Wikipedia",
            "category": "Physics",
            "language": "en"
        }
    },
    {
        "text": """Nguyễn Trãi (chữ Hán: 阮廌; 1380 – 19 tháng 9 năm 1442), hiệu là Ức Trai (抑齋), là một nhà chính trị, nhà văn, nhà văn hóa lớn của dân tộc Việt Nam. Ông đã tham gia tích cực cuộc Khởi nghĩa Lam Sơn do Lê Lợi lãnh đạo chống lại sự xâm lược của nhà Minh (Trung Quốc) với Đại Việt. Khi cuộc khởi nghĩa thành công vào năm 1428, Nguyễn Trãi trở thành một trong những khai quốc công thần của triều đại quân chủ nhà Hậu Lê trong Lịch sử Việt Nam.""",
        "document_id": "nguyentrai_bio",
        "metadata": {
            "source": "Wikipedia",
            "category": "History",
            "language": "vi",
            "translation": "Nguyen Trai (Chinese characters: 阮廌; 1380 - September 19, 1442), pen name Uc Trai (抑齋), was a great politician, writer, and cultural figure of Vietnam. He actively participated in the Lam Son Uprising led by Le Loi against the Ming Dynasty's (China) invasion of Dai Viet. When the uprising succeeded in 1428, Nguyen Trai became one of the founding officials of the Later Le Dynasty in Vietnamese history."
        }
    },
    {
        "text": """Hà Nội, thủ đô của Việt Nam, có lịch sử hơn 1000 năm. Thành phố mang trong mình di sản văn hóa phong phú với 36 phố phường cổ, hơn 600 đền chùa và di tích lịch sử. Hà Nội cũng là trung tâm chính trị, văn hóa, khoa học và kinh tế của Việt Nam. Hồ Gươm (Hồ Hoàn Kiếm) nằm ở trung tâm thành phố, là một trong những biểu tượng nổi tiếng nhất của Hà Nội.""",
        "document_id": "hanoi_intro",
        "metadata": {
            "source": "Travel Guide",
            "category": "Geography",
            "language": "vi",
            "translation": "Hanoi, the capital of Vietnam, has a history of over 1000 years. The city carries a rich cultural heritage with 36 ancient streets and wards, more than 600 temples, pagodas, and historical sites. Hanoi is also the political, cultural, scientific, and economic center of Vietnam. Sword Lake (Hoan Kiem Lake) located in the center of the city is one of the most famous symbols of Hanoi."
        }
    },
    {
        "text": """Hoàng Mậu Trung (hiệu là Ngọc Tự Hàn) sinh năm 1998 năm nay 27 tuổi học tại PTIT. Sinh ra trong một gia đình thuần nông tại Hà Lĩnh, Hà Trung, Thanh Hóa.""",
        "document_id": "hoangmautrung_bio",
        "metadata": {
            "source": "Self introduce",
            "category": "Person",
            "language": "vi",
            "translation": "Hoang Mau Trung (pen name Ngoc Tu Han) was born in 1998 and is 27 years old. He is currently studying at PTIT. Born in a pure farming family in Ha Linh, Ha Trung, Thanh Hoa."
        }
    }
]

async def get_builder():
    # Initialize embedding provider
    embedding_provider = create_embedding(
        provider_type="google",
        model_name="text-embedding-004",
        api_key=os.getenv("GOOGLE_API_KEY"),
        cache=False
    )

    # Initialize LLM provider
    llm = LLMProviderFactory.create_provider(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    graph_extractor = GraphExtractor(llm=llm, 
                                     embedding_provider=embedding_provider)
    
    neo4j_connection = SimpleNeo4jConnection(
        uri=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password")
    )

    # Initialize GraphBuilder
    builder = GraphBuilder(
        graph_store=neo4j_connection,
        graph_extractor=graph_extractor,
        embedding_provider=embedding_provider,
        max_workers=3,
        batch_size=10
    )
    return builder

async def test_single_document_processing():
    """Test processing a single document"""
    logger.info("=== Starting single document processing test ===")
    
    builder = await get_builder()
    
    # Process the first document
    doc = SAMPLE_DOCUMENTS[0]
    
    try:
        # Initialize GraphBuilder
    
        # Process document
        result = await builder.process_document(
            text=doc["text"],
            document_id=doc["document_id"],
            document_metadata=doc["metadata"]
        )
        
        # Print results
        logger.info(f"Results for document {doc['document_id']}:")
        logger.info(f"- Number of triplets created: {result['triplets']}")
        logger.info(f"- Number of entities with embeddings: {result['entities']}")
        logger.info(f"- Processing time: {result['processing_time']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
    finally:
        # Close connection
        await builder.close()
    
    logger.info("=== Completed single document processing test ===")

async def test_batch_processing():
    """Test batch processing of multiple documents"""
    logger.info("=== Starting batch processing test ===")
    
    # Initialize GraphBuilder
    builder = await get_builder()
    
    try:
        
        # Prepare data for batch processing
        # (deep copy to avoid changing original data)
        import copy
        batch_docs = copy.deepcopy(SAMPLE_DOCUMENTS)
        
        # Batch processing
        start_time = asyncio.get_event_loop().time()
        results = await builder.process_documents(
            documents=batch_docs,
            concurrency=4  # Process 2 documents concurrently
        )
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Print results
        logger.info(f"Results of batch processing {len(results)} documents:")
        for result in results:
            logger.info(f"- Document {result['document_id']}:")
            logger.info(f"  - Triplets: {result['triplets']}")
            logger.info(f"  - Entities: {result['entities']}")
            logger.info(f"  - Time: {result['processing_time']:.2f}s")
        
        logger.info(f"Total batch processing time: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
    finally:
        # Close connection
        await builder.close()
    
    logger.info("=== Completed batch processing test ===")

async def main():
    """Main function to execute the program"""
    logger.info("Starting GraphBuilder example")
    
    # Check required settings
    required_envs = ["GOOGLE_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_envs = [env for env in required_envs if not os.environ.get(env)]
    
    if missing_envs:
        logger.error(f"Missing environment variables: {', '.join(missing_envs)}")
        logger.info("Please set the following environment variables:")
        logger.info("export GOOGLE_API_KEY=<your_google_key>")
        logger.info("export NEO4J_URI=neo4j://localhost:7687")
        logger.info("export NEO4J_USERNAME=neo4j")
        logger.info("export NEO4J_PASSWORD=<your_password>")
        return
    
    # Run tests
    # await test_single_document_processing()
    logger.info("\n" + "="*50 + "\n")
    await test_batch_processing()
    
    logger.info("Completed GraphBuilder example")

if __name__ == "__main__":
    asyncio.run(main()) 