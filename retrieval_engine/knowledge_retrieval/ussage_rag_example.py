from dotenv import load_dotenv
import os
load_dotenv()
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType

# Tạo Graph RAG v2
graph_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.GRAPH_V2,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    embedding_config={
        "provider": "google",  
        "model_name": "text-embedding-004",  
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    retrieval_config={
        "username": "neo4j",
        "password": "password",
        "url": "neo4j://localhost:7687",
        "similarity_top_k": 10
    }
)

# Tạo Vector RAG
vector_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.VECTOR,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    embedding_config={
        "provider": "huggingface",  
        "model_name": "all-MiniLM-L6-v2",  
    },
    retrieval_config={
        "index_name": "example_hf", 
        "similarity_top_k": 10
    }
)

# Tạo Hybrid RAG
hybrid_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.HYBRID,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    retrieval_config={
        "vector_rag": vector_rag,
        "graph_rag": graph_rag,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "combination_strategy": "weighted",  # 'weighted', 'ensemble', hoặc 'cascade'
        "deduplicate": True,
        "max_workers": 2
    }
)

print("\n=== Kết quả từ Graph RAG ===")
graph_results = graph_rag.process("Nguyễn Trãi mất năm nào?", top_k=5)
print(graph_results)

print("\n=== Kết quả từ Vector RAG ===")
vector_results = vector_rag.process("Nguyễn Trãi mất năm nào?", top_k=5)
print(vector_results)

print("\n=== Kết quả từ Hybrid RAG ===")
hybrid_results = hybrid_rag.process("Nguyễn Trãi mất năm nào?", top_k=5)
print(hybrid_results)

# Đóng kết nối
import asyncio
asyncio.run(hybrid_rag.close())

