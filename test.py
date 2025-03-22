from dotenv import load_dotenv
import os
load_dotenv()
# from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType

# graph_rag = RetrievalFactory.create_rag(
#     rag_type=RAGType.GRAPH,
#     llm_config={
#         "provider": "gemini",  # Options: "openai", "gemini"
#         "model_name": "models/gemini-2.0-flash",
#         "api_key": os.getenv("GOOGLE_API_KEY")
#     },
#     embedding_config={
#         "provider": "openai",   # only OpenAI is supported for now
#         "model_name": "text-embedding-ada-002",  
#         "api_key": os.getenv("OPENAI_API_KEY")
#     },
#     retrieval_config={
#         "username": "neo4j",
#         "password": "password",
#         "url": "neo4j://localhost:7687",
#         "similarity_top_k": 10
#     }
# )

# results = graph_rag.process("Nguyễn Trãi mất năm nào?")
# print(results)


from retrieval_engine.knowledge_retrieval.graph_rag_v2.graph_extractor import GraphExtractor
from llm_services import LLMProviderFactory

llm = LLMProviderFactory.create_provider(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

graph_extractor = GraphExtractor(llm=llm)

chunk1 = """
Nguyễn Trãi (chữ Hán: 阮廌; 1380 – 19 tháng 9 năm 1442), hiệu là Ức Trai (抑齋), là một nhà chính trị, nhà văn, nhà văn hóa lớn của dân tộc Việt Nam. 
Ông đã tham gia tích cực cuộc Khởi nghĩa Lam Sơn do Lê Lợi lãnh đạo chống lại sự xâm lược của nhà Minh (Trung Quốc) với Đại Việt. 
Khi cuộc khởi nghĩa thành công vào năm 1428, Nguyễn Trãi trở thành một trong những khai quốc công thần của triều đại quân chủ nhà Hậu Lê trong Lịch sử Việt Nam.[2]
"""

chunk2 = """
Hoàng Mậu TRung sinh năm 1998 năm nay 27 tuổi học tại PTIT
"""

metaclass={
    "source": "Wikipedia",
    "url": "https://en.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i",
    "title": "Nguyễn Trãi",
    "author": "Wikipedia",
    "published_date": "2021-03-01",
    "content_type": "text/plain",
    "language": "vi",
    "format": "text",
    "encoding": "utf-8"
}

import asyncio

triplets = asyncio.run(graph_extractor.extract_triplets([chunk1, chunk2], [metaclass, metaclass]))
print(triplets)