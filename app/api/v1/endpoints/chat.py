from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import json
import os
import logging
import asyncio
import time
from dotenv import load_dotenv

# Import necessary components
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType
from retrieval_engine.text2sql import Text2SQL
from llm_services import LLMProviderFactory

# Import refactored components
from app.utils.websocket.connection import ConnectionManager
from app.utils.llm_helpers.streaming import stream_response_and_track
from app.utils.llm_helpers.processors import ChatProcessor
from app.utils.models import ChatRequest, ChatResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Create a single instance of the connection manager
manager = ConnectionManager()

# Provider initialization functions
def get_llm():
    return LLMProviderFactory.create_provider(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_vector_rag(collection_name=None):
    """Get a Vector RAG instance for the specified collection"""
    collection = collection_name or os.getenv("VECTOR_COLLECTION_NAME", "callcenter")
    
    vector_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.VECTOR,
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
            "collection_name": collection, 
            "similarity_top_k": 10
        }
    )
    
    return vector_rag

def get_graph_rag():
    """Get a Graph RAG instance"""
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
            "username": os.getenv("NEO4J_USERNAME", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password"),
            "url": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            "similarity_top_k": 50,  # Minimum of 10 as requested
            "hybrid": False 
        }
    )
    
    return graph_rag

def get_hybrid_rag(collection_name=None):
    """Get a Hybrid RAG instance combining vector and graph RAG"""
    collection = collection_name or os.getenv("VECTOR_COLLECTION_NAME", "callcenter")
    
    vector_rag = get_vector_rag(collection)
    graph_rag = get_graph_rag()
    
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
            "combination_strategy": "weighted",
            "deduplicate": True,
            "max_workers": 2
        }
    )
    
    return hybrid_rag

def get_text2sql():
    """Get a Text2SQL instance"""
    llm = get_llm()
    
    text2sql = Text2SQL(
        db_type="postgres",
        llm_provider=llm,
        max_retries=2,
        batch_size=3,
        max_concurrency=5
    )
    
    return text2sql

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for interactive chat with different processing modes.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Wait for and parse the incoming message
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Extract question and mode
            question = request_data.get("question", "")
            mode = request_data.get("mode", "vector")
            
            # Initialize start time for overall processing
            start_time = time.time()
            
            # Process in stages to show thinking
            await manager.send_json(
                websocket,
                {
                    "type": "thinking",
                    "step": "start",
                    "content": f"Processing question: {question}"
                }
            )
            
            await manager.send_json(
                websocket,
                {
                    "type": "thinking",
                    "step": "mode",
                    "content": f"Using mode: {mode}"
                }
            )
            
            # Initialize components
            llm = get_llm()
            
            # Initialize results containers
            answer = ""
            sources = []
            sql_query = None
            sql_data = None
            stream_successful = False
            
            try:
                # Configure the right knowledge retrieval system based on mode
                knowledge_retrieval = None
                if mode == "vector":
                    knowledge_retrieval = get_vector_rag()
                elif mode == "graph":
                    knowledge_retrieval = get_graph_rag()
                elif mode == "hybrid_rag":
                    knowledge_retrieval = get_hybrid_rag()
                
                # Configure the text2sql if needed
                text2sql_instance = None
                if mode in ["database", "hybrid"]:
                    text2sql_instance = get_text2sql()
                
                # Create the chat processor
                processor = ChatProcessor(
                    websocket=websocket,
                    connection_manager=manager,
                    llm_provider=llm,
                    knowledge_retrieval=knowledge_retrieval,
                    text2sql=text2sql_instance
                )
                
                # Process based on mode
                if mode == "vector":
                    # Vector knowledge base mode
                    answer, sources, stream_successful = await processor.process_knowledge_mode(question)
                    
                elif mode == "graph":
                    # Graph knowledge base mode
                    answer, sources, stream_successful = await processor.process_graph_mode(question)
                    
                elif mode == "hybrid_rag":
                    # Hybrid RAG mode (vector + graph)
                    answer, sources, stream_successful = await processor.process_hybrid_rag_mode(question)
                    
                elif mode == "database":
                    # Database only mode
                    answer, sql_query, sql_data, stream_successful = await processor.process_database_mode(question)
                    
                elif mode == "hybrid":
                    # Get information from both knowledge base (vector by default) and database
                    knowledge_retrieval = get_vector_rag()  # Use vector RAG for hybrid mode
                    processor.knowledge_retrieval = knowledge_retrieval
                    
                    # First get knowledge base results
                    kb_answer, sources, kb_stream_successful = await processor.process_knowledge_mode(question)
                    
                    # Then get database results
                    db_answer, sql_query, sql_data, db_stream_successful = await processor.process_database_mode(question)
                    
                    # Combine the results
                    if (sources or (sql_data and len(sql_data) > 0)):
                        answer, stream_successful = await processor.process_hybrid_mode(
                            question, sources, sql_query, sql_data
                        )
                    else:
                        # If only one source has data, use its answer
                        if sources and not sql_data:
                            answer = kb_answer
                            stream_successful = kb_stream_successful
                        elif sql_data and not sources:
                            answer = db_answer
                            stream_successful = db_stream_successful
                        else:
                            answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                            stream_successful = False
                else:
                    # Invalid mode
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Invalid mode: {mode}. Valid options are: vector, graph, hybrid_rag, database, hybrid"
                        }
                    )
                    continue
                
                # Calculate total processing time
                total_time = time.time() - start_time
                
                # Send completion message
                await manager.send_json(
                    websocket,
                    {
                        "type": "thinking",
                        "step": "complete",
                        "content": f"Processing complete in {total_time:.2f} seconds."
                    }
                )
                
                # Send the final result
                await manager.send_json(
                    websocket,
                    {
                        "type": "result",
                        "answer": answer,
                        "sources": sources,
                        "sql": sql_query,
                        "data": sql_data
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing WebSocket chat: {str(e)}")
                
                # Send error thinking step
                await manager.send_json(
                    websocket,
                    {
                        "type": "thinking",
                        "step": "error",
                        "content": f"Error: {str(e)}"
                    }
                )
                
                # Send error result
                await manager.send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": f"Error processing question: {str(e)}"
                    }
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    REST API endpoint for chat with support for different modes.
    """
    try:
        # Extract question and mode
        question = request.question
        mode = request.mode
        
        # Initialize start time for overall processing
        start_time = time.time()
        
        # Initialize components based on mode
        llm = get_llm()
        thinking_steps = []
        
        # Add initial thinking steps
        thinking_steps.append({"step": "start", "content": f"Processing question: {question}"})
        thinking_steps.append({"step": "mode", "content": f"Using mode: {mode}"})
        
        # Initialize results containers
        answer = ""
        sources = []
        sql_query = None
        sql_data = None
        
        # Configure the right knowledge retrieval system based on mode
        knowledge_retrieval = None
        if mode == "vector":
            knowledge_retrieval = get_vector_rag()
            thinking_steps.append({"step": "retrieval_type", "content": "Using vector retrieval"})
        elif mode == "graph":
            knowledge_retrieval = get_graph_rag()
            thinking_steps.append({"step": "retrieval_type", "content": "Using graph retrieval"})
        elif mode == "hybrid_rag":
            knowledge_retrieval = get_hybrid_rag()
            thinking_steps.append({"step": "retrieval_type", "content": "Using hybrid RAG retrieval"})
        
        # Configure the text2sql if needed
        text2sql_instance = None
        if mode in ["database", "hybrid"]:
            text2sql_instance = get_text2sql()
            if mode == "database":
                thinking_steps.append({"step": "retrieval_type", "content": "Using database retrieval"})
        
        # Create a processor for non-streaming API
        class NonStreamingProcessor:
            def __init__(self, llm_provider, knowledge_retrieval, text2sql, thinking_steps):
                self.llm = llm_provider
                self.knowledge_retrieval = knowledge_retrieval
                self.text2sql = text2sql
                self.thinking_steps = thinking_steps
            
            def _add_thinking(self, step, content):
                self.thinking_steps.append({"step": step, "content": content})
            
            async def process_vector_mode(self, question):
                """Process vector knowledge retrieval"""
                self._add_thinking("vector_retrieval", "Getting information from vector knowledge base...")
                
                kb_start_time = time.time()
                kb_results = self.knowledge_retrieval.process(question, top_k=5)
                kb_elapsed = time.time() - kb_start_time
                
                # Extract sources
                sources = []
                retrieved_docs = []
                
                if isinstance(kb_results, dict):
                    if 'documents' in kb_results:
                        retrieved_docs = kb_results['documents']
                        self._add_thinking("vector_results", f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds")
                        
                        # Extract sources
                        for doc in retrieved_docs:
                            if 'content' in doc and doc['content'].strip():
                                source = {
                                    "id": doc.get("id", ""),
                                    "content": doc.get("content", ""),
                                    "score": doc.get("score", 0),
                                    "title": doc.get("metadata", {}).get("title", "Unknown document"),
                                }
                                sources.append(source)
                    elif 'sources' in kb_results:
                        sources = kb_results.get("sources", [])
                
                # If we have an answer directly from kb_results, use it
                answer = ""
                if kb_results and isinstance(kb_results, dict) and 'answer' in kb_results and kb_results['answer']:
                    answer = kb_results["answer"]
                elif sources:
                    # Generate an answer using the retrieved documents
                    context = "\n\n".join([
                        f"Document {i+1}:\n{source.get('content', '')}" 
                        for i, source in enumerate(sources[:3]) 
                        if 'content' in source and source['content'].strip()
                    ])
                    
                    answer_prompt = f"""
                    Based on the following information, please answer this question: {question}
                    
                    Information:
                    {context}
                    
                    If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
                    
                    Answer:
                    """
                    
                    self._add_thinking("answer_generation", "Generating answer from retrieved documents...")
                    answer = await self.llm.generate_text(answer_prompt)
                else:
                    answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                
                return answer, sources
            
            async def process_graph_mode(self, question):
                """Process graph knowledge retrieval"""
                self._add_thinking("graph_retrieval", "Getting information from graph knowledge base...")
                
                kb_start_time = time.time()
                
                self._add_thinking("retrieval_config", "Using graph database with minimum 10 results")
                
                # Run the graph query
                try:
                    kb_results = await asyncio.to_thread(
                        self.knowledge_retrieval.process,
                        question, 
                        top_k=10,
                        use_semantic=True,
                        use_graph=True
                    )
                except Exception as e:
                    logger.error(f"Error in graph retrieval: {str(e)}")
                    self._add_thinking("graph_retrieval_error", f"Error: {str(e)}")
                    return "Đã có lỗi khi truy vấn cơ sở dữ liệu đồ thị. Vui lòng thử lại sau.", []
                
                kb_elapsed = time.time() - kb_start_time
                
                # Đơn giản hóa xử lý kết quả - chỉ lấy documents và text field
                sources = []
                
                # Kiểm tra xem kết quả có phải là dict hay không
                if isinstance(kb_results, dict):
                    # Kiểm tra xem kb_results có chứa 'documents' không
                    if 'documents' in kb_results:
                        retrieved_docs = kb_results['documents']
                        self._add_thinking("graph_documents", f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds")
                        
                        # Log cấu trúc document để debug
                        if retrieved_docs and len(retrieved_docs) > 0:
                            self._add_thinking("document_structure", f"Document keys: {list(retrieved_docs[0].keys())}")
                        
                        # Extract sources from documents, chỉ quan tâm đến trường 'text'
                        for doc in retrieved_docs:
                            if 'text' in doc and doc['text'].strip():
                                source = {
                                    "id": doc.get("id", ""),
                                    "content": doc.get("text", ""),  # Lưu ý: sử dụng 'text' field làm content
                                    "score": doc.get("metadata", {}).get("score", 0) if "metadata" in doc else 0,
                                    "title": f"Document {len(sources) + 1}",
                                    "source": "graph"
                                }
                                sources.append(source)
                    # Fallback nếu không tìm thấy documents
                    elif 'query' in kb_results and isinstance(kb_results.get('query'), dict):
                        self._add_thinking("query_info", f"Query: {kb_results['query'].get('query', '')}")
                        
                        # Kiểm tra xem còn field nào khác chứa thông tin
                        self._add_thinking("available_keys", f"Available keys in result: {list(kb_results.keys())}")
                else:
                    self._add_thinking("unexpected_format", f"Received unexpected format: {type(kb_results)}")
                
                # Nếu không tìm thấy sources nào
                if not sources:
                    self._add_thinking("no_sources", "No relevant documents found")
                    return "Tôi không tìm thấy thông tin liên quan từ cơ sở dữ liệu đồ thị.", []
                
                # Generate answer from sources
                answer = ""
                
                # Tạo context từ sources
                context = "\n\n".join([
                    f"Document {i+1}:\n{source.get('content', '')}" 
                    for i, source in enumerate(sources[:5]) 
                    if 'content' in source and source['content'].strip()
                ])
                
                answer_prompt = f"""
                Based on the following information, please answer this question: {question}
                
                Information:
                {context}
                
                If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
                
                Answer:
                """
                
                self._add_thinking("answer_generation", "Generating answer from graph knowledge...")
                answer = await self.llm.generate_text(answer_prompt)
                
                # Add document details to thinking steps for transparency
                if sources:
                    doc_list = ""
                    for i, source in enumerate(sources[:3]):
                        score = source.get('score', 0)
                        doc_list += f"• Document {i+1} (relevance: {score:.2f})\n"
                    
                    if len(sources) > 3:
                        doc_list += f"• ... and {len(sources) - 3} more documents"
                        
                    self._add_thinking("graph_document_details", f"Top retrieved documents:\n{doc_list}")
                
                return answer, sources
            
            async def process_hybrid_rag_mode(self, question):
                """Process hybrid RAG retrieval"""
                self._add_thinking("hybrid_rag_retrieval", "Getting information using Hybrid RAG (vector + graph)...")
                
                kb_start_time = time.time()
                
                self._add_thinking("retrieval_config", "Using weighted combination of vector and graph retrieval (60% vector, 40% graph)")
                
                # Process the query using the hybrid RAG
                kb_results = self.knowledge_retrieval.process(question, top_k=10)
                kb_elapsed = time.time() - kb_start_time
                
                # Đơn giản hóa xử lý kết quả - chỉ tập trung vào vector (content) và graph (text)
                sources = []
                vector_docs = []
                graph_docs = []
                
                # Debug result structure
                if isinstance(kb_results, dict):
                    self._add_thinking("hybrid_result_keys", f"Hybrid RAG result keys: {list(kb_results.keys())}")
                
                if isinstance(kb_results, dict):
                    # Xử lý vector sources (sử dụng field 'content')
                    if 'vector_results' in kb_results:
                        vector_docs = kb_results['vector_results'].get('documents', [])
                        self._add_thinking("vector_results", f"Found {len(vector_docs)} relevant vector documents")
                        
                        for doc in vector_docs:
                            if 'content' in doc and doc['content'].strip():
                                source = {
                                    "id": doc.get("id", ""),
                                    "content": doc.get("content", ""),
                                    "score": doc.get("score", 0),
                                    "title": f"Vector Document {len(sources) + 1}",
                                    "source_type": "vector"
                                }
                                sources.append(source)
                    
                    # Xử lý graph sources (sử dụng field 'text')
                    if 'graph_results' in kb_results:
                        graph_result = kb_results['graph_results']
                        if 'documents' in graph_result:
                            graph_docs = graph_result.get('documents', [])
                            self._add_thinking("graph_results", f"Found {len(graph_docs)} relevant graph documents")
                            
                            for doc in graph_docs:
                                if 'text' in doc and doc['text'].strip():
                                    source = {
                                        "id": doc.get("id", ""),
                                        "content": doc.get("text", ""),  # Sử dụng 'text' field cho graph documents
                                        "score": doc.get("metadata", {}).get("score", 0) if doc.get("metadata") else 0,
                                        "title": f"Graph Document {len(sources) + 1}",
                                        "source_type": "graph"
                                    }
                                    sources.append(source)
                    
                    # Xử lý combined documents nếu có
                    if 'combined_documents' in kb_results:
                        combined_docs = kb_results['combined_documents']
                        self._add_thinking("combined_results", f"Combined and deduplicated to {len(combined_docs)} documents")
                        
                        for doc in combined_docs:
                            content = None
                            # Ưu tiên kiểm tra field 'content' trước
                            if 'content' in doc and doc['content'].strip():
                                content = doc['content']
                            # Sau đó mới đến 'text' nếu không có 'content'
                            elif 'text' in doc and doc['text'].strip():
                                content = doc['text']
                                
                            if content:
                                source = {
                                    "id": doc.get("id", ""),
                                    "content": content,
                                    "score": doc.get("final_score", doc.get("score", 0)),
                                    "title": f"Hybrid Document {len(sources) + 1}",
                                    "source_type": doc.get("source_type", "hybrid")
                                }
                                sources.append(source)
                else:
                    self._add_thinking("unexpected_format", f"Received unexpected format: {type(kb_results)}")
                
                # Nếu không tìm thấy sources nào
                if not sources:
                    self._add_thinking("no_sources", "No relevant documents found")
                    return "Tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này.", []
                
                # Generate answer from sources
                answer = ""
                
                # Sắp xếp sources theo score để lấy các tài liệu liên quan nhất
                sorted_sources = sorted(sources, key=lambda x: x.get('score', 0), reverse=True)
                
                # Tạo context từ sources
                context = "\n\n".join([
                    f"Document {i+1} ({source.get('source_type', 'document')}):\n{source.get('content', '')}" 
                    for i, source in enumerate(sorted_sources[:5]) 
                    if 'content' in source and source['content'].strip()
                ])
                
                answer_prompt = f"""
                Based on the following information from a hybrid knowledge retrieval system (combining vector and graph databases), please answer this question: {question}
                
                Information:
                {context}
                
                If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
                
                Answer:
                """
                
                self._add_thinking("answer_generation", "Generating answer from hybrid knowledge sources...")
                answer = await self.llm.generate_text(answer_prompt)
                
                # Thống kê về sources
                if sources:
                    vector_count = len([s for s in sources if s.get('source_type') == 'vector'])
                    graph_count = len([s for s in sources if s.get('source_type') == 'graph'])
                    hybrid_count = len([s for s in sources if s.get('source_type') == 'hybrid'])
                    self._add_thinking("hybrid_source_stats", f"Source breakdown: {vector_count} vector, {graph_count} graph, {hybrid_count} combined")
                
                return answer, sources
            
            async def process_database_mode(self, question):
                """Process database query"""
                self._add_thinking("database_query", "Converting question to SQL query...")
                
                sql_start_time = time.time()
                sql_result = await self.text2sql.process_query(question)
                sql_elapsed = time.time() - sql_start_time
                
                answer = ""
                sql_query = None
                sql_data = None
                
                if sql_result["success"]:
                    self._add_thinking("sql_generation", f"Generated SQL in {sql_elapsed:.2f} seconds:\n{sql_result['sql']}")
                    
                    sql_query = sql_result["sql"]
                    sql_data = sql_result["data"]
                    
                    if "tables_used" in sql_result and sql_result["tables_used"]:
                        self._add_thinking("sql_tables", f"Tables used: {', '.join(sql_result['tables_used'])}")
                    
                    self._add_thinking("sql_results", f"Query returned {len(sql_data)} rows")
                    
                    if sql_data:
                        # Generate answer from SQL data
                        data_context = f"SQL query: {sql_query}\n\nData results: {json.dumps(sql_data, indent=2)}"
                        answer_prompt = f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{question}'\n\n{data_context}"
                        
                        self._add_thinking("answer_generation", "Generating answer from database results...")
                        answer = await self.llm.generate_text(answer_prompt)
                    else:
                        answer = "No data found in database for your question."
                else:
                    self._add_thinking("sql_error", f"Error generating SQL: {sql_result.get('error', 'Unknown error')}")
                    answer = "Sorry, I couldn't convert your question to a database query."
                
                return answer, sql_query, sql_data
            
            async def process_hybrid_mode(self, question):
                """Process hybrid mode (vector + database)"""
                self._add_thinking("hybrid_mode", "Processing question using both knowledge base and database...")
                
                # First get vector knowledge results
                self._add_thinking("vector_step", "Getting information from vector knowledge base...")
                kb_answer, kb_sources = await self.process_vector_mode(question)
                
                # Then get database results
                self._add_thinking("database_step", "Getting information from database...")
                db_answer, sql_query, sql_data = await self.process_database_mode(question)
                
                # Combine results if both have data
                if (kb_sources or (sql_data and len(sql_data) > 0)):
                    self._add_thinking("combining_sources", "Combining information from knowledge base and database...")
                    
                    kb_context = json.dumps(kb_sources, indent=2) if kb_sources else "No knowledge base results."
                    db_context = f"SQL query: {sql_query}\n\nData results: {json.dumps(sql_data, indent=2)}" if sql_query else "No database results."
                    
                    combined_prompt = f"""
                    The user asked: '{question}'
                    
                    Knowledge base information:
                    {kb_context}
                    
                    Database information:
                    {db_context}
                    
                    Based on all available information, provide a comprehensive and accurate answer to the user's question.
                    """
                    
                    self._add_thinking("answer_generation", "Generating answer from combined sources...")
                    answer = await self.llm.generate_text(combined_prompt)
                else:
                    # If only one source has data, use its answer
                    if kb_sources and not sql_data:
                        answer = kb_answer
                    elif sql_data and not kb_sources:
                        answer = db_answer
                    else:
                        answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                
                return answer, kb_sources, sql_query, sql_data
        
        # Create the processor
        processor = NonStreamingProcessor(
            llm_provider=llm,
            knowledge_retrieval=knowledge_retrieval,
            text2sql=text2sql_instance,
            thinking_steps=thinking_steps
        )
        
        # Process based on mode
        if mode == "vector":
            answer, sources = await processor.process_vector_mode(question)
            
        elif mode == "graph":
            answer, sources = await processor.process_graph_mode(question)
            
        elif mode == "hybrid_rag":
            answer, sources = await processor.process_hybrid_rag_mode(question)
            
        elif mode == "database":
            answer, sql_query, sql_data = await processor.process_database_mode(question)
            
        elif mode == "hybrid":
            answer, sources, sql_query, sql_data = await processor.process_hybrid_mode(question)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid mode: {mode}. Valid options are: vector, graph, hybrid_rag, database, hybrid"
            )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        thinking_steps.append({"step": "complete", "content": f"Processing complete in {total_time:.2f} seconds."})
        
        # Return the response
        return ChatResponse(
            answer=answer,
            sources=sources or [],
            sql=sql_query,
            data=sql_data,
            thinking=thinking_steps
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/debug_stream_gemini")
async def debug_stream_gemini(request: ChatRequest):
    """
    Debug endpoint to test the Gemini streaming API directly
    """
    try:
        llm = get_llm()
        prompt = f"This is a test of the streaming API. Please respond to: {request.question}"
        
        # Create a streaming response using StreamingResponse
        async def stream_generator():
            try:
                yield "Streaming test started. Raw response from Gemini:\n\n"
                
                # Get the raw response from Gemini
                async for chunk in llm.generate_text_stream(prompt):
                    yield f"CHUNK: {chunk}\n---\n"
                
                yield "\nStreaming test completed."
            except Exception as e:
                yield f"\nError occurred: {str(e)}"
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(stream_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error in debug_stream_gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 