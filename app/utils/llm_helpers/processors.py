import json
import logging
import time
import os
import asyncio
from fastapi import WebSocket
from typing import Dict, List, Any, Tuple, Optional

from app.utils.websocket.connection import ConnectionManager
from app.utils.llm_helpers.streaming import stream_response_and_track

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatProcessor:
    """
    Processes chat requests in different modes: knowledgebase, database, or hybrid.
    """
    def __init__(
        self,
        websocket: WebSocket,
        connection_manager: ConnectionManager,
        llm_provider,
        knowledge_retrieval,
        text2sql
    ):
        self.websocket = websocket
        self.manager = connection_manager
        self.llm = llm_provider
        self.knowledge_retrieval = knowledge_retrieval
        self.text2sql = text2sql
    
    async def _send_thinking(self, step: str, content: str):
        """Send a thinking step message"""
        await self.manager.send_json(
            self.websocket,
            {
                "type": "thinking",
                "step": step,
                "content": content
            }
        )
    
    async def process_knowledge_mode(self, question: str) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Process a question using vector knowledge base.
        
        Args:
            question: The user question
            
        Returns:
            Tuple of (answer, sources, stream_successful)
        """
        await self._send_thinking("knowledge_retrieval", "Getting information from vector knowledge base...")
        
        kb_start_time = time.time()
        collection_name = os.getenv("VECTOR_COLLECTION_NAME", "callcenter")
        
        await self._send_thinking(
            "retrieval_config",
            f"Using vector collection: {collection_name}"
        )
        
        kb_results = self.knowledge_retrieval.process(question, top_k=5)
        kb_elapsed = time.time() - kb_start_time
        
        # Process and extract sources
        sources = []
        has_relevant_docs = False
        retrieved_docs = []
        
        # Handle the new response format which might have documents in results.documents
        if isinstance(kb_results, dict):
            # Check if the new format is being used (results.documents structure)
            if 'results' in kb_results and 'documents' in kb_results['results']:
                retrieved_docs = kb_results['results']['documents']
                await self._send_thinking(
                    "knowledge_results",
                    f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds"
                )
                
                # Extract sources from the new format
                for doc in retrieved_docs:
                    if 'content' in doc and doc['content'].strip():
                        source = {
                            "id": doc.get("id", ""),
                            "content": doc.get("content", ""),
                            "score": doc.get("original_score", 0),
                            "title": doc.get("metadata", {}).get("title", "Unknown document"),
                        }
                        # Add other metadata
                        if "metadata" in doc:
                            for key, value in doc["metadata"].items():
                                if key not in ["title"]:  # Skip title as we already included it
                                    source[key] = value
                        sources.append(source)
                
                has_relevant_docs = len(sources) > 0
                
            elif 'documents' in kb_results:
                # Handle format with direct documents array
                retrieved_docs = kb_results['documents']
                await self._send_thinking(
                    "knowledge_results",
                    f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds"
                )
                
                # Extract sources from the format
                for doc in retrieved_docs:
                    if 'content' in doc and doc['content'].strip():
                        source = {
                            "id": doc.get("id", ""),
                            "content": doc.get("content", ""),
                            "score": doc.get("original_score", 0),
                            "title": doc.get("metadata", {}).get("title", "Unknown document"),
                        }
                        # Add other metadata
                        if "metadata" in doc:
                            for key, value in doc["metadata"].items():
                                if key not in ["title"]:  # Skip title as we already included it
                                    source[key] = value
                        sources.append(source)
                
                has_relevant_docs = len(sources) > 0
                
            else:
                # Handle the old format
                await self._send_thinking(
                    "knowledge_results",
                    f"Found {len(kb_results.get('sources', []))} relevant documents in {kb_elapsed:.2f} seconds"
                )
                sources = kb_results.get("sources", [])
                has_relevant_docs = len(sources) > 0
        else:
            await self._send_thinking(
                "knowledge_results_error",
                f"Unexpected format from knowledge retrieval: {type(kb_results)}"
            )
            
        # If we have an answer directly from kb_results, use it
        answer = ""
        stream_successful = False
        if kb_results and isinstance(kb_results, dict) and 'answer' in kb_results and kb_results['answer']:
            answer = kb_results["answer"]
        elif has_relevant_docs:
            # Generate an answer using the fetched documents
            if retrieved_docs:
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc.get('content', '')}" 
                    for i, doc in enumerate(retrieved_docs[:3]) 
                    if 'content' in doc and doc['content'].strip()
                ])
            else:
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
            
            # Stream the answer
            await self._send_thinking(
                "answer_generation_start",
                "Generating answer from retrieved documents..."
            )
            
            await self.manager.send_json(
                self.websocket,
                {
                    "type": "stream_start",
                    "message": ""
                }
            )
            
            answer, stream_successful = await stream_response_and_track(
                self.websocket, self.llm, answer_prompt, logger
            )
        else:
            answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
            
        # Add document details to thinking steps for transparency
        if sources:
            doc_list = ""
            for i, source in enumerate(sources[:3]):
                title = source.get('title', 'Unknown document')
                score = source.get('score', 0)
                doc_list += f"• {title} (relevance: {score:.2f})\n"
            
            if len(sources) > 3:
                doc_list += f"• ... and {len(sources) - 3} more documents"
                
            await self._send_thinking(
                "knowledge_documents",
                f"Top retrieved documents:\n{doc_list}"
            )
        
        return answer, sources, stream_successful
    
    async def process_database_mode(self, question: str) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]], bool]:
        """
        Process a question using database (SQL).
        
        Args:
            question: The user question
            
        Returns:
            Tuple of (answer, sql_query, sql_data, stream_successful)
        """
        await self._send_thinking(
            "database_query",
            "Converting question to SQL query..."
        )
        
        sql_start_time = time.time()
        sql_result = await self.text2sql.process_query(question)
        sql_elapsed = time.time() - sql_start_time
        
        answer = ""
        sql_query = None
        sql_data = None
        stream_successful = False
        
        if sql_result["success"]:
            await self._send_thinking(
                "sql_generation", 
                f"Generated SQL in {sql_elapsed:.2f} seconds:\n{sql_result['sql']}"
            )
            
            sql_query = sql_result["sql"]
            sql_data = sql_result["data"]
            
            if "tables_used" in sql_result and sql_result["tables_used"]:
                await self._send_thinking(
                    "sql_tables", 
                    f"Tables used: {', '.join(sql_result['tables_used'])}"
                )
            
            await self._send_thinking(
                "sql_results", 
                f"Query returned {len(sql_data)} rows"
            )
            
            if sql_data:
                # Generate answer from SQL data
                data_context = f"SQL query: {sql_query}\n\nData results: {json.dumps(sql_data, indent=2)}"
                answer_prompt = f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{question}'\n\n{data_context}"
                
                # Stream the answer
                await self._send_thinking(
                    "answer_generation_start",
                    "Generating answer from database results..."
                )
                
                await self.manager.send_json(
                    self.websocket,
                    {
                        "type": "stream_start",
                        "message": ""
                    }
                )
                
                answer, stream_successful = await stream_response_and_track(
                    self.websocket, self.llm, answer_prompt, logger
                )
            else:
                answer = "No data found in database for your question."
        else:
            await self._send_thinking(
                "sql_error", 
                f"Error generating SQL: {sql_result.get('error', 'Unknown error')}"
            )
            answer = "Sorry, I couldn't convert your question to a database query."
            
        return answer, sql_query, sql_data, stream_successful
    
    async def process_graph_mode(self, question: str) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Process a question using graph knowledge base.
        
        Args:
            question: The user question
            
        Returns:
            Tuple of (answer, sources, stream_successful)
        """
        await self._send_thinking("graph_retrieval", "Getting information from graph knowledge base...")
        
        kb_start_time = time.time()
        
        await self._send_thinking(
            "retrieval_config",
            f"Using graph database with minimum 10 results"
        )
        
        # Run the graph query in a separate thread to avoid blocking
        try:
            kb_results = await asyncio.to_thread(
                self.knowledge_retrieval.process,
                question, 
                top_k=30,
                use_semantic=True,
                use_graph=True
            )
        except Exception as e:
            logger.error(f"Error in graph retrieval: {str(e)}")
            await self._send_thinking(
                "graph_retrieval_error",
                f"Error: {str(e)}"
            )
            return "Đã có lỗi khi truy vấn cơ sở dữ liệu đồ thị. Vui lòng thử lại sau.", [], False
            
        kb_elapsed = time.time() - kb_start_time
        
        # Đơn giản hóa xử lý kết quả - chỉ lấy documents và text field
        sources = []
        
        # Kiểm tra xem kết quả có phải là dict hay không
        if isinstance(kb_results, dict):
            # Kiểm tra xem kb_results có chứa 'documents' không
            if 'documents' in kb_results:
                retrieved_docs = kb_results['documents']
                await self._send_thinking(
                    "graph_documents",
                    f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds"
                )
                
                # Log cấu trúc document để debug
                if retrieved_docs and len(retrieved_docs) > 0:
                    await self._send_thinking(
                        "document_structure",
                        f"Document keys: {list(retrieved_docs[0].keys())}"
                    )
                
                # Extract sources from documents, chỉ quan tâm đến trường 'text'
                for doc in retrieved_docs:
                    if 'text' in doc and doc['text'].strip():
                        source = {
                            "id": doc.get("id", ""),
                            "content": doc.get("text", ""),  # Lưu ý: sử dụng 'text' field làm content
                            "score": doc.get("metadata", {}).get("score", 0) if "metadata" in doc else 0,
                            "title": f"Document {sources.index(source) + 1}" if sources else "Document 1",
                            "source": "graph"
                        }
                        sources.append(source)
            # Fallback nếu không tìm thấy documents
            elif 'query' in kb_results and isinstance(kb_results.get('query'), dict):
                await self._send_thinking(
                    "query_info",
                    f"Query: {kb_results['query'].get('query', '')}"
                )
                
                # Kiểm tra xem còn field nào khác chứa thông tin
                await self._send_thinking(
                    "available_keys",
                    f"Available keys in result: {list(kb_results.keys())}"
                )
        else:
            await self._send_thinking(
                "unexpected_format",
                f"Received unexpected format: {type(kb_results)}"
            )
        
        # Nếu không tìm thấy sources nào
        if not sources:
            await self._send_thinking(
                "no_sources",
                "No relevant documents found"
            )
            return "Tôi không tìm thấy thông tin liên quan từ cơ sở dữ liệu đồ thị.", [], False
        
        # Generate answer from sources
        answer = ""
        stream_successful = False
        
        # Tạo context từ sources
        context = "\n\n".join([
            f"Document {i+1}:\n{source.get('content', '')}" 
            for i, source in enumerate(sources[:5]) 
            if 'content' in source and source['content'].strip()
        ])
        

        print(context)
        answer_prompt = f"""
        Based on the following information, please answer this question: {question}
        
        Information:
        {context}
        
        If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
        
        Answer:
        """
        
        # Stream the answer
        await self._send_thinking(
            "answer_generation_start",
            "Generating answer from graph knowledge..."
        )
        
        await self.manager.send_json(
            self.websocket,
            {
                "type": "stream_start",
                "message": ""
            }
        )
        
        answer, stream_successful = await stream_response_and_track(
            self.websocket, self.llm, answer_prompt, logger
        )
        
        # Add document details to thinking steps for transparency
        if sources:
            doc_list = ""
            for i, source in enumerate(sources[:3]):
                score = source.get('score', 0)
                doc_list += f"• Document {i+1} (relevance: {score:.2f})\n"
            
            if len(sources) > 3:
                doc_list += f"• ... and {len(sources) - 3} more documents"
                
            await self._send_thinking(
                "graph_document_details",
                f"Top retrieved documents:\n{doc_list}"
            )
        
        return answer, sources, stream_successful
        
    async def process_hybrid_rag_mode(self, question: str) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Process a question using hybrid RAG (combining vector and graph retrieval).
        
        Args:
            question: The user question
            
        Returns:
            Tuple of (answer, sources, stream_successful)
        """
        await self._send_thinking("hybrid_rag_retrieval", "Getting information using Hybrid RAG (vector + graph)...")
        
        kb_start_time = time.time()
        
        await self._send_thinking(
            "retrieval_config",
            "Using weighted combination of vector and graph retrieval (60% vector, 40% graph)"
        )
        
        # Process the query using the hybrid RAG
        kb_results = self.knowledge_retrieval.process(question, top_k=10)
        kb_elapsed = time.time() - kb_start_time
        
        # Đơn giản hóa xử lý kết quả
        sources = []
        vector_docs = []
        graph_docs = []
        
        # Debug the returned structure
        if isinstance(kb_results, dict):
            await self._send_thinking(
                "hybrid_result_keys",
                f"Hybrid RAG result keys: {list(kb_results.keys())}"
            )
        
        if isinstance(kb_results, dict):
            # Xử lý vector sources (sử dụng field 'content')
            if 'vector_results' in kb_results:
                vector_docs = kb_results['vector_results'].get('documents', [])
                await self._send_thinking(
                    "vector_results",
                    f"Found {len(vector_docs)} relevant vector documents"
                )
                
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
                    await self._send_thinking(
                        "graph_results",
                        f"Found {len(graph_docs)} relevant graph documents"
                    )
                    
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
                await self._send_thinking(
                    "combined_results",
                    f"Combined and deduplicated to {len(combined_docs)} documents in {kb_elapsed:.2f} seconds"
                )
                
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
            await self._send_thinking(
                "hybrid_results_error",
                f"Unexpected format from hybrid retrieval: {type(kb_results)}"
            )
        
        # Nếu không tìm thấy sources nào
        if not sources:
            await self._send_thinking(
                "no_sources", 
                "No relevant documents found"
            )
            return "Tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này.", [], False
        
        # Generate answer from sources
        answer = ""
        stream_successful = False
        
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
        
        # Stream the answer
        await self._send_thinking(
            "answer_generation_start",
            "Generating answer from hybrid knowledge sources..."
        )
        
        await self.manager.send_json(
            self.websocket,
            {
                "type": "stream_start",
                "message": ""
            }
        )
        
        answer, stream_successful = await stream_response_and_track(
            self.websocket, self.llm, answer_prompt, logger
        )
        
        # Thống kê về sources
        if sources:
            vector_count = len([s for s in sources if s.get('source_type') == 'vector'])
            graph_count = len([s for s in sources if s.get('source_type') == 'graph'])
            hybrid_count = len([s for s in sources if s.get('source_type') == 'hybrid'])
            
            await self._send_thinking(
                "hybrid_source_stats",
                f"Source breakdown: {vector_count} vector, {graph_count} graph, {hybrid_count} combined"
            )
            
            # Hiển thị top sources
            doc_list = ""
            for i, source in enumerate(sorted_sources[:3]):
                title = source.get('title', 'Unknown document')
                source_type = source.get('source_type', 'unknown')
                score = source.get('score', 0)
                doc_list += f"• {title} (source: {source_type}, relevance: {score:.2f})\n"
            
            if len(sources) > 3:
                doc_list += f"• ... and {len(sources) - 3} more sources"
                
            await self._send_thinking(
                "hybrid_sources",
                f"Top retrieved sources:\n{doc_list}"
            )
        
        return answer, sources, stream_successful
    
    async def process_hybrid_mode(
        self, 
        question: str, 
        kb_sources: List[Dict[str, Any]], 
        sql_query: Optional[str], 
        sql_data: Optional[List[Dict[str, Any]]]
    ) -> Tuple[str, bool]:
        """
        Process a question using both knowledge base and database.
        
        Args:
            question: The user question
            kb_sources: Sources from the knowledge base
            sql_query: Generated SQL query
            sql_data: Data returned from the SQL query
            
        Returns:
            Tuple of (answer, stream_successful)
        """
        await self._send_thinking(
            "combining_sources",
            "Combining information from knowledge base and database..."
        )
        
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
        
        # Stream the answer
        await self._send_thinking(
            "answer_generation_start",
            "Generating answer from combined sources..."
        )
        
        await self.manager.send_json(
            self.websocket,
            {
                "type": "stream_start",
                "message": ""
            }
        )
        
        answer, stream_successful = await stream_response_and_track(
            self.websocket, self.llm, combined_prompt, logger
        )
        
        return answer, stream_successful 