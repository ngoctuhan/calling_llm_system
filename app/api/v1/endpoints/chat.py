from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Models for request and response
class ChatRequest(BaseModel):
    question: str
    mode: str = "knowledgebase"  # Options: "knowledgebase", "database", "hybrid"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[List[Dict[str, str]]] = None

# Initialize the tools
def get_llm():
    return LLMProviderFactory.create_provider(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_knowledge_retrieval():
    llm = get_llm()
    
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
            "collection_name": os.getenv("VECTOR_COLLECTION_NAME", "callcenter"), 
            "similarity_top_k": 10
        }
    )
    
    return vector_rag

def get_text2sql():
    llm = get_llm()
    
    text2sql = Text2SQL(
        db_type="postgres",
        llm_provider=llm,
        max_retries=2,
        batch_size=3,
        max_concurrency=5,
        connection_params={
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT"),
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
        }
    )
    
    return text2sql


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)

manager = ConnectionManager()

# Thêm một hàm tiện ích mới để theo dõi trạng thái streaming
async def stream_response_and_track(websocket, llm, answer_prompt, logger):
    """
    Stream response from LLM and track if streaming was successful.
    Returns the generated answer and a boolean indicating if streaming was successful.
    """
    collected_answer = ""
    stream_success = False
    
    try:
        # Try streaming first
        logger.info("Starting text stream generation")
        async for chunk in llm.generate_text_stream(answer_prompt):
            if chunk:
                collected_answer += chunk
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "chunk": chunk
                }))
        stream_success = True
    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        # Fall back to non-streaming if streaming fails
        logger.info("Falling back to non-streaming generation")
        try:
            collected_answer = await llm.generate_text(answer_prompt)
            # Send the full answer as one chunk for the client to display
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "chunk": collected_answer
            }))
            stream_success = True
        except Exception as fallback_error:
            logger.error(f"Fallback generation also failed: {str(fallback_error)}")
            collected_answer = "Tôi không thể xử lý câu hỏi của bạn lúc này. Vui lòng thử lại sau."
    
    # End the stream regardless of method used
    await websocket.send_text(json.dumps({
        "type": "stream_end"
    }))
    
    # Log completion of answer generation
    logger.info(f"Answer generation complete, stream success: {stream_success}")
    
    return collected_answer, stream_success

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Extract question and mode
            question = request_data.get("question", "")
            mode = request_data.get("mode", "knowledgebase")
            
            # Process in stages to show thinking
            await manager.send_message(websocket, json.dumps({
                "type": "thinking",
                "step": "start",
                "content": f"Processing question: {question}"
            }))
            
            await manager.send_message(websocket, json.dumps({
                "type": "thinking",
                "step": "mode",
                "content": f"Using mode: {mode}"
            }))
            
            answer = ""
            sources = []
            sql_query = None
            sql_data = None
            stream_successful = False
            
            start_time = time.time()
            
            try:
                if mode == "knowledgebase" or mode == "hybrid":
                    await manager.send_message(websocket, json.dumps({
                        "type": "thinking",
                        "step": "knowledge_retrieval",
                        "content": "Getting information from knowledge base..."
                    }))
                    
                    kb_start_time = time.time()
                    vector_rag = get_knowledge_retrieval()
                    collection_name = os.getenv("VECTOR_COLLECTION_NAME", "callcenter")
                    
                    await manager.send_message(websocket, json.dumps({
                        "type": "thinking",
                        "step": "retrieval_config",
                        "content": f"Using collection: {collection_name}"
                    }))
                    
                    kb_results = vector_rag.process(question, top_k=5)
                    kb_elapsed = time.time() - kb_start_time
                    
                    # Handle the new response format which might have documents in results.documents
                    has_relevant_docs = False
                    if isinstance(kb_results, dict):
                        # Check if the new format is being used (results.documents structure)
                        if 'results' in kb_results and 'documents' in kb_results['results']:
                            retrieved_docs = kb_results['results']['documents']
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "knowledge_results",
                                "content": f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds"
                            }))
                            
                            # Check if we have meaningful results
                            has_relevant_docs = len(retrieved_docs) > 0
                            
                            # Extract sources from the new format
                            sources = []
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
                            
                            # Continue with normal processing for documents
                            if 'answer' in kb_results and kb_results['answer']:
                                answer = kb_results["answer"]
                            elif not answer and has_relevant_docs:
                                # Generate an answer using the fetched documents
                                llm = get_llm()
                                context = "\n\n".join([f"Document {i+1}:\n{doc.get('content', '')}" 
                                                    for i, doc in enumerate(retrieved_docs[:3]) if 'content' in doc and doc['content'].strip()])
                                
                                answer_prompt = f"""
                                Based on the following information, please answer this question: {question}
                                
                                Information:
                                {context}
                                
                                If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
                                
                                Answer:
                                """
                                
                                # Stream the answer
                                await manager.send_message(websocket, json.dumps({
                                    "type": "thinking",
                                    "step": "answer_generation_start",
                                    "content": "Generating answer from retrieved documents..."
                                }))
                                
                                await manager.send_message(websocket, json.dumps({
                                    "type": "stream_start",
                                    "message": ""
                                }))
                                
                                answer, stream_successful = await stream_response_and_track(
                                    websocket, llm, answer_prompt, logger
                                )
                            else:
                                # No relevant documents found
                                answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                                stream_successful = False
                        elif 'documents' in kb_results:
                            # Handle format with direct documents array
                            retrieved_docs = kb_results['documents']
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "knowledge_results",
                                "content": f"Found {len(retrieved_docs)} relevant documents in {kb_elapsed:.2f} seconds"
                            }))
                            
                            # Check if we have meaningful results
                            has_relevant_docs = len(retrieved_docs) > 0
                            
                            # Extract sources from the format
                            sources = []
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
                            
                            # Generate answer using fetched documents
                            if not answer and has_relevant_docs:
                                llm = get_llm()
                                context = "\n\n".join([f"Document {i+1}:\n{doc.get('content', '')}" 
                                                    for i, doc in enumerate(retrieved_docs[:3]) if 'content' in doc and doc['content'].strip()])
                                
                                answer_prompt = f"""
                                Based on the following information, please answer this question: {question}
                                
                                Information:
                                {context}
                                
                                If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
                                
                                Answer:
                                """
                                
                                # Stream the answer
                                await manager.send_message(websocket, json.dumps({
                                    "type": "thinking",
                                    "step": "answer_generation_start",
                                    "content": "Generating answer from retrieved documents..."
                                }))
                                
                                await manager.send_message(websocket, json.dumps({
                                    "type": "stream_start",
                                    "message": ""
                                }))
                                
                                answer, stream_successful = await stream_response_and_track(
                                    websocket, llm, answer_prompt, logger
                                )
                            else:
                                # No relevant documents found
                                answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                                stream_successful = False
                        else:
                            # Handle the old format
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "knowledge_results",
                                "content": f"Found {len(kb_results.get('sources', []))} relevant documents in {kb_elapsed:.2f} seconds"
                            }))
                            sources = kb_results.get("sources", [])
                            has_relevant_docs = len(sources) > 0
                            
                            # If there's an answer in the results, use it
                            if 'answer' in kb_results and kb_results['answer']:
                                answer = kb_results["answer"]
                            elif has_relevant_docs:
                                # Generate answer from sources
                                llm = get_llm()
                                context = "\n\n".join([f"Document {i+1}:\n{source.get('content', '')}" 
                                                   for i, source in enumerate(sources[:3]) if 'content' in source and source['content'].strip()])
                                
                                answer_prompt = f"""
                                Based on the following information, please answer this question: {question}
                                
                                Information:
                                {context}
                                
                                If the provided information doesn't contain enough details to answer the question confidently, respond with: "Tôi không có đủ thông tin để trả lời câu hỏi này."
                                
                                Answer:
                                """
                                
                                # Stream the answer
                                await manager.send_message(websocket, json.dumps({
                                    "type": "thinking",
                                    "step": "answer_generation_start",
                                    "content": "Generating answer from retrieved documents..."
                                }))
                                
                                await manager.send_message(websocket, json.dumps({
                                    "type": "stream_start",
                                    "message": ""
                                }))
                                
                                answer, stream_successful = await stream_response_and_track(
                                    websocket, llm, answer_prompt, logger
                                )
                            else:
                                answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                                stream_successful = False
                    else:
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "knowledge_results_error",
                            "content": f"Unexpected format from knowledge retrieval: {type(kb_results)}"
                        }))
                        answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                        stream_successful = False
                    
                    # Adding document details to thinking steps for transparency
                    if sources:
                        doc_list = ""
                        for i, source in enumerate(sources[:3]):
                            title = source.get('title', 'Unknown document')
                            score = source.get('score', 0)
                            doc_list += f"• {title} (relevance: {score:.2f})\n"
                        
                        if len(sources) > 3:
                            doc_list += f"• ... and {len(sources) - 3} more documents"
                            
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "knowledge_documents",
                            "content": f"Top retrieved documents:\n{doc_list}"
                        }))
                    
                    # Generate final answer for knowledgebase mode
                    if mode == "knowledgebase":
                        answer_start_time = time.time()
                        if not answer:  # If answer wasn't already generated
                            answer = "Tôi không có đủ thông tin để trả lời câu hỏi này."
                            stream_successful = False
                        
                        # Only send this if we didn't already stream the answer
                        if not has_relevant_docs:
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "answer_generation",
                                "content": f"Generated answer from knowledge base in {time.time() - answer_start_time:.2f} seconds"
                            }))
                
                if mode == "database" or mode == "hybrid":
                    await manager.send_message(websocket, json.dumps({
                        "type": "thinking",
                        "step": "database_query",
                        "content": "Converting question to SQL query..."
                    }))
                    
                    sql_start_time = time.time()
                    text2sql = get_text2sql()
                    sql_result = await text2sql.process_query(question)
                    sql_elapsed = time.time() - sql_start_time
                    
                    if sql_result["success"]:
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "sql_generation", 
                            "content": f"Generated SQL in {sql_elapsed:.2f} seconds:\n{sql_result['sql']}"
                        }))
                        
                        sql_query = sql_result["sql"]
                        sql_data = sql_result["data"]
                        
                        if "tables_used" in sql_result and sql_result["tables_used"]:
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "sql_tables", 
                                "content": f"Tables used: {', '.join(sql_result['tables_used'])}"
                            }))
                        
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "sql_results", 
                            "content": f"Query returned {len(sql_data)} rows"
                        }))
                        
                        if mode == "database":
                            if sql_data:
                                # Chỉ tạo câu trả lời nếu có dữ liệu SQL
                                llm = get_llm()
                                data_context = f"SQL query: {sql_query}\n\nData results: {json.dumps(sql_data, indent=2)}"
                                answer_prompt = f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{question}'\n\n{data_context}"
                                
                                # Stream the answer
                                await manager.send_message(websocket, json.dumps({
                                    "type": "thinking",
                                    "step": "answer_generation_start",
                                    "content": "Generating answer from database results..."
                                }))
                                
                                await manager.send_message(websocket, json.dumps({
                                    "type": "stream_start",
                                    "message": ""
                                }))
                                
                                answer, stream_successful = await stream_response_and_track(
                                    websocket, llm, answer_prompt, logger
                                )
                            else:
                                answer = "No data found in database for your question."
                                stream_successful = False
                        else:
                            # Combine results from both sources
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "combining_sources",
                                "content": "Combining information from knowledge base and database..."
                            }))
                            
                            llm = get_llm()
                            kb_context = json.dumps(sources, indent=2) if sources else "No knowledge base results."
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
                            await manager.send_message(websocket, json.dumps({
                                "type": "thinking",
                                "step": "answer_generation_start",
                                "content": "Generating answer from combined sources..."
                            }))
                            
                            await manager.send_message(websocket, json.dumps({
                                "type": "stream_start",
                                "message": ""
                            }))
                            
                            answer, stream_successful = await stream_response_and_track(
                                websocket, llm, combined_prompt, logger
                            )
                    else:
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "sql_error", 
                            "content": f"Error generating SQL: {sql_result.get('error', 'Unknown error')}"
                        }))
                        
                        if mode == "database":
                            answer = "Sorry, I couldn't convert your question to a database query."
                            stream_successful = False
                
                total_time = time.time() - start_time
                await manager.send_message(websocket, json.dumps({
                    "type": "thinking",
                    "step": "complete",
                    "content": f"Processing complete in {total_time:.2f} seconds."
                }))
                
                # Luôn gửi sources, SQL và data kể cả khi streaming thành công
                # Để frontend có thể hiển thị chúng
                await manager.send_message(websocket, json.dumps({
                    "type": "result",
                    "answer": answer,
                    "sources": sources,
                    "sql": sql_query,
                    "data": sql_data
                }))
                
            except Exception as e:
                logger.error(f"Error processing WebSocket chat: {str(e)}")
                await manager.send_message(websocket, json.dumps({
                    "type": "thinking",
                    "step": "error",
                    "content": f"Error: {str(e)}"
                }))
                
                await manager.send_message(websocket, json.dumps({
                    "type": "error",
                    "message": f"Error processing question: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

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