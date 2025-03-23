from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os
import logging
import asyncio
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
            "provider": "huggingface",  
            "model_name": "all-MiniLM-L6-v2",  
        },
        retrieval_config={
            "index_name": "example_hf", 
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

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat request and return a response using the appropriate tools.
    """
    thinking_steps = []
    answer = ""
    sources = []
    sql_query = None
    sql_data = None
    
    try:
        thinking_steps.append({"step": "start", "content": f"Processing question: {request.question}"})
        thinking_steps.append({"step": "mode", "content": f"Using mode: {request.mode}"})
        
        if request.mode == "knowledgebase" or request.mode == "hybrid":
            thinking_steps.append({"step": "knowledge_retrieval", "content": "Getting information from knowledge base..."})
            vector_rag = get_knowledge_retrieval()
            kb_results = vector_rag.process(request.question, top_k=5)
            
            thinking_steps.append({"step": "knowledge_results", "content": f"Found {len(kb_results.get('sources', []))} relevant documents"})
            sources = kb_results.get("sources", [])
            
            if request.mode == "knowledgebase":
                answer = kb_results.get("answer", "No answer found in knowledge base.")
        
        if request.mode == "database" or request.mode == "hybrid":
            thinking_steps.append({"step": "database_query", "content": "Converting question to SQL query..."})
            text2sql = get_text2sql()
            sql_result = await text2sql.process_query(request.question)
            
            if sql_result["success"]:
                thinking_steps.append({"step": "sql_generation", "content": f"Generated SQL: {sql_result['sql']}"})
                sql_query = sql_result["sql"]
                sql_data = sql_result["data"]
                
                if request.mode == "database":
                    if sql_data:
                        llm = get_llm()
                        data_context = f"SQL query: {sql_query}\n\nData results: {sql_data}"
                        answer_response = await llm.generate_text(
                            f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{request.question}'\n\n{data_context}"
                        )
                        answer = answer_response
                    else:
                        answer = "No data found in database for your question."
            else:
                thinking_steps.append({"step": "sql_error", "content": f"Error generating SQL: {sql_result.get('error', 'Unknown error')}"})
                if request.mode == "database":
                    answer = "Sorry, I couldn't convert your question to a database query."
        
        if request.mode == "hybrid":
            # Combine results from both sources
            llm = get_llm()
            kb_context = json.dumps(sources, indent=2) if sources else "No knowledge base results."
            db_context = f"SQL query: {sql_query}\n\nData results: {json.dumps(sql_data, indent=2)}" if sql_query else "No database results."
            
            combined_prompt = f"""
            The user asked: '{request.question}'
            
            Knowledge base information:
            {kb_context}
            
            Database information:
            {db_context}
            
            Based on all available information, provide a comprehensive and accurate answer to the user's question.
            """
            
            thinking_steps.append({"step": "combining_sources", "content": "Combining information from knowledge base and database..."})
            answer_response = await llm.generate_text(combined_prompt)
            answer = answer_response
            
        thinking_steps.append({"step": "complete", "content": "Processing complete."})
            
        return ChatResponse(
            answer=answer,
            sources=sources,
            sql=sql_query,
            data=sql_data,
            thinking=thinking_steps
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        thinking_steps.append({"step": "error", "content": f"Error: {str(e)}"})
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

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
            
            try:
                if mode == "knowledgebase" or mode == "hybrid":
                    await manager.send_message(websocket, json.dumps({
                        "type": "thinking",
                        "step": "knowledge_retrieval",
                        "content": "Getting information from knowledge base..."
                    }))
                    
                    vector_rag = get_knowledge_retrieval()
                    kb_results = vector_rag.process(question, top_k=5)
                    
                    await manager.send_message(websocket, json.dumps({
                        "type": "thinking",
                        "step": "knowledge_results",
                        "content": f"Found {len(kb_results.get('sources', []))} relevant documents"
                    }))
                    
                    sources = kb_results.get("sources", [])
                    
                    if mode == "knowledgebase":
                        answer = kb_results.get("answer", "No answer found in knowledge base.")
                
                if mode == "database" or mode == "hybrid":
                    await manager.send_message(websocket, json.dumps({
                        "type": "thinking",
                        "step": "database_query",
                        "content": "Converting question to SQL query..."
                    }))
                    
                    text2sql = get_text2sql()
                    sql_result = await text2sql.process_query(question)
                    
                    if sql_result["success"]:
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "sql_generation",
                            "content": f"Generated SQL: {sql_result['sql']}"
                        }))
                        
                        sql_query = sql_result["sql"]
                        sql_data = sql_result["data"]
                        
                        if mode == "database":
                            if sql_data:
                                llm = get_llm()
                                data_context = f"SQL query: {sql_query}\n\nData results: {sql_data}"
                                answer_response = await llm.generate_text(
                                    f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{question}'\n\n{data_context}"
                                )
                                answer = answer_response
                            else:
                                answer = "No data found in database for your question."
                    else:
                        await manager.send_message(websocket, json.dumps({
                            "type": "thinking",
                            "step": "sql_error",
                            "content": f"Error generating SQL: {sql_result.get('error', 'Unknown error')}"
                        }))
                        
                        if mode == "database":
                            answer = "Sorry, I couldn't convert your question to a database query."
                
                if mode == "hybrid":
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
                    
                    answer_response = await llm.generate_text(combined_prompt)
                    answer = answer_response
                
                # Send final result
                await manager.send_message(websocket, json.dumps({
                    "type": "result",
                    "answer": answer,
                    "sources": sources,
                    "sql": sql_query,
                    "data": sql_data
                }))
                
                await manager.send_message(websocket, json.dumps({
                    "type": "thinking",
                    "step": "complete",
                    "content": "Processing complete."
                }))
                
            except Exception as e:
                logger.error(f"Error processing WebSocket chat: {str(e)}")
                await manager.send_message(websocket, json.dumps({
                    "type": "error",
                    "message": f"Error processing question: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket) 