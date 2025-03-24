import os
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import traceback
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType
from retrieval_engine.text2sql import Text2SQL
from llm_services import LLMProviderFactory

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Pydantic models for request/response
class RetrievalRequest(BaseModel):
    query: str
    collection_name: str = "callcenter"
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    model_name: Optional[str] = "gemini-2.0-flash"

class Text2SQLRequest(BaseModel):
    query: str
    model_name: Optional[str] = "gemini-2.0-flash"

class RetrievalResponse(BaseModel):
    success: bool
    query: str
    method: str
    results: Any
    answer: Optional[str] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

# Cache for retrieval instances to avoid recreating them for each request
rag_instances = {}

def get_vector_rag(collection_name: str):
    """Get or create a Vector RAG instance"""
    cache_key = f"vector_{collection_name}"
    if cache_key not in rag_instances:
        rag_instances[cache_key] = RetrievalFactory.create_rag(
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
                "collection_name": collection_name, 
                "similarity_top_k": 10
            }
        )
    return rag_instances[cache_key]

def get_graph_rag():
    """Get or create a Graph RAG instance"""
    cache_key = "graph_rag"
    if cache_key not in rag_instances:
        rag_instances[cache_key] = RetrievalFactory.create_rag(
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
                "similarity_top_k": 10,
                "hybrid": False 
            }
        )
    return rag_instances[cache_key]

def get_hybrid_rag(collection_name: str):
    """Get or create a Hybrid RAG instance"""
    cache_key = f"hybrid_{collection_name}"
    if cache_key not in rag_instances:
        vector_rag = get_vector_rag(collection_name)
        graph_rag = get_graph_rag()
        
        rag_instances[cache_key] = RetrievalFactory.create_rag(
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
    return rag_instances[cache_key]

async def get_text2sql():
    """Get or create a Text2SQL instance"""
    cache_key = "text2sql"
    if cache_key not in rag_instances:
        llm = LLMProviderFactory.create_provider(
            model="gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize the Text2SQL system
        rag_instances[cache_key] = Text2SQL(
            db_type="postgres",
            llm_provider=llm,
            max_retries=2,
            batch_size=3,
            max_concurrency=5
        )
    return rag_instances[cache_key]

@router.post("/test/vector", response_model=RetrievalResponse)
async def test_vector_rag(request: RetrievalRequest):
    """Test Vector RAG retrieval"""
    try:
        import time
        start_time = time.time()
        
        # Get Vector RAG instance
        vector_rag = get_vector_rag(request.collection_name)
        
        # Process the query
        results = vector_rag.process(request.query, top_k=request.top_k)
        
        execution_time = time.time() - start_time
        
        return RetrievalResponse(
            success=True,
            query=request.query,
            method="vector",
            results=results,
            answer=results.get("answer", ""),
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in Vector RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Vector RAG query: {str(e)}")

@router.post("/test/graph", response_model=RetrievalResponse)
async def test_graph_rag(request: RetrievalRequest):
    """Test Graph RAG retrieval"""
    try:
        import time
        start_time = time.time()
        
        # Get Graph RAG instance
        graph_rag = get_graph_rag()
        
        # Process the query
        results = await asyncio.to_thread(graph_rag.process, 
                                          request.query, 
                                          top_k=request.top_k, 
                                          use_semantic=True, 
                                          use_graph=False)
        
        execution_time = time.time() - start_time
        
        return RetrievalResponse(
            success=True,
            query=request.query,
            method="graph",
            results=results,
            answer=results.get("answer", ""),
            execution_time=execution_time
        )
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in Graph RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Graph RAG query: {str(e)}")

@router.post("/test/hybrid", response_model=RetrievalResponse)
async def test_hybrid_rag(request: RetrievalRequest):
    """Test Hybrid RAG retrieval"""
    try:
        import time
        start_time = time.time()
        
        # Get Hybrid RAG instance
        hybrid_rag = get_hybrid_rag(request.collection_name)
        
        # Process the query
        results = hybrid_rag.process(request.query, top_k=request.top_k)
        
        execution_time = time.time() - start_time
        
        return RetrievalResponse(
            success=True,
            query=request.query,
            method="hybrid",
            results=results,
            answer=results.get("answer", ""),
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in Hybrid RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Hybrid RAG query: {str(e)}")

@router.post("/test/text2sql", response_model=RetrievalResponse)
async def test_text2sql(request: Text2SQLRequest):
    """Test Text2SQL retrieval"""
    try:
        import time
        start_time = time.time()
        
        # Get Text2SQL instance
        text2sql = await get_text2sql()
        
        # Process the query
        result = await text2sql.process_query(request.query)
        
        execution_time = time.time() - start_time
        
        return RetrievalResponse(
            success=result["success"],
            query=request.query,
            method="text2sql",
            results={
                "sql": result.get("sql", ""),
                "data": result.get("data", []),
                "tables_used": result.get("tables_used", []),
                "retries": result.get("retries", 0)
            },
            execution_time=execution_time,
            error=result.get("error") if not result["success"] else None
        )
    except Exception as e:
        logger.error(f"Error in Text2SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Text2SQL query: {str(e)}")

@router.on_event("shutdown")
async def cleanup():
    """Close all connections when shutting down"""
    for key, instance in rag_instances.items():
        if hasattr(instance, "close"):
            try:
                if asyncio.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close()
                logger.info(f"Closed connection for {key}")
            except Exception as e:
                logger.error(f"Error closing {key}: {str(e)}") 