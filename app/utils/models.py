from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class ChatRequest(BaseModel):
    """Request model for chat operations"""
    question: str
    mode: Literal["vector", "graph", "hybrid_rag", "database", "hybrid"] = "vector"

class ChatResponse(BaseModel):
    """Response model for chat operations"""
    answer: str
    sources: List[Dict[str, Any]] = []
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[List[Dict[str, str]]] = None

class ThinkingStep(BaseModel):
    """Model for a thinking step in the chat process"""
    type: Literal["thinking"] = "thinking"
    step: str
    content: str
    
class StreamChunk(BaseModel):
    """Model for a stream chunk in the chat process"""
    type: Literal["stream_chunk"] = "stream_chunk"
    chunk: str
    
class StreamStart(BaseModel):
    """Model for the start of a stream in the chat process"""
    type: Literal["stream_start"] = "stream_start"
    message: str = ""
    
class StreamEnd(BaseModel):
    """Model for the end of a stream in the chat process"""
    type: Literal["stream_end"] = "stream_end"
    
class ChatResult(BaseModel):
    """Model for the final result of a chat operation"""
    type: Literal["result"] = "result"
    answer: str
    sources: List[Dict[str, Any]] = []
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    
class ChatError(BaseModel):
    """Model for an error in the chat process"""
    type: Literal["error"] = "error"
    message: str 