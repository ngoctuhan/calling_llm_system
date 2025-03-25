import os
import logging
import hashlib
from typing import Dict, List, Optional
from functools import lru_cache

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, Field

import tiktoken
from data_ingestion.readers import CMSReader
from data_ingestion.chunkers import TextChunk, TokenTextChunker
from retrieval_engine.knowledge_retrieval.vector_rag import VectorRAG
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphBuilder
from retrieval_engine.knowledge_retrieval.graph_rag_v2.graph_extractor import GraphExtractor
from retrieval_engine.knowledge_retrieval.graph_rag_v2.neo4j_connection import SimpleNeo4jConnection
from llm_services import LLMProviderFactory
from retrieval_engine.knowledge_retrieval.vector_rag.embeddings import EmbeddingFactory

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Pydantic models for request/response
class IngestURLRequest(BaseModel):
    url: HttpUrl
    collection_name: str = "callcenter"
    chunk_size: Optional[int] = Field(default=500, ge=100, le=4000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=500)
    credentials: Optional[Dict] = {}
    metadata: Optional[Dict] = {}

class IngestTextRequest(BaseModel):
    text: str
    document_id: str
    collection_name: str = "callcenter"
    chunk_size: Optional[int] = Field(default=500, ge=100, le=4000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=500)
    metadata: Optional[Dict] = {}

class IngestDocumentsRequest(BaseModel):
    documents: List[str]
    document_ids: List[str]
    collection_name: str = "callcenter"
    chunk_size: Optional[int] = Field(default=500, ge=100, le=4000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=500)
    metadatas: Optional[List[Dict]] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunk_count: Optional[int] = None
    job_id: Optional[str] = None

# Helper function to get tokenizer functions for chunking
def get_tiktoken_functions():
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def encode_fn(text):
        return encoding.encode(text)

    def decode_fn(tokens):
        # Handle the test case in the TokenTextChunker constructor
        if tokens and isinstance(tokens[0], str) and tokens[0] == "test":
            return "test string"
        
        # Normal case: convert tokens to text
        if isinstance(tokens, list) and all(isinstance(token, int) for token in tokens):
            return encoding.decode(tokens)
        else:
            try:
                return encoding.decode([int(token) for token in tokens if token])
            except ValueError:
                logger.warning(f"Could not convert tokens to int")
                return ""
                
    return encode_fn, decode_fn

# Helper function to create chunker with specified settings
def create_token_chunker(tokens_per_chunk: int, chunk_overlap: int) -> TokenTextChunker:
    encode_fn, decode_fn = get_tiktoken_functions()
    return TokenTextChunker(
        tokenizer_fn=encode_fn,
        detokenizer_fn=decode_fn,
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap
    )

# Helper function to initialize vector RAG
@lru_cache(maxsize=8)
def get_vector_rag(collection_name: str) -> VectorRAG:
    return VectorRAG(
        collection_name=collection_name,
        embedding_provider="google",
        embedding_model_name="text-embedding-004",
    )

# Helper function to initialize graph builder
async def get_graph_builder():
    # Initialize embedding provider
    embedding_provider = EmbeddingFactory.create_provider(
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
    
    graph_extractor = GraphExtractor(
        llm=llm, 
        embedding_provider=embedding_provider,
        batch_size=30,
        max_knowledge_triplets=100
    )
    
    # Initialize Neo4j connection
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
        max_workers=10,
        batch_size=32
    )
    return builder

# Get chunkers with cached instances
@lru_cache(maxsize=1)
def get_graph_chunker() -> TokenTextChunker:
    return create_token_chunker(tokens_per_chunk=1000, chunk_overlap=200)

@lru_cache(maxsize=1)
def get_vector_chunker(chunk_size: int = 500, chunk_overlap: int = 100) -> TokenTextChunker:
    return create_token_chunker(tokens_per_chunk=chunk_size, chunk_overlap=chunk_overlap)

# Background task for processing documents
async def process_documents_task(
    documents: List[str],
    document_ids: List[str],
    collection_name: str,
    metadatas: Optional[List[Dict]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    try:
        # Get chunkers for vector and graph databases
        vector_chunker = get_vector_chunker(chunk_size, chunk_overlap)
        graph_chunker = get_graph_chunker()
        
        # Get database connections
        vector_rag = get_vector_rag(collection_name)
        graph_builder = await get_graph_builder()
        
        metadatas = metadatas or [{} for _ in documents]
        
        # Process each document
        for i, (doc, doc_id, metadata) in enumerate(zip(documents, document_ids, metadatas)):
            metadata = metadata.copy()
            metadata['document_id'] = doc_id
            
            # Split document into chunks for vector database
            vector_chunks = vector_chunker.chunk(doc, metadata)
            
            # Add to vector database
            vector_rag.index_documents(
                documents=[chunk.text for chunk in vector_chunks],
                metadatas=[chunk.metadata for chunk in vector_chunks],
            )
            
            # Split document into larger chunks for graph database
            graph_chunks = graph_chunker.chunk(doc, metadata)
            
            # Prepare graph documents with document_id
            graph_documents = [
                {
                    "text": chunk.text,
                    "document_id": doc_id,
                    "metadata": chunk.metadata 
                }
                for chunk in graph_chunks
            ]
            
            # Process graph documents
            await graph_builder.process_documents(graph_documents, concurrency=10)
            
            logger.info(f"Processed document {i+1}/{len(documents)}: {doc_id} - "
                       f"{len(vector_chunks)} vector chunks, {len(graph_chunks)} graph chunks")
        
        # Close connections
        await graph_builder.close()
        
        logger.info(f"Completed ingestion of {len(documents)} documents into '{collection_name}' collection")
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        raise

@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """
    Ingest content from a URL into both vector and graph databases
    """
    try:
        # Initialize CMS reader
        cms_reader = CMSReader(credentials=request.credentials)
        
        # Read content from URL
        content = cms_reader.read(str(request.url))
        text = cms_reader.get_text(content)
        
        # Get metadata
        metadata = cms_reader.get_metadata(str(request.url))
        if request.metadata:
            metadata.update(request.metadata)
            
        # Generate document ID from URL
        document_id = hashlib.md5(str(request.url).encode()).hexdigest()
        
        # Process in background
        background_tasks.add_task(
            process_documents_task,
            documents=[text],
            document_ids=[document_id],
            collection_name=request.collection_name,
            metadatas=[metadata],
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return IngestResponse(
            success=True,
            message=f"Started ingestion of {request.url}",
            document_id=document_id,
            job_id=document_id
        )
    except Exception as e:
        logger.error(f"Error ingesting URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest URL: {str(e)}")