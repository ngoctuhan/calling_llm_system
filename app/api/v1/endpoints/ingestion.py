import os
import logging
from typing import Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, Field

from data_ingestion.readers import CMSReader
from data_ingestion.chunkers import TextChunker
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
    chunk_size: Optional[int] = Field(default=500, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=500)
    credentials: Optional[Dict] = {}
    metadata: Optional[Dict] = {}

class IngestTextRequest(BaseModel):
    text: str
    document_id: str
    collection_name: str = "callcenter"
    chunk_size: Optional[int] = Field(default=500, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=500)
    metadata: Optional[Dict] = {}

class IngestDocumentsRequest(BaseModel):
    documents: List[str]
    document_ids: List[str]
    collection_name: str = "callcenter"
    chunk_size: Optional[int] = Field(default=500, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=500)
    metadatas: Optional[List[Dict]] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunk_count: Optional[int] = None
    job_id: Optional[str] = None

# Helper function to initialize vector RAG
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
    
    graph_extractor = GraphExtractor(llm=llm, 
                                     embedding_provider=embedding_provider,
                                     batch_size=30)
    
    # Initialize Neo4j connection
    print(os.environ.get("NEO4J_URI", "neo4j://localhost:7687"))
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

# Helper function to process a single chunk
async def process_chunk(text: str, document_id: str, metadata: Dict, vector_rag: VectorRAG, graph_builder: GraphBuilder):
    # Add to vector database
    vector_rag.index_documents(
        documents=[text],
        metadatas=[metadata],
        document_ids=[document_id]
    )
    
    # Add to graph database
    await graph_builder.process_document(
        text=text,
        document_id=document_id,
        document_metadata=metadata
    )

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
        chunker = TextChunker(min_chunk_size=chunk_size-100, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vector_rag = get_vector_rag(collection_name)
        graph_builder = await get_graph_builder()
        
        metadatas = metadatas or [{} for _ in documents]
        
        # Process each document
        for i, (doc, doc_id, metadata) in enumerate(zip(documents, document_ids, metadatas)):
            # Split document into chunks
            chunks = chunker.chunk(doc, metadata)
            
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Add to vector database
            vector_rag.index_documents(
                documents=chunk_texts,
                metadatas=chunk_metadatas,
            )

            graph_documents = [
                {
                    "text": chunk['text'],
                    "document_id": doc_id,
                    "metadata": chunk['metadata'] 
                }
                for chunk in chunks
            ]
            await graph_builder.process_documents(graph_documents, concurrency=10)
            logger.info(f"Processed document {i+1}/{len(documents)}: {doc_id} into {len(chunks)} chunks")
        
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
        import hashlib
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

@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest, background_tasks: BackgroundTasks):
    """
    Ingest text content into both vector and graph databases
    """
    try:
        # Process in background
        background_tasks.add_task(
            process_documents_task,
            documents=[request.text],
            document_ids=[request.document_id],
            collection_name=request.collection_name,
            metadatas=[request.metadata or {}],
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return IngestResponse(
            success=True,
            message=f"Started ingestion of document {request.document_id}",
            document_id=request.document_id,
            job_id=request.document_id
        )
    except Exception as e:
        logger.error(f"Error ingesting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest text: {str(e)}")
