from fastapi import APIRouter
from app.api.v1.endpoints import chat, ingestion, retrieval

api_router = APIRouter()
# Keep the original chat endpoint for backward compatibility
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
# Add the refactored chat endpoint
api_router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"]) 