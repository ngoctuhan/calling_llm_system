from fastapi import APIRouter
from app.api.v1.endpoints import chat, ingestion, retrieval

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
api_router.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"]) 