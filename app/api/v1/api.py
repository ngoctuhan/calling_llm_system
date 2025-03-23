from fastapi import APIRouter
from app.api.v1.endpoints import chat, ingestion

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"]) 