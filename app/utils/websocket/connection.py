from fastapi import WebSocket
from typing import List
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections and facilitates sending messages to clients.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection established. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed. Remaining connections: {len(self.active_connections)}")

    async def send_message(self, websocket: WebSocket, message: str):
        """Send a text message to a specific WebSocket client"""
        await websocket.send_text(message)
    
    async def send_json(self, websocket: WebSocket, data: dict):
        """Send a JSON message to a specific WebSocket client"""
        await websocket.send_text(json.dumps(data))
        
    async def broadcast(self, message: str):
        """Send a text message to all connected clients"""
        for connection in self.active_connections:
            await connection.send_text(message)
            
    async def broadcast_json(self, data: dict):
        """Send a JSON message to all connected clients"""
        json_data = json.dumps(data)
        for connection in self.active_connections:
            await connection.send_text(json_data) 