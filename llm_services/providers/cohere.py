import json
from typing import List, Optional
import httpx 
from .provider import LLMProvider, Message
from typing import Generator
class CohereProvider(LLMProvider):
    """Cohere provider"""
    
    def __init__(self, api_key: str, model: str = "command"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.api_url = "https://api.cohere.ai/v1/generate"
    
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> str:
        """Generate text from Cohere models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Convert messages to Cohere's chat format
            chat_history = []
            for msg in messages:
                if msg.role == "system":
                    # Prepend system message to first user message
                    continue
                elif msg.role == "user":
                    chat_history.append({"role": "USER", "message": msg.content})
                elif msg.role == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg.content})
            
            # Check for system message
            system_message = next((msg for msg in messages if msg.role == "system"), None)
            
            payload = {
                "model": self.model,
                "chat_history": chat_history,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            if system_message:
                payload["preamble"] = system_message.content
        else:
            # Use raw prompt for non-chat models
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        
        response = await self.client.post(
            self.api_url,
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"Cohere API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result["generations"][0]["text"]
    
     
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from Cohere models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Convert messages to Cohere's chat format
            chat_history = []
            for msg in messages:
                if msg.role == "system":
                    # Prepend system message to first user message
                    continue
                elif msg.role == "user":
                    chat_history.append({"role": "USER", "message": msg.content})
                elif msg.role == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg.content})
            
            # Check for system message
            system_message = next((msg for msg in messages if msg.role == "system"), None)
            
            payload = {
                "model": self.model,
                "chat_history": chat_history,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            
            if system_message:
                payload["preamble"] = system_message.content
        else:
            # Use raw prompt for non-chat models
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
        
        async with self.client.stream(
            "POST",
            self.api_url,
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_msg = f"Cohere API error: {response.status_code}"
                raise Exception(error_msg)
            
            async for chunk in response.aiter_lines():
                if not chunk:
                    continue
                
                try:
                    data = json.loads(chunk)
                    if "text" in data:
                        yield data["text"]
                except (json.JSONDecodeError, KeyError) as e:
                    pass