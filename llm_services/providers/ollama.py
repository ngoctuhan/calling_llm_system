from typing import List, Optional
from .provider import LLMProvider, Message
import httpx 
import json
from typing import Generator
class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM deployment"""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
            },
            timeout=300.0,  # Longer timeout for local models
        )
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
    
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> str:
        """Generate text from Ollama models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Use chat endpoint for messages
            payload_messages = []
            for msg in messages:
                role = msg.role
                # Ollama uses "assistant" role
                if role == "assistant":
                    role = "assistant"
                payload_messages.append({"role": role, "content": msg.content})
            
            payload = {
                "model": self.model,
                "messages": payload_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            response = await self.client.post(
                self.chat_url,
                json=payload,
            )
        else:
            # Use generate endpoint for single prompts
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            response = await self.client.post(
                self.api_url,
                json=payload,
            )
        
        if response.status_code != 200:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result.get("message", {}).get("content", "") if messages else result.get("response", "")
    
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from Ollama models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Use chat endpoint for messages
            payload_messages = []
            for msg in messages:
                role = msg.role
                # Ollama uses "assistant" role
                if role == "assistant":
                    role = "assistant"
                payload_messages.append({"role": role, "content": msg.content})
            
            payload = {
                "model": self.model,
                "messages": payload_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": True
            }
            
            async with self.client.stream(
                "POST",
                self.chat_url,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code}"
                    raise Exception(error_msg)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
        else:
            # Use generate endpoint for single prompts
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": True
            }
            
            async with self.client.stream(
                "POST",
                self.api_url,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code}"
                    raise Exception(error_msg)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
