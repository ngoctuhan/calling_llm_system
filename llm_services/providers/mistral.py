import json
from typing import List, Optional
from .provider import LLMProvider, Message
import httpx 
from typing import Generator
class MistralProvider(LLMProvider):
    """Mistral AI provider"""
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
    
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> str:
        """Generate text from Mistral AI models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_prompt = kwargs.get("system_prompt", "")
        
        if messages:
            # Use message history, extract system message if needed
            payload_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    if not system_prompt:
                        system_prompt = msg.content
                else:
                    payload_messages.append(msg.to_dict())
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        # Insert system prompt if provided
        if system_prompt:
            payload_messages.insert(0, {"role": "system", "content": system_prompt})
            
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = await self.client.post(
            self.api_url,
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"Mistral AI API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from Mistral AI models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_prompt = kwargs.get("system_prompt", "")
        
        if messages:
            # Use message history, extract system message if needed
            payload_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    if not system_prompt:
                        system_prompt = msg.content
                else:
                    payload_messages.append(msg.to_dict())
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        # Insert system prompt if provided
        if system_prompt:
            payload_messages.insert(0, {"role": "system", "content": system_prompt})
            
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        async with self.client.stream(
            "POST",
            self.api_url,
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_msg = f"Mistral AI API error: {response.status_code}"
                raise Exception(error_msg)
            
            async for chunk in response.aiter_lines():
                if not chunk or chunk == "data: [DONE]":
                    continue
                
                if chunk.startswith("data: "):
                    json_str = chunk[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_str)
                        if "choices" in data and data["choices"] and "delta" in data["choices"][0]:
                            delta = data["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                    except (json.JSONDecodeError, KeyError) as e:
                        pass