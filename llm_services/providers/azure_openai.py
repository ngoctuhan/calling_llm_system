from typing import List, Optional
from .provider import LLMProvider, Message
import httpx 
import json
from typing import Generator
class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2023-07-01-preview"):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.client = httpx.AsyncClient(
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.api_url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
    
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> str:
        """Generate text from Azure OpenAI models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Use full conversation history
            payload_messages = [msg.to_dict() for msg in messages]
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        payload = {
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = await self.client.post(
            self.api_url,
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"Azure OpenAI API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from Azure OpenAI models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Use full conversation history
            payload_messages = [msg.to_dict() for msg in messages]
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        payload = {
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
                error_msg = f"Azure OpenAI API error: {response.status_code}"
                raise Exception(error_msg)
            
            async for chunk in response.aiter_lines():
                if not chunk or chunk == "data: [DONE]":
                    continue
                
                if chunk.startswith("data: "):
                    json_str = chunk[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError) as e:
                        pass
