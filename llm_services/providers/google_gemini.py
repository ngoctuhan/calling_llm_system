import json
from typing import List, Optional
import httpx
from .provider import LLMProvider, Message
from typing import Generator
class GoogleGeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> str:
        """Generate text from Google Gemini models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg.role == "system":
                    # For system message, add as a special user message
                    contents.append({
                        "role": "user",
                        "parts": [{"text": f"[System instruction] {msg.content}"}]
                    })
                elif msg.role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [{"text": msg.content}]
                    })
                elif msg.role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [{"text": msg.content}]
                    })
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            }
        else:
            # Use single prompt
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            }
        
        response = await self.client.post(
            f"{self.api_url}?key={self.api_key}",
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"Google Gemini API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from Google Gemini models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg.role == "system":
                    # For system message, add as a special user message
                    contents.append({
                        "role": "user",
                        "parts": [{"text": f"[System instruction] {msg.content}"}]
                    })
                elif msg.role == "user":
                    contents.append({
                        "role": "user",
                        "parts": [{"text": msg.content}]
                    })
                elif msg.role == "assistant":
                    contents.append({
                        "role": "model",
                        "parts": [{"text": msg.content}]
                    })
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            }
        else:
            # Use single prompt
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            }
        
        async with self.client.stream(
            "POST",
            f"{self.api_url}:streamGenerateContent?key={self.api_key}",
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_msg = f"Google Gemini API error: {response.status_code}"
                raise Exception(error_msg)
            
            async for chunk in response.aiter_lines():
                if not chunk:
                    continue
                
                try:
                    data = json.loads(chunk)
                    if "candidates" in data and data["candidates"][0]["content"]["parts"]:
                        yield data["candidates"][0]["content"]["parts"][0]["text"]
                except (json.JSONDecodeError, KeyError) as e:
                    pass
