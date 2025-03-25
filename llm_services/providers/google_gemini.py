import json
from typing import List, Optional, Generator, AsyncGenerator
import httpx
from .provider import LLMProvider, Message
import logging
import re
import asyncio

# Set up logger
logger = logging.getLogger(__name__)

class GoogleGeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.stream_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    
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
            logger.error(error_msg)
            raise Exception(error_msg)
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt - alias for generate to support unified interface"""
        return await self.generate(prompt, **kwargs)
    
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from Google Gemini models"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        try:
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
            
            logger.info(f"Starting stream request to Gemini API: {self.stream_api_url}")
            
            # Use the non-streaming approach for now and emulate streaming while we debug
            logger.info("Using non-streaming approach with chunking for reliability")
            response = await self.client.post(
                f"{self.api_url}?key={self.api_key}",
                json=payload,
                timeout=120.0
            )
            
            if response.status_code != 200:
                error_msg = f"Google Gemini API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            if "candidates" in result and result["candidates"]:
                full_text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Artificially chunk the response to simulate streaming
                # This is more reliable than dealing with Gemini's streaming format
                chunks = []
                if len(full_text) < 50:
                    chunks = [full_text]
                else:
                    # Break by paragraphs first
                    paragraphs = full_text.split('\n\n')
                    for paragraph in paragraphs:
                        if len(paragraph) > 100:
                            # Further break longer paragraphs
                            sentences = re.split(r'([.!?] )', paragraph)
                            current_chunk = ""
                            for i in range(0, len(sentences), 2):
                                if i+1 < len(sentences):
                                    current_chunk += sentences[i] + sentences[i+1]
                                else:
                                    current_chunk += sentences[i]
                                
                                if len(current_chunk) >= 50 or i >= len(sentences)-2:
                                    chunks.append(current_chunk)
                                    current_chunk = ""
                            
                            if current_chunk:
                                chunks.append(current_chunk)
                        else:
                            chunks.append(paragraph)
                
                # Yield the chunks with a slight delay to simulate streaming
                for chunk in chunks:
                    yield chunk
                    await asyncio.sleep(0.1)  # Small delay to simulate streaming
            else:
                logger.error(f"Unexpected response format: {result}")
                yield "Error: Couldn't generate a response"
                        
        except Exception as e:
            logger.error(f"Error in Gemini stream: {str(e)}")
            # Fall back to non-streaming if streaming fails
            try:
                complete_response = await self.generate(prompt, messages, **kwargs)
                yield complete_response
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                yield "Tôi không thể xử lý câu hỏi của bạn lúc này. Vui lòng thử lại sau."
    
    async def generate_text_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream text generation - alias for generate_stream to support unified interface"""
        try:
            async for chunk in self.generate_stream(prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Error in generate_text_stream: {e}")
            # Fall back to non-streaming
            try:
                complete_response = await self.generate_text(prompt, **kwargs)
                yield complete_response
            except Exception as fallback_error:
                logger.error(f"Final fallback also failed: {str(fallback_error)}")
                yield "Tôi không thể xử lý câu hỏi của bạn lúc này. Vui lòng thử lại sau."
