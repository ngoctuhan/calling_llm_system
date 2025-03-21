import os
import json
import httpx
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Optional, List, Union, Tuple, Literal
from dataclasses import dataclass

@dataclass
class Message:
    """Class representing a message in a conversation"""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None

    def to_dict(self):
        """Converts the Message object to a dictionary."""
        if self.role != "function":
            return {"role": self.role, "content": self.content}
        return {
                "role": self.role, 
                "content": self.content,
                "name": self.name 
            }
    
    
class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> str:
        """
        Generate text from prompt or message history
        
        Args:
            prompt: The prompt to generate from (used if messages is None)
            messages: Optional list of messages representing conversation history
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None, **kwargs) -> Generator[str, None, None]:
        """
        Generate streaming response from prompt or message history
        
        Args:
            prompt: The prompt to generate from (used if messages is None)
            messages: Optional list of messages representing conversation history
            **kwargs: Additional arguments for generation
            
        Returns:
            Generator yielding response chunks
        """
        pass
    
    @classmethod
    async def print_stream(cls, response: Generator) -> str:
        """Print streaming response and collect the full response"""
        full_response = ""
        async for chunk in response:
            print(chunk, end="", flush=True)
            if 'content' in chunk:
                full_response += chunk
        print()
        return full_response


