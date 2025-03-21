import os
from typing import Optional
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .cohere import CohereProvider 
from .google_gemini import GoogleGeminiProvider 
from .mistral import MistralProvider 
from .provider import LLMProvider

class LLMProviderFactory:
    """Factory class to create appropriate LLM provider based on model name"""
    
    _PROVIDER_MAP = {
        "gpt-3.5-turbo": OpenAIProvider,
        "gpt-4": OpenAIProvider,
        "gpt-4-turbo": OpenAIProvider,
        "gpt-4o": OpenAIProvider,
        "gpt-4o-mini": OpenAIProvider,
        "claude-3-opus": AnthropicProvider,
        "claude-3-sonnet": AnthropicProvider,
        "claude-3-haiku": AnthropicProvider,
        "claude-3.5-sonnet": AnthropicProvider,
        "command": CohereProvider,
        "command-r": CohereProvider,
        "command-light": CohereProvider,
        "gemini-pro": GoogleGeminiProvider,
        "gemini-ultra": GoogleGeminiProvider,
        "gemini-2.0-flash": GoogleGeminiProvider,
        "mistral-small": MistralProvider,
        "mistral-medium": MistralProvider,
        "mistral-large": MistralProvider,
    }
    
    @classmethod
    def create_provider(cls, model: str, api_key: Optional[str] = None) -> LLMProvider:
        """Create appropriate LLM provider based on model name"""
        for prefix, provider_cls in cls._PROVIDER_MAP.items():
            if model.startswith(prefix):
                # Try to get API key from environment if not provided
                if not api_key:
                    api_key_env_map = {
                        OpenAIProvider: "OPENAI_API_KEY",
                        AnthropicProvider: "ANTHROPIC_API_KEY",
                        CohereProvider: "COHERE_API_KEY",
                        GoogleGeminiProvider: "GOOGLE_API_KEY",
                        MistralProvider: "MISTRAL_API_KEY",
                    }
                    env_key = api_key_env_map.get(provider_cls)
                    if env_key and env_key in os.environ:
                        api_key = os.environ[env_key]
                    else:
                        raise ValueError(f"API key not provided and not found in environment for model {model}")
                
                return provider_cls(api_key=api_key, model=model)
        
        raise ValueError(f"Unknown model: {model}. Please use a supported model.")