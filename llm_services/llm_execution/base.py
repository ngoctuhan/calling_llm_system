from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, prompt: str, **kwargs) -> str:
        pass


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required: Optional[List[str]] = None

@dataclass
class LLMConfig:
    
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    top_k: float = 1
    endpoint: str = "default"
    
