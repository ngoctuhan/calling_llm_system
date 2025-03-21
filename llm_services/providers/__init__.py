import asyncio
from typing import Dict, Generator, List, Literal, Optional
from .provider import Message, LLMProvider
from .llm_provider_factory import LLMProviderFactory

class Conversation:
    """Class to manage conversation history"""
    
    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[Message] = []
        if system_prompt:
            self.add_system_message(system_prompt)
    
    def add_system_message(self, content: str):
        """Add a system message to the conversation"""
        self.messages.append(Message(role="system", content=content))
        return self
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation"""
        self.messages.append(Message(role="user", content=content))
        return self
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation"""
        self.messages.append(Message(role="assistant", content=content))
        return self
    
    def add_message(self, role: Literal["system", "user", "assistant"], content: str):
        """Add a message with specified role to the conversation"""
        self.messages.append(Message(role=role, content=content))
        return self
    
    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation"""
        return self.messages.copy()
    
    def clear(self):
        """Clear all messages except system message"""
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        self.messages = system_messages

class LLMManager:
    """Manager class to handle multiple LLM providers and models"""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.conversations: Dict[str, Conversation] = {}
    
    def add_provider(self, name: str, provider: LLMProvider):
        """Add a provider with a custom name"""
        self.providers[name] = provider
    
    def add_provider_from_model(self, name: str, model: str, api_key: Optional[str] = None):
        """Add a provider based on model name"""
        provider = LLMProviderFactory.create_provider(model, api_key)
        self.providers[name] = provider
    
    def create_conversation(self, conversation_id: str, system_prompt: Optional[str] = None):
        """Create a new conversation"""
        self.conversations[conversation_id] = Conversation(system_prompt)
        return self.conversations[conversation_id]
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    async def generate(self, 
                      provider_name: str, 
                      prompt: Optional[str] = None, 
                      conversation_id: Optional[str] = None,
                      add_to_history: bool = True,
                      **kwargs) -> str:
        """
        Generate text using a specific provider
        
        Args:
            provider_name: Name of the provider to use
            prompt: The prompt to generate from (used if conversation_id is None or add_to_history is True)
            conversation_id: Optional conversation ID to use history
            add_to_history: Whether to add the prompt and response to conversation history
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text response
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        messages = None
        conversation = None
        
        if conversation_id:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation '{conversation_id}' not found")
            
            if prompt and add_to_history:
                conversation.add_user_message(prompt)
            
            messages = conversation.get_messages()
            
        if not prompt and not messages:
            raise ValueError("Either prompt or conversation_id with existing messages must be provided")
            
        response = await self.providers[provider_name].generate(
            prompt=prompt or "", 
            messages=messages,
            **kwargs
        )
        
        # Add response to conversation history if requested
        if conversation and add_to_history:
            conversation.add_assistant_message(response)
            
        return response
    
    async def generate_stream(self, 
                             provider_name: str, 
                             prompt: Optional[str] = None, 
                             conversation_id: Optional[str] = None,
                             add_to_history: bool = True,
                             **kwargs) -> Generator[str, None, None]:
        """
        Generate streaming response using a specific provider
        
        Args:
            provider_name: Name of the provider to use
            prompt: The prompt to generate from (used if conversation_id is None or add_to_history is True)
            conversation_id: Optional conversation ID to use history
            add_to_history: Whether to add the prompt and response to conversation history
            **kwargs: Additional arguments for generation
            
        Returns:
            Generator yielding response chunks
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        messages = None
        conversation = None
        full_response = ""
        
        if conversation_id:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation '{conversation_id}' not found")
            
            if prompt and add_to_history:
                conversation.add_user_message(prompt)
            
            messages

__name__ = ["Message", "LLMProvider"]

# Example usage
async def main():
    # Initialize with environment variables
    llm_manager = LLMManager()
    
    # Add different types of providers
    
    # Standard cloud providers (using environment variables for API keys)
    llm_manager.add_provider_from_model("gpt4", "gpt-4")
    llm_manager.add_provider_from_model("claude", "claude-3-opus-20240229")
    llm_manager.add_provider_from_model("cohere", "command")
    llm_manager.add_provider_from_model("gemini", "gemini-pro")
    
    # Ollama (local models)
    llm_manager.add_provider_from_model("llama3", "llama3", 
                                       provider="ollama", 
                                       base_url="http://localhost:11434")
    
    # Azure OpenAI
    llm_manager.add_provider_from_model("azure-gpt4", "gpt-4",
                                       azure=True,
                                       endpoint="https://your-resource.openai.azure.com",
                                       deployment_name="your-gpt4-deployment",
                                       api_key="YOUR_AZURE_API_KEY")
    
    # Create a conversation with system prompt
    conversation_id = "conversation1"
    llm_manager.create_conversation(
        conversation_id,
        system_prompt="You are a helpful AI assistant that specializes in explaining complex topics simply."
    )
    
    # Generate from a specific model using conversation history
    await llm_manager.generate(
        "claude", 
        "What is quantum computing?", 
        conversation_id=conversation_id
    )
    
    # Continue the conversation with follow-up
    response = await llm_manager.generate(
        "claude", 
        "How is it different from classical computing?", 
        conversation_id=conversation_id
    )
    print(f"Claude response: {response}\n")
    
    # Streaming response with conversation history
    print("Streaming response from GPT-4:")
    full_response = ""
    async for chunk in llm_manager.generate_stream(
        "gpt4", 
        "Give me a concrete example in everyday terms",
        conversation_id=conversation_id
    ):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n")
    
    # Try local model (Ollama)
    print("Response from local Llama3 model:")
    try:
        local_response = await llm_manager.generate(
            "llama3", 
            "Summarize what you know about quantum computing"
        )
        print(local_response)
    except Exception as e:
        print(f"Ollama error (is it running?): {e}")
    print("-" * 50)
    
    # Multi-provider comparison
    responses = await llm_manager.generate_multi(
        "What are three practical applications of quantum computing?",
        providers=["gpt4", "claude", "gemini"],
        temperature=0.5,
        max_tokens=300,
    )
    
    for provider, response in responses.items():
        print(f"{provider.upper()} RESPONSE:")
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())