from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import json
from abc import ABC, abstractmethod
from .base import BaseExecutor
from ..prompts.loader import PromptLoader

class BaseExecutorImpl(BaseExecutor, ABC):
    """Abstract base class for executors implementing common functionality"""
    def __init__(self, model: Any, prompt_loader: PromptLoader):
        self.model = model
        self.prompt_loader = prompt_loader

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Union[str, Dict[str, Any], AsyncGenerator]:
        """Execute the executor's main functionality"""
        pass

    async def _create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Create a chat completion with common parameters"""
        return await self.model.chat.completions.create(
            model=kwargs.get('model', 'gpt-4o-mini'),
            messages=messages,
            temperature=kwargs.get('temperature', 0.7),
            stream=stream
        )

class ChatExecutor(BaseExecutorImpl):
    """Manages chat interactions with conversation history"""
    def __init__(self, model: Any, prompt_loader: PromptLoader):
        super().__init__(model, prompt_loader)
        self._conversation_history: List[Dict[str, str]] = []
        self._system_prompt: Optional[str] = None

    def set_system_prompt(self, template_name: str, **template_vars) -> None:
        """Set system prompt from template"""
        self._system_prompt = self.prompt_loader.render_prompt(template_name, **template_vars)
        self._conversation_history = [{"role": "system", "content": self._system_prompt}]

    async def execute(
        self, 
        user_message: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Execute chat with history context"""
        if not self._system_prompt:
            raise ValueError("System prompt must be set before chat execution")

        self._conversation_history.append({"role": "user", "content": user_message})
        
        if not stream:
            return await self._execute_non_stream(**kwargs)
        return await self._execute_stream(**kwargs)

    async def _execute_non_stream(self, **kwargs) -> str:
        """Handle non-streaming chat execution"""
        response = await self._create_completion(self._conversation_history, **kwargs)
        assistant_message = response.choices[0].message.content
        self._add_to_history("assistant", assistant_message)
        return assistant_message

    async def _execute_stream(self, **kwargs) -> AsyncGenerator[str, None]:
        """Handle streaming chat execution"""
        stream_response = await self._create_completion(
            self._conversation_history,
            stream=True,
            **kwargs
        )

        async def stream_generator():
            full_response = []
            async for chunk in stream_response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    yield content
            
            self._add_to_history("assistant", "".join(full_response))

        return stream_generator()

    def _add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history"""
        self._conversation_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear chat history but keep system prompt"""
        self._conversation_history = (
            [{"role": "system", "content": self._system_prompt}]
            if self._system_prompt
            else []
        )

class OneShotExecutor(BaseExecutorImpl):
    """Handles single prompt executions without context"""
    async def execute(
        self,
        template_name: str,
        template_vars: Dict[str, Any],
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Execute single prompt without context"""
        prompt = self.prompt_loader.render_prompt(template_name, **template_vars)
        messages = [{"role": "user", "content": prompt}]

        if not stream:
            response = await self._create_completion(messages, **kwargs)
            return response.choices[0].message.content

        stream_response = await self._create_completion(messages, stream=True, **kwargs)
        return self._create_stream_generator(stream_response)

    @staticmethod
    async def _create_stream_generator(stream_response: Any) -> AsyncGenerator[str, None]:
        """Create generator for streaming response"""
        async for chunk in stream_response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class FunctionCallExecutor(BaseExecutorImpl):
    """Manages function calling capabilities"""
    def __init__(self, model: Any, prompt_loader: PromptLoader):
        super().__init__(model, prompt_loader)
        self.available_functions: Dict[str, callable] = {}
        self.function_definitions: List[Dict[str, Any]] = []

    def register_function(
        self,
        func: callable,
        name: Optional[str] = None,
        description: str = "",
        parameters: Dict[str, Any] = None,
        required: List[str] = None
    ):
        """Register a function that can be called by the model"""
        func_name = name or func.__name__
        
        # Create function definition for OpenAI
        function_def = {
            "name": func_name,
            "description": description or func.__doc__ or "",
            "parameters": parameters or {}
        }
        if required:
            function_def["parameters"]["required"] = required

        self.available_functions[func_name] = func
        self.function_definitions.append(function_def)

    async def execute(
        self,
        template_name: str,
        template_vars: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute prompt with function calling capability"""
        # Render prompt
        prompt = self.prompt_loader.render_prompt(template_name, **template_vars)

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        try:
            # Call OpenAI API with function definitions
            response = await self.model.chat.completions.create(
                model=kwargs.get('model', 'gpt-4o-mini'),
                messages=messages,
                functions=self.function_definitions,
                function_call="auto",
                temperature=kwargs.get('temperature', 0.7)
            )

            message = response.choices[0].message

            # Check if function call was made
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)

                # Get the registered function
                if function_name not in self.available_functions:
                    raise ValueError(f"Function {function_name} not found")

                function = self.available_functions[function_name]

                # Execute the function
                result = await function(**function_args)

                return {
                    "function_called": function_name,
                    "arguments": function_args,
                    "result": result
                }
            else:
                # No function was called, return normal response
                return {
                    "function_called": None,
                    "content": message.content
                }

        except Exception as e:
            return {
                "error": str(e),
                "function_called": None,
                "content": None
            }