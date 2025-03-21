import asyncio
import json
from typing import Generator, List, Optional, Dict, Any, Callable, Union
import httpx
from .provider import LLMProvider, Message

class OpenAIProvider(LLMProvider):
    """OpenAI API provider with function calling support"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, 
                      functions: Optional[List[Dict[str, Any]]] = None,
                      function_call: Optional[Union[str, Dict[str, str]]] = None,
                      **kwargs) -> Union[str, Dict[str, Any]]:
        """Generate text from OpenAI models with optional function calling"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Use full conversation history
            payload_messages = [msg.to_dict() for msg in messages]
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add function calling if provided
        if functions:
            payload["tools"] = [{"type": "function", "function": func} for func in functions]
            
        if function_call:
            if isinstance(function_call, str) and function_call == "auto":
                payload["tool_choice"] = "auto"
            else:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": function_call
                }
                
        response = await self.client.post(
            self.api_url,
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
            
        result = response.json()
        message = result["choices"][0]["message"]
        
        # Return either the content or the function call
        if "tool_calls" in message:
            # Return function call data
            return {
                "content": message.get("content", ""),
                "tool_calls": message["tool_calls"]
            }
        else:
            # Return just the content
            return message["content"]
            
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None,
                             functions: Optional[List[Dict[str, Any]]] = None,
                             function_call: Optional[Union[str, Dict[str, str]]] = None,
                             **kwargs) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """Generate streaming response from OpenAI models with function calling support"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        if messages:
            # Use full conversation history
            payload_messages = [msg.to_dict() for msg in messages]
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        # Add function calling if provided
        if functions:
            payload["tools"] = [{"type": "function", "function": func} for func in functions]
            
        if function_call:
            if isinstance(function_call, str) and function_call == "auto":
                payload["tool_choice"] = "auto"
            else:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": function_call
                }
        
        async with self.client.stream(
            "POST",
            self.api_url,
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_msg = f"OpenAI API error: {response.status_code}"
                raise Exception(error_msg)
                
            current_tool_calls = []
            
            async for chunk in response.aiter_lines():
                if not chunk or chunk == "data: [DONE]":
                    continue
                    
                if chunk.startswith("data: "):
                    json_str = chunk[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_str)
                        delta = data["choices"][0]["delta"]
                        
                        if "content" in delta and delta["content"]:
                            yield {"type": "content", "content": delta["content"]}
                            
                        if "tool_calls" in delta:
                            # Handle tool calls in stream
                            for tool_call_delta in delta["tool_calls"]:
                                tool_call_id = tool_call_delta.get("id")
                                
                                # Find or create tool call entry
                                tool_call = next((tc for tc in current_tool_calls if tc.get("id") == tool_call_id), None)
                                if not tool_call and "id" in tool_call_delta:
                                    tool_call = {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    }
                                    current_tool_calls.append(tool_call)
                                
                                if tool_call and "function" in tool_call_delta:
                                    func_delta = tool_call_delta["function"]
                                    
                                    if "name" in func_delta:
                                        tool_call["function"]["name"] += func_delta["name"]
                                        
                                    if "arguments" in func_delta:
                                        tool_call["function"]["arguments"] += func_delta["arguments"]
                                        
                                    yield {
                                        "type": "tool_call",
                                        "tool_call": tool_call
                                    }
                                    
                    except (json.JSONDecodeError, KeyError) as e:
                        pass

    async def handle_function_response(self, messages: List[Message], function_name: str, 
                                      function_response: Any, **kwargs) -> str:
        """
        Handle function execution response and continue the conversation
        
        Parameters:
            messages: The conversation history
            function_name: The name of the function that was called
            function_response: The result of the function execution
            
        Returns:
            The model's response after processing the function result
        """
        # Add function response to messages
        messages.append(
            Message(
                role="function",
                content=json.dumps(function_response),
                name=function_name
            )
        )
        
        # Continue conversation with function result included
        response = await self.generate(
            prompt="",  # Not needed as we're using messages
            messages=messages,
            **kwargs
        )
        
        return response

async def main():

    provider = OpenAIProvider(api_key="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", model="gpt-4o-mini")

    # Example 1: Simple text generation
    # response = await provider.generate(
    #     prompt="Explain quantum computing in simple terms"
    # )
    # print("Simple response:", response)

    # Example 2: Using conversation history
    # messages = [
    #     Message(role="system", content="You are a helpful assistant."),
    #     Message(role="user", content="What's the capital of France?"),
    #     Message(role="assistant", content="The capital of France is Paris."),
    #     Message(role="user", content="What's the population there?")
    # ]
    # response = await provider.generate(prompt="", messages=messages)
    # print("Conversation response:", response)

    # Example 3: Function calling
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    # function_response = await provider.generate(
    #     prompt="What's the weather like in New York?",
    #     functions=functions,
    #     function_call="auto"
    # )
    
    # print("Function call response:", function_response)

    # Example 4: Streaming response
    # response = provider.generate_stream(
    #     prompt="What is the grpc?",
    # )
    # await provider.print_stream(response)

    # Example 5: Function calling with streaming
    # stream_response = provider.generate_stream(
    #     prompt="What's the weather like in Seattle?",
    #     functions=functions,
    #     function_call="auto"
    # )
    # await provider.print_stream(stream_response)

    # Example 6: Handle function response
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What's the weather like in Boston?")
    ]
    
    function_result = await provider.generate(
        prompt="",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    
    if isinstance(function_result, dict) and "tool_calls" in function_result:
        # Mock weather data that would come from a real API call
        weather_data = {
            "location": "Boston, MA",
            "temperature": 72,
            "unit": "fahrenheit",
            "condition": "Partly cloudy"
        }
        
        # Get function name from tool call
        tool_call = function_result["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        
        # Handle the function response
        final_response = await provider.handle_function_response(
            messages=messages,
            function_name=function_name,
            function_response=weather_data
        )
        
        print("Final response after function execution:", final_response)

if __name__ == "__main__":
    
    asyncio.run(main())