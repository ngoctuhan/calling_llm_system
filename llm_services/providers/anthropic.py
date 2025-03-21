import asyncio
import json
from typing import List, Optional, Dict, Any, Generator, AsyncGenerator
from .provider import LLMProvider, Message
import httpx
from httpx import AsyncClient

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with function calling support"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        self.client = AsyncClient(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    async def generate(self, prompt: str, messages: Optional[List[Message]] = None, 
                      functions: Optional[List[Dict[str, Any]]] = None,
                      function_call: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """Generate text from Anthropic Claude models with function calling support"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_prompt = kwargs.get("system_prompt", "")
        
        if messages:
            # Extract system message if present
            system_messages = [msg for msg in messages if msg.role == "system"]
            if system_messages and not system_prompt:
                system_prompt = system_messages[0].content
                # Remove system message from conversation
                payload_messages = [msg.to_dict() for msg in messages if msg.role != "system"]
            else:
                payload_messages = [msg.to_dict() for msg in messages]
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        # Add function calling if provided
        if functions:
            tools = [{"type": "function", "function": func} for func in functions]
            payload["tools"] = tools
            
            if function_call:
                if function_call == "auto":
                    # Let the model decide which function to call
                    payload["tool_choice"] = "auto"
                else:
                    # Force the model to call a specific function
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": function_call}
                    }
        
        response = await self.client.post(
            self.api_url,
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        
        # Process the response to match OpenAI's format
        processed_response = {
            "id": result.get("id", ""),
            "model": result.get("model", self.model),
            "created": result.get("created_at", 0),
            "object": "chat.completion",
            "usage": result.get("usage", {}),
        }
        
        # Handle function calling responses
        content = result["content"]
        message = {"role": "assistant"}
        
        # Check if the response contains a tool call
        tool_calls = []
        for block in content:
            if block["type"] == "tool_use":
                tool_call = {
                    "id": block.get("id", f"call_{len(tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"])
                    }
                }
                tool_calls.append(tool_call)
            elif block["type"] == "text":
                message["content"] = block["text"]
        
        if tool_calls:
            message["tool_calls"] = tool_calls
            
        processed_response["choices"] = [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop" if not tool_calls else "tool_calls"
            }
        ]
            
        return processed_response
    
    async def generate_stream(self, prompt: str, messages: Optional[List[Message]] = None,
                             functions: Optional[List[Dict[str, Any]]] = None,
                             function_call: Optional[str] = None,
                             **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response from Anthropic Claude models with function calling support"""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_prompt = kwargs.get("system_prompt", "")
        
        if messages:
            # Extract system message if present
            system_messages = [msg for msg in messages if msg.role == "system"]
            if system_messages and not system_prompt:
                system_prompt = system_messages[0].content
                # Remove system message from conversation
                payload_messages = [msg.to_dict() for msg in messages if msg.role != "system"]
            else:
                payload_messages = [msg.to_dict() for msg in messages]
        else:
            # Use single prompt
            payload_messages = [{"role": "user", "content": prompt}]
            
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        # Add function calling if provided
        if functions:
            tools = [{"type": "function", "function": func} for func in functions]
            payload["tools"] = tools
            
            if function_call:
                if function_call == "auto":
                    # Let the model decide which function to call
                    payload["tool_choice"] = "auto"
                else:
                    # Force the model to call a specific function
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": function_call}
                    }
        
        message_id = None
        current_content = ""
        current_tool_calls = []
        
        async with self.client.stream(
            "POST",
            self.api_url,
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_msg = f"Anthropic API error: {response.status_code}"
                raise Exception(error_msg)
            
            async for chunk in response.aiter_lines():
                if not chunk or chunk.startswith("event:") or chunk == "data: {}":
                    continue
                
                if chunk.startswith("data: "):
                    json_str = chunk[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(json_str)
                        
                        # Set message_id from the first message
                        if not message_id and data.get("message", {}).get("id"):
                            message_id = data["message"]["id"]
                        
                        # Process content delta
                        if "delta" in data:
                            delta = data["delta"]
                            
                            # Process text content
                            if "text" in delta:
                                current_content += delta["text"]
                                
                            # Process tool use delta
                            if "tool_use" in delta:
                                tool_use = delta["tool_use"]
                                
                                # Find or create the tool call
                                tool_id = tool_use.get("id")
                                existing_tool = next((t for t in current_tool_calls if t["id"] == tool_id), None)
                                
                                if not existing_tool and "name" in tool_use:
                                    # Create new tool call
                                    existing_tool = {
                                        "id": tool_id or f"call_{len(current_tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": tool_use["name"],
                                            "arguments": ""
                                        }
                                    }
                                    current_tool_calls.append(existing_tool)
                                
                                # Update tool call arguments
                                if existing_tool and "input" in tool_use:
                                    # Merge with existing arguments
                                    try:
                                        current_args = json.loads(existing_tool["function"]["arguments"] or "{}")
                                    except json.JSONDecodeError:
                                        current_args = {}
                                        
                                    # Update with new input
                                    for key, value in tool_use["input"].items():
                                        current_args[key] = value
                                        
                                    existing_tool["function"]["arguments"] = json.dumps(current_args)
                            
                            # Construct streaming response in OpenAI format
                            stream_resp = {
                                "id": message_id or "stream_msg",
                                "model": self.model,
                                "created": data.get("created_at", 0),
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            
                            # Add text content if present
                            if "text" in delta:
                                stream_resp["choices"][0]["delta"] = {"content": delta["text"]}
                                
                            # Add tool calls if present and completed
                            if "tool_use" in delta and "name" in delta["tool_use"]:
                                stream_resp["choices"][0]["delta"] = {
                                    "tool_calls": [
                                        {
                                            "id": delta["tool_use"].get("id", f"call_{len(current_tool_calls)-1}"),
                                            "type": "function",
                                            "function": {
                                                "name": delta["tool_use"]["name"],
                                                "arguments": "{}"
                                            }
                                        }
                                    ]
                                }
                            elif "tool_use" in delta and "input" in delta["tool_use"]:
                                tool_id = delta["tool_use"].get("id")
                                tool_index = next((i for i, t in enumerate(current_tool_calls) if t["id"] == tool_id), 0)
                                
                                stream_resp["choices"][0]["delta"] = {
                                    "tool_calls": [
                                        {
                                            "index": tool_index,
                                            "function": {
                                                "arguments": json.dumps(delta["tool_use"]["input"])
                                            }
                                        }
                                    ]
                                }
                                
                            # Check if this is the final message
                            if data.get("type") == "message_stop":
                                stream_resp["choices"][0]["finish_reason"] = "stop" if not current_tool_calls else "tool_calls"
                                
                            yield stream_resp
                                
                    except (json.JSONDecodeError, KeyError) as e:
                        pass
                        
    async def handle_function_response(self, messages: List[Message], function_name: str, 
                              function_response: str, **kwargs) -> Dict[str, Any]:
        """Send function execution results back to the model"""
        # Prepare messages including the function response
        payload_messages = [msg.to_dict() for msg in messages]
        
        # Add function response as tool_result
        payload_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{len(payload_messages)}",
                "type": "function",
                "function": {"name": function_name}
            }]
        })
        
        # Add tool results message
        payload_messages.append({
            "role": "tool",
            "tool_call_id": f"call_{len(payload_messages)-1}",
            "content": function_response
        })
        
        # Make the API call
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        system_prompt = kwargs.get("system_prompt", "")
        
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        response = await self.client.post(
            self.api_url,
            json=payload,
        )
        
        if response.status_code != 200:
            error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        
        result = response.json()
        
        # Process the response to match OpenAI's format
        processed_response = {
            "id": result.get("id", ""),
            "model": result.get("model", self.model),
            "created": result.get("created_at", 0),
            "object": "chat.completion",
            "usage": result.get("usage", {}),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["content"][0]["text"] if result.get("content") else None
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        return processed_response
    
async def main():

    provider = AnthropicProvider(api_key, model="claude-3-opus-20240229")

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
    # functions = [
    #     {
    #         "name": "get_weather",
    #         "description": "Get the current weather in a location",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "location": {
    #                     "type": "string",
    #                     "description": "The city and state, e.g. San Francisco, CA"
    #                 },
    #                 "unit": {
    #                     "type": "string",
    #                     "enum": ["celsius", "fahrenheit"],
    #                     "description": "The temperature unit to use"
    #                 }
    #             },
    #             "required": ["location"]
    #         }
    #     }
    # ]

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
    # messages = [
    #     Message(role="system", content="You are a helpful assistant."),
    #     Message(role="user", content="What's the weather like in Boston?")
    # ]
    
    # function_result = await provider.generate(
    #     prompt="",
    #     messages=messages,
    #     functions=functions,
    #     function_call="auto"
    # )
    
    # if isinstance(function_result, dict) and "tool_calls" in function_result:
    #     # Mock weather data that would come from a real API call
    #     weather_data = {
    #         "location": "Boston, MA",
    #         "temperature": 72,
    #         "unit": "fahrenheit",
    #         "condition": "Partly cloudy"
    #     }
        
    #     # Get function name from tool call
    #     tool_call = function_result["tool_calls"][0]
    #     function_name = tool_call["function"]["name"]
        
    #     # Handle the function response
    #     final_response = await provider.handle_function_response(
    #         messages=messages,
    #         function_name=function_name,
    #         function_response=weather_data
    #     )
        
    #     print("Final response after function execution:", final_response)

if __name__ == "__main__":
    
    asyncio.run(main())