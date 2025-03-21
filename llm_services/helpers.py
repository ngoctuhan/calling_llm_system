import sys
import asyncio
from typing import AsyncGenerator, Union, Dict, Any

async def print_stream(
    stream: Union[AsyncGenerator[str, None], AsyncGenerator[Dict[str, Any], None]], 
    end: str = "",
    flush: bool = True,
    delay: float = 0.0,  # Optional delay between chunks
    file=sys.stdout
) -> None:
    """
    Print content from a stream generator.
    
    Args:
        stream: AsyncGenerator that yields strings or dictionaries
        end: String to append after each chunk (default: "")
        flush: Whether to flush the output (default: True)
        delay: Optional delay between chunks in seconds (default: 0.0)
        file: Output file object (default: sys.stdout)
    """
    try:
        async for chunk in stream:
            # Handle different types of chunks
            if isinstance(chunk, str):
                # Direct string output
                print(chunk, end=end, flush=flush, file=file)
            elif isinstance(chunk, dict):
                # Handle dictionary output based on type
                if chunk.get("type") == "content":
                    print(chunk["content"], end=end, flush=flush, file=file)
                elif chunk.get("type") == "function_result":
                    print("\nFunction called:", chunk["function_called"])
                    print("Arguments:", chunk["arguments"])
                    print("Result:", chunk["result"])
                else:
                    # Default dictionary handling
                    print(chunk, end=end, flush=flush, file=file)
            
            if delay > 0:
                await asyncio.sleep(delay)
                
    except Exception as e:
        print(f"\nError during streaming: {str(e)}", file=sys.stderr)
    finally:
        # Add a final newline if end is not specified
        if not end:
            print(file=file)


def cache_function(func):
    """Simple cache decorator for functions"""
    cache = {}
    def wrapper(*args):
        cache_key = hash(f"{func.__name__}_{args}")
        if cache_key not in cache:
            cache[cache_key] = func(*args)
        return cache[cache_key]
    return wrapper
    