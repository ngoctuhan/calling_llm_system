from fastapi import WebSocket
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def stream_response_and_track(websocket: WebSocket, llm, prompt: str, logger=logger):
    """
    Stream response from LLM and track if streaming was successful.
    
    Args:
        websocket: The WebSocket connection to send chunks to
        llm: LLM provider instance with streaming capability
        prompt: The prompt to send to the LLM
        logger: Logger instance to use
        
    Returns:
        Tuple of (collected_answer, stream_success) where:
        - collected_answer: The complete generated text
        - stream_success: Boolean indicating if streaming was successful
    """
    collected_answer = ""
    stream_success = False
    
    try:
        # Try streaming first
        logger.info("Starting text stream generation")
        async for chunk in llm.generate_text_stream(prompt):
            if chunk:
                collected_answer += chunk
                await websocket.send_text(json.dumps({
                    "type": "stream_chunk",
                    "chunk": chunk
                }))
        stream_success = True
    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        # Fall back to non-streaming if streaming fails
        logger.info("Falling back to non-streaming generation")
        try:
            collected_answer = await llm.generate_text(prompt)
            # Send the full answer as one chunk for the client to display
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "chunk": collected_answer
            }))
            stream_success = True
        except Exception as fallback_error:
            logger.error(f"Fallback generation also failed: {str(fallback_error)}")
            collected_answer = "Tôi không thể xử lý câu hỏi của bạn lúc này. Vui lòng thử lại sau."
    
    # End the stream regardless of method used
    await websocket.send_text(json.dumps({
        "type": "stream_end"
    }))
    
    # Log completion of answer generation
    logger.info(f"Answer generation complete, stream success: {stream_success}")
    
    return collected_answer, stream_success 