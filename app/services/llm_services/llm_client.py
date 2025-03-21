import logging
from typing import Dict, Any, List
import openai
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.OPENAI_API_KEY

def generate_response(processed_data: Dict[str, Any], conversation_id: int) -> str:
    """
    Generate a response using LLM based on processed data
    
    Args:
        processed_data: The processed user input data
        conversation_id: The ID of the current conversation
        
    Returns:
        str: The generated response
    """
    try:
        logger.info(f"Generating response for conversation {conversation_id}")
        
        # Extract the cleaned text from processed data
        user_text = processed_data.get("cleaned_text", "")
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": "You are a helpful call center assistant. Provide concise and accurate responses."},
            {"role": "user", "content": user_text}
        ]
        
        # Call the LLM to generate a response
        response = openai.ChatCompletion.create(
            model=settings.DEFAULT_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        
        # Extract the generated text from the response
        generated_text = response.choices[0].message.content
        
        logger.debug(f"Generated response: {generated_text}")
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Return a fallback response in case of errors
        return "I'm sorry, I'm having trouble generating a response right now. Please try again later."

def get_conversation_history(conversation_id: int) -> List[Dict[str, str]]:
    """
    Retrieve conversation history for context
    
    Args:
        conversation_id: The ID of the conversation
        
    Returns:
        List[Dict]: List of messages with role and content
    """
    # This is a placeholder - in a real implementation, this would retrieve
    # messages from the database and format them for the LLM
    # For now, returning an empty list as it will be built in the generate_response function
    return [] 