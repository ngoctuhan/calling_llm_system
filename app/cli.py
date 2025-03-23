#!/usr/bin/env python3
"""
Command-line interface for the Call Center Information System.
This provides a text-based interface to interact with the system.
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from dotenv import load_dotenv
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import necessary components
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType
from retrieval_engine.text2sql import Text2SQL
from llm_services import LLMProviderFactory

class ChatCLI:
    """Command-line interface for the Call Center Information System."""
    
    def __init__(self):
        # Initialize standard LLM for non-thinking tasks
        self.llm = LLMProviderFactory.create_provider(
            model="gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Thinking model client using the newer Google Genai SDK
        self.thinking_client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"), 
            http_options={'api_version':'v1alpha'}
        )
        
        # Initialize vector RAG
        self.vector_rag = RetrievalFactory.create_rag(
            rag_type=RAGType.VECTOR,
            llm_config={
                "model_name": "gemini-2.0-flash",
                "api_key": os.getenv("GOOGLE_API_KEY")
            },
            embedding_config={
                "provider": "google",  
                "model_name": "text-embedding-004",  
                "api_key": os.getenv("GOOGLE_API_KEY")
            },
            retrieval_config={
                "index_name": "callcenter", 
                "similarity_top_k": 10
            }
        )
        
        # Initialize Graph RAG
        self.graph_rag = RetrievalFactory.create_rag(
            rag_type=RAGType.GRAPH_V2,
            llm_config={
                "model_name": "gemini-2.0-flash",
                "api_key": os.getenv("GOOGLE_API_KEY")
            },
            embedding_config={
                "provider": "google",  
                "model_name": "text-embedding-004",  
                "api_key": os.getenv("GOOGLE_API_KEY")
            },
            retrieval_config={
                "username": "neo4j",
                "password": "password",
                "url": "neo4j://localhost:7687",
                "similarity_top_k": 10
            }
        )
        
        # Initialize Hybrid RAG (will use both vector and graph together)
        self.hybrid_rag = RetrievalFactory.create_rag(
            rag_type=RAGType.HYBRID,
            llm_config={
                "model_name": "gemini-2.0-flash",
                "api_key": os.getenv("GOOGLE_API_KEY")
            },
            retrieval_config={
                "vector_rag": self.vector_rag,
                "graph_rag": self.graph_rag,
                "vector_weight": 0.5,
                "graph_weight": 0.5,
                "combination_strategy": "weighted",  # 'weighted', 'ensemble', hoặc 'cascade'
                "deduplicate": True,
                "max_workers": 2
            }
        )
        
        # Initialize Text2SQL
        self.text2sql = Text2SQL(
            db_type="postgres",
            llm_provider=self.llm,
            max_retries=2,
            batch_size=3,
            max_concurrency=5
        )
        
        # Define function declarations for Gemini function calling
        self.tool_definitions = [{
            "functionDeclarations": [
                {
                    "name": "vector_rag",
                    "description": "Retrieves information from a vector database. Use for general knowledge questions and finding semantically similar information.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The user's question"
                            },
                            "top_k": {
                                "type": "INTEGER", 
                                "description": "Number of results to return"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "graph_rag",
                    "description": "Retrieves information from a knowledge graph. Use for questions about relationships between entities (people, places, events).",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The user's question"
                            },
                            "top_k": {
                                "type": "INTEGER",
                                "description": "Number of results to return"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "hybrid_rag",
                    "description": "Combines Graph RAG and Vector RAG. Use for complex or ambiguous questions requiring both relationship and semantic understanding.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The user's question"
                            },
                            "top_k": {
                                "type": "INTEGER",
                                "description": "Number of results to return"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "text2sql",
                    "description": "Translates a natural language question into an SQL query and executes it against a database. Use ONLY for questions explicitly requiring data from the database.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The user's question about database data"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "direct_answer",
                    "description": "Provides a direct answer based on general knowledge. Use when you confidently know the answer without using any tools.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The user's question"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "ask_clarification",
                    "description": "Asks the user for clarification. Use when the question is ambiguous, unclear, or you are unsure which tool is best.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "question": {
                                "type": "STRING",
                                "description": "The user's original question"
                            },
                            "clarification_request": {
                                "type": "STRING",
                                "description": "What specific information you need from the user to answer their question"
                            }
                        },
                        "required": ["question", "clarification_request"]
                    }
                }
            ]
        }]
        
        logger.info("Chat CLI initialized successfully")
    
    async def process_question(self, question, mode="auto"):
        """Process a user question and return the answer."""
        
        print(f"\n[THINKING] Processing question: {question}")
        print(f"[THINKING] Using mode: {mode}")
        
        answer = ""
        sources = []
        sql_query = None
        sql_data = None
        
        # Use function calling with Thinking model when in auto mode
        if mode == "auto":
            print("[THINKING] Using Gemini Flash Thinking model to determine the best tool...")
            
            # System prompt to guide the model
            system_prompt = """
            You are a helpful and intelligent question-answering assistant. Your primary goal is to understand the user's intent 
            and provide the most relevant information. Answer in the same language as the question (Vietnamese or English).
            
            For database questions, look for explicit references to database entities. For graph questions, look for entity
            relationships. For complex questions, use hybrid_rag.
            
            If you're unsure or the question is ambiguous, ask for clarification.
            """
            
            try:
                # Call the Thinking model without function calling first to get thinking process
                thinking_response = await self.thinking_client.aio.models.generate_content(
                    model='gemini-2.0-flash-thinking-exp',
                    contents=f"Think carefully about how to answer this question: {question}",
                )
                
                print(f"[THINKING PROCESS] {thinking_response.text}")
                
                # Call the model with function declarations for actual execution
                function_response = await self.thinking_client.aio.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[
                        {"role": "system", "parts": [{"text": system_prompt}]},
                        {"role": "user", "parts": [{"text": question}]}
                    ],
                    generation_config={"temperature": 0.1},
                    tools=self.tool_definitions
                )
                
                # Process function call
                if hasattr(function_response.candidates[0].content.parts[0], 'function_call'):
                    function_call = function_response.candidates[0].content.parts[0].function_call
                    function_name = function_call.name
                    function_args = json.loads(function_call.args)
                    
                    print(f"[THINKING] Function call: {function_name}")
                    print(f"[THINKING] Arguments: {function_args}")
                    
                    # Set mode based on function call
                    if function_name == "vector_rag":
                        mode = "vector_rag"
                        top_k = function_args.get("top_k", 10)
                    elif function_name == "graph_rag":
                        mode = "graph_rag"
                        top_k = function_args.get("top_k", 10)
                    elif function_name == "hybrid_rag":
                        mode = "hybrid_rag"
                        top_k = function_args.get("top_k", 10)
                    elif function_name == "text2sql":
                        mode = "database"
                    elif function_name == "direct_answer":
                        # Use the thinking result to generate a direct answer
                        thinking_context = thinking_response.text
                        direct_answer_prompt = f"""
                        Question: {question}
                        
                        Your thinking process:
                        {thinking_context}
                        
                        Based on your thinking, provide a clear and concise answer.
                        """
                        
                        direct_answer_response = await self.thinking_client.aio.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=direct_answer_prompt
                        )
                        
                        return {
                            "answer": direct_answer_response.text,
                            "sources": [],
                            "sql": None,
                            "data": None,
                            "thinking": thinking_response.text
                        }
                    elif function_name == "ask_clarification":
                        # Return clarification request
                        clarification = function_args.get("clarification_request", "")
                        return {
                            "answer": f"I need more information to answer your question properly. {clarification}",
                            "sources": [],
                            "sql": None,
                            "data": None,
                            "requires_clarification": True,
                            "thinking": thinking_response.text
                        }
                else:
                    # Model provided a direct response without function calling
                    return {
                        "answer": function_response.candidates[0].content.parts[0].text,
                        "sources": [],
                        "sql": None,
                        "data": None,
                        "thinking": thinking_response.text
                    }
            except Exception as e:
                logger.error(f"Error with Thinking model: {str(e)}")
                print(f"[ERROR] Error using Thinking model: {str(e)}")
                print("[THINKING] Falling back to standard LLM...")
                
                # Fall back to standard LLM
                return await self._process_with_standard_llm(question, mode)
        
        # Process based on the selected mode
        if mode == "vector_rag":
            print("[THINKING] Getting information from vector knowledge base...")
            kb_results = self.vector_rag.process(question, top_k=5)
            sources = kb_results.get("sources", [])
            print(f"[THINKING] Found {len(sources)} relevant documents")
            answer = kb_results.get("answer", "No answer found in knowledge base.")
        
        elif mode == "graph_rag":
            print("[THINKING] Getting information from graph knowledge base...")
            kb_results = self.graph_rag.process(question, top_k=5)
            sources = kb_results.get("sources", [])
            print(f"[THINKING] Found {len(sources)} relevant entities and relationships")
            answer = kb_results.get("answer", "No answer found in knowledge graph.")
        
        elif mode == "hybrid_rag":
            print("[THINKING] Getting information from hybrid knowledge system...")
            kb_results = self.hybrid_rag.process(question, top_k=5)
            sources = kb_results.get("sources", [])
            print(f"[THINKING] Found {len(sources)} relevant information items")
            answer = kb_results.get("answer", "No answer found in knowledge system.")
        
        elif mode == "database":
            print("[THINKING] Converting question to SQL query...")
            sql_result = await self.text2sql.process_query(question)
            
            if sql_result["success"]:
                sql_query = sql_result["sql"]
                sql_data = sql_result["data"]
                print(f"[THINKING] Generated SQL: {sql_query}")
                
                if sql_data:
                    data_context = f"SQL query: {sql_query}\n\nData results: {sql_data}"
                    # Use thinking model for result interpretation
                    try:
                        answer_response = await self.thinking_client.aio.models.generate_content(
                            model='gemini-2.0-flash-thinking-exp',
                            contents=f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{question}'\n\n{data_context}"
                        )
                        answer = answer_response.text
                    except Exception:
                        # Fallback to standard LLM
                        answer_response = await self.llm.generate_text(
                            f"Based on the following SQL query and results, please provide a clear answer to the user's question: '{question}'\n\n{data_context}"
                        )
                        answer = answer_response
                else:
                    answer = "No data found in database for your question."
            else:
                print(f"[THINKING] Error generating SQL: {sql_result.get('error', 'Unknown error')}")
                answer = "Sorry, I couldn't convert your question to a database query."
        
        elif mode == "combined":
            # Use all available tools and combine results
            print("[THINKING] Using all available tools and combining results...")
            
            # Get vector RAG results
            vector_results = self.vector_rag.process(question, top_k=3)
            vector_sources = vector_results.get("sources", [])
            
            # Get graph RAG results
            graph_results = self.graph_rag.process(question, top_k=3)
            graph_sources = graph_results.get("sources", [])
            
            # Try database query
            sql_result = await self.text2sql.process_query(question)
            if sql_result["success"]:
                sql_query = sql_result["sql"]
                sql_data = sql_result["data"]
            
            # Combine sources
            sources = vector_sources + graph_sources
            
            # Prepare context for LLM
            vector_context = json.dumps(vector_sources, indent=2) if vector_sources else "No vector knowledge base results."
            graph_context = json.dumps(graph_sources, indent=2) if graph_sources else "No graph knowledge base results."
            db_context = f"SQL query: {sql_query}\n\nData results: {json.dumps(sql_data, indent=2)}" if sql_query else "No database results."
            
            combined_prompt = f"""
            The user asked: '{question}'
            
            Vector knowledge base information:
            {vector_context}
            
            Graph knowledge base information:
            {graph_context}
            
            Database information:
            {db_context}
            
            Based on all available information, provide a comprehensive and accurate answer to the user's question.
            If the information comes from a specific source, please indicate this briefly.
            Answer in the same language as the question (Vietnamese or English).
            """
            
            # Use thinking model for combined results
            try:
                answer_response = await self.thinking_client.aio.models.generate_content(
                    model='gemini-2.0-flash-thinking-exp',
                    contents=combined_prompt
                )
                answer = answer_response.text
            except Exception:
                # Fallback to standard LLM
                answer_response = await self.llm.generate_text(combined_prompt)
                answer = answer_response
        
        print("[THINKING] Processing complete.")
        
        # Return results
        return {
            "answer": answer,
            "sources": sources,
            "sql": sql_query,
            "data": sql_data
        }
    
    async def _process_with_standard_llm(self, question, mode="auto"):
        """Process question with standard LLM when Thinking model fails"""
        
        # Use function calling when in auto mode
        if mode == "auto":
            print("[THINKING] Using Gemini function calling to determine the best tool...")
            
            # System prompt to guide the model
            system_prompt = """
            You are a helpful and intelligent question-answering assistant. Your primary goal is to understand the user's intent 
            and provide the most relevant information. Answer in the same language as the question (Vietnamese or English).
            
            For database questions, look for explicit references to database entities. For graph questions, look for entity
            relationships. For complex questions, use hybrid_rag.
            
            If you're unsure or the question is ambiguous, ask for clarification.
            """
            
            # Call the model with function declarations
            response = await self.llm.generate_content(
                contents=[
                    {"role": "system", "parts": [{"text": system_prompt}]},
                    {"role": "user", "parts": [{"text": question}]}
                ],
                generation_config={"temperature": 0.1},
                tools=self.tool_definitions
            )
            
            # Process function call
            if hasattr(response.candidates[0].content.parts[0], 'function_call'):
                function_call = response.candidates[0].content.parts[0].function_call
                function_name = function_call.name
                function_args = json.loads(function_call.args)
                
                print(f"[THINKING] Function call: {function_name}")
                print(f"[THINKING] Arguments: {function_args}")
                
                # Set mode based on function call
                if function_name == "vector_rag":
                    mode = "vector_rag"
                elif function_name == "graph_rag":
                    mode = "graph_rag"
                elif function_name == "hybrid_rag":
                    mode = "hybrid_rag"
                elif function_name == "text2sql":
                    mode = "database"
                elif function_name == "direct_answer":
                    # Return direct answer
                    direct_answer_prompt = f"""
                    Question: {question}
                    Please provide a direct answer based on your general knowledge.
                    """
                    answer = await self.llm.generate_text(direct_answer_prompt)
                    return {
                        "answer": answer,
                        "sources": [],
                        "sql": None,
                        "data": None
                    }
                elif function_name == "ask_clarification":
                    # Return clarification request
                    clarification = function_args.get("clarification_request", "")
                    return {
                        "answer": f"I need more information to answer your question properly. {clarification}",
                        "sources": [],
                        "sql": None,
                        "data": None,
                        "requires_clarification": True
                    }
            else:
                # Model provided a direct response without function calling
                return {
                    "answer": response.candidates[0].content.parts[0].text,
                    "sources": [],
                    "sql": None,
                    "data": None
                }
            
        # Continue with the rest of processing based on mode
        return await self.process_question(question, mode)
    
    async def interactive_mode(self):
        """Run the chat interface in interactive mode."""
        
        print("=================================================")
        print("Welcome to the Call Center Information System CLI")
        print("=================================================")
        print("Type 'exit', 'quit', or Ctrl+C to exit")
        print("Type 'mode <auto|vector_rag|graph_rag|hybrid_rag|database|combined>' to change the mode")
        print("Current mode: auto")
        print("=================================================")
        
        mode = "auto"
        
        # Create a chat session with the Thinking model
        try:
            # The proper way to create a chat session with the Google Genai SDK
            thinking_chat = self.thinking_client.aio.chats.create(
                model='gemini-2.0-flash-thinking-exp'
            )
            print("[SYSTEM] Successfully initialized Thinking chat session")
        except Exception as e:
            logger.error(f"Error creating Thinking chat session: {str(e)}")
            print(f"[SYSTEM] Error initializing Thinking chat: {str(e)}")
            thinking_chat = None
        
        conversation_history = []
        
        try:
            while True:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ["exit", "quit", "q"]:
                    break
                
                if question.lower().startswith("mode "):
                    mode_arg = question.lower().split("mode ")[1].strip()
                    if mode_arg in ["auto", "vector_rag", "graph_rag", "hybrid_rag", "database", "combined"]:
                        mode = mode_arg
                        print(f"Mode changed to: {mode}")
                    else:
                        print(f"Invalid mode: {mode_arg}. Available modes: auto, vector_rag, graph_rag, hybrid_rag, database, combined")
                    continue
                
                if not question:
                    continue
                
                # Use Thinking chat for multi-turn conversations if available
                if thinking_chat and mode == "auto":
                    try:
                        # Send message to thinking chat - await the send_message function
                        thinking_response = await thinking_chat.send_message(question)
                        
                        # Get thinking process
                        print("\n[THINKING PROCESS]")
                        print(thinking_response.text)
                        
                        # Process with function calling for tool selection
                        result = await self.process_question(question, mode)
                        
                    except Exception as e:
                        logger.error(f"Error in Thinking chat: {str(e)}")
                        print(f"[ERROR] Error with Thinking chat: {str(e)}")
                        
                        # Fallback to standard processing
                        result = await self._process_with_standard_llm(question, mode)
                else:
                    # Standard processing without Thinking chat
                    result = await self.process_question(question, mode)
                
                # Add question to conversation history
                conversation_history.append({"role": "user", "parts": [{"text": question}]})
                
                # Add answer to conversation history
                conversation_history.append({"role": "assistant", "parts": [{"text": result["answer"]}]})
                
                # Keep conversation history manageable (last 20 turns)
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
                print("\n" + "=" * 50)
                print(f"ANSWER: {result['answer']}")
                
                if result.get("requires_clarification", False):
                    print("\nPlease provide more information to help me answer your question.")
                
                if result.get("thinking", None):
                    print("\nTHINKING PROCESS:")
                    print(result["thinking"])
                
                if result["sources"]:
                    print("\nSOURCES:")
                    for i, source in enumerate(result["sources"]):
                        print(f"{i+1}. {source.get('title', 'Source')} - {source.get('content', '')[:100]}...")
                
                if result["sql"]:
                    print("\nSQL QUERY:")
                    print(result["sql"])
                    
                    if result["data"]:
                        print("\nQUERY RESULTS:")
                        if len(result["data"]) <= 5:
                            for row in result["data"]:
                                print(json.dumps(row, indent=2))
                        else:
                            print(f"{len(result['data'])} rows returned. First 3 rows:")
                            for row in result["data"][:3]:
                                print(json.dumps(row, indent=2))
                
                print("=" * 50)
        
        except KeyboardInterrupt:
            print("\nExiting...")
        
        finally:
            # Clean up resources
            await self.close()
    
    async def close(self):
        """Clean up resources."""
        try:
            # Close database connection
            self.text2sql.db_connector.disconnect()
            # Close RAG systems if they have close methods
            for rag_system in [self.vector_rag, self.graph_rag, self.hybrid_rag]:
                if hasattr(rag_system, 'close'):
                    await rag_system.close()
        except Exception as e:
            logger.error(f"Error closing resources: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="Call Center Information System CLI")
    parser.add_argument("--mode", choices=["auto", "vector_rag", "graph_rag", "hybrid_rag", "database", "combined"], 
                        default="auto", help="Mode of operation")
    parser.add_argument("--question", type=str, help="Question to ask (non-interactive mode)")
    
    args = parser.parse_args()
    
    cli = ChatCLI()
    
    if args.question:
        # Non-interactive mode - process a single question
        result = await cli.process_question(args.question, args.mode)
        
        print("\n" + "=" * 50)
        print(f"ANSWER: {result['answer']}")
        
        if result.get("thinking", None):
            print("\nTHINKING PROCESS:")
            print(result["thinking"])
        
        if result["sources"]:
            print("\nSOURCES:")
            for i, source in enumerate(result["sources"]):
                print(f"{i+1}. {source.get('title', 'Source')} - {source.get('content', '')[:100]}...")
        
        if result["sql"]:
            print("\nSQL QUERY:")
            print(result["sql"])
            
            if result["data"]:
                print("\nQUERY RESULTS:")
                if len(result["data"]) <= 5:
                    for row in result["data"]:
                        print(json.dumps(row, indent=2))
                else:
                    print(f"{len(result['data'])} rows returned. First 3 rows:")
                    for row in result["data"][:3]:
                        print(json.dumps(row, indent=2))
        
        print("=" * 50)
        
        # Close resources
        await cli.close()
    else:
        # Interactive mode
        await cli.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main()) 