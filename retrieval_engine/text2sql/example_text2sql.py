"""
Example usage of the Text2SQL system with an e-commerce database.
"""

import os
import logging
import json
import asyncio
import time
from dotenv import load_dotenv
from retrieval_engine.text2sql import Text2SQL
from llm_services import LLMProviderFactory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

async def main():
    """Run the Text2SQL example with the e-commerce database"""
    
    llm = LLMProviderFactory.create_provider(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Initialize the Text2SQL system
    try:
        text2sql = Text2SQL(
            db_type="postgres",
            llm_provider=llm,
            max_retries=2,
            batch_size=3,
            max_concurrency=5
        )
        logging.info("Text2SQL system initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Text2SQL: {str(e)}")
        return
    
    # Sample queries for the e-commerce database
    sample_queries = [
        # Basic queries
        "Show me all products in the Electronics category",
        "List the top 5 most expensive products",
        "How many customers do we have in each region?",
        
        # # Join queries
        "Show me all orders made by customer John Doe",
        "List products that have been ordered more than 3 times",
        "What are the total sales for each product category?",
        
        # # Analytical queries
        "Show me the average order value by region",
        "Which payment method is most commonly used?",
        "Find products with stock quantity less than 50",
        
        # Complex queries
        "Who are our top 3 customers by total purchase amount?",
        "What is the percentage of orders that have been delivered vs pending?"
    ]
    
    # Process each query individually
    for i, query in enumerate(sample_queries):
        logging.info(f"Processing query {i+1}/{len(sample_queries)}: {query}")
        
        # Process the query
        result = await text2sql.process_query(query)
        
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("-"*80)
        
        if result["success"]:
            print(f"SQL: {result['sql']}")
            print("-"*80)
            print("RESULTS:")
            
            if result["data"]:
                # Pretty print for small result sets
                if len(result["data"]) <= 5:
                    for row in result["data"]:
                        print(json.dumps(row, indent=2))
                else:
                    print(f"{len(result['data'])} rows returned")
                    print("First 3 rows:")
                    for row in result["data"][:3]:
                        print(json.dumps(row, indent=2))
            else:
                print("No data returned")
        else:
            print(f"ERROR: {result['error']}")
            
        print(f"Execution Time: {result['execution_time']:.2f} seconds")
        print(f"Tables Used: {', '.join(result['tables_used'])}")
        print(f"Retries: {result['retries']}")
        print("="*80 + "\n")
    
    # Process a batch of queries concurrently (more efficient)
    logging.info(f"Processing a batch of {len(sample_queries)} queries concurrently")
    batch_start_time = time.time()
    batch_results = await text2sql.process_query_batch(sample_queries)
    batch_total_time = time.time() - batch_start_time
    
    print("\n" + "="*80)
    print(f"BATCH PROCESSING COMPLETED")
    print(f"Total time: {batch_total_time:.2f} seconds")
    print(f"Average time per query: {batch_total_time/len(sample_queries):.2f} seconds")
    print(f"Success rate: {sum(1 for r in batch_results if r['success'])}/{len(batch_results)}")
    print("="*80 + "\n")
    
    # Disconnect from database
    text2sql.db_connector.disconnect()
    logging.info("Example completed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 