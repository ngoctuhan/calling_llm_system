import os
import logging
import time
import re
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
import difflib
import asyncio

from .db_connector import DBConnector, PostgresConnector
from .query_cache import QueryCache

logger = logging.getLogger(__name__)

# Define the SQL conversion prompt template as a constant
DEFAULT_SQL_CONVERT_TMPL = """You are an expert SQL developer specializing in {db_type} databases.
Your task is to convert the natural language query into a valid SQL query.

NATURAL LANGUAGE QUERY:
{query}

DATABASE TYPE: {db_type}

RELEVANT TABLES:
{table_info}

{relations_info}

{similar_queries_info}

IMPORTANT GUIDELINES:
1. Only generate a SELECT query. Do not generate INSERT, UPDATE, or DELETE queries.
2. Make the query as efficient as possible.
3. Use proper table aliases when joining multiple tables.
4. Do not include any explanations, just return the SQL query.
5. Make sure that the generated SQL query is valid for the specified database type.
6. Keep the result concise and clean by selecting only necessary columns.

SQL QUERY:
"""

# Define the SQL correction prompt template as a constant
DEFAULT_SQL_CORRECT_TMPL = """Your previous SQL query had the following error:
{error_message}

Please fix the SQL query and try again. Here's the original request:
{query}

Previous incorrect query:
{sql_query}

Only return the fixed SQL query without any explanations.
"""

class Text2SQL:
    """
    Text2SQL system for converting natural language questions to SQL queries.
    
    Implements the full pipeline for processing natural language queries:
    1. Retrieval of similar queries from cache
    2. Analysis of database schema to identify relevant tables
    3. Generation of SQL for the database
    4. Execution and feedback handling
    """
    
    def __init__(
        self,
        db_connector: Optional[DBConnector] = None,
        query_cache: Optional[QueryCache] = None,
        db_type: str = "postgres",
        llm_provider = None,
        prompt_template: Optional[str] = None,
        correction_template: Optional[str] = None,
        max_retries: int = 2,
        batch_size: int = 5,
        max_concurrency: int = 10
    ):
        """
        Initialize the Text2SQL processor
        
        Args:
            db_connector: Database connector (if None, will be created based on db_type)
            query_cache: Query cache (if None, a new one will be created)
            db_type: Type of database to use (postgres, mysql, etc.)
            llm_provider: LLM provider instance for text generation (similar to graph_extractor)
            prompt_template: Optional custom prompt template for SQL generation
            correction_template: Optional custom template for SQL correction
            max_retries: Maximum number of retries for SQL generation
            batch_size: Number of queries to process in a single batch
            max_concurrency: Maximum number of concurrent LLM calls
        """
        self.db_type = db_type.lower()
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        self.prompt_template = prompt_template or DEFAULT_SQL_CONVERT_TMPL
        self.correction_template = correction_template or DEFAULT_SQL_CORRECT_TMPL
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        
        # Initialize database connector if not provided
        if db_connector is None:
            if self.db_type == "postgres":
                self.db_connector = PostgresConnector()
            # Add support for other databases here in the future
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        else:
            self.db_connector = db_connector
            
        # Initialize query cache if not provided
        self.query_cache = query_cache or QueryCache()
        
    def analyze_schema_for_tables(self, query: str, schema: Dict[str, Any]) -> List[str]:
        """
        Analyze the database schema to identify relevant tables for the query
        
        Args:
            query: The natural language query
            schema: The database schema
            
        Returns:
            List[str]: List of relevant table names, ranked by relevance
        """
        # Extract table names and their columns from schema
        tables = schema.get("tables", {})
        if not tables:
            logger.warning("No tables found in schema")
            return []
            
        # Simple keyword matching for table and column names
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        table_scores = {}
        
        for table_name, table_info in tables.items():
            table_words = set(re.findall(r'\b\w+\b', table_name.lower()))
            
            # Score table name matches
            table_name_match = len(table_words.intersection(query_words))
            
            # Score column name matches
            column_match_score = 0
            for column in table_info.get("columns", []):
                column_name = column.get("name", "").lower()
                column_words = set(re.findall(r'\b\w+\b', column_name))
                column_match_score += len(column_words.intersection(query_words))
            
            # Combined score with table name matches weighted higher
            table_scores[table_name] = (table_name_match * 2) + column_match_score
        
        # Sort tables by score (descending) and filter those with score > 0
        relevant_tables = [table for table, score in 
                          sorted(table_scores.items(), key=lambda x: x[1], reverse=True) 
                          if score > 0]
        
        # If no direct matches, return all tables as potential candidates
        if not relevant_tables:
            logger.info("No direct table matches found, returning all tables")
            return list(tables.keys())
        
        return relevant_tables
    
    def _format_table_info(self, schema: Dict[str, Any], relevant_tables: List[str]) -> str:
        """
        Format table information for the prompt
        
        Args:
            schema: Database schema
            relevant_tables: List of relevant tables
            
        Returns:
            str: Formatted table information
        """
        table_info = ""
        tables = schema.get("tables", {})
        
        for table_name in relevant_tables:
            if table_name in tables:
                table_info += f"\n- {table_name}:\n"
                for column in tables[table_name].get("columns", []):
                    col_name = column.get("name", "")
                    col_type = column.get("data_type", "")
                    primary = " (PRIMARY KEY)" if column.get("is_primary", False) else ""
                    nullable = " NULL" if column.get("is_nullable", True) else " NOT NULL"
                    table_info += f"  - {col_name}: {col_type}{primary}{nullable}\n"
        
        return table_info
    
    def _format_relations_info(self, schema: Dict[str, Any], relevant_tables: List[str]) -> str:
        """
        Format relationship information for the prompt
        
        Args:
            schema: Database schema
            relevant_tables: List of relevant tables
            
        Returns:
            str: Formatted relationship information
        """
        relations = schema.get("relations", [])
        if not relations:
            return ""
            
        relations_info = "TABLE RELATIONSHIPS:\n"
        for relation in relations:
            if relation.get("source_table") in relevant_tables or relation.get("target_table") in relevant_tables:
                source_table = relation.get("source_table", "")
                source_col = relation.get("source_column", "")
                target_table = relation.get("target_table", "")
                target_col = relation.get("target_column", "")
                relations_info += f"- {source_table}.{source_col} -> {target_table}.{target_col}\n"
        
        return relations_info
    
    def _format_similar_queries_info(self, similar_queries: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format similar queries information for the prompt
        
        Args:
            similar_queries: List of similar queries from cache
            
        Returns:
            str: Formatted similar queries information
        """
        if not similar_queries or len(similar_queries) == 0:
            return ""
            
        similar_queries_info = "SIMILAR QUERIES:\n"
        for i, query_data in enumerate(similar_queries[:3]):  # Limit to top 3
            payload = query_data.get("payload", {})
            nat_query = payload.get("natural_query", "")
            sql_query = payload.get("sql_query", "")
            similar_queries_info += f"Example {i+1}:\nQuestion: {nat_query}\nSQL: {sql_query}\n\n"
        
        return similar_queries_info
    
    async def generate_sql(
        self,
        query: str,
        schema: Dict[str, Any],
        relevant_tables: List[str],
        similar_queries: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate SQL for the query using LLM
        
        Args:
            query: Natural language query
            schema: Database schema
            relevant_tables: List of relevant tables for the query
            similar_queries: List of similar queries from cache (optional)
            
        Returns:
            str: The generated SQL query
        """
        if not self.llm_provider:
            raise ValueError("LLM provider must be specified")
            
        # Format information for prompt
        table_info = self._format_table_info(schema, relevant_tables)
        relations_info = self._format_relations_info(schema, relevant_tables)
        similar_queries_info = self._format_similar_queries_info(similar_queries)
        
        # Format the prompt
        prompt = self.prompt_template.format(
            db_type=self.db_type.upper(),
            query=query,
            table_info=table_info,
            relations_info=relations_info,
            similar_queries_info=similar_queries_info
        )
        
        try:
            # Generate SQL using LLM
            generated_sql = await self.llm_provider.generate(prompt)
            
            # Extract SQL from response (in case LLM includes extra text)
            sql_pattern = r"```sql\s*(.*?)\s*```"
            match = re.search(sql_pattern, generated_sql, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            return generated_sql.strip()
            
        except Exception as e:
            logger.error(f"Error generating SQL with LLM: {str(e)}")
            raise
    
    async def correct_sql(
        self,
        query: str,
        sql_query: str,
        error_message: str
    ) -> str:
        """
        Correct an SQL query using LLM feedback
        
        Args:
            query: Original natural language query
            sql_query: SQL query to correct
            error_message: Error message to guide correction
            
        Returns:
            str: Corrected SQL query
        """
        if not self.llm_provider:
            raise ValueError("LLM provider must be specified")
            
        # Format the correction prompt
        prompt = self.correction_template.format(
            error_message=error_message,
            query=query,
            sql_query=sql_query
        )
        
        try:
            # Generate corrected SQL using LLM
            corrected_sql = await self.llm_provider.generate(prompt)
            
            # Extract SQL from response (in case LLM includes extra text)
            sql_pattern = r"```sql\s*(.*?)\s*```"
            match = re.search(sql_pattern, corrected_sql, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            return corrected_sql.strip()
            
        except Exception as e:
            logger.error(f"Error correcting SQL with LLM: {str(e)}")
            raise
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and convert it to SQL
        
        Args:
            query: The natural language query
            
        Returns:
            Dict[str, Any]: Processing results including SQL, data, and metadata
        """
        start_time = time.time()
        
        try:
            # Connect to database if not already connected
            if not getattr(self.db_connector, "conn", None):
                self.db_connector.connect()
            
            # Get database schema
            schema = self.db_connector.get_schema()
            database_name = getattr(self.db_connector, "dbname", "database")
            
            # Step 1: Search for similar queries in cache
            similar_queries = self.query_cache.find_similar_queries(
                natural_query=query,
                database_name=database_name
            )
            
            # Step 2: Analyze schema to identify relevant tables
            relevant_tables = self.analyze_schema_for_tables(query, schema)
            
            # Step 3: Generate SQL
            sql_query = await self.generate_sql(query, schema, relevant_tables, similar_queries)
            
            print(f"Generated SQL: {sql_query}")
            # Step 4: Execute the query directly without validation
            execution_success, execution_result = self.db_connector.execute_query(sql_query)
            execution_error = None if execution_success else execution_result
            
            # Extract used tables from query
            table_pattern = r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)'
            table_matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
            tables_used = []
            for match in table_matches:
                for table in match:
                    if table and table not in tables_used:
                        tables_used.append(table)
            
            # Step 5: Retry if execution failed
            retry_count = 0
            while not execution_success and retry_count < self.max_retries:
                retry_count += 1
                logger.info(f"SQL execution failed: {execution_error}. Retrying ({retry_count}/{self.max_retries})...")
                
                # Correct SQL with execution error feedback
                sql_query = await self.correct_sql(query, sql_query, execution_error)
                
                # Try executing the corrected query
                execution_success, execution_result = self.db_connector.execute_query(sql_query)
                execution_error = None if execution_success else execution_result
                
                # Update used tables for the new query
                table_matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
                tables_used = []
                for match in table_matches:
                    for table in match:
                        if table and table not in tables_used:
                            tables_used.append(table)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Store in cache if successful
            if execution_success:
                self.query_cache.store_query(
                    natural_query=query,
                    sql_query=sql_query,
                    tables_used=tables_used,
                    database_name=database_name,
                    execution_success=execution_success,
                    execution_result=None,  # Don't store actual data, just metadata
                    execution_time=execution_time
                )
            
            # Return results
            result = {
                "query": query,
                "sql": sql_query,
                "success": execution_success,
                "data": execution_result if execution_success else None,
                "error": execution_error if not execution_success else None,
                "tables_used": tables_used,
                "execution_time": execution_time,
                "retries": retry_count
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "query": query,
                "sql": None,
                "success": False,
                "data": None,
                "error": f"Query processing error: {str(e)}",
                "tables_used": [],
                "execution_time": time.time() - start_time,
                "retries": 0
            }
        finally:
            # Keep connection open for future queries, close only if explicitly called
            pass

    async def process_query_batch(
        self,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of natural language queries
        
        Args:
            queries: List of natural language queries
            
        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        if not queries:
            return []
            
        all_results = []
        
        # Process queries in batches for better performance
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i+self.batch_size]
            
            # Extract triplets concurrently with limited concurrency
            sem = asyncio.Semaphore(self.max_concurrency)
            
            async def process_with_semaphore(query):
                async with sem:
                    return await self.process_query(query)
            
            # Run processing tasks concurrently
            processing_tasks = [
                process_with_semaphore(query)
                for query in batch_queries
            ]
            batch_results = await asyncio.gather(*processing_tasks)
            
            all_results.extend(batch_results)
            
            logger.info(f"Processed batch of {len(batch_queries)} queries, {len(all_results)} total")
        
        return all_results 