import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import psycopg2
from psycopg2.extras import RealDictCursor
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class DBConnector(ABC):
    """
    Abstract base class for database connectors.
    
    This provides a common interface for different database systems
    to execute queries and fetch schema information.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """
        Execute the given SQL query and return results
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Tuple[bool, Union[List[Dict], str]]: (success, result/error_message)
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Retrieve the database schema information
        
        Returns:
            Dict[str, Any]: Schema information with tables, columns, and relations
        """
        pass
    
    @abstractmethod
    def verify_query_safety(self, query: str) -> Tuple[bool, str]:
        """
        Verify if a query is safe to execute (read-only and no potential harmful operations)
        
        Args:
            query (str): SQL query to verify
            
        Returns:
            Tuple[bool, str]: (is_safe, reason_if_not_safe)
        """
        pass


class PostgresConnector(DBConnector):
    """
    PostgreSQL database connector implementation
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 dbname: Optional[str] = None, user: Optional[str] = None, 
                 password: Optional[str] = None):
        """
        Initialize the PostgreSQL connector
        
        Args:
            host: Database host (default: from POSTGRES_HOST env var)
            port: Database port (default: from POSTGRES_PORT env var)
            dbname: Database name (default: from POSTGRES_DB env var)
            user: Database user (default: from POSTGRES_USER env var)
            password: Database password (default: from POSTGRES_PASSWORD env var)
        """
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.dbname = dbname or os.getenv("POSTGRES_DB")
        self.user = user or os.getenv("POSTGRES_USER")
        self.password = password or os.getenv("POSTGRES_PASSWORD")
        
        if not all([self.dbname, self.user, self.password]):
            raise ValueError("Missing database credentials. Set environment variables or pass credentials directly.")
        
        self.conn = None
        
    def connect(self) -> None:
        """
        Establish connection to the PostgreSQL database
        """
        if self.conn is not None:
            logger.warning("Connection already established. Disconnecting first.")
            self.disconnect()
            
        try:
            logger.info(f"Connecting to PostgreSQL database: {self.dbname} at {self.host}:{self.port}")
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL database: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """
        Close the PostgreSQL database connection
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Disconnected from PostgreSQL database")
    
    def _convert_for_json(self, value):
        """
        Convert values that are not JSON serializable to serializable types
        
        Args:
            value: The value to convert
            
        Returns:
            JSON serializable value
        """
        if isinstance(value, Decimal):
            return float(value)
        return value
    
    def execute_query(self, query: str) -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """
        Execute the given SQL query and return results
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Tuple[bool, Union[List[Dict], str]]: (success, result/error_message)
        """
        if not self.conn:
            try:
                self.connect()
            except Exception as e:
                return False, f"Connection error: {str(e)}"
        
        # Verify query safety before execution
        is_safe, reason = self.verify_query_safety(query)
        if not is_safe:
            return False, f"Query rejected: {reason}"
        
        try:
            # Use RealDictCursor to get column names in results
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                
                # Check if this is a SELECT query
                if cursor.description is not None:
                    results = cursor.fetchall()
                    # Convert RealDictRow objects to regular dictionaries and handle Decimal values
                    dict_results = []
                    for row in results:
                        json_safe_row = {}
                        for key, value in dict(row).items():
                            json_safe_row[key] = self._convert_for_json(value)
                        dict_results.append(json_safe_row)
                    return True, dict_results
                else:
                    # For non-SELECT queries (which we actually block)
                    self.conn.rollback()  # Always rollback as we only allow read-only queries
                    return False, "Only SELECT queries are allowed"
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Query execution error: {str(e)}")
            return False, f"Query execution error: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Retrieve the PostgreSQL database schema information
        
        Returns:
            Dict[str, Any]: Schema information with tables, columns, and relations
        """
        if not self.conn:
            try:
                self.connect()
            except Exception as e:
                logger.error(f"Connection error while fetching schema: {str(e)}")
                return {"tables": {}, "relations": []}
        
        schema = {"tables": {}, "relations": []}
        
        try:
            # Get tables and their columns
            table_query = """
            SELECT 
                t.table_name, 
                c.column_name, 
                c.data_type,
                c.column_default,
                c.is_nullable,
                tc.constraint_type,
                CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN TRUE ELSE FALSE END as is_primary
            FROM 
                information_schema.tables t
            LEFT JOIN 
                information_schema.columns c ON t.table_name = c.table_name AND t.table_schema = c.table_schema
            LEFT JOIN 
                information_schema.table_constraints tc ON tc.table_name = c.table_name AND tc.table_schema = c.table_schema
            LEFT JOIN 
                information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name 
                AND ccu.table_schema = tc.table_schema AND ccu.column_name = c.column_name
            WHERE 
                t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            ORDER BY 
                t.table_name, c.ordinal_position;
            """
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(table_query)
                results = cursor.fetchall()
                
                for row in results:
                    table_name = row["table_name"]
                    if table_name not in schema["tables"]:
                        schema["tables"][table_name] = {"columns": []}
                    
                    column_info = {
                        "name": row["column_name"],
                        "data_type": row["data_type"],
                        "is_nullable": row["is_nullable"] == "YES",
                        "is_primary": row["is_primary"],
                        "default_value": row["column_default"]
                    }
                    
                    schema["tables"][table_name]["columns"].append(column_info)
            
            # Get foreign key relations
            fk_query = """
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS referenced_table_name,
                ccu.column_name AS referenced_column_name
            FROM
                information_schema.table_constraints AS tc
            JOIN
                information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
            JOIN
                information_schema.constraint_column_usage AS ccu ON tc.constraint_name = ccu.constraint_name
            WHERE
                tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
            """
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(fk_query)
                results = cursor.fetchall()
                
                for row in results:
                    relation = {
                        "source_table": row["table_name"],
                        "source_column": row["column_name"],
                        "target_table": row["referenced_table_name"],
                        "target_column": row["referenced_column_name"]
                    }
                    schema["relations"].append(relation)
            
            return schema
            
        except Exception as e:
            logger.error(f"Error fetching database schema: {str(e)}")
            return {"tables": {}, "relations": []}
    
    def verify_query_safety(self, query: str) -> Tuple[bool, str]:
        """
        Verify if a query is safe to execute (SELECT only, no potentially harmful operations)
        
        Args:
            query (str): SQL query to verify
            
        Returns:
            Tuple[bool, str]: (is_safe, reason_if_not_safe)
        """
        # Normalize the query for easier checking
        normalized_query = query.strip().upper()
        
        # # Only allow SELECT statements
        # if not normalized_query.startswith("SELECT "):
        #     return False, "Only SELECT queries are allowed"
        
        # Check for potentially harmful operations
        dangerous_keywords = [
            "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE", 
            "GRANT", "REVOKE", "WITH", "COPY", "EXECUTE", "DO"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in normalized_query:
                return False, f"Query contains potentially harmful operation: {keyword}"
        
        return True, ""

    
# To implement MySQL connector in the future:
# 
# class MySQLConnector(DBConnector):
#     """MySQL database connector implementation"""
#     
#     def __init__(self):
#         # Initialize MySQL connection parameters
#         pass
#     
#     def connect(self):
#         # Establish MySQL connection
#         pass
#     
#     def disconnect(self):
#         # Close MySQL connection
#         pass
#     
#     def execute_query(self, query):
#         # Execute MySQL query
#         pass
#     
#     def get_schema(self):
#         # Get MySQL schema information
#         pass
#     
#     def verify_query_safety(self, query):
#         # Verify MySQL query safety
#         pass 