# ============================================
# Database Connection Manager
# - Handles PostgreSQL connections with SQLAlchemy
# - Implements connection pooling and lifecycle management
# - Provides clean interface for database operations
# - Includes error handling and connection health monitoring
# ============================================

# Standard library imports
import time
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Union

# Third-party imports
import pandas as pd
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy.pool import QueuePool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass

class DatabaseManager:
    """
    Database connection manager for PostgreSQL with SQLAlchemy.
    
    Features:
    - Connection pooling with configurable settings
    - Automatic connection health checks
    - Transaction management with context managers
    - Batch operations for better performance
    - Error handling and retry logic
    - Connection monitoring and metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database manager.
        
        Args:
            config: Dictionary containing database configuration
                - connection_string: PostgreSQL connection string
                - pool_size: Number of connections to maintain (default: 20)
                - max_overflow: Additional connections when pool is full (default: 30)
                - pool_recycle: Recycle connections after N seconds (default: 3600)
                - pool_pre_ping: Validate connections before use (default: True)
                - application_name: Application name for monitoring (default: 'avalon_dashboard')
        """
        self.config = self._validate_config(config)
        self.engine: Optional[Engine] = None
        self.session_factory = None
        self._initialize()
        
        logger.info("DatabaseManager initialized successfully")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default values for configuration"""
        defaults = {
            'pool_size': 20,
            'max_overflow': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'application_name': 'avalon_dashboard'
        }
        
        # Validate required fields
        if 'connection_string' not in config:
            raise ValueError("'connection_string' is required in database config")
        
        # Merge with defaults
        validated_config = defaults.copy()
        validated_config.update(config)
        
        return validated_config
    
    def _initialize(self):
        """Initialize the database engine and session factory"""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.config['connection_string'],
                future=True,
                poolclass=QueuePool,
                pool_size=self.config['pool_size'],
                max_overflow=self.config['max_overflow'],
                pool_recycle=self.config['pool_recycle'],
                pool_pre_ping=self.config['pool_pre_ping'],
                connect_args={
                    'application_name': self.config['application_name']
                }
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Test connection
            self._test_connection()
            
            logger.info(f"Database engine initialized with pool_size={self.config['pool_size']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise DatabaseError(f"Connection test failed: {str(e)}")
    
    def get_connection(self):
        """Get a database connection from the pool"""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")
        return self.engine.connect()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self.session_factory:
            raise DatabaseError("Session factory not initialized")
        return self.session_factory()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with automatic cleanup"""
        connection = None
        try:
            connection = self.get_connection()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if connection:
                connection.close()
    
    @contextmanager
    def get_db_session(self):
        """Context manager for database sessions with automatic cleanup"""
        session = None
        try:
            session = self.get_session()
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Database session failed: {str(e)}")
            raise DatabaseError(f"Database session failed: {str(e)}")
        finally:
            if session:
                session.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, fetch: bool = False, fetch_one: bool = False) -> Any:
        """
        Execute a SQL query with optional parameter binding.
        
        Args:
            query: SQL query string
            params: Dictionary of parameters for the query
            fetch: When True, fetch and return results (only for SELECT/WITH or DML with RETURNING)
            fetch_one: When True and fetching, return a single row instead of all
            
        Returns:
            - List of rows (or single row) when fetching
            - Integer rowcount for DML without RETURNING
            - None for DDL statements
        """
        try:
            sql = query.lstrip()
            first_kw = sql.split(None, 1)[0].upper() if sql else ''
            is_select_like = first_kw in ('SELECT', 'WITH')
            is_dml = first_kw in ('INSERT', 'UPDATE', 'DELETE')
            is_ddl = first_kw in ('CREATE', 'ALTER', 'DROP', 'TRUNCATE')
            wants_fetch = fetch or is_select_like

            if wants_fetch:
                with self.get_db_connection() as conn:
                    result = conn.execute(text(query), params or {})
                    if fetch_one:
                        row = result.fetchone()
                        return row
                    return result.fetchall()
            else:
                # Non-SELECT path: execute inside an explicit transaction and do not fetch
                with self.engine.begin() as conn:
                    result = conn.execute(text(query), params or {})
                    if is_dml:
                        try:
                            return result.rowcount
                        except Exception:
                            return None
                    # For DDL or other statements, nothing to return
                    return None
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute multiple operations in a single transaction.
        
        Args:
            operations: List of operation dictionaries with 'query' and 'params' keys
            
        Returns:
            List of operation results
        """
        try:
            with self.engine.begin() as conn:
                results = []
                for operation in operations:
                    if 'query' not in operation:
                        raise ValueError("Each operation must have a 'query' key")
                    
                    params = operation.get('params', {})
                    result = conn.execute(text(operation['query']), params)
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Transaction execution failed: {str(e)}")
            raise DatabaseError(f"Transaction execution failed: {str(e)}")
    
    def execute_with_retry(self, func, max_retries: int = 3, delay: float = 1.0):
        """
        Execute a database operation with retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries (doubles each time)
            
        Returns:
            Function result
            
        Raises:
            DatabaseError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except (OperationalError, DisconnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts")
                    raise DatabaseError(f"Operation failed after {max_retries} attempts: {str(e)}")
            except Exception as e:
                # Non-retryable errors
                raise DatabaseError(f"Non-retryable error: {str(e)}")
        
        raise DatabaseError(f"Unexpected retry loop exit: {str(last_exception)}")
    
    def bulk_insert(self, table_name: str, data: List[Dict[str, Any]], if_exists: str = 'append') -> bool:
        """
        Efficiently insert multiple records using pandas to_sql.
        
        Args:
            table_name: Name of the target table
            data: List of dictionaries containing the data to insert
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = pd.DataFrame(data)
            
            with self.get_db_connection() as conn:
                df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists=if_exists,
                    index=False,
                    method='multi',
                    chunksize=1000
                )
            
            logger.info(f"Bulk insert completed: {len(data)} records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            raise DatabaseError(f"Bulk insert failed: {str(e)}")
    
    def read_sql_pandas(self, query: str, params: Optional[Dict[str, Any]] = None, 
                        chunk_size: Optional[int] = None) -> Union[pd.DataFrame, Any]:
        """
        Read SQL query results into a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Dictionary of parameters for the query
            chunk_size: If specified, return iterator of DataFrames
            
        Returns:
            pandas DataFrame or iterator of DataFrames (when chunk_size is specified)
        """
        try:
            if chunk_size:
                return pd.read_sql(query, self.engine, params=params, chunksize=chunk_size)
            else:
                return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Pandas read_sql failed: {str(e)}")
            raise DatabaseError(f"Pandas read_sql failed: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.get_db_connection() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {str(e)}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status information.
        
        Returns:
            Dictionary containing pool status metrics
        """
        if not self.engine:
            return {'error': 'Engine not initialized'}
        
        try:
            pool = self.engine.pool
            return {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid(),
                'total_connections': pool.size() + pool.overflow()
            }
        except Exception as e:
            logger.error(f"Failed to get pool status: {str(e)}")
            return {'error': str(e)}
    
    def close(self):
        """Close all database connections and cleanup resources"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database engine disposed successfully")
        except Exception as e:
            logger.error(f"Error disposing database engine: {str(e)}")
    
    def __enter__(self):
        """Enter context - return self for use in context managers"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup resources"""
        self.close()
    
    def __call__(self) -> Engine:
        """Make the class callable to get the engine directly"""
        return self.engine

# Convenience function for creating database manager with common configuration
def create_database_manager(host: str = 'localhost', port: str = '5432', 
                          database: str = 'avalon', username: str = 'admin', 
                          password: str = 'password!', **kwargs) -> DatabaseManager:
    """
    Convenience function to create a DatabaseManager with common configuration.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        **kwargs: Additional configuration options
        
    Returns:
        Configured DatabaseManager instance
    """
    connection_string = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
    
    config = {
        'connection_string': connection_string,
        **kwargs
    }
    
    return DatabaseManager(config)

# Example usage and testing
if __name__ == "__main__":
    # Test the database manager
    try:
        # Create manager with default config
        db = create_database_manager()
        
        # Test health check
        print(f"Database health: {'✅' if db.health_check() else '❌'}")
        
        # Test pool status
        status = db.get_pool_status()
        print(f"Pool status: {status}")
        
        # Test simple query
        result = db.execute_query("SELECT version()")
        print(f"PostgreSQL version: {result[0][0]}")
        
        # Cleanup
        db.close()
        print("✅ Database manager test completed successfully")
        
    except Exception as e:
        print(f"❌ Database manager test failed: {str(e)}") 