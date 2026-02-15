
#!/usr/bin/env python3
"""
Professional Python Basics - Production-Ready Code Examples
Python Version: 3.11+
Author: Professional Python Team
License: MIT
"""

import sys
import os
import json
import time
import logging
from typing import (
    Any, Dict, List, Optional, Union, 
    Callable, TypeVar, Generic, Protocol
)
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from contextlib import contextmanager
from functools import wraps, lru_cache
import hashlib
import inspect

# ============================================================================
# SECTION 1: ADVANCED VARIABLE HANDLING
# ============================================================================

class VariableScope:
    """Demonstrates advanced variable scope and closure patterns"""
    
    @staticmethod
    def create_counter(initial: int = 0) -> Dict[str, Callable]:
        """
        Create a counter closure with private state
        
        Args:
            initial: Initial counter value
            
        Returns:
            Dictionary with counter operations
        """
        count = initial
        
        def get() -> int:
            """Get current count"""
            return count
        
        def set(value: int) -> None:
            """Set count to value"""
            nonlocal count
            count = value
        
        def increment(step: int = 1) -> int:
            """Increment count by step"""
            nonlocal count
            count += step
            return count
        
        def decrement(step: int = 1) -> int:
            """Decrement count by step"""
            nonlocal count
            count -= step
            return count
        
        return {
            'get': get,
            'set': set,
            'inc': increment,
            'dec': decrement
        }
    
    @staticmethod
    def demonstrate():
        """Demonstrate variable scope patterns"""
        print("\n=== Variable Scope Demonstration ===")
        
        # Create counter
        counter = VariableScope.create_counter(10)
        
        print(f"Initial: {counter['get']()}")
        print(f"Increment: {counter['inc']()}")
        print(f"Increment: {counter['inc'](5)}")
        print(f"Decrement: {counter['dec'](3)}")
        
        counter['set'](100)
        print(f"After set: {counter['get']()}")

# ============================================================================
# SECTION 2: PRODUCTION DATA TYPES
# ============================================================================

class HttpStatus(Enum):
    """HTTP status codes with metadata"""
    OK = (200, "OK")
    CREATED = (201, "Created")
    BAD_REQUEST = (400, "Bad Request")
    UNAUTHORIZED = (401, "Unauthorized")
    NOT_FOUND = (404, "Not Found")
    INTERNAL_ERROR = (500, "Internal Server Error")
    
    def __init__(self, code: int, phrase: str):
        self.code = code
        self.phrase = phrase
    
    @property
    def is_success(self) -> bool:
        """Check if status is success"""
        return 200 <= self.code < 300
    
    @classmethod
    def from_code(cls, code: int) -> Optional['HttpStatus']:
        """Get status from code"""
        for status in cls:
            if status.code == code:
                return status
        return None

@dataclass
class User:
    """Production-ready user model"""
    
    # Required fields
    username: str
    email: str
    
    # Optional fields with defaults
    id: Optional[int] = None
    is_active: bool = True
    roles: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Private fields (not in repr)
    _password_hash: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate after initialization"""
        self.email = self.email.lower()
        self._validate()
    
    def _validate(self):
        """Validate user data"""
        if '@' not in self.email:
            raise ValueError(f"Invalid email: {self.email}")
        
        if len(self.username) < 3:
            raise ValueError("Username too short")
    
    def set_password(self, password: str):
        """Set hashed password"""
        salt = "fixed_salt"  # In production, use random salt
        self._password_hash = hashlib.sha256(
            (password + salt).encode()
        ).hexdigest()
    
    def verify_password(self, password: str) -> bool:
        """Verify password"""
        if not self._password_hash:
            return False
        salt = "fixed_salt"
        test_hash = hashlib.sha256(
            (password + salt).encode()
        ).hexdigest()
        return test_hash == self._password_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_active': self.is_active,
            'roles': self.roles,
            'created_at': self.created_at.isoformat()
        }

# ============================================================================
# SECTION 3: ADVANCED FUNCTIONS
# ============================================================================

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        print(f"Attempt {attempt + 1} failed. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
            
            raise last_error
        return wrapper
    return decorator

def memoize(ttl_seconds: Optional[int] = None):
    """
    Memoization decorator with optional TTL
    
    Args:
        ttl_seconds: Time to live in seconds (None = forever)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl_seconds is None or (time.time() - timestamp) < ttl_seconds:
                    return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        
        return wrapper
    return decorator

# ============================================================================
# SECTION 4: CONTEXT MANAGERS
# ============================================================================

@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing operations
    
    Args:
        name: Name of the operation
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{name} took {elapsed:.2f}ms")

@contextmanager
def database_transaction(connection):
    """
    Context manager for database transactions
    
    Args:
        connection: Database connection object
    """
    try:
        print("Beginning transaction...")
        yield connection
        print("Committing transaction...")
        # connection.commit() in real code
    except Exception as e:
        print(f"Rolling back transaction: {e}")
        # connection.rollback() in real code
        raise

# ============================================================================
# SECTION 5: ERROR HANDLING
# ============================================================================

class ApplicationError(Exception):
    """Base application exception"""
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        super().__init__(message)
        self.code = code
        self.timestamp = datetime.now()

class ValidationError(ApplicationError):
    """Validation error"""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field

class NotFoundError(ApplicationError):
    """Resource not found"""
    def __init__(self, resource: str, identifier: Any):
        super().__init__(f"{resource} not found: {identifier}", "NOT_FOUND")
        self.resource = resource
        self.identifier = identifier

def safe_execute(func: Callable, default: Any = None, *args, **kwargs) -> Any:
    """
    Safely execute a function with fallback
    
    Args:
        func: Function to execute
        default: Default value on error
        *args: Function arguments
        **kwargs: Function keyword arguments
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error executing {func.__name__}: {e}")
        return default

# ============================================================================
# SECTION 6: FILE OPERATIONS
# ============================================================================

class FileHandler:
    """Production file handling utilities"""
    
    @staticmethod
    def read_json(filepath: str, encoding: str = 'utf-8') -> Optional[Dict]:
        """Read JSON file safely"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error reading {filepath}: {e}")
            return None
    
    @staticmethod
    def write_json(filepath: str, data: Dict, encoding: str = 'utf-8') -> bool:
        """Write JSON file safely"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logging.error(f"Error writing {filepath}: {e}")
            return False
    
    @staticmethod
    @contextmanager
    def locked_file(filepath: str, mode: str = 'r'):
        """
        Context manager for file locking
        
        Args:
            filepath: Path to file
            mode: File mode
        """
        import fcntl
        with open(filepath, mode) as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

# ============================================================================
# SECTION 7: DATA PROCESSING
# ============================================================================

class DataPipeline:
    """Data processing pipeline with chainable operations"""
    
    def __init__(self, data: Any):
        self.data = data
        self._operations = []
    
    def map(self, func: Callable) -> 'DataPipeline':
        """Add map operation"""
        self._operations.append(('map', func))
        return self
    
    def filter(self, predicate: Callable) -> 'DataPipeline':
        """Add filter operation"""
        self._operations.append(('filter', predicate))
        return self
    
    def reduce(self, func: Callable, initial: Any = None) -> 'DataPipeline':
        """Add reduce operation"""
        self._operations.append(('reduce', func, initial))
        return self
    
    def execute(self) -> Any:
        """Execute the pipeline"""
        result = self.data
        
        for op in self._operations:
            if op[0] == 'map':
                result = [op[1](x) for x in result]
            elif op[0] == 'filter':
                result = [x for x in result if op[1](x)]
            elif op[0] == 'reduce':
                from functools import reduce
                result = reduce(op[1], result, op[2])
        
        return result

# ============================================================================
# SECTION 8: PERFORMANCE OPTIMIZATION
# ============================================================================

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def fibonacci_cached(n: int) -> int:
        """Cached Fibonacci calculation"""
        if n < 2:
            return n
        return PerformanceOptimizer.fibonacci_cached(n-1) + PerformanceOptimizer.fibonacci_cached(n-2)
    
    @staticmethod
    def batch_process(items: List[Any], processor: Callable, batch_size: int = 100) -> List[Any]:
        """Process items in batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [processor(item) for item in batch]
            results.extend(batch_results)
            
            # Yield control periodically
            if i % (batch_size * 10) == 0:
                time.sleep(0)  # Allow other threads to run
        
        return results
    
    @staticmethod
    def profile(func: Callable) -> Callable:
        """Profiling decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import cProfile
            import pstats
            import io
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            result = func(*args, **kwargs)
            
            profiler.disable()
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
            
            print(f"\nProfile for {func.__name__}:")
            print(stream.getvalue())
            
            return result
        return wrapper

# ============================================================================
# SECTION 9: API CLIENT
# ============================================================================

class APIClient:
    """Production API client with retry and rate limiting"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        self._init_session()
    
    def _init_session(self):
        """Initialize HTTP session"""
        import requests
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Python-APIClient/1.0'
        })
    
    @retry(max_attempts=3, delay=1.0)
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request with retry"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    @retry(max_attempts=3, delay=1.0)
    def post(self, endpoint: str, data: Dict = None) -> Dict:
        """Make POST request with retry"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        response = self.session.post(url, json=data, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    def close(self):
        """Close session"""
        if self.session:
            self.session.close()

# ============================================================================
# SECTION 10: LOGGING CONFIGURATION
# ============================================================================

def setup_logging(
    name: str = __name__,
    level: str = 'INFO',
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '{"time": "%(asctime)s", "name": "%(name)s", '
            '"level": "%(levelname)s", "message": "%(message)s"}'
        ))
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main demonstration function"""
    print("=" * 60)
    print("PROFESSIONAL PYTHON BASICS - DEMONSTRATION")
    print("=" * 60)
    
    # 1. Variable Scope
    VariableScope.demonstrate()
    
    # 2. Data Types
    print("\n=== Data Types Demonstration ===")
    
    # Enum usage
    status = HttpStatus.OK
    print(f"Status: {status.name} - {status.phrase} (Code: {status.code})")
    print(f"Is success: {status.is_success}")
    
    # Dataclass usage
    user = User("johndoe", "John@Example.com")
    user.set_password("secret123")
    print(f"User created: {user.username}, {user.email}")
    print(f"Password valid: {user.verify_password('secret123')}")
    print(f"User dict: {user.to_dict()}")
    
    # 3. Functions
    print("\n=== Functions Demonstration ===")
    
    @memoize(ttl_seconds=5)
    def expensive_function(n: int) -> int:
        """Expensive function for demonstration"""
        print(f"Computing for {n}...")
        time.sleep(1)
        return n * n
    
    # First call computes
    print(f"Result 1: {expensive_function(5)}")
    # Second call uses cache
    print(f"Result 2: {expensive_function(5)}")
    
    # 4. Context Managers
    print("\n=== Context Managers Demonstration ===")
    
    with timer("Data processing"):
        time.sleep(0.5)
        print("Processing data...")
    
    with database_transaction("mock_connection"):
        print("Performing database operations...")
        # raise Exception("Test error")  # Uncomment to test rollback
    
    # 5. Error Handling
    print("\n=== Error Handling Demonstration ===")
    
    def divide(a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValidationError("Division by zero", field="b")
        return a / b
    
    result = safe_execute(divide, default=float('inf'), 10, 0)
    print(f"Safe division result: {result}")
    
    # 6. File Operations
    print("\n=== File Operations Demonstration ===")
    
    # Write test data
    test_data = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    success = FileHandler.write_json("data/test.json", test_data)
    print(f"Write successful: {success}")
    
    # Read test data
    data = FileHandler.read_json("data/test.json")
    print(f"Read data: {data}")
    
    # 7. Data Processing
    print("\n=== Data Processing Demonstration ===")
    
    numbers = list(range(1, 11))
    pipeline = DataPipeline(numbers)
    
    result = (pipeline
        .filter(lambda x: x % 2 == 0)  # Even numbers
        .map(lambda x: x * 2)          # Double them
        .execute()
    )
    print(f"Pipeline result: {result}")
    
    # 8. Performance Optimization
    print("\n=== Performance Optimization Demonstration ===")
    
    with timer("Fibonacci (cached)"):
        fib_result = PerformanceOptimizer.fibonacci_cached(30)
        print(f"Fibonacci(30): {fib_result}")
    
    # 9. API Client (Simulated)
    print("\n=== API Client Demonstration ===")
    
    # Note: This uses a mock endpoint
    client = APIClient("https://api.example.com", api_key="test-key")
    
    @retry(max_attempts=2, delay=0.5)
    def simulate_api_call():
        """Simulate API call"""
        import random
        if random.random() < 0.5:  # 50% failure rate
            raise ConnectionError("API unavailable")
        return {"status": "success", "data": [1, 2, 3]}
    
    try:
        result = simulate_api_call()
        print(f"API result: {result}")
    except Exception as e:
        print(f"API call failed after retries: {e}")
    
    client.close()
    
    # 10. Logging
    print("\n=== Logging Demonstration ===")
    
    logger = setup_logging(__name__, 'DEBUG')
    logger.info("Application started", extra={"user_id": 123})
    logger.debug("Debug message with details", extra={"data": numbers})
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## 🚀 How to Use

1. **Copy the code** from `code_examples.py` into your project
2. **Run the demonstration** to see everything in action:
   ```bash
   python code_examples.py
   ```
3. **Study each section** and modify the examples
4. **Import specific classes** into your own code:
   ```python
   from code_examples import User, APIClient, FileHandler
   
   user = User("john", "john@example.com")
   client = APIClient("https://api.example.com")
   ```

## 📝 Best Practices Demonstrated

✅ **Type hints** for all functions
✅ **Docstrings** for documentation
✅ **Error handling** with custom exceptions
✅ **Context managers** for resource management
✅ **Decorators** for cross-cutting concerns
✅ **Dataclasses** for clean data models
✅ **Enums** for constants
✅ **Logging** for observability
✅ **Retry logic** for resilience
✅ **Caching** for performance

## 🔧 Production Usage

```python
# Example: Using in a production application
from code_examples import setup_logging, User, FileHandler

# Setup logging
logger = setup_logging("myapp", "INFO", "app.log")

# Create user
user = User("admin", "admin@example.com")
user.set_password("secure_password")

# Save to file
FileHandler.write_json("users/admin.json", user.to_dict())

logger.info(f"User created: {user.username}")
