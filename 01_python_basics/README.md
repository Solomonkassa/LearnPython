
# Python Basics - Advanced Professional Guide 🐍

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Type Hints](https://img.shields.io/badge/types-mypy-brightgreen)](http://mypy-lang.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents
- [Advanced Variable Handling](#advanced-variable-handling)
- [Production-Ready Data Types](#production-ready-data-types)
- [Advanced Control Flow](#advanced-control-flow)
- [Professional Functions](#professional-functions)
- [Advanced Error Handling](#advanced-error-handling)
- [Production File Operations](#production-file-operations)
- [Memory Management](#memory-management)
- [Performance Optimization](#performance-optimization)
- [Best Practices & Patterns](#best-practices--patterns)

## Advanced Variable Handling

### Variable Scope and Closure Patterns
```python
"""
advanced_scope.py - Professional variable scope management
"""

from typing import Callable, Any, Dict
import sys

class ScopeManager:
    """Advanced scope management with context tracking"""
    
    def __init__(self):
        self._scopes: Dict[str, Any] = {}
        self._current_scope = "global"
    
    def create_scope(self, name: str) -> 'ScopeManager':
        """Create a new named scope"""
        self._scopes[name] = {}
        return self
    
    def __enter__(self):
        """Context manager for scope"""
        self._previous_scope = self._current_scope
        return self
    
    def __exit__(self, *args):
        """Exit scope context"""
        self._current_scope = self._previous_scope
    
    def set(self, key: str, value: Any):
        """Set variable in current scope"""
        self._scopes.setdefault(self._current_scope, {})[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get variable from current scope"""
        return self._scopes.get(self._current_scope, {}).get(key, default)

# Closure factory pattern
def variable_factory(initial_value: Any) -> Callable:
    """
    Create a closure with private variable state
    
    Args:
        initial_value: Initial value for the closure
        
    Returns:
        Callable function that maintains state
    """
    private_state = {'value': initial_value}
    
    def getter() -> Any:
        """Get current value"""
        return private_state['value']
    
    def setter(new_value: Any) -> None:
        """Set new value with validation"""
        private_state['value'] = new_value
    
    def updater(func: Callable) -> Any:
        """Update value using function"""
        private_state['value'] = func(private_state['value'])
        return private_state['value']
    
    # Return multiple functions as a tuple or dictionary
    return {
        'get': getter,
        'set': setter,
        'update': updater
    }

# Usage
counter = variable_factory(0)
counter['set'](10)
print(counter['get']())  # 10
counter['update'](lambda x: x + 5)
print(counter['get']())  # 15
```

### Advanced Variable Annotations and Type Hints
```python
"""
type_hints.py - Production-grade type hinting
"""

from typing import (
    TypeVar, Generic, Union, Optional, 
    List, Dict, Set, Tuple, Callable,
    Any, Protocol, runtime_checkable,
    Literal, Final, TypedDict, overload
)
from datetime import datetime
from decimal import Decimal
from enum import Enum
import sys

# Type variables for generics
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Constants with Final (Python 3.8+)
MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT: Final[float] = 30.0

# Literal types for specific values
HTTPMethod = Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# TypedDict for structured dictionaries
class UserDict(TypedDict):
    """Typed dictionary for user data"""
    id: int
    name: str
    email: str
    created_at: datetime
    is_active: bool

# Runtime checkable protocol
@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects"""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...

# Generic class example
class Repository(Generic[T]):
    """Generic repository pattern"""
    
    def __init__(self):
        self._items: Dict[str, T] = {}
    
    def add(self, key: str, value: T) -> None:
        """Add item to repository"""
        self._items[key] = value
    
    def get(self, key: str) -> Optional[T]:
        """Get item from repository"""
        return self._items.get(key)
    
    def find_all(self, predicate: Callable[[T], bool]) -> List[T]:
        """Find all items matching predicate"""
        return [item for item in self._items.values() if predicate(item)]

# Overload example
@overload
def process_data(data: str) -> str: ...

@overload
def process_data(data: List[int]) -> List[str]: ...

def process_data(data: Union[str, List[int]]) -> Union[str, List[str]]:
    """Process data with different types"""
    if isinstance(data, str):
        return data.upper()
    else:
        return [str(x) for x in data]
```

## Production-Ready Data Types

### Advanced Enum Patterns
```python
"""
enums_advanced.py - Production-grade enum usage
"""

from enum import Enum, auto, IntEnum, Flag, unique
from typing import Optional, Any, Dict
from dataclasses import dataclass

@unique  # Ensures no duplicate values
class HttpStatus(IntEnum):
    """HTTP status codes with metadata"""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    
    @property
    def is_success(self) -> bool:
        """Check if status is success"""
        return 200 <= self.value < 300
    
    @property
    def is_client_error(self) -> bool:
        """Check if status is client error"""
        return 400 <= self.value < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if status is server error"""
        return 500 <= self.value < 600
    
    @classmethod
    def from_code(cls, code: int) -> Optional['HttpStatus']:
        """Get status enum from code"""
        try:
            return cls(code)
        except ValueError:
            return None

class Permission(Flag):
    """Permission flags using bitwise operations"""
    NONE = 0
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    ADMIN = READ | WRITE | DELETE
    
    def has_permission(self, permission: 'Permission') -> bool:
        """Check if has specific permission"""
        return (self & permission) == permission

# Usage
status = HttpStatus.OK
print(status.is_success)  # True
print(status.name)  # 'OK'
print(status.value)  # 200

# Permission checking
user_perms = Permission.READ | Permission.WRITE
print(user_perms.has_permission(Permission.READ))  # True
print(user_perms.has_permission(Permission.DELETE))  # False
```

### Advanced Data Classes
```python
"""
dataclasses_advanced.py - Production dataclass patterns
"""

from dataclasses import dataclass, field, asdict, astuple
from typing import List, Optional, Any, ClassVar
from datetime import datetime
import json
import hashlib

@dataclass
class BaseModel:
    """Base model with common functionality"""
    
    id: Optional[int] = field(default=None, compare=False)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = field(default=None, compare=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = asdict(self)
        # Handle datetime serialization
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def update_timestamp(self):
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now()

@dataclass
class User(BaseModel):
    """User model with advanced features"""
    
    # Class variables
    MIN_PASSWORD_LENGTH: ClassVar[int] = 8
    VALID_ROLES: ClassVar[List[str]] = ['admin', 'user', 'guest']
    
    # Instance fields with defaults
    username: str
    email: str
    password: str = field(repr=False)  # Don't show in repr
    roles: List[str] = field(default_factory=list)
    is_active: bool = True
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.email = self.email.lower()
        self.validate()
    
    def validate(self) -> bool:
        """Validate user data"""
        if len(self.password) < self.MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters")
        
        if '@' not in self.email:
            raise ValueError("Invalid email format")
        
        for role in self.roles:
            if role not in self.VALID_ROLES:
                raise ValueError(f"Invalid role: {role}")
        
        return True
    
    def hash_password(self) -> str:
        """Return hashed password"""
        return hashlib.sha256(self.password.encode()).hexdigest()
    
    def add_role(self, role: str):
        """Add role if valid"""
        if role in self.VALID_ROLES and role not in self.roles:
            self.roles.append(role)
            self.update_timestamp()
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create user from dictionary"""
        return cls(**data)

# Usage
user = User(
    username="john_doe",
    email="John@Example.com",  # Will be lowercased
    password="secret123",
    roles=["user"]
)

print(user.email)  # john@example.com
print(user.hash_password())  # Hashed password
print(user.to_json())  # JSON representation
```

## Advanced Control Flow

### Pattern Matching (Python 3.10+)
```python
"""
pattern_matching.py - Advanced structural pattern matching
"""

from typing import Any, List, Dict, Union
from dataclasses import dataclass

@dataclass
class Point:
    """Point class for pattern matching"""
    x: float
    y: float

@dataclass
class Circle:
    """Circle class for pattern matching"""
    center: Point
    radius: float

@dataclass
class Rectangle:
    """Rectangle class for pattern matching"""
    top_left: Point
    bottom_right: Point

def process_shape(shape: Any) -> str:
    """
    Process different shapes using pattern matching
    
    Args:
        shape: Shape object to process
        
    Returns:
        Description of the shape
    """
    match shape:
        case Point(x=0, y=0):
            return "Origin point"
        
        case Point(x, y):
            return f"Point at ({x}, {y})"
        
        case Circle(center=Point(x, y), radius=r) if r > 0:
            return f"Circle at ({x}, {y}) with radius {r}"
        
        case Rectangle(Point(x1, y1), Point(x2, y2)):
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            return f"Rectangle {width}x{height}"
        
        case _:
            return f"Unknown shape: {type(shape).__name__}"

def parse_command(command: List[str]) -> Dict[str, Any]:
    """
    Parse command using pattern matching
    
    Args:
        command: List of command parts
        
    Returns:
        Parsed command dictionary
    """
    match command:
        case ["quit"]:
            return {"action": "quit"}
        
        case ["load", filename]:
            return {"action": "load", "filename": filename}
        
        case ["save", *rest]:
            match rest:
                case [filename]:
                    return {"action": "save", "filename": filename}
                case [filename, "force"]:
                    return {"action": "save", "filename": filename, "force": True}
                case _:
                    return {"action": "save", "error": "Invalid arguments"}
        
        case ["search", query, *options] if options:
            return {"action": "search", "query": query, "options": options}
        
        case _:
            return {"action": "unknown", "command": command}

# Advanced matching with guards
def process_http_response(status: int, headers: Dict, body: Any) -> str:
    """
    Process HTTP response with pattern matching
    
    Args:
        status: HTTP status code
        headers: Response headers
        body: Response body
    """
    match (status, headers, body):
        case (200, _, str() as text):
            return f"Success: {text[:100]}..."
        
        case (200, _, dict() as data):
            return f"Success: {len(data)} fields"
        
        case (404, _, _):
            return "Not found"
        
        case (code, _, _) if 400 <= code < 500:
            return f"Client error: {code}"
        
        case (code, _, _) if 500 <= code < 600:
            return f"Server error: {code}"
        
        case _:
            return f"Unknown response: {status}"

# Usage examples
print(process_shape(Point(0, 0)))  # Origin point
print(process_shape(Circle(Point(1, 2), 5)))  # Circle at (1, 2) with radius 5

print(parse_command(["load", "data.txt"]))  # {'action': 'load', 'filename': 'data.txt'}
print(parse_command(["save", "output.json", "force"]))  # {'action': 'save', 'filename': 'output.json', 'force': True}
```

### Advanced Exception Handling
```python
"""
exceptions_advanced.py - Production exception handling patterns
"""

from typing import Optional, Type, Callable, Any
from contextlib import contextmanager
from functools import wraps
import logging
import traceback
import time

# Custom exception hierarchy
class ApplicationError(Exception):
    """Base application exception"""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code or "UNKNOWN_ERROR"
        self.timestamp = time.time()

class ValidationError(ApplicationError):
    """Data validation error"""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field

class DatabaseError(ApplicationError):
    """Database operation error"""
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(message, "DATABASE_ERROR")
        self.query = query

class ConfigurationError(ApplicationError):
    """Configuration error"""
    pass

# Retry decorator with exponential backoff
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator

# Context manager for exception handling
@contextmanager
def error_handler(
    error_types: tuple = (Exception,),
    fallback: Optional[Any] = None,
    reraise: bool = True,
    log: bool = True
):
    """
    Context manager for handling errors
    
    Args:
        error_types: Tuple of exception types to catch
        fallback: Fallback value if error occurs
        reraise: Whether to reraise after handling
        log: Whether to log the error
    """
    try:
        yield
    except error_types as e:
        if log:
            logging.error(f"Error caught: {e}\n{traceback.format_exc()}")
        
        if reraise:
            raise
        
        return fallback

# Safe execution with fallback
def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    on_error: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Safely execute function with fallback
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value if error occurs
        on_error: Error callback
        **kwargs: Function keyword arguments
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if on_error:
            on_error(e)
        return default

# Usage examples
@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
def unstable_network_call():
    """Simulate unstable network call"""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return "Success"

# Safe file operation
with error_handler((IOError, PermissionError), reraise=False):
    with open("config.json", "r") as f:
        data = f.read()

# Result: data will be None if file doesn't exist

# Error handling with fallback
def divide(a: float, b: float) -> float:
    """Divide with zero handling"""
    return safe_execute(
        lambda: a / b,
        default=float('inf'),
        on_error=lambda e: logging.warning(f"Division error: {e}")
    )
```

## Professional Functions

### Advanced Function Patterns
```python
"""
functions_advanced.py - Production function patterns
"""

from typing import Callable, Any, Dict, List, Optional, Union
from functools import wraps, partial, lru_cache, singledispatch
import inspect
import time

# Function composition
def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions: f(g(h(x)))
    
    Args:
        *functions: Functions to compose
        
    Returns:
        Composed function
    """
    def compose_two(f: Callable, g: Callable) -> Callable:
        return lambda x: f(g(x))
    
    return reduce(compose_two, functions, lambda x: x)

# Pipe operator style
def pipe(value: Any, *functions: Callable) -> Any:
    """
    Pipe value through functions: value |> f |> g
    
    Args:
        value: Initial value
        *functions: Functions to apply
        
    Returns:
        Transformed value
    """
    for func in functions:
        value = func(value)
    return value

# Partial application factory
def partial_application(func: Callable, *args, **kwargs) -> Callable:
    """
    Create partial application with inspection
    
    Args:
        func: Function to partially apply
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    partial_func = partial(func, *args, **kwargs)
    
    # Preserve metadata
    @wraps(func)
    def wrapper(*more_args, **more_kwargs):
        return partial_func(*more_args, **more_kwargs)
    
    return wrapper

# Function memoization with TTL
def memoize_ttl(ttl_seconds: int = 60, maxsize: int = 128):
    """
    Memoization decorator with TTL
    
    Args:
        ttl_seconds: Time to live in seconds
        maxsize: Maximum cache size
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            # Manage cache size
            if len(cache) > maxsize:
                # Remove oldest entry
                oldest_key = min(cache.items(), key=lambda x: x[1][1])[0]
                del cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator

# Function validation decorator
def validate_input(**validators):
    """
    Validate function inputs
    
    Args:
        **validators: Validator functions for parameters
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for {param_name}: {value}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Single dispatch for function overloading
@singledispatch
def process_data(data):
    """Base processing function"""
    raise NotImplementedError(f"No handler for {type(data)}")

@process_data.register(str)
def _(data: str) -> str:
    """Process string data"""
    return data.upper()

@process_data.register(list)
def _(data: list) -> list:
    """Process list data"""
    return [x * 2 for x in data if isinstance(x, (int, float))]

@process_data.register(dict)
def _(data: dict) -> dict:
    """Process dict data"""
    return {k: v for k, v in data.items() if v is not None}

# Usage examples
# Validation
@validate_input(age=lambda x: x >= 0, name=lambda x: len(x) > 0)
def create_user(name: str, age: int) -> dict:
    """Create user with validation"""
    return {"name": name, "age": age}

# Memoization
@memoize_ttl(ttl_seconds=30)
def expensive_calculation(n: int) -> int:
    """Expensive calculation with caching"""
    time.sleep(1)  # Simulate work
    return n * n

# Single dispatch
print(process_data("hello"))  # "HELLO"
print(process_data([1, 2, 3, "a"]))  # [2, 4, 6]
print(process_data({"a": 1, "b": None, "c": 3}))  # {"a": 1, "c": 3}

# Function composition
add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x ** 2

composed = compose(square, double, add_one)
print(composed(5))  # square(double(add_one(5))) = square(12) = 144

# Pipe style
result = pipe(
    5,
    add_one,
    double,
    square
)
print(result)  # 144
```

## Memory Management

### Advanced Memory Patterns
```python
"""
memory_management.py - Professional memory management
"""

import sys
import gc
import weakref
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
import tracemalloc
import objgraph

# Slots for memory optimization
class OptimizedClass:
    """
    Class using __slots__ for memory optimization
    
    Saves memory by avoiding __dict__ per instance
    """
    __slots__ = ['name', 'age', 'data']
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.data = {}

# Weak reference patterns
class Cache:
    """
    Cache using weak references to avoid memory leaks
    """
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
    
    def add(self, key: str, value: Any):
        """Add item to cache with weak reference"""
        self._cache[key] = weakref.ref(value)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        ref = self._cache.get(key)
        if ref is not None:
            return ref()  # Returns None if object was garbage collected
        return None
    
    def cleanup(self):
        """Remove dead references"""
        dead_keys = [
            key for key, ref in self._cache.items() 
            if ref() is None
        ]
        for key in dead_keys:
            del self._cache[key]

# Memory context manager
@contextmanager
def memory_tracker(threshold_mb: float = 100):
    """
    Track memory usage and alert if exceeds threshold
    
    Args:
        threshold_mb: Memory threshold in MB
    """
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        print(f"Current memory: {current_mb:.2f} MB")
        print(f"Peak memory: {peak_mb:.2f} MB")
        
        if current_mb > threshold_mb:
            print(f"WARNING: Memory usage exceeded {threshold_mb} MB")
            
            # Print top memory users
            print("\nTop memory users:")
            objgraph.show_most_common_types(limit=10)

# Memory pool pattern
class MemoryPool:
    """
    Object pool pattern for memory-intensive objects
    """
    
    def __init__(self, factory: callable, max_size: int = 10):
        self._factory = factory
        self._pool: List[Any] = []
        self._max_size = max_size
    
    def acquire(self) -> Any:
        """Acquire object from pool"""
        if self._pool:
            return self._pool.pop()
        return self._factory()
    
    def release(self, obj: Any):
        """Release object back to pool"""
        if len(self._pool) < self._max_size:
            self._pool.append(obj)
    
    def __enter__(self):
        """Context manager entry"""
        self._obj = self.acquire()
        return self._obj
    
    def __exit__(self, *args):
        """Context manager exit"""
        self.release(self._obj)

# Memory profiling decorator
def profile_memory(func: callable) -> callable:
    """
    Decorator to profile memory usage of a function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        
        # Get baseline
        baseline = tracemalloc.take_snapshot()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Get snapshot after function
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Compare
        top_stats = snapshot.compare_to(baseline, 'lineno')
        
        print(f"\nMemory profile for {func.__name__}:")
        for stat in top_stats[:5]:
            print(stat)
        
        return result
    
    return wrapper

# Usage examples
@profile_memory
def memory_intensive_operation():
    """Memory intensive operation for profiling"""
    data = [list(range(1000)) for _ in range(1000)]
    return len(data)

# Memory pool usage
def create_connection():
    """Factory function for connections"""
    return {"connection_id": id(object())}

pool = MemoryPool(create_connection, max_size=5)

with pool as conn:
    print(f"Using connection: {conn}")
    # Connection automatically returns to pool

# Cache with weak references
cache = Cache()
data = {"key": "value"}
cache.add("test", data)
print(cache.get("test"))  # Returns the object
del data
print(cache.get("test"))  # May return None if garbage collected

# Memory tracking
with memory_tracker(threshold_mb=10):
    # Perform operations
    large_list = [list(range(1000)) for _ in range(1000)]
    del large_list
```

## Performance Optimization

### Advanced Performance Techniques
```python
"""
performance_optimization.py - Production performance patterns
"""

import time
import functools
from typing import Any, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from numba import jit, vectorize
import cython

# Profiling decorator
def profile_time(threshold_ms: float = 100):
    """
    Profile function execution time
    
    Args:
        threshold_ms: Threshold in milliseconds to log warning
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            if elapsed > threshold_ms:
                print(f"WARNING: {func.__name__} took {elapsed:.2f}ms "
                      f"(threshold: {threshold_ms}ms)")
            
            return result
        return wrapper
    return decorator

# JIT compilation with numba
@jit(nopython=True, parallel=True)
def numba_heavy_computation(data: np.ndarray) -> np.ndarray:
    """
    Heavy computation optimized with numba
    """
    result = np.zeros_like(data)
    for i in range(len(data)):
        # Parallelizable computation
        result[i] = data[i] * data[i] + np.sqrt(data[i])
    return result

# Vectorized operations
@vectorize(['float64(float64)'], target='parallel')
def vectorized_operation(x: float) -> float:
    """
    Vectorized operation for array processing
    """
    return x * x + 2 * x + 1

# Cython optimization (in .pyx file)
"""
# cython_example.pyx
cpdef int fibonacci(int n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Lazy evaluation pattern
class LazyProperty:
    """
    Lazy property descriptor
    """
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value

# Batch processing optimization
class BatchProcessor:
    """
    Optimized batch processing with chunking
    """
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    def process_large_dataset(self, data: List[Any], processor: Callable) -> List[Any]:
        """
        Process large dataset in batches
        """
        results = []
        
        # Process in chunks
        for i in range(0, len(data), self.batch_size):
            chunk = data[i:i + self.batch_size]
            
            # Process chunk in parallel
            with ThreadPoolExecutor() as executor:
                chunk_results = list(executor.map(processor, chunk))
                results.extend(chunk_results)
        
        return results

# Caching strategies
class LRUCache:
    """
    Least Recently Used (LRU) cache implementation
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[Any, Any] = {}
        self.access_order: List[Any] = []
    
    def get(self, key: Any) -> Any:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any):
        """Put value in cache"""
        if key in self.cache:
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)

# Usage examples
class DataProcessor:
    """Example class with optimized methods"""
    
    def __init__(self, data: List[float]):
        self.data = data
        
        # Lazy properties
        self._mean = None
        self._std = None
    
    @LazyProperty
    def mean(self) -> float:
        """Calculate mean lazily"""
        print("Computing mean...")
        return sum(self.data) / len(self.data)
    
    @LazyProperty
    def std(self) -> float:
        """Calculate standard deviation lazily"""
        print("Computing standard deviation...")
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5
    
    @profile_time(threshold_ms=50)
    def heavy_operation(self) -> float:
        """Heavy operation with profiling"""
        time.sleep(0.1)  # Simulate work
        return sum(x * x for x in self.data)

# Numba optimization
@profile_time(threshold_ms=100)
def process_with_numpy(data: np.ndarray) -> np.ndarray:
    """Process with numpy optimization"""
    return numba_heavy_computation(data)

# Usage
processor = DataProcessor(list(range(1000000)))

# First access computes
print(processor.mean)  # Computes and caches
print(processor.mean)  # Uses cached value

# Profile heavy operation
processor.heavy_operation()

# Batch processing
batch_processor = BatchProcessor(batch_size=100)
data = list(range(10000))
results = batch_processor.process_large_dataset(
    data, 
    lambda x: x * x
)

# LRU cache usage
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
cache.get("a")  # Makes "a" most recent
cache.put("d", 4)  # Removes "b" (least recent)
```

## Best Practices & Patterns

### Production-Ready Code Patterns
```python
"""
best_practices.py - Production best practices
"""

from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import logging
from datetime import datetime
import json
import hashlib

# 1. Configuration management
class Config:
    """
    Thread-safe configuration management
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config = {}
            self._initialized = True
    
    def load_from_file(self, filename: str):
        """Load config from JSON file"""
        with open(filename, 'r') as f:
            self._config.update(json.load(f))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set config value"""
        self._config[key] = value

# 2. Structured logging
class StructuredLogger:
    """
    Structured logging with JSON format
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup JSON logging"""
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"time": "%(asctime)s", "name": "%(name)s", '
            '"level": "%(levelname)s", "message": %(message)s}'
        ))
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info with context"""
        log_entry = {
            "message": message,
            "context": kwargs
        }
        self.logger.info(json.dumps(log_entry))
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error with context"""
        log_entry = {
            "message": message,
            "context": kwargs
        }
        self.logger.error(json.dumps(log_entry), exc_info=exc_info)

# 3. Secure data handling
class SecureData:
    """
    Secure data handling with encryption
    """
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def hash_sensitive(self, data: str) -> str:
        """Hash sensitive data"""
        salt = "fixed_salt"  # In production, use random salt per user
        return hashlib.pbkdf2_hmac(
            'sha256',
            data.encode(),
            salt.encode(),
            100000
        ).hex()
    
    def mask_sensitive(self, data: str, visible_chars: int = 4) -> str:
        """Mask sensitive data (e.g., credit cards)"""
        if len(data) <= visible_chars:
            return '*' * len(data)
        return '*' * (len(data) - visible_chars) + data[-visible_chars:]

# 4. API response standardization
class APIResponse:
    """
    Standardized API response format
    """
    
    def __init__(self):
        self.status = "success"
        self.data = None
        self.error = None
        self.metadata = {}
    
    def success(self, data: Any = None, metadata: Dict = None):
        """Create success response"""
        self.status = "success"
        self.data = data
        self.metadata = metadata or {}
        return self.to_dict()
    
    def error(self, message: str, code: str = None, details: Any = None):
        """Create error response"""
        self.status = "error"
        self.error = {
            "message": message,
            "code": code,
            "details": details
        }
        return self.to_dict()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }

# 5. Database connection context manager
@contextmanager
def database_connection(connection_string: str):
    """
    Context manager for database connections
    """
    connection = None
    try:
        # Simulate connection
        connection = {"conn": connection_string, "connected": True}
        print(f"Connected to database: {connection_string}")
        yield connection
    except Exception as e:
        print(f"Database error: {e}")
        raise
    finally:
        if connection:
            connection["connected"] = False
            print("Database connection closed")

# 6. Retry with backoff
def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    backoff_factor: float = 2,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            return None
        return wrapper
    return decorator

# 7. Rate limiting
class RateLimiter:
    """
    Token bucket rate limiter
    """
    
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        
        if new_tokens > 0:
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens"""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    @contextmanager
    def limit(self, tokens: int = 1):
        """Context manager for rate limiting"""
        if not self.acquire(tokens):
            raise Exception("Rate limit exceeded")
        yield

# 8. Circuit breaker pattern
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

# Usage examples
if __name__ == "__main__":
    # Configuration
    config = Config()
    config.set("api_key", "secret123")
    
    # Structured logging
    logger = StructuredLogger("myapp")
    logger.info("User logged in", user_id=123, ip_address="192.168.1.1")
    
    # Secure data
    secure = SecureData("my_secret_key")
    print(secure.mask_sensitive("4111111111111111"))  # ***********1111
    
    # API response
    api = APIResponse()
    print(api.success(data={"user": "john"}, metadata={"timestamp": datetime.now()}))
    
    # Database connection
    with database_connection("postgresql://localhost/mydb"):
        print("Performing database operations...")
    
    # Rate limiting
    limiter = RateLimiter(max_tokens=10, refill_rate=1)
    for _ in range(12):
        with limiter.limit():
            print("Making API call...")
    
    # Circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def unstable_function():
        if time.time() % 2 < 1:
            raise Exception("Service unavailable")
        return "Success"
    
    for i in range(10):
        try:
            result = breaker.call(unstable_function)
            print(f"Call {i}: {result}")
        except Exception as e:
            print(f"Call {i} failed: {e}")
        time.sleep(1)
```

## 🚀 Production Checklist

Before deploying code to production, ensure:

### Code Quality
- [ ] Type hints for all functions
- [ ] Comprehensive docstrings
- [ ] 90%+ test coverage
- [ ] Passes pylint with 10/10
- [ ] Formatted with black
- [ ] Imports sorted with isort

### Security
- [ ] No hardcoded secrets
- [ ] Input validation for all user data
- [ ] Proper error handling (no stack traces to users)
- [ ] Rate limiting for APIs
- [ ] SQL injection prevention
- [ ] XSS protection for web apps

### Performance
- [ ] Profiling completed for hot paths
- [ ] Caching strategy implemented
- [ ] Database queries optimized
- [ ] Memory usage monitored
- [ ] Connection pooling configured

### Monitoring
- [ ] Structured logging implemented
- [ ] Metrics collection configured
- [ ] Health check endpoints
- [ ] Alerting thresholds defined
- [ ] Distributed tracing (if applicable)

### Reliability
- [ ] Retry logic with backoff
- [ ] Circuit breakers for external services
- [ ] Graceful degradation
- [ ] Proper timeout configuration
- [ ] Backup/restore procedures

## 📚 Further Reading

- [Python Official Documentation](https://docs.python.org/3/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PEP 484 Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [Real Python Tutorials](https://realpython.com/)
- [Effective Python](https://effectivepython.com/)
```
