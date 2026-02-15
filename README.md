
# Advanced Python Programming - Professional Collection 🚀

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Type Hints](https://img.shields.io/badge/types-mypy-brightgreen)](http://mypy-lang.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

This is a curated collection of **advanced** Python topics for experienced developers. Each module contains production-ready code examples and in-depth explanations of complex concepts used in real-world applications.

## 📋 Advanced Topics Coverage

| Module | Topic | Complexity | Key Advanced Concepts |
|--------|-------|------------|----------------------|
| [01](./01_metaprogramming) | Metaprogramming | 🔴🔴🔴 | Metaclasses, Descriptors, AST |
| [02](./02_async_concurrency) | Async & Concurrency | 🔴🔴🔴 | asyncio, uvloop, Trio |
| [03](./03_design_patterns_advanced) | Advanced Design Patterns | 🔴🔴⚪ | Architecture Patterns, DDD |
| [04](./04_performance_optimization) | Performance Optimization | 🔴🔴🔴 | Profiling, Cython, JIT |
| [05](./05_memory_management) | Memory Management | 🔴🔴⚪ | Garbage Collection, Slots, WeakRef |
| [06](./06_cython_extensions) | Cython Extensions | 🔴🔴🔴 | C Integration, Performance |
| [07](./07_distributed_systems) | Distributed Systems | 🔴🔴🔴 | Ray, Dask, Celery |
| [08](./08_event_driven_architecture) | Event-Driven Architecture | 🔴🔴⚪ | Event Sourcing, CQRS |
| [09](./09_microservices_patterns) | Microservices Patterns | 🔴🔴⚪ | Service Discovery, Circuit Breaker |
| [10](./10_advanced_testing) | Advanced Testing | 🔴⚪⚪ | Property Testing, Mutation Testing |
| [11](./11_compiler_interpreters) | Compilers & Interpreters | 🔴🔴🔴 | AST, Bytecode, JIT |
| [12](./12_advanced_descriptors) | Advanced Descriptors | 🔴⚪⚪ | Data Validation, ORM Patterns |
| [13](./13_advanced_decorators) | Advanced Decorators | 🔴⚪⚪ | Nesting, Parametrization |
| [14](./14_advanced_generators) | Advanced Generators | 🔴⚪⚪ | Coroutines, Yield From |
| [15](./15_advanced_context_managers) | Advanced Context Managers | 🔴⚪⚪ | Async Context, Nested |
| [16](./16_advanced_typing) | Advanced Typing | 🔴⚪⚪ | Protocols, Generics, TypeVar |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-python-professional.git
cd advanced-python-professional

# Create virtual environment with Python 3.11+
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run any advanced example
python 01_metaprogramming/code_examples.py
```

## 📦 Advanced Requirements

```txt
# requirements.txt
# Core
cython>=3.0.0
numpy>=1.24.0
pandas>=2.0.0

# Async
aiohttp>=3.8.0
uvloop>=0.17.0
trio>=0.22.0
anyio>=3.6.0

# Distributed
ray>=2.3.0
dask>=2023.0.0
celery>=5.2.0

# Performance
numba>=0.57.0
pypy>=7.3.11
py-spy>=0.3.0

# Testing
pytest>=7.3.0
hypothesis>=6.70.0
mutmut>=2.4.0

# Development
mypy>=1.0.0
black>=23.0.0
pylint>=2.17.0
```

## 💡 Advanced Learning Path

1. **Foundation** (Weeks 1-2)
   - Master decorators, generators, context managers
   - Understand Python's data model

2. **Deep Dive** (Weeks 3-4)
   - Metaprogramming and descriptors
   - Async programming patterns

3. **Performance** (Weeks 5-6)
   - Profiling and optimization
   - Cython and C extensions

4. **Architecture** (Weeks 7-8)
   - Distributed systems
   - Event-driven architecture

5. **Production** (Weeks 9-10)
   - Advanced testing strategies
   - Deployment patterns
```

---

## 📄 01_metaprogramming/README.md

```markdown
# Advanced Metaprogramming in Python 🔧

## Table of Contents
- [Metaclasses](#metaclasses)
- [Class Decorators](#class-decorators)
- [Descriptors](#descriptors)
- [AST Manipulation](#ast-manipulation)
- [Dynamic Code Generation](#dynamic-code-generation)
- [Production Patterns](#production-patterns)

## Metaclasses

### Singleton Metaclass (Production Ready)
```python
"""
singleton.py - Thread-safe Singleton metaclass with lazy initialization
"""

import threading
from typing import Any, Dict, Type, TypeVar, Optional

T = TypeVar('T')

class SingletonMeta(type):
    """Thread-safe Singleton metaclass"""
    
    _instances: Dict[Type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Override __call__ to control instance creation"""
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def clear_instance(cls, target_cls: Type[T]) -> None:
        """Clear singleton instance (useful for testing)"""
        with cls._lock:
            cls._instances.pop(target_cls, None)

# Usage
class DatabaseConnection(metaclass=SingletonMeta):
    """Database connection pool manager"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = []
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool"""
        # Production connection logic here
        pass
    
    def get_connection(self):
        """Get connection from pool"""
        # Connection pool logic
        pass

# Testing
db1 = DatabaseConnection("postgresql://localhost:5432/mydb")
db2 = DatabaseConnection("postgresql://localhost:5432/mydb")
assert db1 is db2  # True - same instance
```

### ORM-like Metaclass
```python
"""
orm_metaclass.py - Building an ORM-like system with metaclasses
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Type
import inspect

class ModelMeta(type):
    """Metaclass for building ORM models"""
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> Type:
        if name == 'Model':
            return super().__new__(mcs, name, bases, namespace)
        
        # Extract field definitions
        fields = {}
        for key, value in list(namespace.items()):
            if isinstance(value, Field):
                fields[key] = value
                namespace[key] = None  # Replace with None for instance attributes
        
        namespace['_fields'] = fields
        namespace['_table_name'] = namespace.get('__table__', name.lower())
        
        # Create class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Add validation methods
        cls.validate = classmethod(mcs._create_validator(fields))
        
        return cls
    
    @staticmethod
    def _create_validator(fields: Dict):
        """Create validation method dynamically"""
        def validate(cls, data: Dict[str, Any]) -> List[str]:
            errors = []
            for field_name, field in fields.items():
                if field.required and field_name not in data:
                    errors.append(f"{field_name} is required")
                elif field_name in data:
                    value = data[field_name]
                    if not isinstance(value, field.field_type):
                        errors.append(
                            f"{field_name} must be {field.field_type.__name__}"
                        )
            return errors
        return validate

class Field:
    """Field descriptor for model attributes"""
    
    def __init__(self, field_type: type, required: bool = True, default=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be {self.field_type.__name__}")
        obj.__dict__[self.name] = value

class Model(metaclass=ModelMeta):
    """Base model class"""
    
    def __init__(self, **kwargs):
        errors = self.__class__.validate(kwargs)
        if errors:
            raise ValueError(f"Validation errors: {errors}")
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Set defaults for missing fields
        for field_name, field in self._fields.items():
            if field_name not in kwargs and field.default is not None:
                setattr(self, field_name, field.default)
    
    def save(self):
        """Save model to database"""
        data = {}
        for field_name in self._fields:
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        
        # Production: Insert into database
        print(f"Saving to {self._table_name}: {data}")
        return True

# Usage
class User(Model):
    __table__ = 'users'
    
    id = Field(int, required=True)
    name = Field(str, required=True)
    email = Field(str, required=True)
    created_at = Field(datetime, required=False, default=datetime.now)

# Create user
user = User(id=1, name="John Doe", email="john@example.com")
user.save()
```

## Class Decorators

### Advanced Class Decorator with Dependency Injection
```python
"""
dependency_injection.py - Advanced DI container with decorators
"""

from functools import wraps
from typing import Dict, Type, Any, Callable, Optional
import inspect

class DependencyContainer:
    """Production-ready DI container"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register(self, interface: Type, implementation: Type, singleton: bool = False):
        """Register a service"""
        self._services[interface] = implementation
        if singleton:
            self._singletons[interface] = None
    
    def register_factory(self, interface: Type, factory: Callable):
        """Register a factory for creating instances"""
        self._factories[interface] = factory
    
    def resolve(self, interface: Type) -> Any:
        """Resolve a service with dependency injection"""
        # Check if singleton exists
        if interface in self._singletons:
            if self._singletons[interface] is None:
                self._singletons[interface] = self._create_instance(interface)
            return self._singletons[interface]
        
        # Check factory
        if interface in self._factories:
            return self._factories[interface]()
        
        # Create new instance
        return self._create_instance(interface)
    
    def _create_instance(self, interface: Type) -> Any:
        """Create instance with dependency resolution"""
        if interface not in self._services:
            # Try to create directly if no registration
            return self._instantiate(interface)
        
        implementation = self._services[interface]
        return self._instantiate(implementation)
    
    def _instantiate(self, cls: Type) -> Any:
        """Instantiate class with constructor injection"""
        constructor = cls.__init__
        signature = inspect.signature(constructor)
        
        # Prepare arguments
        args = []
        kwargs = {}
        
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
            
            # Try to resolve parameter type
            if param.annotation != inspect.Parameter.empty:
                try:
                    resolved = self.resolve(param.annotation)
                    if param.kind == param.POSITIONAL_ONLY:
                        args.append(resolved)
                    else:
                        kwargs[name] = resolved
                except Exception as e:
                    if param.default == param.empty:
                        raise ValueError(f"Cannot resolve {name}: {e}")
        
        return cls(*args, **kwargs)

# DI Container decorator
container = DependencyContainer()

def inject(cls: Type) -> Type:
    """Class decorator for dependency injection"""
    original_init = cls.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Bind arguments
        sig = inspect.signature(original_init)
        bound = sig.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()
        
        # Resolve dependencies
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if name not in bound.arguments and param.annotation != param.empty:
                try:
                    bound.arguments[name] = container.resolve(param.annotation)
                except:
                    if param.default == param.empty:
                        raise
        
        original_init(self, *bound.arguments.values())
    
    cls.__init__ = new_init
    return cls

# Service definitions
class EmailService:
    def send(self, to: str, message: str):
        print(f"Sending email to {to}: {message}")

class Logger:
    def log(self, message: str):
        print(f"LOG: {message}")

# Usage
@inject
class UserService:
    def __init__(self, email_service: EmailService, logger: Logger):
        self.email_service = email_service
        self.logger = logger
    
    def register_user(self, email: str):
        self.logger.log(f"Registering user: {email}")
        self.email_service.send(email, "Welcome!")

# Register services
container.register(EmailService, EmailService)
container.register(Logger, Logger)

# Create service with automatic injection
user_service = UserService()
user_service.register_user("user@example.com")
```

## Descriptors

### Advanced Descriptor with Validation and Caching
```python
"""
advanced_descriptors.py - Production-ready descriptors
"""

from typing import Any, Optional, Callable
import hashlib
import json
from datetime import datetime, timedelta

class ValidatedAttribute:
    """Descriptor with validation and type checking"""
    
    def __init__(self, validator: Optional[Callable] = None, 
                 type_check: Optional[type] = None):
        self.validator = validator
        self.type_check = type_check
        self.data = {}
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.data.get(id(obj), None)
    
    def __set__(self, obj, value):
        # Type checking
        if self.type_check and not isinstance(value, self.type_check):
            raise TypeError(f"Expected {self.type_check.__name__}, got {type(value).__name__}")
        
        # Custom validation
        if self.validator and not self.validator(value):
            raise ValueError(f"Validation failed for value: {value}")
        
        self.data[id(obj)] = value
    
    def __delete__(self, obj):
        del self.data[id(obj)]

class CachedProperty:
    """Descriptor for cached properties with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache = {}
    
    def __call__(self, func):
        self.func = func
        return self
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        # Check cache
        cache_key = id(obj)
        if cache_key in self.cache:
            value, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                return value
        
        # Compute and cache
        value = self.func(obj)
        self.cache[cache_key] = (value, datetime.now())
        return value
    
    def invalidate(self, obj):
        """Invalidate cache for an object"""
        cache_key = id(obj)
        self.cache.pop(cache_key, None)

class SecureString:
    """Descriptor for secure string handling"""
    
    def __init__(self, sensitive: bool = False):
        self.sensitive = sensitive
        self.data = {}
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.data.get(id(obj), None)
        if value and self.sensitive:
            return "[REDACTED]"
        return value
    
    def __set__(self, obj, value):
        if not isinstance(value, str):
            raise TypeError("Value must be a string")
        
        if self.sensitive:
            # Store hash for sensitive data
            self.data[id(obj)] = hashlib.sha256(value.encode()).hexdigest()
        else:
            self.data[id(obj)] = value

# Usage
class User:
    name = ValidatedAttribute(type_check=str)
    age = ValidatedAttribute(
        type_check=int,
        validator=lambda x: 0 <= x <= 150
    )
    password = SecureString(sensitive=True)
    
    def __init__(self, name: str, age: int, password: str):
        self.name = name
        self.age = age
        self.password = password
    
    @CachedProperty(ttl_seconds=60)
    def expensive_computation(self):
        """Expensive operation with caching"""
        print("Computing...")
        return sum(i * i for i in range(1000000))

# Usage examples
user = User("Alice", 30, "secret123")
print(user.name)  # "Alice"
print(user.age)   # 30
print(user.password)  # "[REDACTED]"

# Cached property
print(user.expensive_computation)  # Computes
print(user.expensive_computation)  # Returns cached
```

## AST Manipulation

### AST Transformer for Performance Optimization
```python
"""
ast_optimizer.py - AST manipulation for code optimization
"""

import ast
import builtins
from typing import Any, Optional
import inspect

class ConstantFoldingTransformer(ast.NodeTransformer):
    """AST transformer that performs constant folding"""
    
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        # Visit children first
        self.generic_visit(node)
        
        # Check if both sides are constants
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            try:
                # Perform operation at compile time
                if isinstance(node.op, ast.Add):
                    result = node.left.value + node.right.value
                elif isinstance(node.op, ast.Sub):
                    result = node.left.value - node.right.value
                elif isinstance(node.op, ast.Mult):
                    result = node.left.value * node.right.value
                elif isinstance(node.op, ast.Div):
                    result = node.left.value / node.right.value
                else:
                    return node
                
                return ast.Constant(value=result)
            except:
                pass
        
        return node

class LoopUnroller(ast.NodeTransformer):
    """Unroll small loops for performance"""
    
    def visit_For(self, node: ast.For) -> Any:
        self.generic_visit(node)
        
        # Check if iterating over range with small constant
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                args = node.iter.args
                if len(args) == 1 and isinstance(args[0], ast.Constant):
                    count = args[0].value
                    if isinstance(count, int) and 1 <= count <= 10:
                        # Unroll the loop
                        unrolled = []
                        for i in range(count):
                            # Replace loop variable with constant
                            for stmt in node.body:
                                transformed = ReplaceName(node.target.id, i).visit(stmt)
                                unrolled.append(transformed)
                        return unrolled
        
        return node

class ReplaceName(ast.NodeTransformer):
    """Replace variable name with constant value"""
    
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value
    
    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == self.name:
            return ast.Constant(value=self.value)
        return node

def optimize_function(func):
    """Decorator that optimizes a function using AST transformations"""
    
    # Get source code
    source = inspect.getsource(func)
    
    # Parse AST
    tree = ast.parse(source)
    
    # Apply optimizations
    tree = ConstantFoldingTransformer().visit(tree)
    tree = LoopUnroller().visit(tree)
    
    # Fix missing locations
    ast.fix_missing_locations(tree)
    
    # Compile optimized code
    code = compile(tree, '<optimized>', 'exec')
    
    # Extract function from compiled code
    namespace = {}
    exec(code, func.__globals__, namespace)
    
    return namespace[func.__name__]

# Usage
@optimize_function
def expensive_operation(x: int) -> int:
    """This function will be optimized"""
    result = 10 * 5  # Constant folded to 50
    total = 0
    for i in range(5):  # Will be unrolled
        total += i * 2  # Multiplication by constant
    return result + total + x

# Run optimized function
print(expensive_operation(10))
```

## Dynamic Code Generation

### Runtime Class Builder
```python
"""
dynamic_class_builder.py - Create classes dynamically at runtime
"""

from typing import Dict, Any, Type, List, Optional
import types

class DynamicClassBuilder:
    """Factory for creating classes dynamically"""
    
    @staticmethod
    def create_class(class_name: str, 
                     base_classes: List[Type] = None,
                     attributes: Dict[str, Any] = None,
                     methods: Dict[str, callable] = None) -> Type:
        """
        Create a class dynamically
        
        Args:
            class_name: Name of the new class
            base_classes: List of base classes
            attributes: Class attributes (descriptors, properties)
            methods: Instance methods
        """
        base_classes = base_classes or [object]
        attributes = attributes or {}
        methods = methods or {}
        
        # Prepare namespace
        namespace = dict(attributes)
        
        # Add methods
        for method_name, method_func in methods.items():
            namespace[method_name] = method_func
        
        # Create class
        return type(class_name, tuple(base_classes), namespace)
    
    @staticmethod
    def add_method(cls: Type, method_name: str, method_func: callable) -> Type:
        """Add a method to an existing class"""
        setattr(cls, method_name, method_func)
        return cls

class DynamicCodeGenerator:
    """Generate and execute code at runtime"""
    
    @staticmethod
    def execute_dynamic_code(code_string: str, globals_dict: Dict = None) -> Any:
        """Execute dynamically generated code"""
        globals_dict = globals_dict or {}
        exec(code_string, globals_dict)
        return globals_dict
    
    @staticmethod
    def create_function_from_string(function_name: str, 
                                     params: List[str],
                                     body: str) -> callable:
        """Create a function from string definition"""
        
        # Build function code
        param_str = ', '.join(params)
        code = f"""
def {function_name}({param_str}):
    {body}
    return locals()
"""
        # Execute and extract function
        namespace = {}
        exec(code, namespace)
        return namespace[function_name]

# Usage examples
# Create a dynamic class
Person = DynamicClassBuilder.create_class(
    'Person',
    attributes={
        'species': 'Homo sapiens'
    },
    methods={
        '__init__': lambda self, name, age: setattr(self, 'name', name) or setattr(self, 'age', age),
        'greet': lambda self: f"Hello, I'm {self.name}"
    }
)

# Create instance
person = Person("Alice", 30)
print(person.greet())  # "Hello, I'm Alice"
print(person.species)  # "Homo sapiens"

# Create function dynamically
func = DynamicCodeGenerator.create_function_from_string(
    'calculate',
    ['x', 'y'],
    """
result = x * y + 10
print(f"Result: {result}")
    """
)

# Use dynamic function
result = func(5, 3)  # Prints: "Result: 25"
```

## Production Patterns

### Plugin System with Metaclasses
```python
"""
plugin_system.py - Production-ready plugin architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List, Optional
import inspect
import importlib
import pkgutil
from pathlib import Path

class PluginMeta(type):
    """Metaclass for automatic plugin registration"""
    
    plugins: Dict[str, Type] = {}
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> Type:
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip abstract base classes
        if inspect.isabstract(cls):
            return cls
        
        # Register plugin
        if name != 'Plugin' and hasattr(cls, 'plugin_name'):
            plugin_name = cls.plugin_name
            mcs.plugins[plugin_name] = cls
            
            # Add metadata
            cls.plugin_version = getattr(cls, 'plugin_version', '1.0.0')
            cls.plugin_author = getattr(cls, 'plugin_author', 'Unknown')
        
        return cls
    
    @classmethod
    def get_plugin(mcs, name: str) -> Optional[Type]:
        """Get plugin by name"""
        return mcs.plugins.get(name)
    
    @classmethod
    def list_plugins(mcs) -> List[str]:
        """List all registered plugins"""
        return list(mcs.plugins.keys())

class Plugin(ABC, metaclass=PluginMeta):
    """Abstract base class for all plugins"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean shutdown of plugin"""
        pass

class PluginManager:
    """Manages plugin lifecycle and discovery"""
    
    def __init__(self, plugin_dirs: List[Path] = None):
        self.plugin_dirs = plugin_dirs or []
        self._instances: Dict[str, Plugin] = {}
        self._configs: Dict[str, Dict] = {}
    
    def discover_plugins(self, package_name: str = None):
        """Discover plugins in specified packages"""
        if package_name:
            # Discover from package
            package = importlib.import_module(package_name)
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                if not is_pkg:
                    importlib.import_module(f"{package_name}.{name}")
        
        # Discover from directories
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                for py_file in plugin_dir.glob("*.py"):
                    if py_file.name != "__init__.py":
                        spec = importlib.util.spec_from_file_location(
                            py_file.stem, py_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
    
    def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> Plugin:
        """Load and initialize a plugin"""
        plugin_class = PluginMeta.get_plugin(plugin_name)
        if not plugin_class:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        
        # Create instance
        instance = plugin_class()
        
        # Initialize with config
        config = config or {}
        instance.initialize(config)
        
        # Store instance
        self._instances[plugin_name] = instance
        self._configs[plugin_name] = config
        
        return instance
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin"""
        instance = self._instances.get(plugin_name)
        if not instance:
            raise ValueError(f"Plugin '{plugin_name}' not loaded")
        
        return instance.execute(*args, **kwargs)
    
    def unload_plugin(self, plugin_name: str):
        """Unload a plugin"""
        instance = self._instances.pop(plugin_name, None)
        if instance:
            instance.shutdown()
        self._configs.pop(plugin_name, None)
    
    def reload_plugin(self, plugin_name: str) -> Plugin:
        """Reload a plugin with new configuration"""
        config = self._configs.get(plugin_name, {})
        self.unload_plugin(plugin_name)
        return self.load_plugin(plugin_name, config)

# Example plugin implementations
class LoggingPlugin(Plugin):
    """Logging plugin example"""
    
    plugin_name = "logging"
    plugin_version = "1.0.0"
    plugin_author = "System"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        self.log_level = config.get('level', 'INFO')
        self.output = config.get('output', 'stdout')
        print(f"Logging plugin initialized with level: {self.log_level}")
    
    def execute(self, *args, **kwargs) -> Any:
        message = args[0] if args else "No message"
        print(f"[{self.log_level}] {message}")
        return True
    
    def shutdown(self) -> None:
        print("Logging plugin shutting down")

class MetricsPlugin(Plugin):
    """Metrics collection plugin"""
    
    plugin_name = "metrics"
    plugin_version = "1.0.0"
    plugin_author = "System"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        self.metrics = {}
        self.enabled = config.get('enabled', True)
        print("Metrics plugin initialized")
    
    def execute(self, *args, **kwargs) -> Any:
        if not self.enabled:
            return None
        
        metric_name = args[0] if args else "default"
        value = kwargs.get('value', 1)
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        return self.metrics[metric_name]
    
    def shutdown(self) -> None:
        print(f"Metrics plugin shutting down. Collected: {self.metrics}")

# Usage
if __name__ == "__main__":
    # Create plugin manager
    manager = PluginManager()
    
    # Load plugins
    logging_plugin = manager.load_plugin("logging", {"level": "DEBUG"})
    metrics_plugin = manager.load_plugin("metrics", {"enabled": True})
    
    # Execute plugins
    manager.execute_plugin("logging", "Application started")
    manager.execute_plugin("metrics", "requests", value=42)
    
    # List all plugins
    print("Available plugins:", PluginMeta.list_plugins())
    
    # Unload plugin
    manager.unload_plugin("metrics")
```

## Key Takeaways

### When to Use Metaprogramming

✅ **DO Use When:**
- Building frameworks or libraries
- Creating ORMs or serialization systems
- Implementing plugin architectures
- Adding cross-cutting concerns (logging, caching)
- Creating DSLs (Domain Specific Languages)

❌ **AVOID When:**
- Simple CRUD applications
- Performance-critical paths (metaprogramming adds overhead)
- Code that needs to be understood by junior developers
- When static typing is essential

### Performance Considerations

```python
import timeit
from functools import wraps

# Regular function
def regular_function(x):
    return x * 2

# Metaprogramming approach
def create_multiplier(factor):
    @wraps(regular_function)
    def multiplier(x):
        return x * factor
    return multiplier

# Benchmark
setup = "from __main__ import regular_function, create_multiplier"
regular_time = timeit.timeit("regular_function(5)", setup=setup, number=1000000)
dynamic_time = timeit.timeit("create_multiplier(2)(5)", setup=setup, number=1000000)

print(f"Regular: {regular_time:.4f}s")
print(f"Dynamic: {dynamic_time:.4f}s")
```

## 🎯 Practice Exercises

1. **Build a Validation Framework**: Create a system using descriptors that validates data types, ranges, and custom rules

2. **Implement a Cache Decorator**: Build a decorator with TTL, max size, and different eviction policies

3. **Create a Plugin System**: Design a plugin architecture that can discover, load, and manage plugins dynamically

4. **Build a Mini-ORM**: Use metaclasses to create a simple ORM with relationship mapping

5. **AST Optimizer**: Create an AST transformer that optimizes common patterns in mathematical expressions

## 📚 Further Reading

- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [PEP 3115 - Metaclasses in Python 3000](https://www.python.org/dev/peps/pep-3115/)
- [PEP 487 - Simpler customisation of class creation](https://www.python.org/dev/peps/pep-0487/)
- [Descriptors Guide](https://docs.python.org/3/howto/descriptor.html)
- [AST Module Documentation](https://docs.python.org/3/library/ast.html)

## 🚀 Production Checklist

Before using metaprogramming in production:

- [ ] Comprehensive test coverage
- [ ] Performance benchmarks
- [ ] Documentation for complex parts
- [ ] Error handling for edge cases
- [ ] Type hints for better IDE support
- [ ] Logging for debugging
- [ ] Graceful degradation paths
- [ ] Security review for dynamic code
