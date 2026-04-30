# Agent Guidelines for VGGT-SLAM

This document provides guidelines for AI agents working on the uniception codebase to ensure high code quality and maintainability.

## Core Principles

### 1. Simplicity and Clarity Over Complexity
- **Write the simplest code that solves the problem**
- Prefer explicit, straightforward implementations over "clever" solutions
- If a simpler approach exists, use it
- **YAGNI (You Aren't Gonna Need It)** - Avoid speculative features
- Delete dead code immediately - use version control, not comments

### 2. Write Code and Comments in consideration of Cognative Load
- **Cognative load is defined as the hardness for one to understand and operate on a piece of code.**
- **The code you write will be read by future you many many times, and all effort in making it easily readable saves great effort**
- All comments, variables should be written and named for reducing cognative load. They should clearly define its purpose, such that one can skip comprehending the detail while not affacting understanding other part of the code.

With this purpose, you should:
- Code should be self-documenting when possible
- Code should be separated into logical blocks with spaces
- Comments should be added for each important logical block such that it captures what the block is doing.
- variable names must carry understandable meaning, and they can be long if necessary. Avoid short, single letter names unless in for loops.
- All functions should have docstring that states its purpose, input, and output such that one understands the io format without running your code.
- Type annotations are required for functions.
- Each function should do ONE thing well
- If a function is getting long, break it into smaller helper functions
- Long functions are harder to test, debug, and maintain

### 3. Code Reuse as a Principal to Reduce Complexity and Cognative Load
- **Reuse existing implementation through structural inheritance, extract common functions.** The less code we have, the less code we need to test, maintain, and generate less cognative load. For example:
    - I need to turn quantity A into B, is A and B clearly defined and this conversion is also needed in other places? if so, consider extracting a method.
    - Am I defining a component that may have many implementations that they share the same input/output interface? if so, define a dataclass that makes this interface uniform and easily changable.
    - I want to design a slight variant of a class A, can I reuse some of its helper function or initializations?

## Examples for Reducing Cognative Load

#### Avoid Excessive Nesting
- **Maximum nesting depth**: 3 levels
- Deep nesting makes code hard to follow
- Use early returns, guard clauses, or extract functions to reduce nesting

**Bad Example:**
```python
def process_item(item):
    if item is not None:
        if item.is_valid():
            if item.has_data():
                if item.data.is_processable():
                    # Do work 4 levels deep
                    pass
```

**Good Example:**
```python
def process_item(item):
    if item is None:
        return
    if not item.is_valid():
        return
    if not item.has_data():
        return
    if not item.data.is_processable():
        return

    # Do work at top level
    pass
```

#### No Unnecessary Wrapper Functions
- Avoid functions that just directly call another function without adding value
- Each function should have a clear purpose and add meaningful logic

**Bad Example:**
```python
def run_colmap(config):
    return execute_colmap(config)  # Adds no value
```

**Good Example:**
```python
def run_colmap(config):
    """Run COLMAP with validation and error handling."""
    validate_config(config)
    try:
        result = execute_colmap(config)
        log_completion(result)
        return result
    except ColmapError as e:
        handle_error(e)
        raise
```

## Python-Specific Guidelines

### Style and Conventions
- Follow PEP 8
- **Use type hints on all function signatures**
- Use f-strings for formatting (not `%` or `.format()`)
- List comprehensions over `map`/`filter` when clearer

### Recommended Patterns
- **Context managers for resources** (`with` statements)
- **Dataclasses for simple data structures** (instead of dicts or classes with just `__init__`)
- **Early returns** to reduce nesting and improve readability
- **Guard clauses at function entry** to handle edge cases upfront

**Example - Dataclasses:**
```python
from dataclasses import dataclass

@dataclass
class SceneConfig:
    root: Path
    max_images: int
    use_gpu: bool = False
```

**Example - Generators:**
```python
def process_large_dataset(dataset_path: Path):
    """Process scenes lazily to avoid loading everything into memory."""
    for scene_dir in dataset_path.iterdir():
        if scene_dir.is_dir():
            yield process_scene(scene_dir)
```

**Example - Context Managers:**
```python
# Good - automatic cleanup
with open(file_path, 'r') as f:
    data = f.read()

# Good - custom resource management
with SceneProcessLock(scene_root):
    process_scene(scene_root)
```
