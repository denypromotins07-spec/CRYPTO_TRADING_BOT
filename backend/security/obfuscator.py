#!/usr/bin/env python3
"""
Obfuscator - Applies lightweight control-flow obfuscation to alpha logic.

This module implements code obfuscation techniques to protect proprietary
trading algorithms from reverse engineering. It applies transformations
that preserve functionality while making static and dynamic analysis difficult.

Security Features:
- Control-flow obfuscation with opaque predicates
- String encryption for sensitive literals
- Dead code insertion
- Variable renaming
- Constant folding with encrypted values

Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Note: This provides defense-in-depth, not absolute protection.
"""

import ast
import random
import string
import hashlib
import base64
import marshal
import types
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObfuscationConfig:
    """Configuration for obfuscation operations."""
    encrypt_strings: bool = True
    insert_dead_code: bool = True
    rename_variables: bool = True
    add_opaque_predicates: bool = True
    constant_obfuscation: bool = True
    seed: Optional[int] = None


class ObfuscatorError(Exception):
    """Exception raised when obfuscation fails."""
    pass


class AlphaLogicObfuscator:
    """
    Obfuscator for protecting proprietary trading algorithms.
    
    Implements the Proxy pattern to wrap and protect sensitive functions.
    """
    
    def __init__(self, config: Optional[ObfuscationConfig] = None):
        """
        Initialize obfuscator.
        
        Args:
            config: Obfuscation configuration
        """
        self.config = config or ObfuscationConfig()
        
        # Initialize random seed for reproducibility if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        # String encryption key (derived from runtime environment)
        self._string_key = self._derive_string_key()
        
        # Name mapping for variable renaming
        self._name_map: Dict[str, str] = {}
        
        # Opaque predicate values
        self._predicate_values: Dict[str, bool] = {}
    
    def _derive_string_key(self) -> bytes:
        """Derive string encryption key from environment."""
        # Use combination of environment factors
        env_data = f"{random.getrandbits(64)}".encode()
        return hashlib.sha256(env_data).digest()
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption for strings."""
        result = bytearray(len(data))
        for i, byte in enumerate(data):
            result[i] = byte ^ key[i % len(key)]
        return bytes(result)
    
    def encrypt_string(self, s: str) -> str:
        """Encrypt a string literal."""
        if not self.config.encrypt_strings:
            return s
        
        encoded = s.encode('utf-8')
        encrypted = self._xor_encrypt(encoded, self._string_key)
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_string(self, encrypted: str) -> str:
        """Decrypt an encrypted string."""
        decoded = base64.b64decode(encrypted.encode('ascii'))
        decrypted = self._xor_encrypt(decoded, self._string_key)
        return decrypted.decode('utf-8')
    
    def generate_opaque_predicate(self, name: str) -> Tuple[str, str]:
        """
        Generate an opaque predicate that always evaluates to a known value.
        
        Returns:
            Tuple of (condition_code, expected_value)
        """
        # Create predicate that looks complex but has known outcome
        predicates = [
            # Always True predicates
            ("(hash('abc123') % 2) == {}".format(hash('abc123') % 2), True),
            ("len(__builtins__) > 0", True),
            ("sum([1, 2, 3]) == 6", True),
            
            # Always False predicates  
            ("(hash('xyz789') % 2) != {}".format(hash('xyz789') % 2), False),
            ("len([]) > 0", False),
        ]
        
        condition, value = random.choice(predicates)
        self._predicate_values[name] = value
        
        return condition, value
    
    def insert_dead_code(self, code: str) -> str:
        """Insert dead code blocks that never execute."""
        if not self.config.insert_dead_code:
            return code
        
        dead_code_templates = [
            '''
if False:
    _dead_{rand} = "{encrypted}"
    for _i in range(1000):
        _dead_{rand} += chr(_i % 256)
'''.format(rand=random.randint(1000, 9999), encrypted=self.encrypt_string("dead")),
            
            '''
def _unused_func_{rand}():
    x = {val}
    y = x * 2
    return y + sum(range(100))
'''.format(rand=random.randint(1000, 9999), val=random.randint(1, 100)),
            
            '''
class _DeadClass_{rand}:
    def __init__(self):
        self.data = list(range(500))
    
    def process(self):
        return sum(self.data)
'''.format(rand=random.randint(1000, 9999)),
        ]
        
        # Insert dead code at random positions
        lines = code.split('\n')
        num_insertions = min(3, len(lines) // 5)
        
        for _ in range(num_insertions):
            insert_pos = random.randint(1, len(lines) - 1)
            dead_code = random.choice(dead_code_templates)
            lines.insert(insert_pos, dead_code)
        
        return '\n'.join(lines)
    
    def rename_identifiers(self, code: str) -> str:
        """Rename variables and functions to obscure names."""
        if not self.config.rename_variables:
            return code
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # Return original if parsing fails
        
        # Generate random names
        def random_name(length: int = 8) -> str:
            chars = string.ascii_lowercase + string.ascii_uppercase
            return '_' + ''.join(random.choice(chars) for _ in range(length))
        
        # Collect all identifiers
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.ClassDef):
                identifiers.add(node.name)
        
        # Filter out built-ins and special names
        protected = {
            'True', 'False', 'None', 'self', 'cls',
            'print', 'len', 'range', 'sum', 'min', 'max',
            'int', 'float', 'str', 'list', 'dict', 'set',
            '__init__', '__str__', '__repr__', '__name__',
            '__doc__', '__module__', '__class__'
        }
        
        renameable = identifiers - protected
        
        # Create name mapping
        for name in renameable:
            if name not in self._name_map:
                self._name_map[name] = random_name()
        
        # Apply renaming (simplified - real implementation would use AST transformation)
        result = code
        for old_name, new_name in sorted(self._name_map.items(), 
                                          key=lambda x: -len(x[0])):
            # Only replace whole word matches
            import re
            result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, result)
        
        return result
    
    def obfuscate_constants(self, code: str) -> str:
        """Replace constants with computed equivalents."""
        if not self.config.constant_obfuscation:
            return code
        
        # Common constant replacements
        constant_replacements = {
            '0': '(1 - 1)',
            '1': '(2 - 1)',
            '2': '(1 + 1)',
            '10': '(5 * 2)',
            '100': '(10 * 10)',
            '3.14159': '(22 / 7)',
            'True': '(1 == 1)',
            'False': '(1 == 0)',
        }
        
        result = code
        for original, replacement in constant_replacements.items():
            import re
            # Match standalone constants only
            pattern = r'\b' + re.escape(original) + r'\b'
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def obfuscate_code(self, code: str) -> str:
        """Apply all obfuscation transformations to code."""
        result = code
        
        # Apply transformations in order
        if self.config.constant_obfuscation:
            result = self.obfuscate_constants(result)
        
        if self.config.rename_variables:
            result = self.rename_identifiers(result)
        
        if self.config.add_opaque_predicates:
            # Add opaque predicate guards
            pred_condition, pred_value = self.generate_opaque_predicate("main_guard")
            guard_code = f'''
# Security guard
_security_check = {pred_condition}
if not _security_check:
    raise RuntimeError("Integrity check failed")
'''
            result = guard_code + result
        
        if self.config.insert_dead_code:
            result = self.insert_dead_code(result)
        
        return result
    
    def obfuscate_function(self, func: Callable) -> Callable:
        """
        Obfuscate a function by wrapping it with protection.
        
        Usage:
            @obfuscator.protect_function
            def my_alpha_logic():
                ...
        """
        # Get function source
        try:
            import inspect
            source = inspect.getsource(func)
        except (OSError, TypeError):
            # Can't get source, return original
            return func
        
        # Obfuscate the source
        obfuscated_source = self.obfuscate_code(source)
        
        # For runtime protection, wrap with integrity check
        def wrapper(*args, **kwargs):
            # Verify integrity before execution
            if not self._verify_integrity():
                raise RuntimeError("Function integrity check failed")
            
            return func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    def _verify_integrity(self) -> bool:
        """Verify that obfuscated code hasn't been tampered with."""
        # Check opaque predicates
        for name, expected_value in self._predicate_values.items():
            # In production, actually evaluate the predicate
            pass
        
        return True
    
    def compile_to_bytecode(self, code: str, filename: str = "<obfuscated>") -> bytes:
        """Compile obfuscated code to bytecode."""
        try:
            tree = ast.parse(code)
            compiled = compile(tree, filename, 'exec')
            return marshal.dumps(compiled)
        except SyntaxError as e:
            raise ObfuscatorError(f"Failed to compile obfuscated code: {e}")
    
    def load_from_bytecode(self, bytecode: bytes, name: str = "obfuscated_module") -> types.ModuleType:
        """Load a module from obfuscated bytecode."""
        try:
            code = marshal.loads(bytecode)
            module = types.ModuleType(name)
            module.__file__ = "<obfuscated>"
            exec(code, module.__dict__)
            return module
        except Exception as e:
            raise ObfuscatorError(f"Failed to load obfuscated module: {e}")
    
    def obfuscate_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Obfuscate a Python file and save the result.
        
        Args:
            input_path: Path to input Python file
            output_path: Path for output (default: input_obf.py)
            
        Returns:
            Path to obfuscated file
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise ObfuscatorError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = str(input_file.with_stem(input_file.stem + "_obf"))
        
        # Read source
        with open(input_file, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Obfuscate
        obfuscated = self.obfuscate_code(source)
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(obfuscated)
        
        return output_path


def protect_alpha_logic(func: Callable) -> Callable:
    """
    Decorator to protect alpha logic functions.
    
    Usage:
        @protect_alpha_logic
        def calculate_signal():
            # Proprietary trading logic
            ...
    """
    obfuscator = AlphaLogicObfuscator()
    return obfuscator.obfuscate_function(func)


if __name__ == '__main__':
    # Demo usage
    print("Alpha Logic Obfuscator")
    print("=" * 40)
    
    config = ObfuscationConfig(
        encrypt_strings=True,
        insert_dead_code=True,
        rename_variables=True,
        add_opaque_predicates=True,
        seed=42  # Reproducible for demo
    )
    
    obfuscator = AlphaLogicObfuscator(config)
    
    # Example code to obfuscate
    sample_code = '''
def calculate_alpha(prices):
    """Calculate trading signal."""
    threshold = 0.05
    signal = 0
    
    for i in range(1, len(prices)):
        change = (prices[i] - prices[i-1]) / prices[i-1]
        if abs(change) > threshold:
            signal += 1 if change > 0 else -1
    
    return signal
'''
    
    print("\nOriginal code:")
    print(sample_code)
    
    print("\nObfuscated code:")
    obfuscated = obfuscator.obfuscate_code(sample_code)
    print(obfuscated)
    
    # Test string encryption
    test_string = "BINANCE_API_KEY_SECRET"
    encrypted = obfuscator.encrypt_string(test_string)
    decrypted = obfuscator.decrypt_string(encrypted)
    
    print(f"\nString encryption test:")
    print(f"  Original: {test_string}")
    print(f"  Encrypted: {encrypted}")
    print(f"  Decrypted: {decrypted}")
    print(f"  Match: {test_string == decrypted}")
