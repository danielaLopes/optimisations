import sys
from pympler import asizeof
import numpy as np
from collections import defaultdict


# Basic class with simple attributes
class SimpleClass:
    def __init__(self, x: int, y: str):
        self.x = x
        self.y = y


# Class with nested objects
class NestedClass:
    def __init__(self, name: str):
        self.name = name
        self.data = {
            "values": [1, 2, 3],
            "text": ["hello", "world"]
        }
        self.cache = defaultdict(list)


def compare_sizes(name: str, obj) -> None:
    """Compare and print sizes from different measurement methods"""
    sys_size = sys.getsizeof(obj)
    pympler_size = asizeof.asizeof(obj)

    print(f"\n{name}:")
    print(f"sys.getsizeof:     {sys_size:.4f}")
    print(f"asizeof.asizeof:   {pympler_size:.4f}")


def main():
    # 1. Simple class comparison
    simple = SimpleClass(42, "hello")
    compare_sizes("Simple Class", simple)

    # 2. Nested class comparison
    nested = NestedClass("test")
    for i in range(10):
        nested.cache[f"key_{i}"].extend([i, i + 1, i + 2])
    compare_sizes("Nested Class", nested)

    # 3. List comparison
    list = [1] * 100_000
    compare_sizes("List", list)

    # 4. Numpy arrays comparison
    numpy_arr = np.ones(100_000)
    compare_sizes("Numpy array", numpy_arr)


if __name__ == "__main__":
    main()