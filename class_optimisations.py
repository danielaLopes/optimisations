import time
import sys
from dataclasses import dataclass
from pympler import asizeof

N_ITERS = 100_000_000


# Baseline
class Empty:
    pass


# Regular class
class RegularPerson:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


# Data class
@dataclass
class DataPerson:
    name: str
    age: int
    email: str


# Optimized class with slots
class SlottedPerson:
    __slots__ = ['name', 'age', 'email']

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


def compare_classes():
    args = {"name": "John", "age": 30, "email": "john@example.com"}

    empty = Empty()
    print(f"Empty class size: {asizeof.asizeof(empty):.4f} bytes")

    regular = RegularPerson(**args)
    data = DataPerson(**args)
    slotted = SlottedPerson(**args)

    print(f"Regular class size: {asizeof.asizeof(regular):.4f} bytes")
    print(f"Data class size: {asizeof.asizeof(data):.4f} bytes")
    print(f"Slotted class size: {asizeof.asizeof(slotted):.4f} bytes")

    # Performance comparison
    start_time = time.time()
    regular_objects = [RegularPerson(**args) for _ in range(N_ITERS)]
    regular_time = time.time() - start_time

    start_time = time.time()
    data_objects = [DataPerson(**args) for _ in range(N_ITERS)]
    data_time = time.time() - start_time

    start_time = time.time()
    slotted_objects = [SlottedPerson(**args) for _ in range(N_ITERS)]
    slotted_time = time.time() - start_time

    print(f"\nTime to create {N_ITERS} objects:")
    print(f"Regular class: {regular_time:.4f} seconds")
    print(f"Data class: {data_time:.4f} seconds")
    print(f"Slotted class: {slotted_time:.4f} seconds")

if __name__ == "__main__":
    compare_classes()