from __future__ import annotations

import time
import typing

from utils import get_object_size_mb, get_process_memory_usage_mb

if typing.TYPE_CHECKING:
    from typing import *


N_ELS = 1_000_000

def create_list(n: int) -> List[int]:
    return [i * 2 for i in range(n)]


def create_generator(n: int) -> Generator[int, None, None]:
    for i in range(n):
        yield i * 2


def compare(fn1: Callable[[int], Any], fn2: Callable[[int], Any]) -> None:
    print(f"Memory usage at the beginning: {get_process_memory_usage_mb():.4f} MB")
    start_time = time.time()
    el1 = fn1(N_ELS)
    end_time = time.time()
    print(f"Memory usage after {fn1.__name__}: {get_process_memory_usage_mb():.4f} MB")
    print(f"Memory usage of el1: {get_object_size_mb(el1):.4f} MB")
    print(f"{fn1.__name__} took {end_time - start_time:.4f} seconds to execute")

    start_time = time.time()
    el2 = fn2(N_ELS)
    end_time = time.time()
    print(f"Memory usage after {fn2.__name__}: {get_process_memory_usage_mb():.4f} MB")
    print(f"Memory usage of el1: {get_object_size_mb(el2):.4f} MB")
    print(f"{fn2.__name__} took {end_time - start_time:.4f} seconds to execute")


if __name__ == "__main__":
    compare(create_list, create_generator)