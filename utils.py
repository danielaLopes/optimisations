from __future__ import annotations

import gc
import multiprocessing
import time
import tracemalloc
import typing
import os
from functools import wraps

from memory_profiler import profile
import psutil
import sys

if typing.TYPE_CHECKING:
    from typing import *


MB_SIZE = 1024 * 1024

def get_object_size_mb(obj: Any) -> float:
	"""Get total memory occupied by object in MB"""
	size_bytes = sys.getsizeof(obj)
	return size_bytes / MB_SIZE

def get_process_memory_usage_mb() -> float:
	"""Get total memory occupied by process in MB"""
	process = psutil.Process(os.getpid())
	return process.memory_info().rss / MB_SIZE


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6  # in MB

# def log_memory_usage(func):
#     def wrapper(*args, **kwargs):
#         gc.collect()
#         memory_before = get_process_memory()
#         result = func(*args, **kwargs)
#         memory_after = get_process_memory()
#         gc.collect()
#         print(f"[{func.__name__}] Memory used: {memory_after - memory_before:.2f} MB")
#         return result
#     return wrapper


def log_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[{wrapper.__module__}.{func.__name__}] Peak memory usage: {peak / MB_SIZE:.2f} MB")
        return result

    return wrapper

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[{wrapper.__module__}.{func.__name__}] Execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def run_in_process(func: Callable, *args, **kwargs) -> None:
    """Run a function in a separate process"""
    gc.collect()  # Clean before running
    process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
    process.start()
    process.join()
