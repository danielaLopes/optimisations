# numpy_example.py
import gc
import sys
import time
from memory_profiler import profile
import numpy as np
import os

from utils import MB_SIZE

NUM_ELS = 100_000_000


def create_sample_data(filename, num_rows=NUM_ELS):
    """Create sample data with different data types"""
    # Using full float64 datatypes (memory-intensive)
    data = np.random.randn(num_rows, 4)
    print("data", data)
    np.save(filename, data)

    del data
    gc.collect()


def compare_memory_usage(n=NUM_ELS):
    """Compare memory usage of different NumPy datatypes"""
    # Create arrays with different dtypes
    dtypes = {
        'float64': np.zeros(n, dtype=np.float64),
        'float32': np.zeros(n, dtype=np.float32),
        'float16': np.zeros(n, dtype=np.float16),
        'int64': np.zeros(n, dtype=np.int64),
        'int32': np.zeros(n, dtype=np.int32),
        'int16': np.zeros(n, dtype=np.int16),
        'int8': np.zeros(n, dtype=np.int8),
        'bool': np.zeros(n, dtype=np.bool_),
        'uint8': np.zeros(n, dtype=np.uint8)
    }

    # Calculate and display memory usage
    print("\nMemory usage per dtype (MB):")
    for dtype_name, arr in dtypes.items():
        memory_mb = sys.getsizeof(arr) / MB_SIZE
        print(f"{dtype_name:<8}: {memory_mb:>6.2f} MB")
        del arr

    gc.collect()

# numpy_example.py
@profile
def demonstrate_slice():
    """Demonstrate memory savings using slices"""
    original = np.random.randn(NUM_ELS)

    # Slice the array (no additional memory)
    slice = original[: 1_000]

@profile
def demonstrate_view_vs_copy():
    """Demonstrate memory savings using views vs copies"""
    original = np.random.randn(NUM_ELS)

    # Create a view (no additional memory)
    view = original.view()

    # Create a copy (additional memory)
    copy = original.copy()

    print("\nMemory usage - views vs copies:")
    print(f"Original array: {original.nbytes / (1024 * 1024):.2f} MB")
    print(f"View of array: {view.nbytes / (1024 * 1024):.2f} MB (shares memory)")
    print(f"Copy of array: {copy.nbytes / (1024 * 1024):.2f} MB (separate memory)")

    del original, view, copy
    gc.collect()

@profile
def inefficient_softmax() -> np.array:
    x = np.random.randn(NUM_ELS)
    max_val = np.max(x)
    sub_result = np.subtract(x, max_val)
    exp_result = np.exp(sub_result)
    sum_result = np.sum(exp_result)
    result = np.divide(exp_result, sum_result)
    return result


@profile
def process_without_optimization(filename):
    """Baseline approach with default datatypes"""
    try:
        data = np.load(filename)

        # Calculations using default dtypes (float64)
        result = {
            'mean': data.mean(axis=0),
            'integers': np.floor(data * 100),
            'small_decimals': data * 0.001,
            'binary_flags': data > 0
        }
        return result

    finally:
        del data
        gc.collect()


def main():
    print("\nProcessing softmax without optimization:")
    start_time = time.time()
    inefficient_softmax()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    filename = 'large_array.npy'
    create_sample_data(filename)

    # Compare memory usage of different dtypes
    compare_memory_usage()

    # Demonstrate slice view usage
    demonstrate_slice()

    # Demonstrate view vs copy memory usage
    demonstrate_view_vs_copy()

    gc.collect()

    print("\nProcessing without optimization:")
    start_time = time.time()
    result1 = process_without_optimization(filename)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print("result1", result1)

    # Clean up
    # os.remove(filename)


if __name__ == "__main__":
    main()