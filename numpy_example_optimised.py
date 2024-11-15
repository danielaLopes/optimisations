# numpy_example_optimised.py
import gc
import time
from memory_profiler import profile
import numpy as np
import os

NUM_ELS = 100_000_000


def create_sample_data(filename, num_rows=NUM_ELS):
    """Create sample data with different data types"""
    # Using full float64 datatypes (memory-intensive)
    data = np.random.randn(num_rows, 4)
    np.save(filename, data)

    del data
    gc.collect()


@profile
def efficient_softmax() -> np.array:
    x = np.random.randn(NUM_ELS)
    max_val = np.max(x)
    np.subtract(x, max_val, out=x)
    np.exp(x, out=x)
    sum_result = np.sum(x)
    np.divide(x, sum_result, out=x)
    return x


@profile
def process_with_optimization(filename):
    """Optimized approach using appropriate datatypes"""
    try:
        # Load data with specific dtype to reduce memory
        data = np.load(filename).astype(np.float32, copy=False)

        # Preallocate arrays with appropriate dtypes
        n_cols = data.shape[1]
        means = np.zeros(n_cols, dtype=np.float32)
        integers = np.zeros(data.shape, dtype=np.int16)  # Smaller integer type
        small_decimals = np.zeros(data.shape, dtype=np.float16)  # Half-precision
        binary_flags = np.zeros(data.shape, dtype=np.bool_)  # Boolean type

        # Optimize computations with views and in-place operations
        np.mean(data, axis=0, out=means)  # In-place mean calculation

        # Use where and in-place operations instead of direct computation
        np.floor(data * 100, out=integers, casting='unsafe')  # Store directly as integers
        np.multiply(data, 0.001, out=small_decimals)  # In-place multiplication
        np.greater(data, 0, out=binary_flags)  # In-place comparison

        result = {
            'mean': means,
            'integers': integers,
            'small_decimals': small_decimals,
            'binary_flags': binary_flags
        }
        return result

    finally:
        del data
        gc.collect()


def main():
    print("\nProcessing softmax with optimization:")
    start_time = time.time()
    efficient_softmax()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    filename = 'large_array.npy'
    # create_sample_data(filename)

    print("\nProcessing with optimization:")
    start_time = time.time()
    result2 = process_with_optimization(filename)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print("result2", result2)

    # Clean up
    os.remove(filename)


if __name__ == "__main__":
    main()