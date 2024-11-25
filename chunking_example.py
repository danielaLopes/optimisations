# chunking_example.py
import gc
import numpy as np
import os
from memory_profiler import profile


SHAPE = (100_000, 100_000)  # Define array dimensions
CHUNK_SIZE = 1_000  # Process 1000 rows at a time


@profile
def process_regular_array():
    """Process data using regular NumPy array loading"""
    print("Processing with regular NumPy array...")
    filename = 'large_array.npy'

    # Create and save large array
    array = np.random.rand(*SHAPE)
    np.save(filename, array)
    del array
    gc.collect()

    # Load and process
    array = np.load(filename)
    result = np.mean(array)

    # Cleanup
    del array
    gc.collect()
    os.remove(filename)
    return result


@profile
def process_chunked_array():
    """Process data in chunks to minimize memory usage"""
    print("\nProcessing with chunked approach...")
    filename = 'large_chunked_array.npy'

    # Create initial empty array file
    temp_array = np.zeros((CHUNK_SIZE, SHAPE[1]), dtype=np.float64)
    np.save(filename, temp_array)
    del temp_array
    gc.collect()

    # Process in chunks
    running_sum = 0
    count = 0

    for start_idx in range(0, SHAPE[0], CHUNK_SIZE):
        # Generate chunk data
        end_idx = min(start_idx + CHUNK_SIZE, SHAPE[0])
        chunk_shape = (end_idx - start_idx, SHAPE[1])
        chunk = np.random.rand(*chunk_shape)

        # Process chunk
        running_sum += np.sum(chunk)
        count += chunk.size

        # Clear memory
        del chunk
        gc.collect()

        # Progress indicator
        print(f"Processed rows {start_idx} to {end_idx}")

    # Calculate final result
    result = running_sum / count

    # Cleanup
    os.remove(filename)
    return result


def compare_approaches():
    result1 = process_regular_array()

    result2 = process_chunked_array()

    print(f"\nRegular array mean: {result1:.6f}")
    print(f"Chunked array mean: {result2:.6f}")
    print(f"Difference: {abs(result1 - result2):.6f}")


if __name__ == "__main__":
    compare_approaches()