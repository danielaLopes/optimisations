import gc

import numpy as np
import os
from memory_profiler import profile

# Define array dimensions
shape = (10000, 10000)


@profile
def create_regular_array():
    print("Creating regular NumPy array...")
    filename = 'large_regular_array.npy'
    array = np.random.rand(*shape)  # Create a large array
    print(f"Regular array created with shape: {array.shape}")
    np.save(filename, array)
    del array
    gc.collect()
    array = np.load(filename)
    _ = array[:100, :100]

    # Cleanup
    del array
    gc.collect()
    os.remove(filename)

@profile
def create_memory_mapped_array():
    print("\nCreating memory-mapped NumPy array...")
    filename = 'large_mmaped_array.npy'
    array_mmap = np.memmap(filename, dtype='float64', mode='w+', shape=shape)
    array_mmap[:] = np.random.rand(*shape)  # Write data to memory-mapped array
    array_mmap.flush()
    print("Memory-mapped array created and written to disk.")
    del array_mmap
    gc.collect()
    array_mmap = np.memmap(filename, dtype='float64', mode='r', shape=shape)
    # Access a small part of the array to simulate partial loading into memory
    _ = array_mmap[:100, :100]  # Access part of the array to load it into memory
    print("Accessed a small portion of the memory-mapped array.")

    # Cleanup
    del array_mmap
    gc.collect()
    os.remove(filename)


if __name__ == "__main__":
    create_regular_array()
    create_memory_mapped_array()