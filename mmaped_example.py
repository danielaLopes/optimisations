import time

from memory_profiler import profile
import numpy as np


N_SAMPLES = 10_000_000
CHUNK_SIZE = 1_000_000

filename = 'mmap_array.dat'

@profile
def create_memmap_array():
    # Create a memory-mapped array
    fp = np.memmap(filename, dtype='float64', mode='w+', shape=(10_000_000,))

    # Fill it with data
    for i in range(N_SAMPLES):
        fp[i] = i

    # Flush to disk
    fp.flush()
    return fp


@profile
def process_memmap_array():
    # Open existing memory-mapped array
    fp = np.memmap(filename, dtype='float64', mode='r', shape=(N_SAMPLES,))

    # Process in chunks
    chunk_size = CHUNK_SIZE
    total = 0
    for i in range(0, len(fp), chunk_size):
        chunk = fp[i:i + chunk_size]
        total += np.sum(chunk)
    return total


if __name__ == "__main__":
    # Create and measure
    start_time = time.time()
    fp = create_memmap_array()
    print(f"Creation time: {time.time() - start_time:.2f} seconds")

    # Process and measure
    start_time = time.time()
    total = process_memmap_array()
    print(f"Processing time: {time.time() - start_time:.2f} seconds")