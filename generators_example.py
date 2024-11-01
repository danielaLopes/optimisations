import gc
import time
from pathlib import Path

from memory_profiler import profile
import os


N_SAMPLES = 10_000_000

dataset_file = Path('large_numbers.txt')

@profile
def read_numbers_generator() -> int:
    def number_generator():
        with open('large_numbers.txt', 'r') as f:
            for line in f:
                yield int(line.strip())

    # Process numbers using generator
    total = 0
    for num in number_generator():
        total += num

    return total


# The naive approach
@profile
def read_numbers_naive() -> int:
    numbers = []
    with open(dataset_file, 'r') as f:
        for line in f:
            numbers.append(int(line.strip()))
    return sum(numbers)


if __name__ == "__main__":
    # Creating a sample dataset
    with open(dataset_file, 'w') as f:
        for i in range(N_SAMPLES):
            f.write(f"{i}\n")

    # Generator approach
    start_time = time.time()
    total = read_numbers_generator()
    print(f"Total: {total}")
    print(f"Time taken (generators): {time.time() - start_time:.2f} seconds")
    gc.collect()

    # Naive approach
    start_time = time.time()
    total = read_numbers_naive()
    print(f"Total: {total}")
    print(f"Time taken (naive): {time.time() - start_time:.2f} seconds")

    os.remove(dataset_file)