import gc
import time

from memory_profiler import profile


N_SAMPLES = 10_000

class ExpensiveObject:
    def __init__(self):
        # Simulate expensive object creation
        self.data = [i * i for i in range(10000)]  # Create some memory overhead


@profile
def process_expensive_generator():
    def object_generator():
        for i in range(N_SAMPLES):
            # Create objects one at a time
            obj = ExpensiveObject()
            yield sum(obj.data)

    # Process values using generator
    total = 0
    for value in object_generator():
        total += value
    return total

@profile
def process_expensive_naive():
    def create_objects():
        return [ExpensiveObject() for _ in range(N_SAMPLES)]

    # Create all objects at once
    total = 0
    for obj in create_objects():
        total += sum(obj.data)
    return total

@profile
def read_numbers_generator() -> int:
    def number_generator():
        for i in range(N_SAMPLES):
            yield i * i

    # Process numbers using generator
    total = 0
    for num in number_generator():
        total += num

    return total


# The naive approach
@profile
def read_numbers_naive() -> int:
    total = 0
    for i in range(N_SAMPLES):
        total += i * i
    return total


if __name__ == "__main__":
    # When not to use generators
    # Generator approach
    # start_time = time.time()
    # total = read_numbers_generator()
    # print(f"Total: {total}")
    # print(f"Time taken (generators): {time.time() - start_time:.2f} seconds")
    # gc.collect()
    #
    # # Naive approach
    # start_time = time.time()
    # total = read_numbers_naive()
    # print(f"Total: {total}")
    # print(f"Time taken (naive): {time.time() - start_time:.2f} seconds")
    # gc.collect()

    # When to use generators
    # Generator approach
    start_time = time.time()
    total = process_expensive_generator()
    print(f"Total (generator): {total}")
    print(f"Time taken (generator): {time.time() - start_time:.2f} seconds")
    gc.collect()

    # Naive approach
    start_time = time.time()
    total = process_expensive_naive()
    print(f"Total (naive): {total}")
    print(f"Time taken (naive): {time.time() - start_time:.2f} seconds")
