import gc
import time

import jax.numpy as jnp
from jax import random, jit
import numpy as np


CHUNK_SIZE = int(1e8)

def process_with_jax():
    start_time = time.time()

    # Create and process data using JAX
    key = random.PRNGKey(0)

    @jit
    def process_chunk(key):
        data = random.normal(key, (CHUNK_SIZE,))
        return jnp.sum(data)

    # Process in chunks
    total = 0
    for i in range(10):
        key, subkey = random.split(key)
        total += process_chunk(subkey)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    gc.collect()

    return float(total)


def process_with_numpy():
    start_time = time.time()

    # Set random seed for reproducibility
    np.random.seed(0)

    def process_chunk(size):
        data = np.random.normal(size=size)
        return np.sum(data)

    # Process in chunks
    total = 0
    for i in range(10):
        total += process_chunk(CHUNK_SIZE)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    gc.collect()

    return float(total)


if __name__ == "__main__":
    process_with_jax()
    process_with_numpy()