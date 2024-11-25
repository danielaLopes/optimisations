# chunking_pandas_example.py
import gc
import time

import pandas as pd
import numpy as np
from memory_profiler import profile
import os


NUM_ROWS = 10_000_000
CHUNK_SIZE = 1_000


# Create a large sample dataset
def create_sample_data(filename, num_rows=NUM_ROWS) -> None:
    df = pd.DataFrame({
        'id': range(num_rows),
        'value': np.random.randn(num_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], num_rows),
        'timestamp': pd.date_range(start='2024-01-01', periods=num_rows, freq='s')
    })
    df.to_csv(filename, index=False)

    del df
    gc.collect()


@profile
def process_without_chunking(filename):
    try:
        df = pd.read_csv(filename)
        # Simulate some data processing
        result = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
        return result
    finally:
        del df
        gc.collect()


@profile
def process_with_chunking(filename, chunk_size=CHUNK_SIZE):
    # Initialize variables to store running calculations
    value_sum = {}
    value_sum_sq = {}
    counts = {}

    # Process data in chunks
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process each chunk
        for category in chunk['category'].unique():
            category_data = chunk[chunk['category'] == category]['value']

            if category not in value_sum:
                value_sum[category] = 0
                value_sum_sq[category] = 0
                counts[category] = 0

            value_sum[category] += category_data.sum()
            value_sum_sq[category] += (category_data ** 2).sum()
            counts[category] += len(category_data)

    # Calculate final statistics
    results = []
    for category in value_sum.keys():
        mean = value_sum[category] / counts[category]
        variance = (value_sum_sq[category] / counts[category]) - (mean ** 2)
        std = np.sqrt(variance)
        results.append({
            'category': category,
            'mean': mean,
            'std': std,
            'count': counts[category]
        })

    return pd.DataFrame(results)


def main():
    # Create sample data
    filename = 'large_dataset.csv'
    create_sample_data(filename)

    print("Processing without chunking:")
    start_time = time.time()
    result1 = process_without_chunking(filename)
    end_time = time.time()
    print(f"\nTook {end_time - start_time} seconds")
    print("\nResult without chunking:")
    print(result1)

    del result1
    gc.collect()

    print("\nProcessing with chunking:")
    start_time = time.time()
    result2 = process_with_chunking(filename)
    end_time = time.time()
    print(f"\nTook {end_time - start_time} seconds")
    print("\nResult with chunking:")
    print(result2.sort_values('category'))

    # Clean up
    os.remove(filename)


if __name__ == "__main__":
    main()