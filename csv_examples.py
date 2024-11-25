# csv_examples.py
from __future__ import annotations

import csv
import gc
from functools import partial
import mmap
from multiprocessing import Pool
import os
from pathlib import Path
import typing

from utils import log_memory_usage, log_execution_time

if typing.TYPE_CHECKING:
    from typing import *


def generate_large_csv(filename: Union[Path, str], num_rows: int) -> None:
    """
    Helper function to generate a large CSV file

    :param filename:
    :param num_rows:
    :return:
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'value'])
        for i in range(num_rows):
            writer.writerow([i, i * 2])

@log_memory_usage
@log_execution_time
def process_csv_non_optimised(filename: Union[Path, str]) -> None:
    """ Read entire file to memory at once. """
    with open(filename, 'r') as csvfile:
        all_data = list(csv.DictReader(csvfile))

    total = 0

    for row in all_data:
        total += int(row['value'])

    print(f"Non optimised method - Total sum: {total}")

# 1. Chunking
@log_memory_usage
@log_execution_time
def process_csv_in_chunks(filename: Union[Path, str], chunk_size=1000) -> None:
    total = 0
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        chunk = []
        for row in reader:
            chunk.append(int(row['value']))
            if len(chunk) == chunk_size:
                total += sum(chunk)
                chunk = []
        if chunk:
            total += sum(chunk)

    print(f"Chunking method - Total sum: {total}")


# 2. Streaming
@log_memory_usage
@log_execution_time
def process_csv_streaming(filename: Union[Path, str]) -> None:
    total = 0
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total += int(row['value'])

    print(f"Streaming method - Total sum: {total}")


# 3. Memory-mapped files
@log_memory_usage
@log_execution_time
def process_csv_mmap(filename: Union[Path, str]) -> None:
    total = 0
    with open(filename, 'r') as csvfile:
        mm = mmap.mmap(csvfile.fileno(), 0, prot=mmap.PROT_READ)
        for line in iter(mm.readline, b''):
            try:
                value = int(line.decode().split(',')[1])
                total += value
            except ValueError:
                pass  # Skip header or invalid lines

    print(f"Memory-mapped method - Total sum: {total}")


# 4. Distributed processing
def process_chunk(filename: Union[Path, str], start: int, end: int) -> None:
    total = 0
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if start <= i < end:
                total += int(row['value'])
    return total

@log_memory_usage
@log_execution_time
def process_csv_distributed(filename: Union[Path, str], num_processes: int) -> None:
    # Get total number of rows
    with open(filename, 'r') as csvfile:
        num_rows = sum(1 for _ in csvfile) - 1  # Subtract 1 for header

    chunk_size = num_rows // num_processes
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    chunks[-1] = (chunks[-1][0], num_rows)  # Adjust last chunk to include remaining rows

    with Pool(num_processes) as pool:
        results = pool.starmap(partial(process_chunk, filename), chunks)

    total = sum(results)

    print(f"Distributed method - Total sum: {total}")


# Main execution
if __name__ == "__main__":
    filename = "large_data.csv"
    num_rows = 10_000_000

    print(f"Generating CSV file with {num_rows} rows...")
    generate_large_csv(filename, num_rows)
    gc.collect()

    print("\nProcessing large CSV file using different methods:")
    process_csv_non_optimised(filename)
    gc.collect()
    process_csv_in_chunks(filename)
    gc.collect()
    process_csv_streaming(filename)
    gc.collect()
    process_csv_mmap(filename)
    gc.collect()
    process_csv_distributed(filename, num_processes=4)
    gc.collect()

    # Clean up
    os.remove(filename)