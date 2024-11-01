import numpy as np
import time
import os
import h5py
from scipy import sparse
import dask.array as da

from utils import log_memory_usage, log_execution_time


def create_large_array(shape, filename):
    """
    Helper function to create a large array

    :param shape:
    :param filename:
    :return:
    """
    arr = np.random.rand(*shape)
    np.save(filename, arr)
    return arr


# 1. Memory-mapped arrays
@log_memory_usage
@log_execution_time
def process_mmap_array(filename, shape):
    start_time = time.time()
    mmap_array = np.load(filename, mmap_mode='r')
    result = np.mean(mmap_array, axis=0)
    end_time = time.time()
    print(f"Memory-mapped method - Mean: {result.mean():.4f}")


# 2. Chunked processing
@log_memory_usage
@log_execution_time
def process_chunked(filename, shape, chunk_size):
    result = np.zeros(shape[1])
    mmap_array = np.load(filename, mmap_mode='r')

    for i in range(0, shape[0], chunk_size):
        chunk = mmap_array[i:i + chunk_size]
        result += np.sum(chunk, axis=0)

    result /= shape[0]

    print(f"Chunked method - Mean: {result.mean():.4f}")


# 3. Out-of-core computation with Dask
@log_memory_usage
@log_execution_time
def process_dask(filename, shape):
    dask_array = da.from_npy_stack(filename)
    result = dask_array.mean(axis=0).compute()

    print(f"Dask method - Mean: {result.mean():.4f}")


# 4. HDF5 for large datasets
@log_memory_usage
@log_execution_time
def process_hdf5(h5_filename, dataset_name, shape):
    with h5py.File(h5_filename, 'r') as f:
        dataset = f[dataset_name]
        result = np.mean(dataset, axis=0)

    print(f"HDF5 method - Mean: {result.mean():.4f}")


# 5. Sparse arrays
@log_memory_usage
@log_execution_time
def process_sparse(shape):
    sparse_matrix = sparse.random(shape[0], shape[1], density=0.01)
    result = sparse_matrix.mean(axis=0)

    print(f"Sparse method - Mean: {result.mean():.4f}")


# Main execution
if __name__ == "__main__":
    shape = (10000, 10000)  # 100 million elements
    filename = "large_array.npy"
    h5_filename = "large_array.h5"
    dataset_name = "large_dataset"

    print(f"Creating large array with shape {shape}...")
    arr = create_large_array(shape, filename)

    print("\nProcessing large NumPy array using different methods:")

    # 1. Memory-mapped arrays
    process_mmap_array(filename, shape)

    # 2. Chunked processing
    process_chunked(filename, shape, chunk_size=1000)

    # 3. Out-of-core computation with Dask
    process_dask(filename, shape)

    # 4. HDF5 for large datasets
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset(dataset_name, data=arr)
    process_hdf5(h5_filename, dataset_name, shape)

    # 5. Sparse arrays
    process_sparse(shape)

    # Clean up
    os.remove(filename)
    os.remove(h5_filename)