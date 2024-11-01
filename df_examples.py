import gc

import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import dask.dataframe as dd

from utils import log_memory_usage, log_execution_time


def create_large_csv(filename, num_rows):
    """
    Helper function to create a large CSV file

    :param filename:
    :param num_rows:
    :return:
    """
    df = pd.DataFrame({
        'id': range(num_rows),
        'value': np.random.randn(num_rows),
        'category': np.random.choice(['A', 'B', 'C'], num_rows)
    })
    df.to_csv(filename, index=False)
    return df


@log_memory_usage
@log_execution_time
def process_csv_chunks(filename, chunksize=100000):
    """
    Chunking with read_csv. We read the CSV file in chunks and process each chunk separately.
    This method can only be used if we don't need to process the whole dataset at the same time.

    + Simple to implement.

    - Only supports pandas operations within each chunk.
    - Can be slow for very large datasets.

    Execution time: 13.31 seconds
    Memory used: 122.29 MB

    :param filename:
    :param chunksize:
    :return:
    """
    total_sum = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        total_sum += chunk['value'].sum()

    print(f"Chunking method - Total sum: {total_sum:.2f}")


@log_memory_usage
@log_execution_time
def process_sql(filename, db_filename):
    """
    SQL databases with SQLAlchemy. We load the data into a SQLite database and use SQL queries to process it, which can
    be more memory-efficient for very large datasets.

    + Leverages SQL optimisations.

    - Requires writing SQL queries.
    - Not as flexible as pandas for data manipulation.

     Execution time: 194.42 seconds
     Memory used: 105.97 MB

    :param filename:
    :param db_filename:
    :return:
    """
    engine = create_engine(f'sqlite:///{db_filename}')

    # Load data into SQLite database
    for chunk in pd.read_csv(filename, chunksize=100000):
        chunk.to_sql('large_data', engine, if_exists='append', index=False)

    # Query the database
    result = pd.read_sql_query("SELECT SUM(value) as total_sum FROM large_data", engine)
    total_sum = result['total_sum'].iloc[0]

    print(f"SQL method - Total sum: {total_sum:.2f}")


@log_memory_usage
@log_execution_time
def process_hdf5(filename, hdf_filename):
    """
    HDF5 with HDFStore. HDF5 file format is designed for storing and managing large and complex datasets efficiently.

    + Fast read/write operations.
    + Supports most pandas operations, including filtering and merging.

    - Not as scalable as Dask for very large datasets.
    - Less efficient than Parquet for column-based operations.

    Execution time: 44.44 seconds
    Memory used: 459.00 MB

    :param filename:
    :param hdf_filename:
    :return:
    """
    # Load data into HDF5 file
    with pd.HDFStore(hdf_filename, mode='w') as store:
        for chunk in pd.read_csv(filename, chunksize=100000):
            store.append('large_data', chunk, index=False)

    # Read and process data
    with pd.HDFStore(hdf_filename, mode='r') as store:
        total_sum = store.select('large_data', columns=['value'])['value'].sum()

    print(f"HDF5 method - Total sum: {total_sum:.2f}")


@log_memory_usage
@log_execution_time
def process_dask(filename):
    """
    Dask DataFrames. Dask is a library for parallel computing, to handle large datasets that don't fit in memory. Most
    appropriate when you need to make operations that require the whole dataset, and to perform pandas-like operations.

    + Supports most pandas operations, including filtering and merging.
    + Scales to datasets much larger than memory.
    + Leverages multi-core processing for faster computations.
    + Lazy evaluation allows for optimized query plans.

    - Not all pandas operations are supported.

    Execution time: 10.88 seconds
    Memory used: 1695.53 MB

    :param filename:
    :return:
    """
    ddf = dd.read_csv(filename)
    total_sum = ddf['value'].sum().compute()

    print(f"Dask method - Total sum: {total_sum:.2f}")

    del ddf


@log_memory_usage
@log_execution_time
def process_parquet(filename, parquet_filename):
    """
    Parquet files. We convert CSV to the Parquet format, which is a columnar storage file format that can be much more
    efficient for certain types of queries.

    + Extremely efficient for column-based operations and filtering.
    + Compact storage format, which can reduce disk I/O.

    - Not as efficient for row-based operations.
    - Merging multiple Parquet files is not straightforward.
    - Requires additional libraries (like pyarrow or fastparquet).

    Execution time: 19.78 seconds
    Memory used: 1099.83 MB

    :param filename:
    :param parquet_filename:
    :return:
    """
    # Convert CSV to Parquet
    df = pd.read_csv(filename)
    df.to_parquet(parquet_filename, engine='pyarrow')

    # Read and process Parquet file
    pq_df = pd.read_parquet(parquet_filename, engine='pyarrow')
    total_sum = pq_df['value'].sum()

    print(f"Parquet method - Total sum: {total_sum:.2f}")

    del df
    del pq_df


@log_memory_usage
@log_execution_time
def process_efficient_types(filename):
    """
    Memory-efficient datatypes. Use appropriate datatypes (e.g., 'category' for categorical data) to reduce memory usage
    when loading the data.

    + Maintains full pandas functionality.
    + Can be combined with other approaches like Dask.

    - Might not be enough if the dataset is too large.
    - Requires careful consideration of data types. If the dtype is to small for the data, might lead to overflow or
    loss of precision.

    Data types:
        * Integers:
            * int8: -128 to 127
            * int16: -32,768 to 32,767
            * int32: -2,147,483,648 to 2,147,483,647
            * int64: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
            * uint8: 0 to 255
            * uint16: 0 to 65,535
            * uint32: 0 to 4,294,967,295
            * uint64: 0 to 18,446,744,073,709,551,615

        * Floats:
            * float16: ±6.1e-05 to ±65504 (half precision)
            * float32: ±1.18e-38 to ±3.4e+38 (single precision)
            * float64: ±2.23e-308 to ±1.80e+308 (double precision)

        * Boolean:
            * bool: True or False (typically uses 1 byte of memory)

        * Categorical:
            * category: Efficient for repeated string values or integers with low cardinality

        * Object:
            * object: Can hold any Python object, but is memory-inefficient

    Execution time: 13.95 seconds
    Memory used: 663.37 MB

    :param filename:
    :return:
    """
    dtypes = {
        'id': 'int32',
        'value': 'float32',
        'category': 'category'
    }

    df = pd.read_csv(filename, dtype=dtypes)
    total_sum = df['value'].sum()

    print(f"Efficient types method - Total sum: {total_sum:.2f}")

    del df


# Main execution
if __name__ == "__main__":
    filename = "large_data.csv"
    num_rows = 100_000_000
    db_filename = "large_data.db"
    hdf_filename = "large_data.h5"
    parquet_filename = "large_data.parquet"

    print(f"Creating CSV file with {num_rows} rows...")
    create_large_csv(filename, num_rows)

    print("\nProcessing large dataset using different methods:")

    process_csv_chunks(filename)
    gc.collect()
    process_sql(filename, db_filename)
    gc.collect()
    process_hdf5(filename, hdf_filename)
    gc.collect()
    process_dask(filename)
    gc.collect()
    process_parquet(filename, parquet_filename)
    gc.collect()
    process_efficient_types(filename)
    gc.collect()

    # Clean up
    os.remove(filename)
    os.remove(db_filename)
    os.remove(hdf_filename)
    os.remove(parquet_filename)


