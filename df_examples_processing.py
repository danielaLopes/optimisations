from __future__ import annotations

from pathlib import Path
import typing

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import create_engine

from utils import log_memory_usage, log_execution_time

if typing.TYPE_CHECKING:
    from typing import *


@log_memory_usage
@log_execution_time
def process_df_non_optimised(filename: Union[str, Path]) -> None:
    df = pd.read_csv(filename)
    total_sum = df['value'].sum()

    print(f"Non optimised method - Total sum: {total_sum:.2f}")
    df.info(verbose=False, memory_usage="deep")
    del df


@log_memory_usage
@log_execution_time
def process_csv_chunks(filename: Union[str, Path], chunksize: int = 1000) -> None:
    """
    Chunking with read_csv. We read the CSV file in chunks and process each chunk separately.
    This method can only be used if we don't need to process the whole dataset at the same time.

    + Simple to implement.

    - Only supports pandas operations within each chunk.
    - Can be slow for very large datasets.

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
def process_sql(filename: Union[str, Path], db_filename: Union[str, Path], chunksize: int = 100_000) -> None:
    """
    SQL databases with SQLAlchemy. We load the data into a SQLite database and use SQL queries to process it, which can
    be more memory-efficient for very large datasets.

    + Leverages SQL optimisations.

    - Requires writing SQL queries.
    - Not as flexible as pandas for data manipulation.

    :param filename:
    :param db_filename:
    :return:
    """
    engine = create_engine(f'sqlite:///{db_filename}')

    # Load data into SQLite database
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        chunk.to_sql('large_data', engine, if_exists='append', index=False)

    # Query the database
    result = pd.read_sql_query("SELECT SUM(value) as total_sum FROM large_data", engine)
    total_sum = result['total_sum'].iloc[0]

    print(f"SQL method - Total sum: {total_sum:.2f}")


@log_memory_usage
@log_execution_time
def process_hdf5(hdf_filename: Union[str, Path], chunksize: int = 100_000) -> None:
    """
    HDF5 with HDFStore. HDF5 file format is designed for storing and managing large and complex datasets efficiently.

    + Fast read/write operations.
    + Supports most pandas operations, including filtering and merging.

    - Not as scalable as Dask for very large datasets.
    - Less efficient than Parquet for column-based operations.

    :param filename:
    :param hdf_filename:
    :return:
    """
    # Read and process data
    with pd.HDFStore(hdf_filename, mode='r') as store:
        total_sum = 0
        for chunk in store.select('large_data', chunksize=chunksize, columns=['value']):
            total_sum += chunk['value'].sum()

    print(f"HDF5 method - Total sum: {total_sum:.2f}")


@log_memory_usage
@log_execution_time
def process_dask(filename: Union[str, Path]) -> None:
    """
    Dask DataFrames. Dask is a library for parallel computing, to handle large datasets that don't fit in memory. Most
    appropriate when you need to make operations that require the whole dataset, and to perform pandas-like operations.

    + Supports most pandas operations, including filtering and merging.
    + Scales to datasets much larger than memory.
    + Leverages multi-core processing for faster computations.
    + Lazy evaluation allows for optimized query plans.

    - Not all pandas operations are supported.

    :param filename:
    :return:
    """
    partition_size = '10MB'  # Adjust based on your memory constraints
    ddf = dd.read_csv(
        filename,
        blocksize=partition_size,
    )
    total_sum = ddf['value'].sum().compute()

    print(f"Dask method - Total sum: {total_sum:.2f}")

    del ddf


@log_memory_usage
@log_execution_time
def process_parquet(parquet_filename: Union[str, Path]) -> None:
    """
    Parquet files. We convert CSV to the Parquet format, which is a columnar storage file format that can be much more
    efficient for certain types of queries.

    + Extremely efficient for column-based operations and filtering.
    + Compact storage format, which can reduce disk I/O.

    - Not as efficient for row-based operations.
    - Merging multiple Parquet files is not straightforward.
    - Requires additional libraries (like pyarrow or fastparquet).

    :param filename:
    :param parquet_filename:
    :return:
    """
    # Read and process Parquet file
    pq_df = pd.read_parquet(parquet_filename, engine='pyarrow')
    total_sum = pq_df['value'].sum()

    print(f"Parquet method - Total sum: {total_sum:.2f}")

    del pq_df


@log_memory_usage
@log_execution_time
def process_efficient_types(filename: Union[str, Path], dtypes: Dict[str, Any]) -> None:
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

    :param filename:
    :return:
    """
    df = pd.read_csv(filename, dtype=dtypes, low_memory=True)
    total_sum = df['value'].sum()

    print(f"Efficient types method - Total sum: {total_sum:.2f}")
    df.info(verbose=False, memory_usage="deep")
    del df
