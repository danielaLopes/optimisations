from __future__ import annotations

import gc
import os
from pathlib import Path
import typing

import numpy as np
import pandas as pd

from df_examples_processing import process_df_non_optimised, process_csv_chunks, process_sql, process_hdf5, \
    process_dask, process_parquet, process_efficient_types
from utils import run_in_process

if typing.TYPE_CHECKING:
    from typing import *


def create_large_csv(
        filename: Union[str, Path],
        hdf_filename: Union[str, Path],
        parquet_filename: Union[str, Path],
        num_rows: int
) -> None:
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

    # Load data into HDF5 file
    with pd.HDFStore(hdf_filename, mode='w') as store:
        for chunk in pd.read_csv(filename, chunksize=100000):
            store.append('large_data', chunk, index=False)

    df.to_parquet(parquet_filename, engine='pyarrow')

    del df


# Main execution
if __name__ == "__main__":
    filename = Path("large_data.csv")

    num_rows = 10_000_000
    chunksize = 100_000

    db_filename = Path("large_data.db")
    hdf_filename = Path("large_data.h5")
    parquet_filename = Path("large_data.parquet")

    dtypes = {
        'id': 'int32',
        'value': 'float32',
        'category': 'category'
    }

    print(f"Creating CSV file with {num_rows} rows...")
    create_large_csv(
        filename,
        hdf_filename,
        parquet_filename,
        num_rows,
    )
    gc.collect()

    print("\nProcessing large dataset using different methods:")

    run_in_process(process_df_non_optimised, filename)
    gc.collect()

    run_in_process(process_csv_chunks, filename)
    gc.collect()

    run_in_process(process_sql, filename, db_filename, chunksize=chunksize)
    gc.collect()

    run_in_process(process_hdf5, hdf_filename, chunksize=chunksize)
    gc.collect()

    run_in_process(process_dask, filename)
    gc.collect()

    run_in_process(process_parquet, parquet_filename)
    gc.collect()

    run_in_process(process_efficient_types, filename, dtypes)
    gc.collect()

    # Clean up
    if filename.exists():
        os.remove(filename)
    if db_filename.exists():
        os.remove(db_filename)
    if hdf_filename.exists():
        os.remove(hdf_filename)
    if parquet_filename.exists():
        os.remove(parquet_filename)


