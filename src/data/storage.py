"""Parquet-based data storage utilities.

All pipeline stages read and write Parquet.  This module centralises the
I/O logic so that serialisation format, compression, and partitioning
can be changed in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("wiki_vandalism.storage")


def load_parquet_dir(
    directory: str | Path,
    glob_pattern: str = "**/*.parquet",
) -> pd.DataFrame:
    """Load all Parquet files in *directory* into a single DataFrame.

    Args:
        directory: Root directory to scan.
        glob_pattern: Glob to match Parquet files (recursive by default).

    Returns:
        Concatenated DataFrame, or an empty DataFrame when nothing is found.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob_pattern))

    if not files:
        logger.warning(f"No Parquet files found in {directory}")
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} records from {len(files)} file(s) in {directory}")
    return df


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    partition_col: str | None = None,
) -> None:
    """Save a DataFrame to Parquet.

    Args:
        df: Data to persist.
        path: Destination file (or directory when partitioning).
        partition_col: Optional column name to partition by.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if partition_col and partition_col in df.columns:
        df.to_parquet(path, engine="pyarrow", index=False, partition_cols=[partition_col])
    else:
        df.to_parquet(path, engine="pyarrow", index=False)

    logger.info(f"Saved {len(df):,} records → {path}")
