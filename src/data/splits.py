"""Temporal train / validation / test splitting.

Unlike random splitting, temporal splits ensure no future data leaks into
training.  A configurable **gap** between splits prevents label leakage from
the delayed-labelling window.

    ┌─── train ───┐ gap ┌── val ──┐ gap ┌── test ──┐
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓      ▓▓▓▓▓▓▓▓▓      ▓▓▓▓▓▓▓▓▓▓
    t_min                                        t_max
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.storage import save_parquet

logger = logging.getLogger("wiki_vandalism.splits")


class TemporalSplitter:
    """Creates temporal train/val/test splits with gap periods."""

    def __init__(self, config: dict):
        scfg = config["splits"]
        self.train_ratio = scfg["train_ratio"]
        self.val_ratio = scfg["val_ratio"]
        self.test_ratio = scfg["test_ratio"]
        self.gap_hours = scfg["gap_hours"]
        self.strategy = scfg.get("strategy", "temporal")
        self.splits_dir = Path(config["data"]["splits_dir"])

    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> dict[str, pd.DataFrame]:
        """Split *df* temporally into train / val / test.

        Rows with ``label=NaN`` (pending) are dropped before splitting.
        Rows that fall into gap windows are discarded.

        Returns:
            ``{"train": …, "val": …, "test": …}``
        """
        df = df.dropna(subset=["label"]).copy()
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        if df.empty:
            raise ValueError("No labelled data available for splitting")

        t_min = df[timestamp_col].min()
        t_max = df[timestamp_col].max()
        total_span = t_max - t_min
        gap_sec = self.gap_hours * 3600

        usable_span = total_span - 2 * gap_sec
        if usable_span <= 0:
            logger.warning(
                f"Data span ({total_span / 3600:.1f} h) < 2 × gap "
                f"({2 * gap_sec / 3600:.1f} h).  Splitting without gaps."
            )
            gap_sec = 0
            usable_span = total_span

        train_end = t_min + usable_span * self.train_ratio
        val_start = train_end + gap_sec
        val_end = val_start + usable_span * self.val_ratio
        test_start = val_end + gap_sec

        splits = {
            "train": df[df[timestamp_col] <= train_end].copy(),
            "val": df[
                (df[timestamp_col] >= val_start) & (df[timestamp_col] <= val_end)
            ].copy(),
            "test": df[df[timestamp_col] >= test_start].copy(),
        }

        for name, sdf in splits.items():
            n_pos = (sdf["label"] == 1).sum()
            n_neg = (sdf["label"] == 0).sum()
            rate = n_pos / len(sdf) * 100 if len(sdf) else 0
            logger.info(
                f"  {name:>5s}: {len(sdf):>7,} edits │ "
                f"{n_pos:>5,} pos ({rate:.1f}%) │ {n_neg:>5,} neg"
            )

        gap_n = len(df) - sum(len(s) for s in splits.values())
        logger.info(f"    gap: {gap_n:>7,} edits discarded (label-leakage buffer)")
        return splits

    # ------------------------------------------------------------------

    def save_splits(self, splits: dict[str, pd.DataFrame]) -> None:
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        for name, sdf in splits.items():
            save_parquet(sdf, self.splits_dir / f"{name}.parquet")

    def load_splits(self) -> dict[str, pd.DataFrame]:
        splits: dict[str, pd.DataFrame] = {}
        for name in ("train", "val", "test"):
            p = self.splits_dir / f"{name}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"Split not found: {p}")
            splits[name] = pd.read_parquet(p)
            logger.info(f"Loaded {name} split: {len(splits[name]):,} rows")
        return splits
