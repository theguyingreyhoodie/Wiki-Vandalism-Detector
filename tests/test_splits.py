import pandas as pd
import pytest

from src.data.splits import TemporalSplitter


def test_temporal_splitter():
    config = {
        "splits": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "gap_hours": 1,
            "strategy": "temporal",
        },
        "data": {
            "splits_dir": "data/splits",
        },
    }

    splitter = TemporalSplitter(config)

    # 100 hourly timestamps
    timestamps = [1000000 + i * 3600 for i in range(100)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "label": [0] * 95 + [1] * 5,
        "revision_new": list(range(100)),
    })

    splits = splitter.split(df)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Temporal ordering verification (no time overlap)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()

    # Gap verification
    val_gap_sec = val_df["timestamp"].min() - train_df["timestamp"].max()
    test_gap_sec = test_df["timestamp"].min() - val_df["timestamp"].max()
    assert val_gap_sec >= 3600
    assert test_gap_sec >= 3600
