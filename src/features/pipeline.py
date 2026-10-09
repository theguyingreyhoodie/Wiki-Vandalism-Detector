"""Feature engineering pipeline orchestration.

Combines metadata features, text embeddings, and historical features into
a single feature matrix ``X`` plus label vector ``y``.  Tracks feature names
so that downstream consumers (model, evaluation, drift) can introspect
which columns correspond to which semantics.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features.edit_features import extract_edit_features, extract_historical_features
from src.features.text_features import TextFeatureExtractor
from src.data.storage import save_parquet

logger = logging.getLogger("wiki_vandalism.features.pipeline")

# Columns that should NOT be used as features
_META_COLS = frozenset({
    "event_id", "timestamp", "user", "title", "comment", "wiki",
    "revision_new", "revision_old", "length_new", "length_old",
    "bot", "minor", "namespace",
    "ingestion_ts", "is_revert", "reverted_revision",
    "label", "label_timestamp",
})


class FeaturePipeline:
    """End-to-end feature engineering: metadata → embeddings → matrix."""

    def __init__(self, config: dict):
        self.config = config
        self.features_dir = Path(config["data"]["features_dir"])
        self.text_extractor = TextFeatureExtractor(config)
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # fit_transform  /  transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit on training data and return ``(X, y)``.

        This fits PCA (if configured) and records the canonical feature
        name list.
        """
        logger.info(f"fit_transform on {len(df):,} edits …")
        t0 = time.time()

        df = extract_edit_features(df)
        df = extract_historical_features(
            df, window=self.config["features"].get("rolling_window_edits", 100)
        )
        df, embeddings = self.text_extractor.extract(df, fit_pca=True)

        # Identify numeric feature columns (everything not in _META_COLS)
        feature_cols = [
            c
            for c in df.columns
            if c not in _META_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.feature_names = feature_cols + [
            f"emb_{i}" for i in range(embeddings.shape[1])
        ]

        X_tab = df[feature_cols].values.astype(np.float32)
        X = np.hstack([X_tab, embeddings])
        y = df["label"].values.astype(np.float32)

        logger.info(
            f"fit_transform done: {X.shape[1]} features, {time.time() - t0:.1f} s"
        )
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform new data using the already-fitted pipeline."""
        logger.info(f"transform on {len(df):,} edits …")
        t0 = time.time()

        df = extract_edit_features(df)
        df = extract_historical_features(
            df, window=self.config["features"].get("rolling_window_edits", 100)
        )
        df, embeddings = self.text_extractor.extract(df, fit_pca=False)

        feature_cols = [c for c in self.feature_names if not c.startswith("emb_")]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X_tab = df[feature_cols].values.astype(np.float32)
        X = np.hstack([X_tab, embeddings])
        y = (
            df["label"].values.astype(np.float32)
            if "label" in df.columns
            else np.zeros(len(df), dtype=np.float32)
        )

        logger.info(f"transform done: {time.time() - t0:.1f} s")
        return X, y

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_names": self.feature_names,
            "pca": self.text_extractor.pca,
        }
        joblib.dump(state, path / "pipeline_state.joblib")
        logger.info(f"Pipeline state saved → {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        state = joblib.load(path / "pipeline_state.joblib")
        self.feature_names = state["feature_names"]
        if state["pca"] is not None:
            self.text_extractor.pca = state["pca"]
        logger.info(f"Pipeline state loaded ← {path}")
