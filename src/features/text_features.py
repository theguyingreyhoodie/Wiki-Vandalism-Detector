"""Text embedding features using a small transformer model.

Uses ``sentence-transformers`` to encode edit comments (and optionally
diffs) into dense vectors, then optionally reduces dimensionality with PCA.

The default model ``all-MiniLM-L6-v2`` (22 M params, 384-dim output) is
chosen for its speed/quality trade-off — fast enough for per-edit inference
while still capturing semantic similarity.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger("wiki_vandalism.features.text")


class TextFeatureExtractor:
    """Extracts dense embedding features from edit comments."""

    def __init__(self, config: dict):
        fcfg = config["features"]
        self.model_name = fcfg["embedding_model"]
        self.embedding_dim = fcfg["embedding_dim"]
        self.pca_components = fcfg.get("pca_components")
        self._model = None
        self._pca: Optional[PCA] = None

    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} …")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded ✓")

    # ------------------------------------------------------------------

    def extract(
        self,
        df: pd.DataFrame,
        text_col: str = "comment",
        batch_size: int = 256,
        fit_pca: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Encode *text_col* and return a dense embedding matrix.

        Args:
            df: Source DataFrame.
            text_col: Column to embed.
            batch_size: Inference batch size.
            fit_pca: ``True`` to fit PCA (training); ``False`` to re-use a
                previously fitted PCA (val / test).

        Returns:
            ``(df, embeddings)`` where *embeddings* has shape ``(n, dim)``.
        """
        self._load_model()

        texts = df[text_col].fillna("").tolist()

        logger.info(f"Encoding {len(texts):,} texts with {self.model_name} …")
        t0 = time.time()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        elapsed = time.time() - t0
        ms_per_edit = elapsed / max(len(texts), 1) * 1000
        logger.info(
            f"Encoding done: {elapsed:.1f} s total, {ms_per_edit:.1f} ms/edit"
        )

        # Optional PCA reduction
        if self.pca_components and self.pca_components < embeddings.shape[1]:
            if fit_pca:
                logger.info(
                    f"Fitting PCA: {embeddings.shape[1]} → {self.pca_components} dims"
                )
                self._pca = PCA(
                    n_components=self.pca_components, random_state=42
                )
                embeddings = self._pca.fit_transform(embeddings)
                ev = self._pca.explained_variance_ratio_.sum()
                logger.info(f"PCA explained variance: {ev:.3f}")
            elif self._pca is not None:
                embeddings = self._pca.transform(embeddings)
            else:
                logger.warning(
                    "PCA requested but not fitted; returning full embeddings"
                )

        return df, embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # PCA state management (for pipeline serialisation)
    # ------------------------------------------------------------------

    @property
    def pca(self) -> Optional[PCA]:
        return self._pca

    @pca.setter
    def pca(self, pca_model: PCA) -> None:
        self._pca = pca_model
