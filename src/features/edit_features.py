"""Hand-crafted features from edit metadata.

These features capture structural patterns of vandalism edits without
requiring text embeddings, making them fast to compute and interpretable.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("wiki_vandalism.features.edit")


def extract_edit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive metadata-based features from raw edit columns.

    All features are designed to be computable **at edit time** (no future
    information) and are intentionally cheap to compute so they work at
    stream throughput.
    """
    df = df.copy()

    # ── Size features ────────────────────────────────────────────────
    df["size_delta"] = df["length_new"] - df["length_old"]
    df["size_delta_abs"] = df["size_delta"].abs()
    df["size_ratio"] = np.where(
        df["length_old"] > 0,
        df["length_new"] / df["length_old"],
        np.where(df["length_new"] > 0, 10.0, 1.0),
    )
    df["is_large_deletion"] = (df["size_delta"] < -500).astype(int)
    df["is_blanking"] = (
        (df["length_new"] < 50) & (df["length_old"] > 500)
    ).astype(int)

    # ── Temporal features ────────────────────────────────────────────
    df["hour_of_day"] = (df["timestamp"] % 86400) // 3600
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s").dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── User features ───────────────────────────────────────────────
    # Recognizes IPv4, IPv6, and Wikimedia Temporary Accounts (prefixed with ~)
    is_ipv4 = df["user"].str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$").fillna(False)
    is_ipv6 = df["user"].str.contains(r":", na=False) & df["user"].str.match(r"^[0-9a-fA-F:]+$").fillna(False)
    is_temp_account = df["user"].str.startswith("~").fillna(False)
    df["is_anonymous"] = (is_ipv4 | is_ipv6 | is_temp_account).astype(int)
    df["is_bot"] = df["bot"].astype(int)
    df["is_minor"] = df["minor"].astype(int)
    df["username_length"] = df["user"].str.len().fillna(0)

    # ── Comment features ────────────────────────────────────────────
    df["comment_length"] = df["comment"].str.len().fillna(0)
    df["has_comment"] = (df["comment_length"] > 0).astype(int)
    df["comment_has_url"] = (
        df["comment"].str.contains(r"https?://", na=False).astype(int)
    )

    # ── Namespace features ──────────────────────────────────────────
    df["is_main_namespace"] = (df["namespace"] == 0).astype(int)
    df["is_talk_namespace"] = (df["namespace"] == 1).astype(int)
    df["is_user_namespace"] = (df["namespace"] == 2).astype(int)

    # ── Derived interaction features ────────────────────────────────
    df["comment_to_size_ratio"] = np.where(
        df["size_delta_abs"] > 0,
        df["comment_length"] / df["size_delta_abs"],
        df["comment_length"].clip(upper=1),
    )

    n_feat = len(df.columns)
    logger.info(f"Extracted edit features → {n_feat} total columns")
    return df


def extract_historical_features(
    df: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """Derive rolling historical features per user and per page.

    All aggregations use **expanding windows with shift(1)** so they never
    include the current row (no target leakage).
    """
    df = df.sort_values("timestamp").copy()

    if "label" in df.columns:
        # User's cumulative revert rate (shifted to avoid leakage)
        df["user_revert_rate"] = (
            df.groupby("user")["label"]
            .transform(lambda s: s.expanding().mean().shift(1))
        )
        df["user_revert_rate"] = df["user_revert_rate"].fillna(0.0)

        # Page's cumulative revert rate
        df["page_revert_rate"] = (
            df.groupby("title")["label"]
            .transform(lambda s: s.expanding().mean().shift(1))
        )
        df["page_revert_rate"] = df["page_revert_rate"].fillna(0.0)

    # Cumulative edit counts
    df["user_edit_count"] = df.groupby("user").cumcount()
    df["page_edit_count"] = df.groupby("title").cumcount()

    logger.info("Extracted historical features (user + page)")
    return df
