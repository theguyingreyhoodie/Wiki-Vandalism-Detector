"""Concept drift detection using Population Stability Index (PSI).

Monitors numerical and categorical feature distributions between a baseline
(e.g., training split or historical window) and production/target evaluation batches.

PSI Threshold Guidelines:
- PSI < 0.10: No significant distribution change (stable)
- 0.10 <= PSI < 0.20: Moderate drift; monitor closely
- PSI >= 0.20: Significant drift; trigger retraining/alerting
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("wiki_vandalism.model.drift")


def calculate_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) for a continuous variable.

    Args:
        expected: Baseline / reference distribution values.
        actual: Production / target distribution values.
        num_buckets: Number of quantiles for binning.
        epsilon: Small smoothing constant to avoid log(0) or division by zero.

    Returns:
        PSI float value.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Filter out NaNs and infinities
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins from baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Extend edge boundaries to ensure full coverage
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Re-normalize with epsilon
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_value))


class DriftDetector:
    """Detects concept and feature drift across pipeline iterations."""

    def __init__(self, config: dict):
        drift_cfg = config.get("drift", {})
        self.enabled = drift_cfg.get("enabled", True)
        self.psi_threshold = drift_cfg.get("psi_threshold", 0.20)
        self.features_to_monitor = drift_cfg.get("features_to_monitor", [])

    def compute_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: list[str] | None = None,
        report_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compute PSI across features between baseline and current dataframes.

        Args:
            baseline_df: Reference (e.g., train set) dataframe.
            current_df: Current (e.g., test set or streaming window) dataframe.
            features: Optional list of column names to check.
            report_dir: Optional path to save drift report.

        Returns:
            Dict containing per-feature PSI, drift status, and summary counts.
        """
        features = features or self.features_to_monitor
        if not features:
            features = [
                col
                for col in baseline_df.columns
                if col in current_df.columns
                and pd.api.types.is_numeric_dtype(baseline_df[col])
            ]

        feature_psi: dict[str, dict[str, Any]] = {}
        drifted_features: list[str] = []

        for feat in features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            psi = calculate_psi(baseline_df[feat].values, current_df[feat].values)
            is_drifted = psi >= self.psi_threshold
            if is_drifted:
                drifted_features.append(feat)

            feature_psi[feat] = {
                "psi": round(psi, 4),
                "drift_detected": is_drifted,
                "status": "DRIFT" if is_drifted else "STABLE",
            }

        report = {
            "psi_threshold": self.psi_threshold,
            "total_monitored": len(feature_psi),
            "drift_detected_count": len(drifted_features),
            "overall_drift_status": "ALERT" if len(drifted_features) > 0 else "STABLE",
            "features": feature_psi,
        }

        logger.info(
            f"Drift assessment: {len(drifted_features)}/{len(feature_psi)} features drifted "
            f"(threshold={self.psi_threshold})"
        )

        if report_dir:
            out_path = Path(report_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            with open(out_path / "drift_report.json", "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Drift report saved → {out_path / 'drift_report.json'}")

        return report
