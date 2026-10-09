"""Model evaluation with comprehensive metrics and visualizations.

Focuses on **PR-AUC** as the primary metric because accuracy is
meaningless with heavy class imbalance (~4 % positive rate).

Generates:
* Precision-Recall curve
* ROC curve
* Confusion-matrix heatmap
* Score-distribution histogram (pos vs neg)
* Per-edit inference latency histogram
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("wiki_vandalism.model.evaluate")


class ModelEvaluator:
    """Evaluation suite for vandalism detection models."""

    def __init__(self, config: dict):
        ecfg = config.get("evaluation", {})
        self.primary_metric = ecfg.get("primary_metric", "pr_auc")
        self.threshold_strategy = ecfg.get("threshold_strategy", "f1_optimal")
        self.target_recall = ecfg.get("target_recall", 0.80)
        self.latency_pcts = config.get("latency", {}).get(
            "report_percentiles", [50, 90, 95, 99]
        )

    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        split_name: str = "test",
        report_dir: str | Path | None = None,
        latencies_ms: np.ndarray | None = None,
    ) -> dict:
        """Run the full evaluation suite and return a metrics dict."""
        logger.info(f"Evaluating on {split_name} ({len(y_true):,} samples) …")

        pr_auc = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        threshold = self._find_threshold(y_true, y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        metrics: dict = {
            "split": split_name,
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "n_negative": int((y_true == 0).sum()),
            "positive_rate": float(y_true.mean()),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "f1": float(f1),
            "threshold": float(threshold),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        if latencies_ms is not None:
            lat = {f"p{p}_ms": float(np.percentile(latencies_ms, p)) for p in self.latency_pcts}
            lat["mean_ms"] = float(np.mean(latencies_ms))
            lat["std_ms"] = float(np.std(latencies_ms))
            metrics["latency"] = lat

        logger.info(
            f"  PR-AUC {pr_auc:.4f} │ ROC-AUC {roc_auc:.4f} │ "
            f"F1 {f1:.4f} │ threshold {threshold:.4f}"
        )

        if report_dir:
            rd = Path(report_dir)
            rd.mkdir(parents=True, exist_ok=True)
            with open(rd / f"{split_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            self._plot_pr_curve(y_true, y_prob, pr_auc, rd, split_name)
            self._plot_roc_curve(y_true, y_prob, roc_auc, rd, split_name)
            self._plot_confusion_matrix(cm, rd, split_name)
            self._plot_score_distribution(y_true, y_prob, threshold, rd, split_name)
            if latencies_ms is not None:
                self._plot_latency(latencies_ms, rd, split_name)
            logger.info(f"  Reports saved → {rd}")

        return metrics

    # ------------------------------------------------------------------
    # Threshold selection
    # ------------------------------------------------------------------

    def _find_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)

        if self.threshold_strategy == "f1_optimal":
            f1s = np.where(
                (prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0
            )
            idx = int(np.argmax(f1s))
            return float(thresholds[idx]) if idx < len(thresholds) else 0.5

        if self.threshold_strategy == "precision_at_recall":
            valid = rec >= self.target_recall
            if valid.any():
                idx = np.where(valid)[0]
                best = idx[int(np.argmax(prec[idx]))]
                return float(thresholds[best]) if best < len(thresholds) else 0.5
        return 0.5

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_pr_curve(self, y_true, y_prob, auc, d, name):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(rec, prec, lw=2, color="#2196F3")
        ax.fill_between(rec, prec, alpha=0.10, color="#2196F3")
        ax.axhline(y=y_true.mean(), color="grey", ls="--", label="random baseline")
        ax.set(xlabel="Recall", ylabel="Precision",
               title=f"PR Curve ({name}) — AUC = {auc:.4f}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"{name}_pr_curve.png", dpi=150); plt.close(fig)

    def _plot_roc_curve(self, y_true, y_prob, auc, d, name):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, lw=2, color="#4CAF50")
        ax.fill_between(fpr, tpr, alpha=0.10, color="#4CAF50")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", label="random")
        ax.set(xlabel="FPR", ylabel="TPR",
               title=f"ROC Curve ({name}) — AUC = {auc:.4f}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"{name}_roc_curve.png", dpi=150); plt.close(fig)

    def _plot_confusion_matrix(self, cm, d, name):
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Clean", "Vandalism"],
                    yticklabels=["Clean", "Vandalism"], ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual",
               title=f"Confusion Matrix ({name})")
        fig.tight_layout(); fig.savefig(d / f"{name}_confusion_matrix.png", dpi=150); plt.close(fig)

    def _plot_score_distribution(self, y_true, y_prob, thr, d, name):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="#4CAF50",
                label="Clean", density=True)
        ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="#F44336",
                label="Vandalism", density=True)
        ax.axvline(thr, color="k", ls="--", lw=2, label=f"thr={thr:.3f}")
        ax.set(xlabel="Predicted Probability", ylabel="Density",
               title=f"Score Distribution ({name})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"{name}_score_dist.png", dpi=150); plt.close(fig)

    def _plot_latency(self, lat, d, name):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lat, bins=50, color="#FF9800", alpha=0.7, edgecolor="white")
        for p in self.latency_pcts:
            v = np.percentile(lat, p)
            ax.axvline(v, color="red", ls="--", alpha=0.7)
            ax.text(v, ax.get_ylim()[1] * 0.9, f"p{p}={v:.1f}ms", fontsize=9)
        ax.set(xlabel="Latency (ms)", ylabel="Count",
               title=f"Inference Latency ({name})")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(d / f"{name}_latency.png", dpi=150); plt.close(fig)
