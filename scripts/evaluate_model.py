#!/usr/bin/env python3
"""Evaluate model on test split, measure latency, and evaluate concept drift."""

import time
from pathlib import Path
import click
import numpy as np
import pandas as pd
from rich.console import Console

from src.model.drift import DriftDetector
from src.model.evaluate import ModelEvaluator
from src.model.trainer import VandalismModelTrainer
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.option("--experiment", default=None, help="Experiment YAML to override config")
@click.option("--model-dir", default="models", help="Directory containing saved model")
@click.option("--reports-dir", default="reports", help="Directory to save evaluation reports and plots")
def main(config_path, experiment, model_dir, reports_dir):
    """Run rigorous evaluation on the temporal test set and assess drift."""
    setup_logging()
    cfg = load_config(config_path, experiment_path=experiment)

    features_dir = Path(cfg["data"]["features_dir"])
    test_npz = features_dir / "test_features.npz"
    train_npz = features_dir / "train_features.npz"

    if not test_npz.exists():
        console.print(f"[bold red]Test features not found:[/bold red] {test_npz}")
        return

    m_dir = Path(model_dir)
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Loading model from:[/bold cyan] {m_dir}")
    trainer = VandalismModelTrainer(cfg)
    model = trainer.load_model(m_dir)

    console.print(f"[bold cyan]Loading test features from:[/bold cyan] {test_npz}")
    test_data = np.load(test_npz, allow_pickle=True)
    X_test, y_test = test_data["X"], test_data["y"]
    feature_names = list(test_data["feature_names"])

    # Measure per-edit inference latency
    console.print("[bold cyan]Measuring per-edit inference latency ...[/bold cyan]")
    latencies_ms = []
    # Benchmark sample of edits individually to measure production-like single-edit latency
    benchmark_size = min(500, len(X_test))
    for i in range(benchmark_size):
        single_x = X_test[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(single_x)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)

    # Full batch scoring
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator(cfg)
    metrics = evaluator.evaluate(
        y_true=y_test,
        y_prob=y_prob,
        split_name="test",
        report_dir=rep_dir,
        latencies_ms=latencies_arr,
    )

    console.print("\n[bold green]Test Set Evaluation Summary:[/bold green]")
    console.print(f"  • Samples: {metrics['n_samples']:,} (Positive rate: {metrics['positive_rate']*100:.2f}%)")
    console.print(f"  • [bold]PR-AUC:[/bold] {metrics['pr_auc']:.4f}")
    console.print(f"  • ROC-AUC: {metrics['roc_auc']:.4f}")
    console.print(f"  • F1 Score (optimal thr={metrics['threshold']:.3f}): {metrics['f1']:.4f}")
    if "latency" in metrics:
        console.print(
            f"  • Inference Latency: p50={metrics['latency']['p50_ms']:.2f}ms, "
            f"p95={metrics['latency']['p95_ms']:.2f}ms, p99={metrics['latency']['p99_ms']:.2f}ms"
        )

    # Concept Drift evaluation: baseline (Train) vs target (Test)
    if cfg.get("drift", {}).get("enabled", True) and train_npz.exists():
        console.print("\n[bold cyan]Assessing Concept & Feature Drift (Train vs Test) ...[/bold cyan]")
        train_data = np.load(train_npz, allow_pickle=True)
        X_train = train_data["X"]

        train_df = pd.DataFrame(X_train, columns=feature_names)
        test_df = pd.DataFrame(X_test, columns=feature_names)

        drift_detector = DriftDetector(cfg)
        drift_report = drift_detector.compute_drift(
            baseline_df=train_df,
            current_df=test_df,
            report_dir=rep_dir,
        )

        console.print(
            f"  • Drift Status: [bold]{drift_report['overall_drift_status']}[/bold] "
            f"({drift_report['drift_detected_count']} features flagged)"
        )

    console.print(f"\n[bold green]✓ Reports & plots saved to:[/bold green] {rep_dir}")


if __name__ == "__main__":
    main()
