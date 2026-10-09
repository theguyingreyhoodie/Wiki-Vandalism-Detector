#!/usr/bin/env python3
"""Train vandalism classification model using LightGBM."""

from pathlib import Path
import click
import numpy as np
from rich.console import Console

from src.model.trainer import VandalismModelTrainer
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.option("--experiment", default=None, help="Experiment YAML to override config")
@click.option("--tune/--no-tune", default=None, help="Force enable/disable Optuna hyperparameter tuning")
@click.option("--output-dir", default="models", help="Directory to save trained model artifacts")
def main(config_path, experiment, tune, output_dir):
    """Train the vandalism detector model on engineered train/val splits."""
    setup_logging()
    cfg = load_config(config_path, experiment_path=experiment)

    if tune is not None:
        cfg["tuning"]["enabled"] = tune

    features_dir = Path(cfg["data"]["features_dir"])
    train_npz = features_dir / "train_features.npz"
    val_npz = features_dir / "val_features.npz"

    if not train_npz.exists() or not val_npz.exists():
        console.print(
            f"[bold red]Features not found in {features_dir}![/bold red] "
            "Run build_features.py first."
        )
        return

    console.print(f"[bold cyan]Loading features from:[/bold cyan] {features_dir}")
    train_data = np.load(train_npz, allow_pickle=True)
    val_data = np.load(val_npz, allow_pickle=True)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    feature_names = list(train_data["feature_names"])

    trainer = VandalismModelTrainer(cfg)

    if cfg.get("tuning", {}).get("enabled", False):
        console.print("[bold yellow]Starting Optuna hyperparameter optimization ...[/bold yellow]")
        trainer.tune(X_train, y_train, X_val, y_val, feature_names=feature_names)
    else:
        console.print("[bold green]Starting LightGBM model training ...[/bold green]")
        trainer.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    out_dir = Path(output_dir)
    trainer.save_model(out_dir)
    console.print(f"[bold green]✓ Model training complete![/bold green] Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
