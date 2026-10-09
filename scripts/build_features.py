#!/usr/bin/env python3
"""Build feature matrices and temporal train/val/test splits."""

from pathlib import Path
import click
import numpy as np
import pandas as pd
from rich.console import Console

from src.data.splits import TemporalSplitter
from src.data.storage import save_parquet
from src.features.pipeline import FeaturePipeline
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.option("--experiment", default=None, help="Experiment YAML to override config")
@click.option("--input-file", default=None, help="Path to labeled_edits.parquet")
def main(config_path, experiment, input_file):
    """Split labeled edits temporally and construct feature matrices."""
    setup_logging()
    cfg = load_config(config_path, experiment_path=experiment)

    in_file = input_file or Path(cfg["data"]["labeled_dir"]) / "labeled_edits.parquet"
    if not Path(in_file).exists():
        console.print(f"[bold red]Labeled file not found:[/bold red] {in_file}")
        return

    console.print(f"[bold cyan]Loading labeled data from:[/bold cyan] {in_file}")
    df = pd.read_parquet(in_file)

    console.print("[bold cyan]Applying temporal split with label-delay gap buffer ...[/bold cyan]")
    splitter = TemporalSplitter(cfg)
    splits = splitter.split(df)
    splitter.save_splits(splits)

    features_dir = Path(cfg["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Fitting feature pipeline on Train split ...[/bold cyan]")
    pipe = FeaturePipeline(cfg)
    X_train, y_train = pipe.fit_transform(splits["train"])
    pipe.save(features_dir)

    console.print("[bold cyan]Transforming Validation split ...[/bold cyan]")
    X_val, y_val = pipe.transform(splits["val"])

    console.print("[bold cyan]Transforming Test split ...[/bold cyan]")
    X_test, y_test = pipe.transform(splits["test"])

    # Save feature matrices and labels as compressed npz files
    for name, X_arr, y_arr in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out_npz = features_dir / f"{name}_features.npz"
        np.savez_compressed(
            out_npz,
            X=X_arr,
            y=y_arr,
            feature_names=np.array(pipe.feature_names),
        )
        console.print(f"  [green]Saved {name} features:[/green] shape={X_arr.shape} → {out_npz}")

    console.print("[bold green]✓ Feature engineering complete![/bold green]")


if __name__ == "__main__":
    main()
