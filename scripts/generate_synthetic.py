#!/usr/bin/env python3
"""Script to generate realistic synthetic Wikipedia edit data."""

from pathlib import Path
import click
from rich.console import Console

from src.data.synthetic import generate_and_save_synthetic
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.option("--samples", default=None, type=int, help="Number of edit events")
@click.option("--rate", default=None, type=float, help="Vandalism rate (e.g. 0.04)")
@click.option("--days", default=None, type=int, help="Time span in days")
@click.option("--output", default=None, type=str, help="Output parquet file path")
def main(config_path, samples, rate, days, output):
    """Generate synthetic Wikipedia edits for local experimentation."""
    setup_logging()
    cfg = load_config(config_path)

    synth_cfg = cfg.get("synthetic", {})
    n_samples = samples or synth_cfg.get("n_samples", 10000)
    vandalism_rate = rate or synth_cfg.get("vandalism_rate", 0.04)
    time_span = days or synth_cfg.get("time_span_days", 30)

    out_file = output or Path(cfg["data"]["raw_dir"]) / "date=synthetic" / "synthetic_edits.parquet"

    console.print(
        f"[bold green]Generating {n_samples:,} synthetic edits[/bold green] "
        f"(rate={vandalism_rate*100:.1f}%, span={time_span} days) → {out_file}"
    )

    df = generate_and_save_synthetic(
        output_path=out_file,
        n_samples=n_samples,
        vandalism_rate=vandalism_rate,
        time_span_days=time_span,
    )

    console.print(f"[bold green]✓ Complete![/bold green] Total rows (including reverts): {len(df):,}")


if __name__ == "__main__":
    main()
