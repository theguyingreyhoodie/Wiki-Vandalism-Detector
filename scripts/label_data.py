#!/usr/bin/env python3
"""Run revert labeler across raw collected edits."""

from pathlib import Path
import click
from rich.console import Console

from src.data.labeler import RevertLabeler
from src.data.storage import load_parquet_dir, save_parquet
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.option("--experiment", default=None, help="Experiment YAML to override config")
@click.option("--input-dir", default=None, help="Input directory of raw Parquets")
@click.option("--output-file", default=None, help="Output labeled Parquet path")
@click.option("--api-backfill/--no-api-backfill", default=False, help="Perform MediaWiki API backfill")
def main(config_path, experiment, input_dir, output_file, api_backfill):
    """Detect reverts, apply time-window delayed labels, and output labeled dataset."""
    setup_logging()
    cfg = load_config(config_path, experiment_path=experiment)

    if api_backfill:
        cfg["labeling"]["use_api_backfill"] = True

    in_dir = input_dir or cfg["data"]["raw_dir"]
    out_path = output_file or Path(cfg["data"]["labeled_dir"]) / "labeled_edits.parquet"

    console.print(f"[bold cyan]Loading raw data from:[/bold cyan] {in_dir}")
    df = load_parquet_dir(in_dir)

    if df.empty:
        console.print("[bold red]No records found![/bold red] Run collection or synthetic generator first.")
        return

    console.print(f"[bold cyan]Running RevertLabeler on {len(df):,} records ...[/bold cyan]")
    labeler = RevertLabeler(cfg)
    labeled_df = labeler.label(df)

    if cfg["labeling"].get("use_api_backfill", False):
        console.print("[bold yellow]Executing API backfill for pending edits ...[/bold yellow]")
        labeled_df = labeler.api_backfill(labeled_df)

    save_parquet(labeled_df, out_path)
    console.print(f"[bold green]✓ Labeled data written to:[/bold green] {out_path}")


if __name__ == "__main__":
    main()
