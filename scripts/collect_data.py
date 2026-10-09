#!/usr/bin/env python3
"""Run real-time Wikipedia edit stream collection from Wikimedia EventStreams."""

import click
from rich.console import Console

from src.stream.collector import EditStreamCollector
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

console = Console()


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Path to config YAML")
@click.option("--experiment", default=None, help="Experiment YAML to override config")
@click.option("--wiki", default=None, help="Target wiki database (e.g. enwiki)")
@click.option("--max-events", default=None, type=int, help="Stop after N events")
def main(config_path, experiment, wiki, max_events):
    """Collect live edits from Wikimedia EventStreams without an API key."""
    setup_logging()

    overrides = {}
    if wiki or max_events:
        overrides["stream"] = {}
        if wiki:
            overrides["stream"]["wiki"] = wiki
        if max_events:
            overrides["stream"]["max_events"] = max_events

    cfg = load_config(config_path, experiment_path=experiment, overrides=overrides or None)

    console.print(
        f"[bold cyan]Connecting to Wikimedia EventStreams[/bold cyan] "
        f"(wiki={cfg['stream']['wiki']}, max_events={cfg['stream'].get('max_events') or 'unlimited'})"
    )
    console.print("Press [bold yellow]Ctrl+C[/bold yellow] at any time to gracefully stop and flush buffer.\n")

    collector = EditStreamCollector(cfg)
    collector.collect()


if __name__ == "__main__":
    main()
