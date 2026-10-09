"""Structured logging setup using Rich console output."""

import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure structured logging with Rich console output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.

    Returns:
        Configured logger for the wiki_vandalism namespace.
    """
    console = Console(stderr=True)

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
        )
    ]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("wiki_vandalism")
    logger.setLevel(getattr(logging, level.upper()))
    return logger
