"""Wikimedia EventStreams SSE collector.

Connects to the Wikimedia EventStreams endpoint and collects edit events
in real time.  Events are buffered in memory and flushed to date-partitioned
Parquet files on disk.

Wikimedia docs: https://stream.wikimedia.org/?doc
No API key required.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sseclient import SSEClient

logger = logging.getLogger("wiki_vandalism.collector")


class EditStreamCollector:
    """Collects Wikipedia edit events from Wikimedia EventStreams (SSE)."""

    # Schema of the flat record we persist
    SCHEMA_COLUMNS = [
        "event_id", "timestamp", "user", "bot", "minor",
        "title", "namespace", "comment",
        "revision_new", "revision_old",
        "length_new", "length_old",
        "wiki", "ingestion_ts",
    ]

    def __init__(self, config: dict):
        stream_cfg = config["stream"]
        self.url = stream_cfg["url"]
        self.wiki_filter = stream_cfg["wiki"]
        self.event_type = stream_cfg["event_type"]
        self.buffer_size = stream_cfg["buffer_size"]
        self.flush_interval = stream_cfg["flush_interval_sec"]
        self.max_events = stream_cfg.get("max_events")
        self.raw_dir = Path(config["data"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list[dict] = []
        self._total_collected = 0
        self._last_flush = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self) -> None:
        """Connect to EventStreams and collect edits until interrupted."""
        logger.info(
            f"Connecting to EventStreams: {self.url} "
            f"(wiki={self.wiki_filter}, buffer={self.buffer_size})"
        )

        try:
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            client = SSEClient(response)

            for event in client.events():
                if self.max_events and self._total_collected >= self.max_events:
                    logger.info(f"Reached max events limit: {self.max_events}")
                    break

                if event.event != "message":
                    continue

                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                if data.get("type") != self.event_type:
                    continue
                if data.get("wiki") != self.wiki_filter:
                    continue

                record = self._parse_event(data)
                if record:
                    self._buffer.append(record)
                    self._total_collected += 1

                    if self._total_collected % 100 == 0:
                        logger.info(
                            f"Collected {self._total_collected} edits "
                            f"(buffer: {len(self._buffer)})"
                        )

                if self._should_flush():
                    self._flush()

        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        except requests.exceptions.RequestException as e:
            logger.error(f"Stream connection error: {e}")
        finally:
            if self._buffer:
                self._flush()
            logger.info(f"Collection complete. Total edits: {self._total_collected}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_event(self, data: dict) -> dict | None:
        """Flatten an SSE event into a tabular record."""
        try:
            return {
                "event_id": f"{data['wiki']}_{data['revision']['new']}",
                "timestamp": data["timestamp"],
                "user": data.get("user", ""),
                "bot": data.get("bot", False),
                "minor": data.get("minor", False),
                "title": data.get("title", ""),
                "namespace": data.get("namespace", 0),
                "comment": data.get("comment", ""),
                "revision_new": data["revision"]["new"],
                "revision_old": data["revision"].get("old", 0),
                "length_new": data.get("length", {}).get("new", 0),
                "length_old": data.get("length", {}).get("old", 0),
                "wiki": data["wiki"],
                "ingestion_ts": time.time(),
            }
        except (KeyError, TypeError) as e:
            logger.debug(f"Skipping malformed event: {e}")
            return None

    def _should_flush(self) -> bool:
        if len(self._buffer) >= self.buffer_size:
            return True
        if time.time() - self._last_flush >= self.flush_interval:
            return True
        return False

    def _flush(self) -> None:
        """Write the in-memory buffer to a date-partitioned Parquet file."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        partition_dir = self.raw_dir / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        filename = f"edits_{int(time.time() * 1000)}.parquet"
        filepath = partition_dir / filename

        df.to_parquet(filepath, engine="pyarrow", index=False)

        logger.info(f"Flushed {len(self._buffer)} edits → {filepath}")
        self._buffer.clear()
        self._last_flush = time.time()
