"""Revert-based labeling for Wikipedia edits.

An edit is labelled **positive** (vandalism) when it is reverted within the
configured time window.  Reverts are detected via:

1. **Comment pattern matching** — regex on the edit summary
   (``Reverted``, ``Undid revision``, ``rollback``, …).
2. **Explicit revision reference** — ``Undid revision <id>`` lets us link the
   revert directly to the vandalised revision.
3. **Fallback heuristic** — a revert edit's ``revision_old`` points to the
   edit it undid.
4. **MediaWiki API back-fill** — queries the revision history for edits whose
   revert status could not be determined from the stream alone.

Edits younger than ``revert_window_hours`` receive ``label=NaN`` (pending).
"""

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("wiki_vandalism.labeler")


class RevertLabeler:
    """Labels edits as *reverted* (vandalism) or *not reverted*."""

    def __init__(self, config: dict):
        lcfg = config["labeling"]
        self.revert_window_sec = lcfg["revert_window_hours"] * 3600
        self.revert_patterns = [
            re.compile(p, re.IGNORECASE) for p in lcfg["revert_patterns"]
        ]
        self.use_api = lcfg.get("use_api_backfill", False)
        self.api_batch_size = lcfg.get("api_batch_size", 50)
        self.api_rate_limit = lcfg.get("api_rate_limit_sec", 1.0)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign revert labels to every edit in *df*.

        Columns added
        --------------
        ``is_revert``          True if this edit **is** a revert of a prior edit.
        ``reverted_revision``  The revision ID that was reverted (if known).
        ``label``              1 = reverted (vandalism), 0 = clean, NaN = pending.
        ``label_timestamp``    UNIX epoch when the label was assigned.
        """
        df = df.copy()

        # Step 1 — identify revert edits by comment pattern
        logger.info("Step 1/3 · Detecting revert edits via comment patterns …")
        df["is_revert"] = df["comment"].apply(self._is_revert_comment)
        df["reverted_revision"] = df.apply(self._extract_reverted_revision, axis=1)
        logger.info(
            f"  Found {df['is_revert'].sum():,} revert edits out of {len(df):,}"
        )

        # Step 2 — mark reverted edits
        logger.info("Step 2/3 · Marking reverted edits …")
        df["label"] = np.nan
        df["label_timestamp"] = np.nan

        reverted_ids = set(
            df.loc[df["reverted_revision"].notna(), "reverted_revision"].astype(int)
        )
        df = self._label_by_revision_match(df, reverted_ids)

        # Step 3 — time-window negatives
        logger.info("Step 3/3 · Applying time-window labels for non-reverted edits …")
        now = time.time()
        old_enough = (now - df["timestamp"]) > self.revert_window_sec
        unlabeled = df["label"].isna()
        df.loc[old_enough & unlabeled, "label"] = 0
        df.loc[old_enough & unlabeled, "label_timestamp"] = now

        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        n_pend = df["label"].isna().sum()
        logger.info(
            f"  Labels: {n_pos:,} positive · {n_neg:,} negative · {n_pend:,} pending"
        )
        return df

    # ------------------------------------------------------------------
    # Comment analysis
    # ------------------------------------------------------------------

    def _is_revert_comment(self, comment: str) -> bool:
        if not comment:
            return False
        return any(p.search(comment) for p in self.revert_patterns)

    def _extract_reverted_revision(self, row: pd.Series) -> float | None:
        if not row.get("is_revert", False):
            return None
        comment = row.get("comment", "")
        if not comment:
            return None

        # "Undid revision 12345678"
        match = re.search(r"[Uu]ndid revision (\d+)", comment)
        if match:
            return float(match.group(1))

        # Fallback: the revert's revision_old is the rev it restored from
        if row.get("revision_old"):
            return float(row["revision_old"])
        return None

    # ------------------------------------------------------------------
    # Labelling helpers
    # ------------------------------------------------------------------

    def _label_by_revision_match(
        self, df: pd.DataFrame, reverted_ids: set[int]
    ) -> pd.DataFrame:
        """Set label=1 for edits whose ``revision_new`` is in *reverted_ids*."""
        mask = df["revision_new"].isin(reverted_ids)
        df.loc[mask, "label"] = 1
        df.loc[mask, "label_timestamp"] = time.time()

        # Heuristic for reverts without an explicit target revision:
        # mark the revision_old of the revert edit as reverted
        reverts_no_target = df[df["is_revert"] & df["reverted_revision"].isna()]
        for _, rev_row in reverts_no_target.iterrows():
            prior = (
                (df["title"] == rev_row["title"])
                & (df["revision_new"] == rev_row["revision_old"])
                & df["label"].isna()
            )
            df.loc[prior, "label"] = 1
            df.loc[prior, "label_timestamp"] = time.time()
        return df

    # ------------------------------------------------------------------
    # Optional API back-fill
    # ------------------------------------------------------------------

    def api_backfill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Query the MediaWiki API for pending edits whose revert status
        could not be determined from the stream alone."""
        if not self.use_api:
            logger.info("API backfill disabled in config")
            return df

        pending = df[df["label"].isna()]
        if pending.empty:
            logger.info("No pending edits to backfill")
            return df

        logger.info(f"API backfill for {len(pending):,} pending edits …")
        api_url = "https://en.wikipedia.org/w/api.php"

        for i, (idx, row) in enumerate(pending.iterrows()):
            if i > 0 and i % self.api_batch_size == 0:
                logger.info(f"  Backfilled {i}/{len(pending)}")
            try:
                params = {
                    "action": "query",
                    "prop": "revisions",
                    "titles": row["title"],
                    "rvstartid": row["revision_new"],
                    "rvlimit": 10,
                    "rvprop": "ids|comment|tags",
                    "format": "json",
                }
                resp = requests.get(api_url, params=params, timeout=10)
                resp.raise_for_status()
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    revisions = page.get("revisions", [])
                    reverted = self._check_revisions_for_revert(
                        row["revision_new"], revisions
                    )
                    df.loc[idx, "label"] = 1 if reverted else 0
                    df.loc[idx, "label_timestamp"] = time.time()
                time.sleep(self.api_rate_limit)
            except requests.exceptions.RequestException as e:
                logger.debug(f"API error for {row['title']}: {e}")
        logger.info("API backfill complete")
        return df

    def _check_revisions_for_revert(
        self, target_rev: int, revisions: list[dict]
    ) -> bool:
        found = False
        for rev in revisions:
            if rev.get("revid") == target_rev:
                found = True
                continue
            if found:
                comment = rev.get("comment", "")
                tags = set(rev.get("tags", []))
                if any(p.search(comment) for p in self.revert_patterns):
                    return True
                if tags & {"mw-revert", "mw-rollback", "mw-undo"}:
                    return True
        return False
