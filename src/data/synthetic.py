"""Realistic synthetic Wikipedia edit generator.

Generates realistic edit events with realistic distributions matching
English Wikipedia:
- Heavy class imbalance (typically ~3-5% vandalism)
- Delayed revert behavior (edits reverted minutes or hours later)
- Anonymity, bot flags, namespaces, comment summaries
- Distinct edit patterns for normal edits vs various vandalism types:
  (page blanking, link spamming, profanity/insults, large deletions)
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.storage import save_parquet

logger = logging.getLogger("wiki_vandalism.synthetic")

SAMPLE_TITLES = [
    "Machine_learning", "Artificial_intelligence", "World_War_II", "United_States",
    "Python_(programming_language)", "Quantum_computing", "Albert_Einstein",
    "Solar_System", "Climate_change", "Renaissance", "Taylor_Swift",
    "Cristiano_Ronaldo", "Barack_Obama", "DNA", "Global_warming", "Internet",
    "Linux", "Evolution", "COVID-19", "Philosophy"
]

NORMAL_COMMENTS = [
    "Fixed typo in introduction",
    "Updated statistics per 2024 report",
    "Added citation for claim in second paragraph",
    "Cleaned up formatting and infobox markup",
    "Grammar correction",
    "Added internal wiki link",
    "Clarified wording in history section",
    "Copyedit per MOS",
    "Updated external links",
    "Reorganized references section",
    "Removed redundant sentence",
    "Expanded section on early career with reliable sources",
]

VANDALISM_COMMENTS = [
    "",
    "haha",
    "lol",
    "subscribe to my channel",
    "test edit please ignore",
    "check out this cool site http://freemoneyfast.example.com",
    "this article is dumb",
    "i was here 2024",
    "yolo",
    "best person ever",
]

REVERT_COMMENT_TEMPLATES = [
    "Reverted edits by {vandal} (talk) to last version by {clean_user}",
    "Undid revision {rev_id} by {vandal} (talk)",
    "revert vandalism",
    "rvv",
    "rv vandalism by IP",
    "Reverted good faith edits; please provide WP:RS",
    "rollback edit war / disruptive edits",
]

CLEAN_USERS = [
    "ClueBot_NG", "WikipedianEditor", "KnowledgeSeeker42", "HistoryBuff99",
    "ScienceWriter", "GrammarEnthusiast", "NeutralObserver", "AcademicRef"
]


def generate_synthetic_edits(
    n_samples: int = 10000,
    vandalism_rate: float = 0.04,
    time_span_days: int = 30,
    start_timestamp: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic Wikipedia edit stream events."""
    np.random.seed(seed)
    random.seed(seed)

    if start_timestamp is None:
        start_timestamp = time.time() - (time_span_days * 86400)

    # Monotonically increasing timestamps across time_span_days
    raw_times = np.sort(np.random.uniform(0, time_span_days * 86400, size=n_samples))
    timestamps = start_timestamp + raw_times

    edits: list[dict] = []
    reverts_to_inject: list[dict] = []

    current_rev_id = 100000000
    page_lengths = {title: random.randint(15000, 80000) for title in SAMPLE_TITLES}

    for i in range(n_samples):
        ts = timestamps[i]
        title = random.choice(SAMPLE_TITLES)
        current_len = page_lengths[title]
        rev_id = current_rev_id + i

        is_vandalism = (random.random() < vandalism_rate)
        is_bot = False
        is_anon = False
        is_minor = False

        if is_vandalism:
            # Vandalism characteristics: often anonymous IP, erratic size deltas
            is_anon = (random.random() < 0.75)
            user = (
                f"{random.randint(24, 210)}.{random.randint(10, 255)}."
                f"{random.randint(10, 255)}.{random.randint(1, 254)}"
                if is_anon
                else f"Vandal_{random.randint(100, 999)}"
            )

            v_type = random.choice(["blanking", "spam", "nonsense", "deletion"])
            if v_type == "blanking":
                new_len = random.randint(0, 30)
                comment = ""
            elif v_type == "deletion":
                new_len = max(50, current_len - random.randint(800, 5000))
                comment = random.choice(VANDALISM_COMMENTS)
            elif v_type == "spam":
                new_len = current_len + random.randint(50, 400)
                comment = "added link http://promopage.example.org"
            else:
                new_len = current_len + random.randint(-50, 50)
                comment = random.choice(VANDALISM_COMMENTS)

            old_len = current_len

            # Schedule a revert edit shortly after (between 30 seconds and 3 hours later)
            revert_delay = random.uniform(30, 3 * 3600)
            revert_ts = ts + revert_delay
            clean_user = random.choice(CLEAN_USERS)
            revert_comment = random.choice(REVERT_COMMENT_TEMPLATES).format(
                vandal=user,
                clean_user=clean_user,
                rev_id=rev_id,
            )

            reverts_to_inject.append({
                "target_rev_id": rev_id,
                "revert_ts": revert_ts,
                "title": title,
                "user": clean_user,
                "comment": revert_comment,
                "restored_length": old_len,
            })
        else:
            # Clean edit characteristics
            is_bot = (random.random() < 0.15)
            is_anon = (random.random() < 0.20) if not is_bot else False
            is_minor = (random.random() < 0.35)

            if is_bot:
                user = f"Bot_{random.choice(['Archive', 'Clue', 'Task', 'Cleaner'])}"
            elif is_anon:
                user = (
                    f"{random.randint(24, 210)}.{random.randint(10, 255)}."
                    f"{random.randint(10, 255)}.{random.randint(1, 254)}"
                )
            else:
                user = random.choice(CLEAN_USERS)

            delta = int(np.random.normal(loc=15, scale=120))
            old_len = current_len
            new_len = max(500, old_len + delta)
            page_lengths[title] = new_len
            comment = random.choice(NORMAL_COMMENTS)

        edits.append({
            "event_id": f"enwiki_{rev_id}",
            "timestamp": ts,
            "user": user,
            "bot": is_bot,
            "minor": is_minor,
            "title": title,
            "namespace": 0,
            "comment": comment,
            "revision_new": rev_id,
            "revision_old": rev_id - 1,
            "length_new": new_len,
            "length_old": old_len,
            "wiki": "enwiki",
            "ingestion_ts": ts + random.uniform(0.1, 0.8),
        })

    # Now create the actual revert edits in the stream
    rev_counter = current_rev_id + n_samples + 1
    for rev_item in reverts_to_inject:
        rev_id = rev_counter
        rev_counter += 1
        edits.append({
            "event_id": f"enwiki_{rev_id}",
            "timestamp": rev_item["revert_ts"],
            "user": rev_item["user"],
            "bot": "Bot" in rev_item["user"],
            "minor": True,
            "title": rev_item["title"],
            "namespace": 0,
            "comment": rev_item["comment"],
            "revision_new": rev_id,
            "revision_old": rev_item["target_rev_id"],
            "length_new": rev_item["restored_length"],
            "length_old": 0,
            "wiki": "enwiki",
            "ingestion_ts": rev_item["revert_ts"] + random.uniform(0.1, 0.5),
        })

    df = pd.DataFrame(edits)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"Generated {len(df):,} synthetic edits with ~{vandalism_rate*100:.1f}% positive target rate "
        f"and {len(reverts_to_inject)} injected revert events"
    )
    return df


def generate_and_save_synthetic(
    output_path: str | Path,
    n_samples: int = 10000,
    vandalism_rate: float = 0.04,
    time_span_days: int = 30,
) -> pd.DataFrame:
    """Generate synthetic edits and save to Parquet."""
    df = generate_synthetic_edits(
        n_samples=n_samples,
        vandalism_rate=vandalism_rate,
        time_span_days=time_span_days,
    )
    save_parquet(df, output_path)
    return df
