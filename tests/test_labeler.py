import time
import pandas as pd
import pytest

from src.data.labeler import RevertLabeler


@pytest.fixture
def sample_config():
    return {
        "labeling": {
            "revert_window_hours": 48,
            "revert_patterns": [
                "Reverted",
                "Undid revision",
                "rollback",
                "rv ",
                "rvv",
                "revert",
            ],
            "use_api_backfill": False,
        }
    }


def test_revert_comment_identification(sample_config):
    labeler = RevertLabeler(sample_config)
    assert labeler._is_revert_comment("Reverted edits by TestUser")
    assert labeler._is_revert_comment("Undid revision 1234567")
    assert labeler._is_revert_comment("rvv - blatant spam")
    assert not labeler._is_revert_comment("Fixed typo in introduction")
    assert not labeler._is_revert_comment("Added citation")


def test_revert_labeling_flow(sample_config):
    labeler = RevertLabeler(sample_config)
    now = time.time()
    three_days_ago = now - (72 * 3600)

    # 1 vandal edit, 1 revert of that edit, 1 clean edit
    data = [
        {
            "event_id": "enwiki_101",
            "timestamp": three_days_ago,
            "user": "192.168.1.1",
            "bot": False,
            "minor": False,
            "title": "Python",
            "namespace": 0,
            "comment": "haha deleted all",
            "revision_new": 101,
            "revision_old": 100,
            "length_new": 10,
            "length_old": 5000,
            "wiki": "enwiki",
            "ingestion_ts": three_days_ago,
        },
        {
            "event_id": "enwiki_102",
            "timestamp": three_days_ago + 300,
            "user": "ClueBot_NG",
            "bot": True,
            "minor": True,
            "title": "Python",
            "namespace": 0,
            "comment": "Undid revision 101 by 192.168.1.1 (talk)",
            "revision_new": 102,
            "revision_old": 100,
            "length_new": 5000,
            "length_old": 10,
            "wiki": "enwiki",
            "ingestion_ts": three_days_ago + 300,
        },
        {
            "event_id": "enwiki_103",
            "timestamp": three_days_ago + 600,
            "user": "GoodEditor",
            "bot": False,
            "minor": False,
            "title": "Python",
            "namespace": 0,
            "comment": "Added section on asyncio",
            "revision_new": 103,
            "revision_old": 102,
            "length_new": 5800,
            "length_old": 5000,
            "wiki": "enwiki",
            "ingestion_ts": three_days_ago + 600,
        },
    ]

    df = pd.DataFrame(data)
    labeled = labeler.label(df)

    # Edit 101 was reverted -> label 1
    rev101_row = labeled[labeled["revision_new"] == 101].iloc[0]
    assert rev101_row["label"] == 1

    # Edit 103 is older than 48h and was never reverted -> label 0
    rev103_row = labeled[labeled["revision_new"] == 103].iloc[0]
    assert rev103_row["label"] == 0
