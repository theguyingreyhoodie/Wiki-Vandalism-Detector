import pandas as pd
from src.features.edit_features import extract_edit_features


def test_extract_edit_features():
    data = [
        {
            "event_id": "enwiki_1",
            "timestamp": 1700000000,
            "user": "203.0.113.195",  # Anonymous IP
            "bot": False,
            "minor": False,
            "title": "Machine_learning",
            "namespace": 0,
            "comment": "check out https://spam.com",
            "length_new": 10,
            "length_old": 5000,
        },
        {
            "event_id": "enwiki_2",
            "timestamp": 1700003600,
            "user": "RegularUser",
            "bot": False,
            "minor": True,
            "title": "Machine_learning",
            "namespace": 0,
            "comment": "Fixed typo",
            "length_new": 5005,
            "length_old": 5000,
        },
    ]

    df = pd.DataFrame(data)
    features_df = extract_edit_features(df)

    assert "size_delta" in features_df.columns
    assert "is_anonymous" in features_df.columns
    assert "is_blanking" in features_df.columns
    assert "comment_has_url" in features_df.columns

    # Row 0: anonymous IP, massive deletion (blanking), URL present
    assert features_df.loc[0, "is_anonymous"] == 1
    assert features_df.loc[0, "is_blanking"] == 1
    assert features_df.loc[0, "comment_has_url"] == 1
    assert features_df.loc[0, "size_delta"] == -4990

    # Row 1: registered user, small change, no URL
    assert features_df.loc[1, "is_anonymous"] == 0
    assert features_df.loc[1, "is_blanking"] == 0
    assert features_df.loc[1, "comment_has_url"] == 0
    assert features_df.loc[1, "size_delta"] == 5
