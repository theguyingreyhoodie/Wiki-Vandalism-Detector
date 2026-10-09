import numpy as np
from src.model.drift import calculate_psi, DriftDetector


def test_calculate_psi_stable():
    np.random.seed(42)
    # Both samples drawn from same normal distribution
    baseline = np.random.normal(loc=0.0, scale=1.0, size=1000)
    current = np.random.normal(loc=0.0, scale=1.0, size=1000)

    psi = calculate_psi(baseline, current)
    # Same distribution should have very low PSI (< 0.1)
    assert psi < 0.10


def test_calculate_psi_drifted():
    np.random.seed(42)
    # Current distribution shifted significantly
    baseline = np.random.normal(loc=0.0, scale=1.0, size=1000)
    current = np.random.normal(loc=3.0, scale=1.0, size=1000)

    psi = calculate_psi(baseline, current)
    # Large shift should result in significant PSI (>= 0.2)
    assert psi >= 0.20


def test_drift_detector():
    import pandas as pd

    cfg = {
        "drift": {
            "enabled": True,
            "psi_threshold": 0.20,
            "features_to_monitor": ["feature_a", "feature_b"],
        }
    }

    detector = DriftDetector(cfg)

    df_base = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 500),
        "feature_b": np.random.uniform(0, 10, 500),
    })
    df_curr = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 500),  # stable
        "feature_b": np.random.uniform(20, 50, 500),  # drifted
    })

    report = detector.compute_drift(df_base, df_curr)
    assert report["total_monitored"] == 2
    assert report["features"]["feature_a"]["status"] == "STABLE"
    assert report["features"]["feature_b"]["status"] == "DRIFT"
    assert report["overall_drift_status"] == "ALERT"
