"""LightGBM training with class-imbalance handling and Optuna tuning.

Key design choices
------------------
* **PR-AUC** as primary metric — accuracy is meaningless at ~4 % prevalence.
* **scale_pos_weight** for imbalance — set to ``"auto"`` in config for
  automatic ``n_neg / n_pos`` computation.
* **Early stopping** on validation PR-AUC to prevent overfitting.
* **Optuna** for optional hyperparameter search with a per-trial budget.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("wiki_vandalism.model.trainer")


class VandalismModelTrainer:
    """Trains and (optionally) tunes a LightGBM vandalism classifier."""

    def __init__(self, config: dict):
        self.config = config
        self.model_params = config["model"]["params"].copy()
        self.early_stopping = config["model"].get("early_stopping_rounds", 50)
        self.tuning_cfg = config.get("tuning", {})
        self.seed = config["project"]["seed"]
        self.model_params["random_state"] = self.seed

        self.model: lgb.LGBMClassifier | None = None
        self.training_metadata: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> lgb.LGBMClassifier:
        """Train the model and evaluate on the held-out validation set."""

        # Auto-compute class weight
        if self.model_params.get("scale_pos_weight") == "auto":
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            self.model_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            logger.info(
                f"Auto scale_pos_weight = {self.model_params['scale_pos_weight']:.1f} "
                f"(neg={n_neg:,}, pos={n_pos:,})"
            )

        logger.info(
            f"Training LightGBM — {X_train.shape[0]:,} train / "
            f"{X_val.shape[0]:,} val / {X_train.shape[1]} features"
        )

        t0 = time.time()
        self.model = lgb.LGBMClassifier(**self.model_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(self.early_stopping, verbose=True),
                lgb.log_evaluation(period=50),
            ],
            feature_name=feature_names or "auto",
        )
        elapsed = time.time() - t0

        # Metrics
        y_tr_prob = self.model.predict_proba(X_train)[:, 1]
        y_vl_prob = self.model.predict_proba(X_val)[:, 1]
        tr_prauc = average_precision_score(y_train, y_tr_prob)
        vl_prauc = average_precision_score(y_val, y_vl_prob)
        vl_rocauc = roc_auc_score(y_val, y_vl_prob)

        self.training_metadata = {
            "train_pr_auc": float(tr_prauc),
            "val_pr_auc": float(vl_prauc),
            "val_roc_auc": float(vl_rocauc),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_features": int(X_train.shape[1]),
            "best_iteration": int(self.model.best_iteration_),
            "train_time_sec": float(elapsed),
            "params": {k: str(v) for k, v in self.model_params.items()},
        }

        logger.info(
            f"Done in {elapsed:.1f} s │ Train PR-AUC {tr_prauc:.4f} │ "
            f"Val PR-AUC {vl_prauc:.4f} │ Val ROC-AUC {vl_rocauc:.4f} │ "
            f"Best iter {self.model.best_iteration_}"
        )
        return self.model

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> lgb.LGBMClassifier:
        """Run Optuna hyperparameter search, then re-train with best params."""
        n_trials = self.tuning_cfg.get("n_trials", 50)
        timeout = self.tuning_cfg.get("timeout_sec", 3600)
        logger.info(f"Optuna tuning: {n_trials} trials / {timeout} s budget")

        def _objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "average_precision",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "random_state": self.seed,
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
            }
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
                feature_name=feature_names or "auto",
            )
            return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

        logger.info(
            f"Best PR-AUC {study.best_value:.4f} — params: {study.best_params}"
        )
        self.model_params.update(study.best_params)
        return self.train(X_train, y_train, X_val, y_val, feature_names)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        self.model.booster_.save_model(str(path / "model.txt"))
        with open(path / "metadata.json", "w") as f:
            json.dump(self.training_metadata, f, indent=2)
        logger.info(f"Model saved → {path}")

    def load_model(self, path: str | Path) -> lgb.LGBMClassifier:
        path = Path(path)
        self.model = joblib.load(path / "model.joblib")
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.training_metadata = json.load(f)
        logger.info(f"Model loaded ← {path}")
        return self.model
