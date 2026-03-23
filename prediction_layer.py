"""
prediction_layer.py
-------------------
The novel contribution layer that sits on top of QuantAgent.

What it does:
  - Takes QuantAgent's three agent reports + raw decision as input
  - Extracts a feature vector from those outputs
  - Runs it through a logistic regression trained on backtest data
  - Returns a refined confidence score and (optionally) a flipped decision

Training:
  - Call PredictionLayer.train(backtest_result) with a loaded backtest result dict
  - The model is saved to data/prediction_model.json (sklearn-free, pure numpy)
  - Call PredictionLayer.load() to restore a trained model

Inference:
  - Call PredictionLayer.predict(agent_outputs) during live analysis
  - Returns {"decision", "confidence", "raw_decision", "was_flipped"}

Why logistic regression and not a neural net:
  - Interpretable — you can show the professor which features matter most
  - Works with small datasets (50-200 backtest samples)
  - No extra dependencies beyond numpy/scipy (already in requirements)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from qa_logger import get_logger

_log = get_logger("Prediction")

MODEL_PATH = Path("data/prediction_model.json")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Keywords that carry directional signal in the reports
BULLISH_WORDS = [
    "bullish",
    "upward",
    "uptrend",
    "breakout",
    "bounce",
    "support holding",
    "golden cross",
    "oversold",
    "buy signal",
    "long",
    "positive momentum",
    "higher high",
    "higher low",
    "ascending",
    "reversal upward",
]
BEARISH_WORDS = [
    "bearish",
    "downward",
    "downtrend",
    "breakdown",
    "resistance",
    "overbought",
    "death cross",
    "sell signal",
    "short",
    "negative momentum",
    "lower high",
    "lower low",
    "descending",
    "reversal downward",
]

# Risk-reward ratio parser
_RR_RE = re.compile(r"(\d+\.?\d*)\s*[:/]\s*(\d+\.?\d*)|(\d+\.?\d+)")


def _count_sentiment(text: str) -> tuple[int, int]:
    """Count bullish and bearish keyword hits in a text."""
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    return bull, bear


def _parse_rr(rr_raw: Any) -> float:
    """Parse risk-reward ratio to a float. Returns 1.5 as default.
    FIX: was returning denom/numer (inverted). Now correctly returns numer/denom.
    e.g. '2:1' -> 2.0, not 0.5.
    """
    if isinstance(rr_raw, (int, float)):
        return float(rr_raw)
    s = str(rr_raw)
    m = _RR_RE.search(s)
    if not m:
        return 1.5
    if m.group(1) and m.group(2):
        try:
            # group(1) is numerator, group(2) is denominator: 2:1 -> 2.0
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return 1.5
    if m.group(3):
        return float(m.group(3))
    return 1.5


def extract_features(agent_outputs: dict) -> np.ndarray:
    """
    Structured + numeric features (NO keyword parsing).
    """

    def enc_dir(d):
        d = (d or "").upper()
        return 1.0 if d == "BULLISH" else -1.0 if d == "BEARISH" else 0.0

    def enc_conf(c):
        c = (c or "").upper()
        return {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}.get(c, 0.5)

    # Structured inputs
    ind_dir = enc_dir(agent_outputs.get("indicator_direction"))
    ind_conf = enc_conf(agent_outputs.get("indicator_confidence"))

    pat_dir = enc_dir(agent_outputs.get("pattern_direction"))
    pat_conf = enc_conf(agent_outputs.get("pattern_confidence"))

    trn_dir = enc_dir(agent_outputs.get("trend_direction"))
    trn_conf = enc_conf(agent_outputs.get("trend_confidence"))

    # Raw decision
    raw_decision = (agent_outputs.get("decision") or "").upper()
    is_long = 1.0 if raw_decision == "LONG" else 0.0

    # Agreement score
    dirs = [ind_dir, pat_dir, trn_dir]
    agreement = sum((d > 0 and is_long) or (d < 0 and not is_long) for d in dirs) / 3.0

    return np.array(
        [
            is_long,
            ind_dir,
            ind_conf,
            pat_dir,
            pat_conf,
            trn_dir,
            trn_conf,
            agreement,
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "raw_long",
    "ind_dir",
    "ind_conf",
    "pat_dir",
    "pat_conf",
    "trn_dir",
    "trn_conf",
    "agreement",
]

N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Logistic Regression (pure numpy — no sklearn needed)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _train_logistic(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 500, l2: float = 0.01
) -> tuple[np.ndarray, float]:
    """
    Train a binary logistic regression with L2 regularisation.
    y: 1 = LONG, 0 = SHORT
    Returns (weights, bias).
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        p = _sigmoid(z)
        err = p - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


# ---------------------------------------------------------------------------
# PredictionLayer class
# ---------------------------------------------------------------------------


class PredictionLayer:
    """
    Wraps the trained logistic regression model.
    Use .train() to fit from backtest records.
    Use .predict() during live inference.
    """

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_importance: dict = {}
        self.trained: bool = False
        self.train_meta: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, backtest_result: dict) -> dict:
        """
        Fit the model from a backtest result dict.
        FIX: HOLD records are now skipped — they corrupt training because
        signal='HOLD' maps to is_long=0.0 (same as SHORT), misleading the model.
        """
        records = backtest_result.get("records", [])
        # Filter out HOLD signals before training
        directional_records = [
            r for r in records if r.get("signal") in ("LONG", "SHORT")
        ]
        if len(directional_records) < 10:
            return {
                "error": f"Need at least 10 directional records, got {len(directional_records)} (total {len(records)})"
            }

        X_list, y_list = [], []
        for r in directional_records:
            features = extract_features(
                {
                    "decision": r.get("signal", ""),
                    "indicator_direction": r.get("indicator_direction"),
                    "indicator_confidence": r.get("indicator_confidence"),
                    "pattern_direction": r.get("pattern_direction"),
                    "pattern_confidence": r.get("pattern_confidence"),
                    "trend_direction": r.get("trend_direction"),
                    "trend_confidence": r.get("trend_confidence"),
                }
            )
            label = 1.0 if r.get("outcome") == "LONG" else 0.0
            X_list.append(features)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)

        # Normalise features (mean/std)
        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.feat_mean) / self.feat_std

        self.weights, self.bias = _train_logistic(X_norm, y)
        self.trained = True

        # Feature importance (absolute weight magnitude)
        self.feature_importance = {
            name: round(float(abs(w)), 4)
            for name, w in zip(FEATURE_NAMES, self.weights)
        }

        # Training accuracy
        preds = (_sigmoid(X_norm @ self.weights + self.bias) >= 0.5).astype(float)
        train_acc = float((preds == y).mean())

        meta = backtest_result.get("meta", {})
        self.train_meta = {
            "ticker": meta.get("ticker"),
            "interval": meta.get("interval"),
            "n_samples": len(directional_records),
            "hold_excluded": len(records) - len(directional_records),
            "train_accuracy": round(train_acc, 4),
        }

        self.save()
        return {
            "success": True,
            "train_accuracy": round(train_acc, 4),
            "n_samples": len(directional_records),
            "hold_excluded": len(records) - len(directional_records),
            "feature_importance": self.feature_importance,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, agent_outputs: dict) -> dict:
        """
        Refine a QuantAgent decision with the trained model.

        agent_outputs should contain:
          decision, indicator_report, pattern_report, trend_report, risk_reward_ratio

        Returns:
          {
            decision:      "LONG" or "SHORT"  (may differ from raw)
            confidence:    float 0-1
            raw_decision:  original QuantAgent decision
            was_flipped:   bool
          }
        """
        raw_decision = (agent_outputs.get("decision") or "").upper()

        if not self.trained or self.weights is None:
            # No model yet — pass through raw decision with neutral confidence
            return {
                "decision": raw_decision,
                "confidence": 0.5,
                "raw_decision": raw_decision,
                "was_flipped": False,
                "model_used": False,
            }

        features = extract_features(agent_outputs).astype(np.float64)
        feat_norm = (features - self.feat_mean) / self.feat_std
        prob_long = float(_sigmoid(feat_norm @ self.weights + self.bias))

        # Confidence = distance from 0.5, mapped to [0.5, 1.0]
        confidence = 0.5 + abs(prob_long - 0.5)

        refined = "LONG" if prob_long >= 0.5 else "SHORT"
        was_flipped = refined != raw_decision

        return {
            "decision": refined,
            "confidence": round(confidence, 4),
            "prob_long": round(prob_long, 4),
            "raw_decision": raw_decision,
            "was_flipped": was_flipped,
            "model_used": True,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH):
        """Save model weights to JSON."""
        if not self.trained:
            return
        data = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feat_mean": self.feat_mean.tolist(),
            "feat_std": self.feat_std.tolist(),
            "feature_importance": self.feature_importance,
            "feature_names": FEATURE_NAMES,
            "train_meta": self.train_meta,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        _log.ok(
            f"Model saved → {path.name} (accuracy={self.train_meta.get('train_accuracy',0):.1%})"
        )

    def load(self, path: Path = MODEL_PATH) -> bool:
        """Load model weights from JSON. Returns True if successful."""
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.weights = np.array(data["weights"])
            self.bias = data["bias"]
            self.feat_mean = np.array(data["feat_mean"])
            self.feat_std = np.array(data["feat_std"])
            self.feature_importance = data.get("feature_importance", {})
            self.train_meta = data.get("train_meta", {})
            self.trained = True
            _log.ok(f"Model loaded from {path.name}")
            return True
        except Exception as e:
            _log.warning(f"No saved model found — starting untrained")
            return False

    def status(self) -> dict:
        """Return a summary of the model's current state."""
        return {
            "trained": self.trained,
            "train_meta": self.train_meta,
            "feature_importance": self.feature_importance,
        }
