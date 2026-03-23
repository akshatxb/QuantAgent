"""
ablation.py
-----------
Ablation study engine for QuantAgent.

Runs three variants over the same historical records and compares them:

  Variant 1 — Random baseline
    Coin-flip LONG/SHORT. The floor every real agent must beat.

  Variant 2 — Indicators only (no LLM synthesis)
    Takes the raw indicator values from backtest records and uses a simple
    majority-vote heuristic: if more indicators are bearish than bullish → SHORT,
    else → LONG. No LLM reasoning, no pattern/trend agents.

  Variant 3 — Full QuantAgent
    The complete pipeline as recorded in the backtest (indicator + pattern +
    trend agents → LLM decision maker). Already computed in backtest records.

  Variant 4 — QuantAgent + Prediction Layer  (optional)
    If a trained prediction model is available, re-runs inference on each record
    and checks whether the refined decision improves accuracy.

Results saved to data/ablation_results/ as JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

import backtest as bt
from qa_logger import get_logger

_log = get_logger("Ablation")

ABLATION_DIR = Path("data/ablation_results")
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Indicator heuristic (variant 2 — no LLM)
# ---------------------------------------------------------------------------

BULLISH_KW = [
    "bullish",
    "upward",
    "uptrend",
    "above",
    "oversold",
    "positive",
    "buy",
    "long",
    "support",
    "bounce",
    "rising",
    "higher",
    "ascending",
    "crossover above",
    "golden cross",
]
BEARISH_KW = [
    "bearish",
    "downward",
    "downtrend",
    "below",
    "overbought",
    "negative",
    "sell",
    "short",
    "resistance",
    "breakdown",
    "falling",
    "lower",
    "descending",
    "crossover below",
    "death cross",
]


def _indicator_only_decision(record: dict) -> str:
    """
    Structured baseline using same signals as real system (no LLM).
    """

    def score(direction, confidence, weight):
        if direction == "BULLISH":
            return weight * confidence
        if direction == "BEARISH":
            return -weight * confidence
        return 0.0

    conf_map = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.5}

    ind = score(
        record.get("indicator_direction"),
        conf_map.get(record.get("indicator_confidence"), 1.0),
        2,
    )
    pat = score(
        record.get("pattern_direction"),
        conf_map.get(record.get("pattern_confidence"), 1.0),
        1,
    )
    trn = score(
        record.get("trend_direction"),
        conf_map.get(record.get("trend_confidence"), 1.0),
        1,
    )

    total = ind + pat + trn
    return "LONG" if total >= 0 else "SHORT"


# ---------------------------------------------------------------------------
# Stats helper (reused from backtest.py but standalone here)
# ---------------------------------------------------------------------------


def _stats(signals: list[str], outcomes: list[str], pnl_mags: list[float]) -> dict:
    """
    Compute stats for an ablation variant.
    HOLD signals are excluded from accuracy. Directional signals on
    FLAT outcomes are tracked separately as filtered_accuracy.
    """
    n = len(signals)
    if n == 0:
        return {"total_signals": 0, "accuracy": 0.0}

    # Separate HOLD from directional
    active_sigs = [
        (s, o, m) for s, o, m in zip(signals, outcomes, pnl_mags) if s != "HOLD"
    ]
    hold_sigs = [
        (s, o, m) for s, o, m in zip(signals, outcomes, pnl_mags) if s == "HOLD"
    ]

    n_active = len(active_sigs)

    # Directional accuracy
    correct_active = sum(1 for s, o, _ in active_sigs if s == o)
    accuracy = correct_active / n_active if n_active else 0.0

    # Filtered accuracy — exclude FLAT outcomes from directional evaluation
    meaningful = [(s, o, m) for s, o, m in active_sigs if o != "FLAT"]
    correct_meaningful = sum(1 for s, o, _ in meaningful if s == o)
    filtered_accuracy = correct_meaningful / len(meaningful) if meaningful else 0.0

    active_signals_list = [s for s, _, _ in active_sigs]
    active_outcomes = [o for _, o, _ in active_sigs]
    active_mags = [m for _, _, m in active_sigs]

    pnl_arr = (
        np.array([m if s == o else -m for s, o, m in active_sigs])
        if active_sigs
        else np.array([0.0])
    )

    tp = sum(
        s == "LONG" and o == "LONG"
        for s, o in zip(active_signals_list, active_outcomes)
    )
    tn = sum(
        s == "SHORT" and o == "SHORT"
        for s, o in zip(active_signals_list, active_outcomes)
    )
    fp = sum(
        s == "LONG" and o == "SHORT"
        for s, o in zip(active_signals_list, active_outcomes)
    )
    fn = sum(
        s == "SHORT" and o == "LONG"
        for s, o in zip(active_signals_list, active_outcomes)
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    sharpe = (
        float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252))
        if np.std(pnl_arr) > 0
        else 0.0
    )

    long_sigs = [
        (s, o) for s, o in zip(active_signals_list, active_outcomes) if s == "LONG"
    ]
    short_sigs = [
        (s, o) for s, o in zip(active_signals_list, active_outcomes) if s == "SHORT"
    ]
    long_acc = sum(s == o for s, o in long_sigs) / len(long_sigs) if long_sigs else 0.0
    short_acc = (
        sum(s == o for s, o in short_sigs) / len(short_sigs) if short_sigs else 0.0
    )

    return {
        "total_signals": n,
        "active_signals": n_active,
        "hold_signals": len(hold_sigs),
        "correct": int(correct_active),
        "accuracy": round(accuracy, 4),
        "filtered_accuracy": round(filtered_accuracy, 4),
        "sharpe_ratio": round(sharpe, 4),
        "cumulative_pnl": round(float(pnl_arr.sum()), 6),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "long_accuracy": round(long_acc, 4),
        "short_accuracy": round(short_acc, 4),
        "long_signals": len(long_sigs),
        "short_signals": len(short_sigs),
    }


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------


def run_ablation(
    backtest_filename: str,
    prediction_layer=None,
) -> dict:
    """
    Run all ablation variants on a saved backtest result.

    Parameters
    ----------
    backtest_filename : str
        Filename inside data/backtest_results/ (e.g. "AAPL_1h_20240101_20240601.json")
    prediction_layer : PredictionLayer | None
        If provided and trained, variant 4 will be computed.

    Returns
    -------
    Full ablation result dict (also saved to data/ablation_results/).
    """
    result = bt.load_backtest_result(backtest_filename)
    if result is None:
        return {"error": f"Backtest file not found: {backtest_filename}"}

    records = result.get("records", [])
    if len(records) < 5:
        return {"error": f"Need at least 5 records, got {len(records)}"}

    meta = result.get("meta", {})

    # Ground truth outcomes and P&L magnitudes
    outcomes = [r["outcome"] for r in records]
    pnl_mags = [r.get("pnl_magnitude", abs(r.get("pnl", 0.01))) for r in records]

    # ------------------------------------------------------------------
    # Variant 1 — Random baseline
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    rand_signals = ["LONG" if rng.random() > 0.5 else "SHORT" for _ in records]
    v1 = _stats(rand_signals, outcomes, pnl_mags)
    v1["variant"] = "random_baseline"
    v1["label"] = "Random baseline"
    v1["description"] = "Coin-flip LONG/SHORT — the floor all agents must beat"

    # ------------------------------------------------------------------
    # Variant 2 — Indicator heuristic only
    # ------------------------------------------------------------------
    ind_signals = [_indicator_only_decision(r) for r in records]
    v2 = _stats(ind_signals, outcomes, pnl_mags)
    v2["variant"] = "indicators_only"
    v2["label"] = "Indicators only"
    v2["description"] = "Keyword-majority heuristic over indicator report — no LLM"

    # ------------------------------------------------------------------
    # Variant 3 — Full QuantAgent (already recorded in backtest)
    # Step 3: Only measure accuracy over directional calls — exclude HOLD
    # windows to make the comparison fair against always-directional baselines.
    # ------------------------------------------------------------------
    directional_records = [r for r in records if r["signal"] in ("LONG", "SHORT")]
    hold_count = len(records) - len(directional_records)
    hold_rate = hold_count / len(records) if records else 0.0

    agent_signals_dir = [r["signal"] for r in directional_records]
    outcomes_dir = [r["outcome"] for r in directional_records]
    pnl_mags_dir = [
        r.get("pnl_magnitude", abs(r.get("pnl", 0.01))) for r in directional_records
    ]

    v3 = _stats(agent_signals_dir, outcomes_dir, pnl_mags_dir)
    v3["variant"] = "full_quantagent"
    v3["label"] = "Full QuantAgent"
    v3["description"] = (
        "Complete pipeline: indicator + pattern + trend agents → LLM decision (directional calls only)"
    )

    # ------------------------------------------------------------------
    # Variant 4 — QuantAgent + Prediction Layer (optional)
    # ------------------------------------------------------------------
    v4 = None
    if prediction_layer is not None and prediction_layer.trained:
        pred_signals = []
        for r in records:
            pred = pred = prediction_layer.predict(
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
            pred_signals.append(pred.get("decision", r.get("signal", "")))

        v4 = _stats(pred_signals, outcomes, pnl_mags)
        v4["variant"] = "quantagent_plus_prediction"
        v4["label"] = "QuantAgent + Prediction Layer"
        v4["description"] = "LLM pipeline refined by trained logistic regression model"

        # Flip rate: how often the model changed the LLM's decision
        all_agent_signals = [r["signal"] for r in records]
        flips = sum(p != a for p, a in zip(pred_signals, all_agent_signals))
        v4["flip_rate"] = round(flips / len(records), 4)
        v4["flip_count"] = flips

    # ------------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------------
    variants = [v for v in [v1, v2, v3, v4] if v is not None]

    comparison = {
        "accuracy": {v["variant"]: v["accuracy"] for v in variants},
        "sharpe_ratio": {v["variant"]: v["sharpe_ratio"] for v in variants},
        "f1_score": {v["variant"]: v["f1_score"] for v in variants},
        "cum_pnl": {v["variant"]: v["cumulative_pnl"] for v in variants},
    }

    # Best variant by accuracy
    best = max(variants, key=lambda v: v["accuracy"])

    # Lift: how much better full QuantAgent is vs random (percentage points)
    lift_vs_random = round((v3["accuracy"] - v1["accuracy"]) * 100, 2)
    lift_vs_indicators = round((v3["accuracy"] - v2["accuracy"]) * 100, 2)

    ablation_result = {
        "meta": {
            "backtest_file": backtest_filename,
            "ticker": meta.get("ticker"),
            "display_name": meta.get("display_name"),
            "market": meta.get("market"),
            "interval": meta.get("interval"),
            "n_records": len(records),
            "n_directional": len(directional_records),
            "hold_count": hold_count,
            "hold_rate": round(hold_rate, 4),
            "run_at": datetime.now().isoformat(),
        },
        "variants": variants,
        "comparison": comparison,
        "best_variant": best["variant"],
        "lift_vs_random_pp": lift_vs_random,
        "lift_vs_indicators_pp": lift_vs_indicators,
        "has_prediction_layer": v4 is not None,
    }

    # Save
    safe_name = backtest_filename.replace(".json", "")
    out_path = ABLATION_DIR / f"ablation_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(ablation_result, f, indent=2, default=str)

    _log.ok(f"Results saved → {out_path.name}")
    _log.ok(
        f"random={v1['accuracy']:.1%}  indicators={v2['accuracy']:.1%}  agent={v3['accuracy']:.1%}"
        + (f"  +pred={v4['accuracy']:.1%}" if v4 else "")
    )

    return ablation_result


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def list_ablation_results() -> list[dict]:
    """Return summaries of all saved ablation results."""
    results = []
    for path in sorted(ABLATION_DIR.glob("ablation_*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            m = data.get("meta", {})
            v3 = next(
                (
                    v
                    for v in data.get("variants", [])
                    if v["variant"] == "full_quantagent"
                ),
                {},
            )
            v1 = next(
                (
                    v
                    for v in data.get("variants", [])
                    if v["variant"] == "random_baseline"
                ),
                {},
            )
            results.append(
                {
                    "filename": path.name,
                    "ticker": m.get("ticker"),
                    "market": m.get("market"),
                    "interval": m.get("interval"),
                    "n_records": m.get("n_records"),
                    "agent_accuracy": v3.get("accuracy"),
                    "random_accuracy": v1.get("accuracy"),
                    "lift_pp": data.get("lift_vs_random_pp"),
                    "best_variant": data.get("best_variant"),
                    "run_at": m.get("run_at"),
                }
            )
        except Exception:
            pass
    return results


def load_ablation_result(filename: str) -> dict | None:
    path = ABLATION_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
