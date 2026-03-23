"""
test_fixes.py — verify QuantAgent gate logic, scoring, and thresholds.
Run with:  python test_fixes.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_agent import (
    _structured_to_score,
    _confidence_multiplier,
    _check_agreement,
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1


# ======================================================================
print("\n=== Structured scoring ===")
# ======================================================================

test("BULLISH → 1",  _structured_to_score("BULLISH") == 1)
test("BEARISH → -1", _structured_to_score("BEARISH") == -1)
test("NEUTRAL → 0",  _structured_to_score("NEUTRAL") == 0)
test("empty → None",  _structured_to_score("") is None)
test("None → None",   _structured_to_score(None) is None)


# ======================================================================
print("\n=== Confidence multiplier ===")
# ======================================================================

test("HIGH → 1.5x",   _confidence_multiplier("HIGH") == 1.5)
test("MEDIUM → 1.0x", _confidence_multiplier("MEDIUM") == 1.0)
test("LOW → 0.5x",    _confidence_multiplier("LOW") == 0.5)
test("None → 1.0x",   _confidence_multiplier(None) == 1.0)


# ======================================================================
print("\n=== Gate: All bullish → LONG ===")
# ======================================================================

state = {
    "indicator_direction": "BULLISH", "indicator_confidence": "HIGH",
    "pattern_direction":   "BULLISH", "pattern_confidence":   "HIGH",
    "trend_direction":     "BULLISH", "trend_confidence":     "HIGH",
}
sig, unan = _check_agreement(state)
test("All BULLISH HIGH → LONG", sig == "LONG")
test("All BULLISH HIGH → unanimous (short-circuit)", unan == True)


# ======================================================================
print("\n=== Gate: All bearish → SHORT ===")
# ======================================================================

state = {
    "indicator_direction": "BEARISH", "indicator_confidence": "HIGH",
    "pattern_direction":   "BEARISH", "pattern_confidence":   "HIGH",
    "trend_direction":     "BEARISH", "trend_confidence":     "HIGH",
}
sig, unan = _check_agreement(state)
test("All BEARISH HIGH → SHORT", sig == "SHORT")
test("All BEARISH HIGH → unanimous", unan == True)


# ======================================================================
print("\n=== Gate: No structured signals → None ===")
# ======================================================================

sig, _ = _check_agreement({})
test("Empty state → None (LLM decides)", sig is None)


# ======================================================================
print("\n=== Gate: Indicator 2x weighting ===")
# ======================================================================

# Indicator BULLISH (2x) vs Pattern BEARISH (1x) → indicator wins
state = {
    "indicator_direction": "BULLISH",
    "pattern_direction":   "BEARISH",
}
sig, _ = _check_agreement(state)
test("Indicator BULLISH(2x) vs Pattern BEARISH(1x) → LONG", sig == "LONG")

# Indicator BEARISH (2x) vs Pattern BULLISH (1x) → indicator wins
state = {
    "indicator_direction": "BEARISH",
    "pattern_direction":   "BULLISH",
}
sig, _ = _check_agreement(state)
test("Indicator BEARISH(2x) vs Pattern BULLISH(1x) → SHORT", sig == "SHORT")


# ======================================================================
print("\n=== Gate: Neutral halving ===")
# ======================================================================

# Indicator NEUTRAL (weight = 2 * 1.0 * 0.5 = 1.0)
# Pattern BULLISH (weight = 1 * 1.0 = 1.0)
# Trend BULLISH (weight = 1 * 1.0 = 1.0)
# bull=2.0 > neut=1.0 → LONG (neutral doesn't block)
state = {
    "indicator_direction": "NEUTRAL",
    "pattern_direction":   "BULLISH",
    "trend_direction":     "BULLISH",
}
sig, _ = _check_agreement(state)
test("NEUTRAL indicator doesn't block 2x BULLISH (neutral halved)", sig == "LONG")


# ======================================================================
print("\n=== Gate: Pattern-only rejection ===")
# ======================================================================

state = {
    "indicator_direction": "NEUTRAL",
    "pattern_direction":   "BULLISH",
    "trend_direction":     "NEUTRAL",
}
sig, _ = _check_agreement(state)
test("Pattern-only BULLISH + both NEUTRAL → rejected (None)", sig is None)

state = {
    "indicator_direction": "NEUTRAL",
    "pattern_direction":   "BEARISH",
    "trend_direction":     "NEUTRAL",
}
sig, _ = _check_agreement(state)
test("Pattern-only BEARISH + both NEUTRAL → rejected (None)", sig is None)


# ======================================================================
print("\n=== Gate: Asymmetric LONG filter (ind == -1) ===")
# ======================================================================

# LONG majority but indicator BEARISH → rejected
state = {
    "indicator_direction":  "BEARISH",
    "pattern_direction":    "BULLISH", "pattern_confidence": "HIGH",
    "trend_direction":      "BULLISH", "trend_confidence":   "HIGH",
}
sig, _ = _check_agreement(state)
test("LONG majority + indicator BEARISH → None (filtered)", sig is None)

# LONG majority with indicator NEUTRAL → passes (ind != -1)
state = {
    "indicator_direction": "NEUTRAL",
    "pattern_direction":   "BULLISH", "pattern_confidence": "HIGH",
    "trend_direction":     "BULLISH", "trend_confidence":   "HIGH",
}
sig, _ = _check_agreement(state)
test("LONG majority + indicator NEUTRAL → LONG (passes filter)", sig == "LONG")

# SHORT is never filtered by the asymmetric LONG filter
state = {
    "indicator_direction":  "BEARISH", "indicator_confidence": "HIGH",
    "pattern_direction":    "BEARISH", "pattern_confidence":   "HIGH",
    "trend_direction":      "BULLISH", "trend_confidence":     "LOW",
}
sig, _ = _check_agreement(state)
test("SHORT signals NOT filtered by asymmetric filter", sig == "SHORT")


# ======================================================================
print("\n=== Gate: Strong-gate short-circuit ===")
# ======================================================================

# All same direction, zero opposition → short-circuit
state = {
    "indicator_direction": "BULLISH", "indicator_confidence": "HIGH",
    "pattern_direction":   "BULLISH", "pattern_confidence":   "MEDIUM",
    "trend_direction":     "BULLISH", "trend_confidence":     "MEDIUM",
}
sig, unan = _check_agreement(state)
test("All BULLISH (zero opposition) → short-circuit", unan == True)


# ======================================================================
print("\n=== Gate: Confidence weighting in action ===")
# ======================================================================

# LOW-confidence bear indicator vs HIGH-confidence bull pattern+trend
state = {
    "indicator_direction":  "BEARISH", "indicator_confidence": "LOW",
    "pattern_direction":    "BULLISH", "pattern_confidence":   "HIGH",
    "trend_direction":      "BULLISH", "trend_confidence":     "HIGH",
}
sig, _ = _check_agreement(state)
# bull > bear, but ind == -1 → asymmetric filter triggers → None
test("LOW-conf bear ind + HIGH-conf bull: asymmetric filter → None", sig is None)


# ======================================================================
print("\n=== Backtest: HOLD scoring removed ===")
# ======================================================================

from backtest import _compute_stats

records = [
    {"signal": "LONG",  "outcome": "LONG",  "correct": True,  "pnl": 0.01, "pnl_magnitude": 0.01},
    {"signal": "SHORT", "outcome": "LONG",  "correct": False, "pnl": -0.01, "pnl_magnitude": 0.01},
    {"signal": "HOLD",  "outcome": "SHORT", "correct": False, "pnl": 0.0, "pnl_magnitude": 0.005},
    {"signal": "HOLD",  "outcome": "LONG",  "correct": False, "pnl": 0.0, "pnl_magnitude": 0.02},
]
stats = _compute_stats(records)
test("hold_correct NOT in stats", "hold_correct" not in stats)
test("hold_rate in stats", "hold_rate" in stats)
test("hold_rate = 0.5", stats.get("hold_rate") == 0.5)
test("total_signals = 2 (directional only)", stats.get("total_signals") == 2)
test("hold_signals = 2", stats.get("hold_signals") == 2)

# Hold rate warning at >50%
records2 = [
    {"signal": "HOLD", "outcome": "LONG",  "correct": False, "pnl": 0.0, "pnl_magnitude": 0.01},
    {"signal": "HOLD", "outcome": "LONG",  "correct": False, "pnl": 0.0, "pnl_magnitude": 0.01},
    {"signal": "HOLD", "outcome": "SHORT", "correct": False, "pnl": 0.0, "pnl_magnitude": 0.01},
    {"signal": "LONG", "outcome": "LONG",  "correct": True,  "pnl": 0.01, "pnl_magnitude": 0.01},
]
stats2 = _compute_stats(records2)
test("75% HOLD rate → warning", stats2.get("hold_rate_warning") == True)


# ======================================================================
print("\n=== Backtest: MIN_MOVE_THRESHOLD ===")
# ======================================================================

from backtest import MIN_MOVE_THRESHOLD
test("MIN_MOVE_THRESHOLD = 0.001", MIN_MOVE_THRESHOLD == 0.001)


# ======================================================================
print("\n=== Prediction layer ===")
# ======================================================================

from prediction_layer import PredictionLayer, extract_features

pred = PredictionLayer()
test("PredictionLayer starts untrained", pred.trained == False)

features = extract_features({
    "decision": "LONG",
    "indicator_report": "bullish momentum golden cross",
    "pattern_report": "bearish breakdown",
    "trend_report": "neutral sideways",
    "risk_reward_ratio": 1.5,
})
test("Feature vector = 12 dimensions", len(features) == 12)
test("Feature[0] is_long = 1.0", features[0] == 1.0)


# ======================================================================
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)
else:
    print("All tests passed! ✓")
    sys.exit(0)
