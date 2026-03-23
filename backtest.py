"""
backtest.py
-----------
Backtesting engine for QuantAgent.

How it works:
  1. Fetch a long window of historical OHLCV data for a ticker + interval
  2. Slide a 45-candle window forward one candle at a time
  3. Run the full QuantAgent pipeline on each window
  4. Label the signal: did price actually go UP or DOWN over the next N candles?
  5. Collect all (signal, outcome) pairs and compute statistics

Statistics produced:
  - Overall accuracy, win rate
  - Sharpe ratio of simulated P&L
  - Confusion matrix (TP/FP/TN/FN)
  - Accuracy broken down by: market (NSE/US), timeframe, signal type (LONG/SHORT)
  - Per-agent contribution scores (ablation)

Results are saved as JSON to data/backtest_results/ for the frontend to consume.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp
from qa_logger import get_logger
log = get_logger("Backtest")

RESULTS_DIR = Path("data/backtest_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# How many candles ahead to measure the outcome
# Extended horizons so trends have more time to develop.
# At 2 candles (2h for 1h charts), price is near-random noise.
OUTCOME_HORIZON: dict[str, int] = {
    "1m":  5,
    "5m":  5,
    "15m": 5,
    "30m": 4,
    "1h":  5,
    "4h":  4,
    "1d":  3,
    "1wk": 2,
    "1mo": 2,
}

# Step 6: Lowered from 0.15% to 0.10% — 0.15% was filtering too many
# valid signals on NSE 1h data (TCS avg move ~0.18-0.25%).
MIN_MOVE_THRESHOLD = 0.0010   # 0.10% of entry price

WINDOW_SIZE = 45   # candles fed to QuantAgent per run
MIN_STEP    = 6    # slide 6 candles forward — less overlap, more independent signals
                   # (was 3: consecutive windows shared 42/45 candles — near-identical)

# Sharpe annualisation: sqrt(bars per trading year)
# Different for each timeframe — using sqrt(252 * daily_bars):
SHARPE_ANNUALISATION: dict[str, float] = {
    "1m":   (252 * 390) ** 0.5,   # US: 6.5h * 60m
    "5m":   (252 * 78)  ** 0.5,
    "15m":  (252 * 26)  ** 0.5,
    "30m":  (252 * 13)  ** 0.5,
    "1h":   (252 * 6.5) ** 0.5,   # ~40.5  (was 15.9 — factor 2.5x understated)
    "4h":   (252 * 1.625) ** 0.5,
    "1d":   252 ** 0.5,
    "1wk":  52  ** 0.5,
    "1mo":  12  ** 0.5,
}


# ---------------------------------------------------------------------------
# Outcome labelling
# ---------------------------------------------------------------------------

def _label_outcome(df: pd.DataFrame, signal_idx: int, horizon: int) -> str | None:
    """
    Given the index of the last candle in the analysis window,
    look `horizon` candles ahead and decide if price went UP or DOWN.

    Returns "LONG", "SHORT", or None if not enough future data.
    """
    future_end = signal_idx + horizon
    if future_end >= len(df):
        return None

    entry_price = df["Close"].iloc[signal_idx]
    exit_price  = df["Close"].iloc[future_end]

    if exit_price > entry_price:
        return "LONG"
    elif exit_price < entry_price:
        return "SHORT"
    else:
        return None  # flat — skip


# ---------------------------------------------------------------------------
# Signal parser
# ---------------------------------------------------------------------------

def _parse_decision(raw: str) -> dict | None:
    """
    Extract the JSON decision block from QuantAgent's raw output string.
    HOLD is accepted as a valid third outcome.
    Returns None only if the JSON is completely unparseable.
    """
    if not raw:
        return None
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        data = json.loads(raw[start:end])
        decision = data.get("decision", "").upper().strip()
        if decision not in ("LONG", "SHORT", "HOLD"):
            return None
        return {
            "decision":          decision,
            "justification":     data.get("justification", ""),
            "risk_reward_ratio": data.get("risk_reward_ratio", "N/A"),
            "forecast_horizon":  data.get("forecast_horizon", "N/A"),
            "signal_strength":   data.get("signal_strength", "UNKNOWN"),
        }
    except (json.JSONDecodeError, Exception):
        return None


# ---------------------------------------------------------------------------
# Statistics calculator
# ---------------------------------------------------------------------------

def _compute_stats(records: list[dict], interval: str = "") -> dict:
    """
    Given a list of {signal, outcome, correct, pnl} records,
    compute the full statistics dict.

    Steps 4+7:
    - HOLD signals are tracked but excluded from ALL accuracy calculations.
      HOLD = no position = no P&L = no accuracy contribution.
    - hold_rate and hold_rate_warning are surfaced for monitoring.
    """
    if not records:
        return {"error": "no records"}

    df = pd.DataFrame(records)

    # Separate HOLD signals from directional signals
    hold_df = df[df["signal"] == "HOLD"]
    dir_df  = df[df["signal"].isin(["LONG", "SHORT"])]
    n       = len(dir_df)  # accuracy is only over directional signals

    # Step 7: Compute hold rate
    total_all = len(df)
    hold_rate = len(hold_df) / total_all if total_all > 0 else 0.0
    hold_rate_warning = hold_rate > 0.50

    if n == 0:
        return {
            "error": "no directional signals (all HOLD)",
            "hold_signals": len(hold_df),
            "hold_rate": round(hold_rate, 4),
            "hold_rate_warning": hold_rate_warning,
        }

    correct   = dir_df["correct"].sum()
    accuracy  = correct / n if n else 0.0
    win_rate  = accuracy

    # Confusion matrix (directional signals only)
    tp = int(((dir_df["signal"] == "LONG")  & (dir_df["outcome"] == "LONG")).sum())
    tn = int(((dir_df["signal"] == "SHORT") & (dir_df["outcome"] == "SHORT")).sum())
    fp = int(((dir_df["signal"] == "LONG")  & (dir_df["outcome"] == "SHORT")).sum())
    fn = int(((dir_df["signal"] == "SHORT") & (dir_df["outcome"] == "LONG")).sum())

    # Precision / Recall for LONG signals
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    # Sharpe ratio on per-trade P&L (directional trades only)
    pnl_series  = dir_df["pnl"].values
    ann_factor  = SHARPE_ANNUALISATION.get(interval, 252 ** 0.5)
    sharpe = (float(np.mean(pnl_series) / np.std(pnl_series) * ann_factor)
              if np.std(pnl_series) > 0 else 0.0)

    cumulative_pnl = float(dir_df["pnl"].sum())

    long_df  = dir_df[dir_df["signal"] == "LONG"]
    short_df = dir_df[dir_df["signal"] == "SHORT"]

    long_acc  = float(long_df["correct"].mean())  if len(long_df)  else 0.0
    short_acc = float(short_df["correct"].mean()) if len(short_df) else 0.0

    # Unanimous accuracy — how well the system performs when all
    # 3 agents agreed on direction (no conflict)
    if "unanimous" in dir_df.columns:
        unan_df = dir_df[dir_df["unanimous"] == True]
        unan_acc = float(unan_df["correct"].mean()) if len(unan_df) else None
        unan_n   = len(unan_df)
    else:
        unan_acc, unan_n = None, 0

    # Step 4: No hold_correct — HOLD means no position, no accuracy contribution.

    return {
        "total_signals":       n,
        "hold_signals":        len(hold_df),
        "hold_rate":           round(hold_rate, 4),
        "hold_rate_warning":   hold_rate_warning,
        "correct":             int(correct),
        "accuracy":            round(accuracy, 4),
        "win_rate":            round(win_rate, 4),
        "sharpe_ratio":        round(sharpe, 4),
        "cumulative_pnl":      round(cumulative_pnl, 6),
        "confusion_matrix":    {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "precision":           round(precision, 4),
        "recall":              round(recall, 4),
        "f1_score":            round(f1, 4),
        "long_accuracy":       round(long_acc, 4),
        "short_accuracy":      round(short_acc, 4),
        "long_signals":        len(long_df),
        "short_signals":       len(short_df),
        "unanimous_signals":   unan_n,
        "unanimous_accuracy":  round(unan_acc, 4) if unan_acc is not None else None,
    }


# ---------------------------------------------------------------------------
# Random baseline (for comparison)
# ---------------------------------------------------------------------------

def run_random_baseline(records: list[dict]) -> dict:
    """
    Simulate a coin-flip agent over the same set of outcomes.
    Returns stats in the same format as _compute_stats.
    """
    rng = np.random.default_rng(42)
    fake_records = []
    for r in records:
        signal = "LONG" if rng.random() > 0.5 else "SHORT"
        outcome = r["outcome"]
        correct = signal == outcome
        pnl     = r["pnl_magnitude"] if correct else -r["pnl_magnitude"]
        fake_records.append({
            "signal":  signal,
            "outcome": outcome,
            "correct": correct,
            "pnl":     pnl,
        })
    stats = _compute_stats(fake_records)
    stats["variant"] = "random_baseline"
    return stats


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Runs the full QuantAgent pipeline over a historical window
    and collects signal vs outcome statistics.
    """

    def __init__(self, trading_graph):
        self.trading_graph = trading_graph

    def run(
        self,
        ticker:    str,
        interval:  str,
        start:     datetime,
        end:       datetime,
        step:      int = MIN_STEP,
        max_runs:  int = 50,          # cap for demo / time budget
        progress_cb = None,           # optional callback(pct, msg)
    ) -> dict:
        """
        Execute the backtest.

        Parameters
        ----------
        ticker    : internal code or raw yf symbol
        interval  : e.g. "15m", "1h", "1d"
        start/end : historical date range
        step      : how many candles to slide the window each iteration
        max_runs  : maximum number of pipeline runs (each takes ~30-60 s)
        progress_cb : callback(float pct, str message) for streaming progress

        Returns
        -------
        Full results dict (also saved to JSON on disk)
        """
        run_start_time = time.time()

        def _progress(pct: float, msg: str, run_num: int = 0, total_runs: int = 0):
            if progress_cb:
                progress_cb(pct, msg)
            # Build a clean progress bar
            filled = int(pct / 5)   # 20 chars wide
            bar    = "█" * filled + "░" * (20 - filled)
            if run_num and total_runs:
                elapsed = time.time() - run_start_time
                eta     = (elapsed / run_num * (total_runs - run_num)) if run_num else 0
                eta_str = f"  ETA {int(eta//60)}m{int(eta%60):02d}s" if eta > 0 else ""
                log.info(f"|{bar}| {pct:5.1f}%  run {run_num}/{total_runs}{eta_str}  — {msg}")
            else:
                log.info(f"|{bar}| {pct:5.1f}%  — {msg}")

        _progress(0, f"Fetching {ticker} {interval} data…")

        df, yf_symbol, display_name = dp.fetch_ohlcv(ticker, interval, start, end)
        if df.empty:
            return {"error": f"No data for {ticker} ({interval})"}

        market   = dp.get_market(yf_symbol)
        horizon  = OUTCOME_HORIZON.get(interval, 2)
        n_rows   = len(df)

        # How many windows can we actually run?
        possible_starts = list(range(0, n_rows - WINDOW_SIZE - horizon, step))
        run_indices     = possible_starts[:max_runs]
        total_runs      = len(run_indices)

        if total_runs == 0:
            return {"error": f"Not enough data for backtest (need at least {WINDOW_SIZE + horizon} rows, got {n_rows})"}

        _progress(5, f"Starting {total_runs} pipeline runs on {display_name} ({market})")

        records:         list[dict] = []
        errors:          int        = 0
        skipped_windows: int        = 0
        start_time:      float      = time.time()

        for run_num, window_start in enumerate(run_indices):
            window_end   = window_start + WINDOW_SIZE
            signal_idx   = window_end - 1          # last candle in window
            outcome_label = _label_outcome(df, signal_idx, horizon)

            if outcome_label is None:
                continue

            # Slice the window
            window_df = df.iloc[window_start:window_end].copy().reset_index(drop=True)

            # Build kline_data dict (same format QuantAgent expects)
            kline_data = {
                "Datetime": window_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                "Open":     window_df["Open"].tolist(),
                "High":     window_df["High"].tolist(),
                "Low":      window_df["Low"].tolist(),
                "Close":    window_df["Close"].tolist(),
            }

            # Build display timeframe string
            tf_display = interval
            if interval.endswith("h"):   tf_display += "our"
            elif interval.endswith("m"): tf_display += "in"
            elif interval.endswith("d"): tf_display += "ay"

            # Pre-generate images (same as web_interface.run_analysis)
            try:
                import static_util
                p_img = static_util.generate_kline_image(kline_data)
                t_img = static_util.generate_trend_image(kline_data)
            except Exception:
                p_img = {"pattern_image": None}
                t_img = {"trend_image": None}

            initial_state = {
                "kline_data":    kline_data,
                "analysis_results": None,
                "messages":      [],
                "time_frame":    tf_display,
                "stock_name":    display_name,
                "pattern_image": p_img.get("pattern_image"),
                "trend_image":   t_img.get("trend_image"),
            }

            try:
                final_state = self.trading_graph.graph.invoke(initial_state)
                raw_decision = final_state.get("final_trade_decision", "")
                parsed = _parse_decision(raw_decision)

                if parsed is None:
                    errors += 1
                    continue

                signal  = parsed["decision"]
                entry   = float(df["Close"].iloc[signal_idx])
                exit_p  = float(df["Close"].iloc[signal_idx + horizon])
                pnl_mag = abs(exit_p - entry) / entry  # fractional move

                # Step 6: Skip trivially small moves — they are noise, not signal.
                if signal != "HOLD" and pnl_mag < MIN_MOVE_THRESHOLD:
                    skipped_windows += 1
                    log.info(f"Run {run_num+1}: skipped (move {pnl_mag:.4%} < threshold)")
                    continue

                # Step 4: HOLD = no position, no P&L, no accuracy contribution.
                # Just record it for hold_rate tracking. No hold_correct logic.
                if signal == "HOLD":
                    records.append({
                        "run":             run_num,
                        "window_start":    str(df["Datetime"].iloc[window_start]),
                        "signal_time":     str(df["Datetime"].iloc[signal_idx]),
                        "signal":          "HOLD",
                        "outcome":         outcome_label,
                        "correct":         False,
                        "pnl":             0.0,
                        "pnl_magnitude":   pnl_mag,
                        "entry_price":     entry,
                        "exit_price":      exit_p,
                        "unanimous":       final_state.get("unanimous_signal", False),
                        "indicator_report": final_state.get("indicator_report", ""),
                        "pattern_report":   final_state.get("pattern_report",  ""),
                        "trend_report":     final_state.get("trend_report",    ""),
                    })
                    continue

                correct = signal == outcome_label
                pnl     = pnl_mag if correct else -pnl_mag

                # Track whether all 3 agents unanimously agreed direction
                unanimous = final_state.get("unanimous_signal", False)

                records.append({
                    "run":             run_num,
                    "window_start":    str(df["Datetime"].iloc[window_start]),
                    "signal_time":     str(df["Datetime"].iloc[signal_idx]),
                    "signal":          signal,
                    "outcome":         outcome_label,
                    "correct":         correct,
                    "pnl":             pnl,
                    "pnl_magnitude":   pnl_mag,
                    "entry_price":     entry,
                    "exit_price":      exit_p,
                    "unanimous":       unanimous,
                    "indicator_report": final_state.get("indicator_report", ""),
                    "pattern_report":   final_state.get("pattern_report",  ""),
                    "trend_report":     final_state.get("trend_report",    ""),
                })

            except Exception as exc:
                errors += 1
                log.warning(f"Run {run_num+1} failed: {exc}")

            pct = 5 + 90 * (run_num + 1) / total_runs
            elapsed = time.time() - start_time
            eta = (elapsed / (run_num + 1)) * (total_runs - run_num - 1)
            _progress(pct, f"{len(records)} signals collected", run_num+1, total_runs)

        # Step 6: Log skip rate and warn if too high
        total_evaluated = len(records) + skipped_windows
        skip_rate = skipped_windows / total_evaluated if total_evaluated > 0 else 0.0
        if skip_rate > 0.20:
            log.warning(
                f"⚠ {skipped_windows}/{total_evaluated} windows ({skip_rate:.0%}) "
                f"skipped by MIN_MOVE_THRESHOLD ({MIN_MOVE_THRESHOLD:.4%}). "
                f"Consider lowering the threshold."
            )
        else:
            log.info(f"Skip rate: {skipped_windows}/{total_evaluated} ({skip_rate:.0%})")

        _progress(95, "Computing statistics and saving results…")

        if not records:
            return {"error": "All pipeline runs failed or produced no parseable decisions"}

        # Full-agent stats
        agent_stats = _compute_stats(records, interval)
        agent_stats["variant"] = "full_quantagent"

        # Step 7: Log hold rate prominently
        hold_rate = agent_stats.get("hold_rate", 0)
        if agent_stats.get("hold_rate_warning"):
            log.warning(
                f"⚠ HOLD rate is {hold_rate:.0%} — more than 50% of signals are HOLD. "
                f"The gate may be too conservative."
            )

        # Random baseline (same outcomes, random signals)
        baseline_stats = run_random_baseline(records)

        # Fix 4: Auto-train prediction layer on this backtest and compute refined stats
        prediction_stats = None
        prediction_meta  = None
        try:
            from prediction_layer import PredictionLayer
            pred_layer = PredictionLayer()
            train_result = pred_layer.train(result_for_train := {"records": records, "meta": {
                "ticker": ticker, "interval": interval,
            }})
            if train_result.get("success"):
                # Re-predict each directional record with the trained model
                pred_records = []
                for r in records:
                    if r["signal"] not in ("LONG", "SHORT"):
                        pred_records.append(r)  # HOLD passes through
                        continue
                    pred = pred_layer.predict({
                        "decision":          r["signal"],
                        "indicator_report":  r.get("indicator_report", ""),
                        "pattern_report":    r.get("pattern_report",  ""),
                        "trend_report":      r.get("trend_report",    ""),
                        "risk_reward_ratio": 1.5,
                    })
                    refined_signal = pred["decision"]
                    correct = refined_signal == r["outcome"]
                    pnl     = r["pnl_magnitude"] if correct else -r["pnl_magnitude"]
                    pred_records.append({**r,
                        "signal":  refined_signal,
                        "correct": correct,
                        "pnl":     pnl,
                    })
                prediction_stats = _compute_stats(pred_records, interval)
                prediction_stats["variant"] = "quantagent_plus_prediction"
                prediction_meta = train_result
                log.ok(f"Prediction layer: train_acc={train_result.get('train_accuracy', 0):.1%}")
        except Exception as exc:
            log.warning(f"Prediction layer skipped: {exc}")

        # P&L curve (cumulative, for chart)
        pnl_curve = list(pd.Series([r["pnl"] for r in records]).cumsum().round(6))

        # Rolling accuracy (window of 10)
        correct_series = pd.Series([int(r["correct"]) for r in records])
        rolling_acc = correct_series.rolling(10, min_periods=1).mean().round(4).tolist()

        result = {
            "meta": {
                "ticker":           ticker,
                "yf_symbol":        yf_symbol,
                "display_name":     display_name,
                "market":           market,
                "interval":         interval,
                "start":            str(start.date()),
                "end":              str(end.date()),
                "total_runs":       total_runs,
                "errors":           errors,
                "outcome_horizon":  horizon,
                "skipped_windows":  skipped_windows,
                "skip_rate":        round(skip_rate, 4),
                "run_at":           datetime.now().isoformat(),
            },
            "agent_stats":      agent_stats,
            "baseline_stats":   baseline_stats,
            "prediction_stats": prediction_stats,
            "prediction_meta":  prediction_meta,
            "pnl_curve":        pnl_curve,
            "rolling_accuracy": rolling_acc,
            "records":          records,
        }

        # Save to disk
        filename = (
            f"{ticker}_{interval}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
        )
        out_path = RESULTS_DIR / filename
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        _progress(100, f"Complete — {len(records)} signals  |  accuracy={agent_stats['accuracy']:.1%}  sharpe={agent_stats['sharpe_ratio']:.2f}", total_runs, total_runs)
        log.ok(f"Results saved → {out_path.name}")

        return result


# ---------------------------------------------------------------------------
# Results loader (used by web API)
# ---------------------------------------------------------------------------

def list_backtest_results() -> list[dict]:
    """Return a list of available saved backtest result summaries."""
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            meta  = data.get("meta", {})
            stats = data.get("agent_stats", {})
            results.append({
                "filename":     path.name,
                "ticker":       meta.get("ticker"),
                "display_name": meta.get("display_name"),
                "market":       meta.get("market"),
                "interval":     meta.get("interval"),
                "start":        meta.get("start"),
                "end":          meta.get("end"),
                "total_signals": stats.get("total_signals"),
                "accuracy":     stats.get("accuracy"),
                "sharpe_ratio": stats.get("sharpe_ratio"),
                "run_at":       meta.get("run_at"),
            })
        except Exception:
            pass
    return results


def load_backtest_result(filename: str) -> dict | None:
    """Load a full backtest result by filename."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
