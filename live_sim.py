"""
live_sim.py
-----------
Live simulation engine for QuantAgent.

SSE event types:
  signal   - new analysis result
  tick     - lightweight price update
  status   - engine state change
  progress - step-by-step pipeline progress
  accuracy - outcome resolved for a past signal (live accuracy tracker)
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import data_pipeline as dp
from qa_logger import get_logger
_log = get_logger("LiveSim")
import static_util

SIGNAL_LOG_DIR = Path("data/live_signals")
SIGNAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIGNAL_HISTORY = 200
MIN_POLL_SECONDS   = 30

POLL_INTERVALS: dict[str, int] = {
    "1m":  60,
    "5m":  60,
    "15m": 120,
    "30m": 180,
    "1h":  300,
    "4h":  600,
    "1d":  900,
    "1wk": 1800,
    "1mo": 3600,
}

# How many candles ahead to check the outcome
OUTCOME_HORIZON: dict[str, int] = {
    "1m":  3,
    "5m":  3,
    "15m": 3,
    "30m": 2,
    "1h":  2,
    "4h":  2,
    "1d":  1,
    "1wk": 1,
    "1mo": 1,
}

# Minutes per candle (for computing outcome check time)
CANDLE_MINUTES: dict[str, int] = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "30m": 30,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
    "1wk": 10080,
    "1mo": 43200,
}


def _poll_interval(interval: str) -> int:
    return max(POLL_INTERVALS.get(interval, 300), MIN_POLL_SECONDS)


def _candles_to_kline_dict(df: pd.DataFrame) -> dict:
    required = ["Datetime", "Open", "High", "Low", "Close"]
    return {
        col: (df[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
              if col == "Datetime" else df[col].tolist())
        for col in required if col in df.columns
    }


def _tf_display(interval: str) -> str:
    if interval.endswith("h"):   return interval + "our"
    if interval.endswith("m"):   return interval + "in"
    if interval.endswith("d"):   return interval + "ay"
    if interval == "1wk":        return "1 week"
    if interval == "1mo":        return "1 month"
    return interval


def sse_event(event_type: str, payload) -> str:
    data = json.dumps({"type": event_type, "payload": payload}, default=str)
    return f"data: {data}\n\n"


class LiveSimEngine:

    def __init__(self, trading_graph, prediction_layer):
        self.trading_graph    = trading_graph
        self.prediction_layer = prediction_layer

        self._ticker:  str  = ""
        self._interval: str = ""
        self._running: bool = False
        self._thread:  threading.Thread | None = None
        self._lock:    threading.Lock = threading.Lock()

        self._subscribers: list[queue.Queue] = []
        self._sub_lock:    threading.Lock    = threading.Lock()

        self.signal_history:  list[dict] = []
        self.pending_outcomes: list[dict] = []   # signals awaiting outcome check

        self.session_stats: dict = {
            "total_signals":    0,
            "long_signals":     0,
            "short_signals":    0,
            "resolved_signals": 0,
            "correct_signals":  0,
            "live_accuracy":    None,   # None until at least 1 resolved
            "session_start":    None,
            "ticker":           "",
            "interval":         "",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, ticker: str, interval: str) -> dict:
        with self._lock:
            if self._running:
                self.stop()

            self._ticker   = ticker
            self._interval = interval
            self._running  = True

            self.signal_history   = []
            self.pending_outcomes = []
            self.session_stats    = {
                "total_signals":    0,
                "long_signals":     0,
                "short_signals":    0,
                "resolved_signals": 0,
                "correct_signals":  0,
                "live_accuracy":    None,
                "session_start":    datetime.now().isoformat(),
                "ticker":           ticker,
                "interval":         interval,
            }

            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name=f"LiveSim-{ticker}-{interval}",
            )
            self._thread.start()

        return {"started": True, "ticker": ticker, "interval": interval}

    def stop(self) -> dict:
        with self._lock:
            self._running = False
            self._broadcast("status", {
                "state":   "stopped",
                "ticker":  self._ticker,
                "message": "Simulation stopped",
            })
        return {"stopped": True}

    def status(self) -> dict:
        return {
            "running":       self._running,
            "ticker":        self._ticker,
            "interval":      self._interval,
            "session_stats": self.session_stats,
            "history_count": len(self.signal_history),
        }

    def subscribe(self):
        q: queue.Queue = queue.Queue(maxsize=50)
        with self._sub_lock:
            self._subscribers.append(q)

        yield sse_event("status", {
            "state":         "connected",
            "running":       self._running,
            "ticker":        self._ticker,
            "interval":      self._interval,
            "session_stats": self.session_stats,
        })

        for sig in self.signal_history[-5:]:
            yield sse_event("signal", sig)

        try:
            while True:
                try:
                    event_str = q.get(timeout=20)
                    yield event_str
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with self._sub_lock:
                self._subscribers.remove(q)

    def get_history(self) -> list[dict]:
        return [
            {k: v for k, v in s.items() if k not in ("pattern_chart", "trend_chart")}
            for s in self.signal_history
        ]

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def _broadcast(self, event_type: str, payload):
        msg = sse_event(event_type, payload)
        with self._sub_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def _progress(self, step: str, detail: str = ""):
        self._broadcast("progress", {"step": step, "detail": detail})

    # ------------------------------------------------------------------
    # Outcome tracker
    # ------------------------------------------------------------------

    def _check_pending_outcomes(self):
        """
        Check any signals whose outcome window has closed.
        Fetches current price and compares against entry price.
        Broadcasts an 'accuracy' event for each resolved signal.
        """
        if not self.pending_outcomes:
            return

        now = datetime.now()
        still_pending = []

        for pending in self.pending_outcomes:
            check_after = datetime.fromisoformat(pending["check_after"])
            if now < check_after:
                still_pending.append(pending)
                continue

            # Time to resolve — fetch current price
            try:
                fetch_start = now - timedelta(hours=2)
                df, _, _ = dp.fetch_ohlcv(
                    self._ticker, self._interval, fetch_start, now
                )
                if df.empty:
                    still_pending.append(pending)
                    continue

                exit_price   = float(df["Close"].iloc[-1])
                entry_price  = pending["entry_price"]
                signal_dir   = pending["decision"]

                # Determine outcome
                price_moved_up = exit_price > entry_price
                if signal_dir == "LONG":
                    correct = price_moved_up
                elif signal_dir == "SHORT":
                    correct = not price_moved_up
                else:
                    still_pending.append(pending)
                    continue

                pnl_pct = ((exit_price - entry_price) / entry_price * 100)
                if signal_dir == "SHORT":
                    pnl_pct = -pnl_pct

                # Update session stats
                self.session_stats["resolved_signals"] += 1
                if correct:
                    self.session_stats["correct_signals"] += 1

                resolved = self.session_stats["resolved_signals"]
                correct_count = self.session_stats["correct_signals"]
                live_acc = round(correct_count / resolved, 4) if resolved else None
                self.session_stats["live_accuracy"] = live_acc

                # Update signal in history
                for sig in self.signal_history:
                    if sig["id"] == pending["signal_id"]:
                        sig["outcome"] = "correct" if correct else "incorrect"
                        sig["exit_price"] = round(exit_price, 4)
                        sig["pnl_pct"]    = round(pnl_pct, 3)
                        break

                # Broadcast accuracy update
                self._broadcast("accuracy", {
                    "signal_id":        pending["signal_id"],
                    "decision":         signal_dir,
                    "entry_price":      round(entry_price, 4),
                    "exit_price":       round(exit_price, 4),
                    "correct":          correct,
                    "pnl_pct":          round(pnl_pct, 3),
                    "resolved_signals": resolved,
                    "correct_signals":  correct_count,
                    "live_accuracy":    live_acc,
                    "live_accuracy_pct": round(live_acc * 100, 1) if live_acc else None,
                    "session_stats":    self.session_stats,
                })

                outcome_str = "CORRECT" if correct else "WRONG"
                _log.signal(f"{signal_dir} → {outcome_str}  entry={entry_price:.4f}  exit={exit_price:.4f}  pnl={pnl_pct:+.2f}%  |  live acc: {live_acc:.1%} ({correct_count}/{resolved})")

            except Exception as e:
                _log.warning(f"Outcome check error: {e}")
                still_pending.append(pending)

        self.pending_outcomes = still_pending

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self):
        poll_secs = _poll_interval(self._interval)

        self._broadcast("status", {
            "state":              "started",
            "ticker":             self._ticker,
            "interval":           self._interval,
            "poll_every_seconds": poll_secs,
            "message":            f"Live simulation started — analysing every {poll_secs}s",
        })

        while self._running:
            try:
                # Check outcomes of past signals first
                self._check_pending_outcomes()
                # Run new analysis cycle
                self._run_one_cycle()
            except Exception as exc:
                _log.error(f"Cycle error: {exc}")
                self._broadcast("status", {
                    "state":   "error",
                    "message": str(exc),
                })

            for _ in range(poll_secs):
                if not self._running:
                    return
                time.sleep(1)

    def _run_one_cycle(self):
        ticker   = self._ticker
        interval = self._interval
        now      = datetime.now()

        self._progress("fetching", f"Fetching latest {interval} candles for {ticker}…")

        lookback_map = {
            "1m": 5, "5m": 5, "15m": 5, "30m": 7,
            "1h": 10, "4h": 14, "1d": 90, "1wk": 365, "1mo": 730,
        }
        lookback_days = lookback_map.get(interval, 5)
        start = now - timedelta(days=lookback_days)

        df, yf_symbol, display_name = dp.fetch_ohlcv(ticker, interval, start, now)

        if df.empty or len(df) < 10:
            msg = (
                f"No data for {ticker} ({interval}). "
                "Market may be closed or interval too short. "
                "Try a longer interval (1h/1d) or crypto (BTC). Retrying next cycle."
            )
            self._broadcast("status", {"state": "waiting", "message": msg})
            return

        # Tick update
        latest = df.iloc[-1]
        self._broadcast("tick", {
            "ticker":          ticker,
            "time":            str(latest["Datetime"]),
            "open":            round(float(latest["Open"]),  4),
            "high":            round(float(latest["High"]),  4),
            "low":             round(float(latest["Low"]),   4),
            "close":           round(float(latest["Close"]), 4),
            "candles_fetched": len(df),
        })

        window     = df.tail(45).copy().reset_index(drop=True)
        kline_data = _candles_to_kline_dict(window)
        tf_display = _tf_display(interval)

        self._progress("charting", "Generating candlestick and trend charts…")
        try:
            p_img = static_util.generate_kline_image(kline_data)
            t_img = static_util.generate_trend_image(kline_data)
        except Exception as e:
            p_img = {"pattern_image": None}
            t_img = {"trend_image": None}
            _log.warning(f"Chart generation error: {e}")

        self._progress("indicators", "Running indicator agent…")
        initial_state = {
            "kline_data":       kline_data,
            "analysis_results": None,
            "messages":         [],
            "time_frame":       tf_display,
            "stock_name":       display_name,
            "pattern_image":    p_img.get("pattern_image"),
            "trend_image":      t_img.get("trend_image"),
        }

        self._progress("pattern",  "Running pattern agent…")
        self._progress("trend",    "Running trend agent…")
        self._progress("decision", "Running decision maker…")

        final_state = self.trading_graph.graph.invoke(initial_state)

        raw_decision = final_state.get("final_trade_decision", "")
        parsed_decision = {}
        try:
            s = raw_decision.find("{")
            e = raw_decision.rfind("}") + 1
            if s != -1 and e > 0:
                parsed_decision = json.loads(raw_decision[s:e])
        except Exception:
            parsed_decision = {"decision": "N/A", "justification": raw_decision}

        self._progress("prediction", "Running prediction layer…")
        prediction = self.prediction_layer.predict({
            "decision":          parsed_decision.get("decision", ""),
            "indicator_report":  final_state.get("indicator_report", ""),
            "pattern_report":    final_state.get("pattern_report",  ""),
            "trend_report":      final_state.get("trend_report",    ""),
            "risk_reward_ratio": parsed_decision.get("risk_reward_ratio", 1.5),
        })

        entry_price  = round(float(df["Close"].iloc[-1]), 4)
        signal_time  = str(df["Datetime"].iloc[-1])
        final_decision = (prediction.get("decision") or
                          parsed_decision.get("decision", "N/A")).upper()

        # Compute when to check the outcome
        horizon_candles = OUTCOME_HORIZON.get(interval, 2)
        candle_mins     = CANDLE_MINUTES.get(interval, 60)
        check_after     = now + timedelta(minutes=horizon_candles * candle_mins)

        signal_id = f"{ticker}_{now.strftime('%Y%m%d_%H%M%S')}"

        signal = {
            "id":               signal_id,
            "timestamp":        now.isoformat(),
            "ticker":           ticker,
            "yf_symbol":        yf_symbol,
            "display_name":     display_name,
            "interval":         interval,
            "signal_time":      signal_time,
            "close_at_signal":  entry_price,
            "raw_decision":     parsed_decision,
            "indicator_report": final_state.get("indicator_report", ""),
            "pattern_report":   final_state.get("pattern_report",  ""),
            "trend_report":     final_state.get("trend_report",    ""),
            "prediction":       prediction,
            "pattern_chart":    p_img.get("pattern_image", ""),
            "trend_chart":      t_img.get("trend_image",   ""),
            "outcome":          "pending",     # will be updated when resolved
            "exit_price":       None,
            "pnl_pct":          None,
            "check_after":      check_after.isoformat(),
            "outcome_horizon":  horizon_candles,
            "ohlcv": {
                "datetime": kline_data["Datetime"],
                "open":     [round(v, 4) for v in kline_data["Open"]],
                "high":     [round(v, 4) for v in kline_data["High"]],
                "low":      [round(v, 4) for v in kline_data["Low"]],
                "close":    [round(v, 4) for v in kline_data["Close"]],
            },
            "session_stats": self.session_stats,
        }

        # Queue for outcome tracking
        if final_decision in ("LONG", "SHORT"):
            self.pending_outcomes.append({
                "signal_id":   signal_id,
                "decision":    final_decision,
                "entry_price": entry_price,
                "check_after": check_after.isoformat(),
            })
            _log.signal(f"{final_decision} @ {entry_price}  |  outcome check at {check_after.strftime('%H:%M')}  ({horizon_candles} candle horizon)")

        self.signal_history.append(signal)
        if len(self.signal_history) > MAX_SIGNAL_HISTORY:
            self.signal_history.pop(0)

        self.session_stats["total_signals"] += 1
        if final_decision == "LONG":
            self.session_stats["long_signals"] += 1
        elif final_decision == "SHORT":
            self.session_stats["short_signals"] += 1

        self._save_signal(signal)
        self._progress("done", "Analysis complete")
        self._broadcast("signal", signal)

    def _save_signal(self, signal: dict):
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            log_path = SIGNAL_LOG_DIR / f"{self._ticker}_{self._interval}_{date_str}.jsonl"
            slim = {k: v for k, v in signal.items()
                    if k not in ("pattern_chart", "trend_chart", "ohlcv")}
            with open(log_path, "a") as f:
                f.write(json.dumps(slim, default=str) + "\n")
        except Exception as e:
            _log.warning(f"Failed to save signal log: {e}")
