"""
Indicator agent — computes and interprets technical indicators.
Optimised for token efficiency: tools are called once, no redundant passes.

Step 5 fix: outputs structured DIRECTION/CONFIDENCE footer for the gate.
"""

import copy
import json
from langchain_core.messages import HumanMessage

from qa_logger import get_logger
_log = get_logger("IndicatorAgent")

N_LAST = 5  # how many trailing values to send to the LLM (was full 28 — caused confusion)


def _trim_series(val: dict | list | any, n: int = N_LAST) -> dict | list | any:
    """Keep only the last N values of any list inside a result dict."""
    if isinstance(val, dict):
        return {k: (v[-n:] if isinstance(v, list) else v) for k, v in val.items()}
    if isinstance(val, list):
        return val[-n:]
    return val


def _parse_structured_fields(text: str) -> tuple[str, str]:
    """
    Extract DIRECTION: and CONFIDENCE: from agent response text.
    Returns (direction, confidence) — defaults to empty string if not found.
    """
    direction = ""
    confidence = ""
    for line in text.splitlines():
        line_upper = line.strip().upper()
        if line_upper.startswith("DIRECTION:"):
            val = line_upper.split(":", 1)[1].strip()
            if val in ("BULLISH", "BEARISH", "NEUTRAL"):
                direction = val
        elif line_upper.startswith("CONFIDENCE:"):
            val = line_upper.split(":", 1)[1].strip()
            if val in ("HIGH", "MEDIUM", "LOW"):
                confidence = val
    return direction, confidence


def create_indicator_agent(llm, toolkit):

    def indicator_agent_node(state):
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        tool_map = {t.name: t for t in tools}

        time_frame  = state["time_frame"]
        stock_name  = state["stock_name"]
        kline_data  = state["kline_data"]

        # —— Step 1: compute all indicators locally (no LLM call needed)
        kd = copy.deepcopy(kline_data)
        results = {}
        for tool in tools:
            try:
                raw = tool.invoke({"kline_data": kd})
                results[tool.name] = _trim_series(raw)  # trim to last 5 values
            except Exception as e:
                results[tool.name] = {"error": str(e)}

        # —— Step 2: single LLM call with trimmed results
        # Sending only last 5 values prevents the LLM from averaging the array
        # or picking the wrong time index when arrays are 28 elements long.
        summary = "\n".join(
            f"{name}: {json.dumps(val)}" for name, val in results.items()
        )

        prompt = (
            f"You are a technical analyst reviewing {stock_name} on a {time_frame} chart.\n\n"
            f"Most recent indicator values (last 5 bars, rightmost = current):\n{summary}\n\n"
            "Write a concise analysis (max 200 words) covering:\n"
            "1. Momentum signals (MACD, ROC)\n"
            "2. Oscillator readings (RSI, Stochastic, Williams %R)\n"
            "3. Overall directional bias: state exactly one of BULLISH / BEARISH / NEUTRAL\n\n"
            "Focus on the CURRENT (rightmost) values. State the exact current RSI and MACD histogram value.\n\n"
            "IMPORTANT: End your response with exactly these two lines (no markdown, no extra text):\n"
            "DIRECTION: BULLISH|BEARISH|NEUTRAL\n"
            "CONFIDENCE: HIGH|MEDIUM|LOW"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        _log.info(f"{stock_name} indicator analysis complete")

        # Parse structured fields from response
        direction, confidence = _parse_structured_fields(response.content)
        if direction:
            _log.info(f"  Structured: DIRECTION={direction}  CONFIDENCE={confidence}")
        else:
            _log.warning("  Could not parse structured DIRECTION from indicator response")

        return {
            "messages":              state.get("messages", []) + [response],
            "indicator_report":      response.content,
            "indicator_direction":   direction,
            "indicator_confidence":  confidence,
        }

    return indicator_agent_node
