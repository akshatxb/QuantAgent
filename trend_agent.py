"""
Trend agent — trendline chart analysis.
Uses precomputed trend image from state.
Optimised: single LLM call, no tool round trip.

Step 5 fix: outputs structured DIRECTION/CONFIDENCE footer for the gate.
"""

import time
from langchain_core.messages import HumanMessage, SystemMessage

from qa_logger import get_logger
_log = get_logger("TrendAgent")

SYSTEM = (
    "You are a technical analyst specialising in support/resistance trendline analysis. "
    "The chart shows a blue support line and red resistance line computed from recent price data. "
    "Analyse price interaction with these lines only — do not invent indicator values."
)


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


def invoke_with_retry(fn, *args, retries=3, wait=6):
    for attempt in range(retries):
        try:
            return fn(*args)
        except Exception as e:
            if attempt < retries - 1:
                _log.warning(f"Attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise


def create_trend_agent(tool_llm, graph_llm, toolkit):

    def trend_agent_node(state):
        time_frame     = state["time_frame"]
        trend_image_b64 = state.get("trend_image")

        if not trend_image_b64:
            _log.warning("No precomputed trend image — generating via tool")
            try:
                result = toolkit.generate_trend_image.invoke(
                    {"kline_data": state["kline_data"]}
                )
                trend_image_b64 = result.get("trend_image")
            except Exception as e:
                _log.error(f"Trend chart generation failed: {e}")
                return {
                    "messages":          state.get("messages", []),
                    "trend_report":      "Trend analysis unavailable — chart generation failed.",
                    "trend_image":       None,
                    "trend_direction":   "",
                    "trend_confidence":  "",
                }

        prompt_content = [
            {
                "type": "text",
                "text": (
                    f"This is a {time_frame} candlestick chart with support (blue) and resistance (red) trendlines.\n\n"
                    "Analyse and answer:\n"
                    "1. Support line slope: rising / flat / falling\n"
                    "2. Resistance line slope: rising / flat / falling\n"
                    "3. Price position: near support / near resistance / mid-range\n"
                    "4. Compression: are the lines converging? If so, estimate breakout proximity\n"
                    "5. Predicted short-term direction: UPWARD / DOWNWARD / SIDEWAYS\n"
                    "6. Confidence: HIGH / MEDIUM / LOW\n\n"
                    "Keep response under 150 words. "
                    "Only describe what you see in the chart — do not reference ADX, RSI, or any indicator not visible in this image.\n\n"
                    "IMPORTANT: End your response with exactly these two lines (no markdown, no extra text):\n"
                    "DIRECTION: BULLISH|BEARISH|NEUTRAL\n"
                    "CONFIDENCE: HIGH|MEDIUM|LOW"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
            },
        ]

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=prompt_content),
        ]

        response = invoke_with_retry(graph_llm.invoke, messages)
        _log.info("Trend analysis complete")

        # Parse structured fields from response
        direction, confidence = _parse_structured_fields(response.content)
        if direction:
            _log.info(f"  Structured: DIRECTION={direction}  CONFIDENCE={confidence}")
        else:
            _log.warning("  Could not parse structured DIRECTION from trend response")

        return {
            "messages":          state.get("messages", []) + [response],
            "trend_report":      response.content,
            "trend_image":       trend_image_b64,
            "trend_direction":   direction,
            "trend_confidence":  confidence,
        }

    return trend_agent_node
