"""
Pattern agent — vision-based candlestick pattern recognition.
Uses precomputed chart image from state (generated before pipeline runs).
Optimised: single LLM call, no tool-calling round trip.

Step 5 fix: outputs structured DIRECTION/CONFIDENCE footer for the gate.
"""

import time
from langchain_core.messages import HumanMessage, SystemMessage

from qa_logger import get_logger
_log = get_logger("PatternAgent")

PATTERN_LIST = """
1. Double Bottom / Double Top — W or M shape, strong reversal signal
2. Head and Shoulders / Inverse H&S — three peaks/troughs, reliable reversal
3. Bullish/Bearish Flag — brief consolidation after strong move, continuation
4. Ascending/Descending/Symmetrical Triangle — converging price action
5. Rising/Falling Wedge — narrowing range, often reversal
6. Rectangle — horizontal consolidation between clear S/R levels
7. V-shaped Reversal — sharp turn with no base
8. Rounded Bottom/Top — gradual arc-shaped turn
9. No clear pattern — price action is choppy or inconclusive
"""

SYSTEM = (
    "You are a candlestick pattern recognition specialist. "
    "Analyse the chart image provided and identify the single most prominent pattern. "
    "Be concise and specific. Do not describe every candle — focus on the overall structure."
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


def create_pattern_agent(tool_llm, graph_llm, toolkit):

    def pattern_agent_node(state):
        time_frame       = state["time_frame"]
        pattern_image_b64 = state.get("pattern_image")

        if not pattern_image_b64:
            _log.warning("No precomputed pattern image — generating via tool")
            try:
                result = toolkit.generate_kline_image.invoke(
                    {"kline_data": state["kline_data"]}
                )
                pattern_image_b64 = result.get("pattern_image")
            except Exception as e:
                _log.error(f"Chart generation failed: {e}")
                return {
                    "messages":            state.get("messages", []),
                    "pattern_report":      "Pattern analysis unavailable — chart generation failed.",
                    "pattern_direction":   "",
                    "pattern_confidence":  "",
                }

        prompt_content = [
            {
                "type": "text",
                "text": (
                    f"This is a {time_frame} candlestick chart.\n\n"
                    f"Known patterns to check against:\n{PATTERN_LIST}\n\n"
                    "Instructions:\n"
                    "- Identify the single most prominent pattern visible\n"
                    "- State your confidence: HIGH / MEDIUM / LOW\n"
                    "- Explain in 3-4 sentences why this pattern is present\n"
                    "- State directional implication: BULLISH / BEARISH / NEUTRAL\n"
                    "- If no clear pattern exists, say so explicitly\n\n"
                    "IMPORTANT: End your response with exactly these two lines (no markdown, no extra text):\n"
                    "DIRECTION: BULLISH|BEARISH|NEUTRAL\n"
                    "CONFIDENCE: HIGH|MEDIUM|LOW"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
            },
        ]

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=prompt_content),
        ]

        response = invoke_with_retry(graph_llm.invoke, messages)
        _log.info(f"Pattern analysis complete")

        # Parse structured fields from response
        direction, confidence = _parse_structured_fields(response.content)
        if direction:
            _log.info(f"  Structured: DIRECTION={direction}  CONFIDENCE={confidence}")
        else:
            _log.warning("  Could not parse structured DIRECTION from pattern response")

        return {
            "messages":            state.get("messages", []) + [response],
            "pattern_report":      response.content,
            "pattern_direction":   direction,
            "pattern_confidence":  confidence,
        }

    return pattern_agent_node
