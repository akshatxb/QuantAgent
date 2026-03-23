"""
decision_agent.py — Weighted gate + LLM synthesis for QuantAgent.

Gate logic:
- Structured DIRECTION/CONFIDENCE fields from agents
- Indicator agent base weight 2x
- Confidence multiplier: HIGH=1.5x, MEDIUM=1.0x, LOW=0.5x
- Neutral votes halved (NEUTRAL ≠ directional)
- Pattern-only trades rejected (pattern is noisy)
- Asymmetric LONG filter: indicator must not be BEARISH
- Strong-gate short-circuit: ≥3x majority skips LLM
"""

import json
from langchain_core.messages import HumanMessage

from qa_logger import get_logger
_log = get_logger("DecisionAgent")

HORIZON_MAP = {
    "1min":   "next 5 minutes (5 candles)",
    "5min":   "next 15 minutes (3 candles)",
    "15min":  "next 45 minutes (3 candles)",
    "30min":  "next 1 hour (2 candles)",
    "1hour":  "next 2 hours (2 candles)",
    "4hour":  "next 8 hours (2 candles)",
    "1day":   "next 2 trading days",
    "1 week": "next 2 weeks",
}

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _structured_to_score(direction: str):
    """Map structured direction string to numeric score."""
    d = (direction or "").upper().strip()
    if d == "BULLISH":
        return 1
    if d == "BEARISH":
        return -1
    if d == "NEUTRAL":
        return 0
    return None


def _confidence_multiplier(conf: str) -> float:
    """Map confidence string to vote multiplier."""
    c = (conf or "").upper().strip()
    if c == "HIGH":
        return 1.5
    if c == "LOW":
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def _check_agreement(state: dict) -> tuple[str | None, bool]:
    """
    Weighted consensus gate.

    Returns (direction_or_None, unanimous).
    - direction: "LONG", "SHORT", or None (let LLM decide)
    - unanimous: True when strong-gate short-circuit fired
    """
    ind = _structured_to_score(state.get("indicator_direction"))
    pat = _structured_to_score(state.get("pattern_direction"))
    trn = _structured_to_score(state.get("trend_direction"))

    ind_conf = _confidence_multiplier(state.get("indicator_confidence"))
    pat_conf = _confidence_multiplier(state.get("pattern_confidence"))
    trn_conf = _confidence_multiplier(state.get("trend_confidence"))

    _log.info(f"Scores: ind={ind} pat={pat} trn={trn}  "
              f"conf=[{state.get('indicator_confidence','?')},"
              f"{state.get('pattern_confidence','?')},"
              f"{state.get('trend_confidence','?')}]")

    # If nothing valid → let LLM decide
    if ind is None and pat is None and trn is None:
        _log.info("Gate: no structured signals — deferring to LLM")
        return None, False

    # ----- Weighted voting -----
    weights = [
        (ind, 2 * ind_conf),
        (pat, 1 * pat_conf),
        (trn, 1 * trn_conf),
    ]

    bull = bear = neut = 0.0

    for score, w in weights:
        if score is None:
            continue
        if score == 1:
            bull += w
        elif score == -1:
            bear += w
        else:
            neut += w * 0.5   # Neutral votes halved — NEUTRAL ≠ directional

    _log.info(f"Weighted: bull={bull:.2f} bear={bear:.2f} neut={neut:.2f}")

    # ----- HARD FILTERS (edge creators) -----

    # Reject pattern-only trades (pattern agent is noisy)
    if pat == 1 and ind == 0 and trn == 0:
        _log.info("Reject: pattern-only LONG — no confirmation")
        return None, False
    if pat == -1 and ind == 0 and trn == 0:
        _log.info("Reject: pattern-only SHORT — no confirmation")
        return None, False

    # Asymmetric LONG filter: indicator must not be BEARISH for LONG
    if bull > bear and ind == -1:
        _log.info("Reject LONG: indicator is BEARISH — asymmetric filter")
        return None, False

    # ----- Majority decision -----
    if bull > bear and bull > neut:
        majority = "LONG"
        majority_w = bull
        minority_w = bear + neut
    elif bear > bull and bear > neut:
        majority = "SHORT"
        majority_w = bear
        minority_w = bull + neut
    else:
        _log.info("Gate: no clear majority — deferring to LLM")
        return None, False

    # ----- Strong signal short-circuit -----
    if minority_w == 0 or majority_w >= 3 * minority_w:
        _log.info(f"Strong {majority} ({majority_w:.1f} vs {minority_w:.1f}) — skipping LLM")
        return majority, True

    return majority, False


# ---------------------------------------------------------------------------
# Decision node factory
# ---------------------------------------------------------------------------

def create_final_trade_decider(llm):
    """Create the LangGraph node for final trade decisions."""

    def trade_decision_node(state: dict) -> dict:

        indicator_report = state["indicator_report"]
        pattern_report   = state["pattern_report"]
        trend_report     = state["trend_report"]
        time_frame       = state["time_frame"]
        stock_name       = state["stock_name"]

        horizon = HORIZON_MAP.get(time_frame, f"next 2-3 {time_frame} candles")

        gate_signal, unanimous = _check_agreement(state)

        # ---- Gate hint for LLM ----
        if gate_signal:
            gate_hint = (f"\nConsensus direction: {gate_signal}. "
                         f"Prefer this unless clearly invalid.")
        else:
            gate_hint = "\nNo clear consensus. Use judgement."

        prompt = f"""You are a quantitative trader.

Stock: {stock_name}
Timeframe: {time_frame}
Horizon: {horizon}

INDICATOR:
{indicator_report}

PATTERN:
{pattern_report}

TREND:
{trend_report}
{gate_hint}

Rules:
- Majority wins
- HOLD only if no direction exists
- Do NOT bias LONG
- Require strong reasoning

Return JSON only:
{{"decision": "LONG|SHORT|HOLD",
"forecast_horizon": "{horizon}",
"justification": "brief reasoning",
"risk_reward_ratio": 1.5,
"signal_strength": "STRONG|MODERATE|WEAK"}}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        try:
            parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
            if parsed.get("decision") not in ("LONG", "SHORT", "HOLD"):
                parsed["decision"] = "HOLD"
            content = json.dumps(parsed)
        except Exception:
            _log.warning("LLM response parse failed — defaulting to HOLD")

        decision = "HOLD"
        try:
            decision = json.loads(content).get("decision", "HOLD")
        except Exception:
            pass

        _log.info(f"Decision: {decision}  gate={gate_signal}  unanimous={unanimous}")

        return {
            "final_trade_decision": content,
            "unanimous_signal": unanimous,
            "messages": [response],
        }

    return trade_decision_node
