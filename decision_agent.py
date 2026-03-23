"""
decision_agent.py — Weighted gate + LLM synthesis for QuantAgent.

Gate logic:
- Structured DIRECTION/CONFIDENCE fields from agents
- Indicator agent base weight 2x
- Pattern downweighted (0.7x — less reliable)
- Confidence multiplier: HIGH=1.5x, MEDIUM=1.0x, LOW=0.5x
- Neutral votes halved (NEUTRAL ≠ directional)
- Pattern-only trades rejected
- Asymmetric LONG filter (indicator must not be BEARISH)
- Weak-signal filter → defer to LLM (allows HOLD)
- Strong-gate short-circuit (absolute + relative thresholds)
"""

import json
from langchain_core.messages import HumanMessage
from qa_logger import get_logger

_log = get_logger("DecisionAgent")


HORIZON_MAP = {
    "1min": "next 5 minutes (5 candles)",
    "5min": "next 15 minutes (3 candles)",
    "15min": "next 45 minutes (3 candles)",
    "30min": "next 1 hour (2 candles)",
    "1hour": "next 2 hours (2 candles)",
    "4hour": "next 8 hours (2 candles)",
    "1day": "next 2 trading days",
    "1 week": "next 2 weeks",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structured_to_score(direction: str):
    d = (direction or "").upper().strip()
    if d == "BULLISH":
        return 1
    if d == "BEARISH":
        return -1
    if d == "NEUTRAL":
        return 0
    return None


def _confidence_multiplier(conf: str) -> float:
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
    Returns:
        (signal, unanimous)
        signal: LONG / SHORT / None (LLM decides)
        unanimous: True if gate skipped LLM
    """

    ind = _structured_to_score(state.get("indicator_direction"))
    pat = _structured_to_score(state.get("pattern_direction"))
    trn = _structured_to_score(state.get("trend_direction"))

    ind_conf = _confidence_multiplier(state.get("indicator_confidence"))
    pat_conf = _confidence_multiplier(state.get("pattern_confidence"))
    trn_conf = _confidence_multiplier(state.get("trend_confidence"))

    _log.info(
        f"Scores: ind={ind} pat={pat} trn={trn} "
        f"conf=[{state.get('indicator_confidence','?')},"
        f"{state.get('pattern_confidence','?')},"
        f"{state.get('trend_confidence','?')}]"
    )

    if ind is None and pat is None and trn is None:
        _log.info("Gate: no structured signals — deferring to LLM")
        return None, False

    # ----- Weighted voting -----
    weights = [
        (ind, 1.7 * ind_conf),
        (pat, 0.7 * pat_conf),  # ↓ pattern weaker
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
            neut += w * 0.5

    _log.info(f"Weighted: bull={bull:.2f} bear={bear:.2f} neut={neut:.2f}")

    # ----- HARD FILTERS -----

    if pat == 1 and ind == 0 and trn == 0:
        _log.info("Reject: pattern-only LONG")
        return None, False

    if pat == -1 and ind == 0 and trn == 0:
        _log.info("Reject: pattern-only SHORT")
        return None, False

    if bull > bear and ind == -1:
        _log.info("Reject LONG: indicator bearish")
        return None, False

    # ----- Majority -----

    if bull > bear and bull > (bear + 0.5 * neut):
        majority = "LONG"
        majority_w = bull
        minority_w = bear + neut

    elif bear > bull and bear > (bull + 0.5 * neut):
        majority = "SHORT"
        majority_w = bear
        minority_w = bull + neut

    else:
        _log.info("No clear majority — defer to LLM")
        return None, False

    # ----- Weak signal filter -----

    if abs(bull - bear) < 0.5:
        _log.info("Weak signal — defer to LLM (possible HOLD)")
        return None, False

    # ----- Strong signal short-circuit -----

    # Absolute strength
    if majority_w >= 2.5:
        _log.info(f"Strong {majority} ({majority_w:.2f}) — skip LLM")
        return majority, True

    # Relative dominance
    if minority_w == 0 or majority_w >= 2.5 * minority_w:
        _log.info(
            f"Dominant {majority} ({majority_w:.2f} vs {minority_w:.2f}) — skip LLM"
        )
        return majority, True

    return majority, False


# ---------------------------------------------------------------------------
# Decision node
# ---------------------------------------------------------------------------


def create_final_trade_decider(llm):

    def trade_decision_node(state: dict):

        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        horizon = HORIZON_MAP.get(time_frame, f"next 2-3 {time_frame} candles")

        gate_signal, unanimous = _check_agreement(state)

        if gate_signal:
            gate_hint = f"\nConsensus: {gate_signal}. Prefer unless invalid."
        elif (
            state.get("pattern_direction") in ("BULLISH", "BEARISH")
            and state.get("indicator_direction") == "NEUTRAL"
            and state.get("trend_direction") == "NEUTRAL"
        ):
            _log.info("Hard override → HOLD (pattern-only case)")
            return {
                "final_trade_decision": json.dumps(
                    {
                        "decision": "HOLD",
                        "forecast_horizon": horizon,
                        "justification": "Pattern-only signal rejected",
                        "risk_reward_ratio": 1.5,
                        "signal_strength": "WEAK",
                    }
                ),
                "unanimous_signal": False,
                "messages": [],
            }
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

Return JSON:
{{"decision":"LONG|SHORT|HOLD",
"forecast_horizon":"{horizon}",
"justification":"brief",
"risk_reward_ratio":1.5,
"signal_strength":"STRONG|MODERATE|WEAK"}}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        try:
            parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
            if parsed.get("decision") not in ("LONG", "SHORT", "HOLD"):
                parsed["decision"] = "HOLD"
            content = json.dumps(parsed)
        except Exception:
            _log.warning("Parse failed → HOLD")
            content = json.dumps({"decision": "HOLD"})

        decision = "HOLD"
        try:
            decision = json.loads(content).get("decision", "HOLD")
        except Exception:
            pass

        _log.info(f"Decision: {decision} | gate={gate_signal} | unanimous={unanimous}")

        return {
            "final_trade_decision": content,
            "unanimous_signal": unanimous,
            "messages": [response],
        }

    return trade_decision_node
