"""
qa_logger.py
------------
Centralised logging for QuantAgent.

All modules import from here:
    from qa_logger import get_logger
    log = get_logger("Backtest")

Output format:
    16:42:03  [Backtest   ]  INFO   Run 3/20 complete — accuracy so far 60.0% (3 signals)
    16:42:03  [LiveSim    ]  SIGNAL SHORT @ 3847.20 — outcome check at 18:42
    16:42:03  [DataPipeline]  OK     TCS.NS — 312 rows fetched
    16:42:03  [Server     ]  ERROR  backtest/run: ticker required
"""

import logging
import sys

# ── colour codes for terminal (Windows 10+ supports ANSI via CMD/PowerShell)
_RESET  = "\033[0m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"

LEVEL_COLORS = {
    "DEBUG":  _GREY,
    "INFO":   _CYAN,
    "OK":     _GREEN,
    "SIGNAL": _BOLD + _CYAN,
    "WARN":   _YELLOW,
    "ERROR":  _RED,
    "FATAL":  _BOLD + _RED,
}

# Custom level numbers
OK_LEVEL     = 25   # between INFO(20) and WARNING(30)
SIGNAL_LEVEL = 26
logging.addLevelName(OK_LEVEL,     "OK")
logging.addLevelName(SIGNAL_LEVEL, "SIGNAL")


class _QAFormatter(logging.Formatter):
    def format(self, record):
        time_str  = self.formatTime(record, "%H:%M:%S")
        name      = record.name[:12].ljust(12)
        level     = record.levelname[:6].ljust(6)
        color     = LEVEL_COLORS.get(record.levelname, "")
        msg       = record.getMessage()
        return f"{_GREY}{time_str}{_RESET}  [{_CYAN}{name}{_RESET}]  {color}{level}{_RESET}  {msg}"


class _QALogger(logging.Logger):
    def ok(self, msg, *args, **kwargs):
        if self.isEnabledFor(OK_LEVEL):
            self._log(OK_LEVEL, msg, args, **kwargs)

    def signal(self, msg, *args, **kwargs):
        if self.isEnabledFor(SIGNAL_LEVEL):
            self._log(SIGNAL_LEVEL, msg, args, **kwargs)


logging.setLoggerClass(_QALogger)

# Root handler — set up once
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_QAFormatter())
_handler.setLevel(logging.DEBUG)

# Suppress noisy third-party loggers
for noisy in ["urllib3", "yfinance", "peewee", "httpcore",
               "httpx", "openai", "anthropic", "langchain",
               "langsmith", "werkzeug"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# werkzeug request log — keep but simplify to only show non-200 or errors
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def get_logger(name: str) -> _QALogger:
    """Return a named QALogger, creating it if needed."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger  # type: ignore
