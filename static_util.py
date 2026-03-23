import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

import color_style as color
from graph_util import (
    fit_trendlines_high_low,
    fit_trendlines_single,
    get_line_points,
    split_line_into_segments,
)

matplotlib.use("Agg")


def generate_kline_image(kline_data) -> dict:
    """
    Generate a candlestick (K-line) chart from OHLCV data and return a base64-encoded image.
    FIX: plt.close() is now called AFTER base64 encoding, not before.
    """
    df = pd.DataFrame(kline_data)
    df = df.tail(45)  # use full window, not just 40

    try:
        df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
    except (ValueError, KeyError):
        df.index = pd.to_datetime(df["Datetime"])

    fig, axlist = mpf.plot(
        df[["Open", "High", "Low", "Close"]],
        type="candle",
        style=color.my_color_style,
        figsize=(12, 6),
        returnfig=True,
        block=False,
    )
    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")

    # Encode to base64 BEFORE closing the figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # close AFTER encode

    return {
        "pattern_image": img_b64,
        "pattern_image_description": "Candlestick chart returned as base64 string.",
    }



def generate_trend_image(kline_data) -> dict:
    """
    Generate a candlestick chart with trendlines from OHLCV data and return a base64-encoded image.
    FIX: plt.close() is now called AFTER base64 encoding, not before.
    """
    data = pd.DataFrame(kline_data)
    candles = data.iloc[-45:].copy()  # use exact window size

    candles["Datetime"] = pd.to_datetime(candles["Datetime"])
    candles.set_index("Datetime", inplace=True)

    support_coefs_c, resist_coefs_c = fit_trendlines_single(candles["Close"])
    support_coefs, resist_coefs = fit_trendlines_high_low(
        candles["High"], candles["Low"], candles["Close"]
    )

    support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
    resist_line_c  = resist_coefs_c[0]  * np.arange(len(candles)) + resist_coefs_c[1]
    support_line   = support_coefs[0]   * np.arange(len(candles)) + support_coefs[1]
    resist_line    = resist_coefs[0]    * np.arange(len(candles)) + resist_coefs[1]

    s_seq  = get_line_points(candles, support_line)
    r_seq  = get_line_points(candles, resist_line)
    s_seq2 = get_line_points(candles, support_line_c)
    r_seq2 = get_line_points(candles, resist_line_c)

    s_segments  = split_line_into_segments(s_seq)
    r_segments  = split_line_into_segments(r_seq)
    s2_segments = split_line_into_segments(s_seq2)
    r2_segments = split_line_into_segments(r_seq2)

    all_segments = s_segments + r_segments + s2_segments + r2_segments
    colors = (
        ["white"] * len(s_segments)
        + ["white"] * len(r_segments)
        + ["blue"]  * len(s2_segments)
        + ["red"]   * len(r2_segments)
    )

    apds = [
        mpf.make_addplot(support_line_c, color="blue", width=1, label="Close Support"),
        mpf.make_addplot(resist_line_c,  color="red",  width=1, label="Close Resistance"),
    ]

    fig, axlist = mpf.plot(
        candles,
        type="candle",
        style=color.my_color_style,
        addplot=apds,
        alines=dict(alines=all_segments, colors=colors, linewidths=1),
        returnfig=True,
        figsize=(12, 6),
        block=False,
    )
    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")

    # Encode to base64 BEFORE closing the figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # close AFTER encode

    return {
        "trend_image": img_b64,
        "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines.",
    }
