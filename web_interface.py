import json
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from flask import Flask, Response, jsonify, render_template, request, send_file

import static_util
from trading_graph import TradingGraph
import data_pipeline as dp
import backtest as bt
from prediction_layer import PredictionLayer
from live_sim import LiveSimEngine
import ablation as abl

from qa_logger import get_logger
log = get_logger("Server")

app = Flask(__name__)


class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer."""
        from default_config import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG.copy()
        self.trading_graph = TradingGraph(config=self.config)
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Thin shim: build asset_mapping from the data_pipeline catalogue
        # so old code that references self.asset_mapping still works
        self.asset_mapping = {
            entry["code"]: entry["name"] for entry in dp.list_assets()
        }

        # Prediction layer (loads saved model if available)
        self.prediction_layer = PredictionLayer()
        self.prediction_layer.load()

        # Live simulation engine (needs prediction_layer, so must come after)
        self.live_sim = LiveSimEngine(self.trading_graph, self.prediction_layer)

        # Backtest engine
        self.backtest_engine = bt.BacktestEngine(self.trading_graph)

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()

    def fetch_yfinance_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Legacy string-date wrapper - delegates to data_pipeline."""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
            df, _, _ = dp.fetch_ohlcv(symbol, interval, start_dt, end_dt)
            return df
        except Exception as e:
            log.error(f"fetch_yfinance_data: {e}")
            return pd.DataFrame()

    def fetch_yfinance_data_with_datetime(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Datetime wrapper - delegates to data_pipeline."""
        df, _, _ = dp.fetch_ohlcv(symbol, interval, start_datetime, end_datetime)
        return df

    def get_available_assets(self) -> list:
        """Return sorted list of asset codes from the data_pipeline catalogue."""
        codes = [e["code"] for e in dp.list_assets()]
        return sorted(codes)

    def get_available_files(self, asset: str, timeframe: str) -> list:
        """Get available data files for a specific asset and timeframe."""
        asset_dir = self.data_dir / asset.lower()
        if not asset_dir.exists():
            return []

        pattern = f"{asset}_{timeframe}_*.csv"
        files = list(asset_dir.glob(pattern))
        return sorted(files)

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str
    ) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # Debug: Check DataFrame structure

            # Prepare data for analysis
            # if len(df) > 49:
            #     df_slice = df.tail(49).iloc[:-3]
            # else:
            #     df_slice = df.tail(45)

            df_slice = df.tail(45)

            # Ensure DataFrame has the expected structure
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df_slice.columns for col in required_columns):
                return {
                    "success": False,
                    "error": f"Missing required columns. Available: {list(df_slice.columns)}",
                }

            # Reset index to avoid any MultiIndex issues
            df_slice = df_slice.reset_index(drop=True)

            # Debug: Check the slice before conversion

            # Convert to dict for tool input - use explicit conversion to avoid tuple keys
            df_slice_dict = {}
            for col in required_columns:
                if col == "Datetime":
                    # Convert datetime objects to strings for JSON serialization
                    df_slice_dict[col] = (
                        df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                    )
                else:
                    df_slice_dict[col] = df_slice[col].tolist()

            # Debug: Check the resulting dictionary

            # Format timeframe for display
            display_timeframe = timeframe
            if timeframe.endswith("h"):
                display_timeframe += "our"
            elif timeframe.endswith("m"):
                display_timeframe += "in"
            elif timeframe.endswith("d"):
                display_timeframe += "ay"
            elif timeframe == "1w":
                display_timeframe = "1 week"
            elif timeframe == "1mo":
                display_timeframe = "1 month"

            p_image = static_util.generate_kline_image(df_slice_dict)
            t_image = static_util.generate_trend_image(df_slice_dict)

            # Create initial state
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": display_timeframe,
                "stock_name": asset_name,
                "pattern_image": p_image["pattern_image"],
                "trend_image": t_image["trend_image"],
            }

            # Run the trading graph
            final_state = self.trading_graph.graph.invoke(initial_state)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "timeframe": display_timeframe,
                "data_length": len(df_slice),
            }

        except Exception as e:
            error_msg = str(e)
            
            # Get current provider from config
            provider = self.config.get("agent_llm_provider", "azure")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            else:
                provider_name = "Qwen"

            # Check for specific API key authentication errors
            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "success": False,
                    "error": f"[ERROR] Invalid API Key: The {provider_name} API key you provided is invalid or has expired. Please check your API key in the Settings section and try again.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "success": False,
                    "error": f"[WARN] Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"[BILLING] Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"[NETWORK] Network Error: Unable to connect to {provider_name} servers. Please check your internet connection and try again.",
                }
            else:
                return {"success": False, "error": f"[ERROR] Analysis Error: {error_msg}"}

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": results["error"]}

        final_state = results["final_state"]

        # Extract analysis results from state fields
        technical_indicators = final_state.get("indicator_report", "")
        pattern_analysis = final_state.get("pattern_report", "")
        trend_analysis = final_state.get("trend_report", "")
        final_decision_raw = final_state.get("final_trade_decision", "")

        # Extract chart data if available
        pattern_chart = final_state.get("pattern_image", "")
        trend_chart = final_state.get("trend_image", "")
        pattern_image_filename = final_state.get("pattern_image_filename", "")
        trend_image_filename = final_state.get("trend_image_filename", "")

        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find("{")
                end = final_decision_raw.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": decision_data.get("decision", "N/A"),
                        "risk_reward_ratio": decision_data.get(
                            "risk_reward_ratio", "N/A"
                        ),
                        "forecast_horizon": decision_data.get(
                            "forecast_horizon", "N/A"
                        ),
                        "justification": decision_data.get("justification", "N/A"),
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": final_decision_raw}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": final_decision_raw}

        # Refine decision through prediction layer
        prediction = {}
        if isinstance(final_decision, dict) and "decision" in final_decision:
            prediction = self.prediction_layer.predict({
                "decision":          final_decision.get("decision", ""),
                "indicator_report":  technical_indicators,
                "pattern_report":    pattern_analysis,
                "trend_report":      trend_analysis,
                "risk_reward_ratio": final_decision.get("risk_reward_ratio", 1.5),
            })

        return {
            "success": True,
            "asset_name": results["asset_name"],
            "timeframe": results["timeframe"],
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision,
            "prediction": prediction,
        }

    def get_timeframe_date_limits(self, timeframe: str) -> Dict[str, Any]:
        """Get valid date range limits from data_pipeline."""
        all_limits = dp.get_interval_limits()
        max_days = all_limits.get(timeframe, 730)
        return {"max_days": max_days, "description": f"{timeframe} data: max {max_days} days"}

    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
    ) -> Dict[str, Any]:
        """Validate date and time range for the given timeframe."""
        try:
            # Create datetime objects with time
            start_datetime_str = f"{start_date} {start_time}"
            end_datetime_str = f"{end_date} {end_time}"

            start = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")

            if start >= end:
                return {
                    "valid": False,
                    "error": "Start date/time must be before end date/time",
                }

            # Get timeframe limits
            limits = self.get_timeframe_date_limits(timeframe)
            max_days = limits["max_days"]

            # Calculate time difference in days (including fractional days)
            time_diff = end - start
            days_diff = time_diff.total_seconds() / (24 * 3600)  # Convert to days

            if days_diff > max_days:
                return {
                    "valid": False,
                    "error": f"Time range too large. {limits['description']}. Please select a smaller range.",
                    "max_days": max_days,
                    "current_days": round(days_diff, 2),
                }

            return {"valid": True, "days": round(days_diff, 2)}

        except ValueError as e:
            return {"valid": False, "error": f"Invalid date/time format: {str(e)}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current provider credentials by making a simple test call."""
        try:
            if provider is None:
                provider = self.config.get("agent_llm_provider", "azure")

            if provider == "azure":
                from langchain_openai import AzureChatOpenAI
                llm = AzureChatOpenAI(
                    azure_endpoint=self.config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                    azure_deployment=self.config.get("azure_deployment", "gpt-4o"),
                    api_version=self.config.get("azure_api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", ""),
                    api_key=self.config.get("azure_api_key") or os.environ.get("AZURE_OPENAI_API_KEY", ""),
                    temperature=0,
                )
                _ = llm.invoke([("user", "Hi")])
                provider_name = "Azure OpenAI"
            elif provider == "anthropic":
                from anthropic import Anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic_api_key", "")
                if not api_key or api_key == "ANTHROPIC_API_KEY":
                    return {"valid": False, "error": "Anthropic API key is not set."}
                client = Anthropic(api_key=api_key)
                _ = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}],
                )
                provider_name = "Anthropic"
            else:  # qwen
                from langchain_qwq import ChatQwen
                api_key = os.environ.get("DASHSCOPE_API_KEY") or self.config.get("qwen_api_key", "")
                if not api_key or api_key == "DASHSCOPE_API_KEY":
                    return {"valid": False, "error": "Qwen API key is not set."}
                llm = ChatQwen(model="qwen-flash", api_key=api_key)
                _ = llm.invoke([("user", "Hello")])
                provider_name = "Qwen"

            return {"valid": True, "message": f"{provider_name} connection verified"}

        except Exception as e:
            error_msg = str(e)
            if provider is None:
                provider = self.config.get("agent_llm_provider", "azure")
            provider_name = {"azure": "Azure OpenAI", "anthropic": "Anthropic", "qwen": "Qwen"}.get(provider, provider)

            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "valid": False,
                    "error": f"[ERROR] Invalid API Key: The {provider_name} API key is invalid or has expired. Please update it in the Settings section.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "valid": False,
                    "error": f"[WARN] Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"[BILLING] Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"[NETWORK] Network Error: Unable to connect to {provider_name} servers. Please check your internet connection.",
                }
            else:
                return {"valid": False, "error": f"[ERROR] API Key Error: {error_msg}"}

    def load_custom_assets(self) -> list:
        """Load custom assets from persistent JSON file."""
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except Exception as e:
            log.warning(f"Could not load custom assets: {e}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        """Save a custom asset symbol persistently (avoid duplicates)."""
        try:
            symbol = symbol.strip()
            if not symbol:
                return False
            if symbol in self.custom_assets:
                return True  # already present
            self.custom_assets.append(symbol)
            # write to file
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_assets, f, indent=2)
            return True
        except Exception as e:
            log.warning(f"Could not save custom asset {symbol}: {e}")
            return False


# Initialize the analyzer
try:
    analyzer = WebTradingAnalyzer()
    log.ok("QuantAgent ready   Azure provider, all modules loaded")
except Exception as _init_err:
    log.critical(f"FATAL: WebTradingAnalyzer failed to initialize: {_init_err}", exc_info=True)
    raise



@app.before_request
def log_request():
    log.debug(f"  {request.method} {request.path}")

@app.route("/")
def index():
    """Main landing page - redirect to demo."""
    return render_template("demo_new.html")


@app.route("/demo")
def demo():
    """Demo page with new interface."""
    return render_template("demo_new.html")


@app.route("/output")
def output():
    """Output page with analysis results."""
    # Get results from session or query parameters
    results = request.args.get("results")
    if results:
        try:
            # Handle URL-encoded results
            results = urllib.parse.unquote(results)
            results_data = json.loads(results)
            return render_template("output.html", results=results_data)
        except (json.JSONDecodeError, Exception) as e:
            log.warning(f"Could not parse results from URL: {e}")
            # Fall back to default results

    # Default results if none provided
    default_results = {
        "asset_name": "BTC",
        "timeframe": "1h",
        "data_length": 1247,
        "technical_indicators": "RSI (14): 65.4 - Neutral to bullish momentum\nMACD: Bullish crossover with increasing histogram\nMoving Averages: Price above 50-day and 200-day MA\nBollinger Bands: Price in upper band, showing strength\nVolume: Above average volume supporting price action",
        "pattern_analysis": "Bull Flag Pattern: Consolidation after strong upward move\nGolden Cross: 50-day MA crossing above 200-day MA\nHigher Highs & Higher Lows: Uptrend confirmation\nVolume Pattern: Increasing volume on price advances",
        "trend_analysis": "Primary Trend: Bullish (Long-term)\nSecondary Trend: Bullish (Medium-term)\nShort-term Trend: Consolidating with bullish bias\nADX: 28.5 - Moderate trend strength\nPrice Action: Higher highs and higher lows maintained\nMomentum: Positive divergence on RSI",
        "pattern_chart": "",
        "trend_chart": "",
        "pattern_image_filename": "",
        "trend_image_filename": "",
        "final_decision": {
            "decision": "LONG",
            "risk_reward_ratio": "1:2.5",
            "forecast_horizon": "24-48 hours",
            "justification": "Based on comprehensive analysis of technical indicators, pattern recognition, and trend analysis, the system recommends a LONG position on BTC. The analysis shows strong bullish momentum with key support levels holding, and multiple technical indicators confirming upward movement.",
        },
    }

    return render_template("output.html", results=default_results)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        data_source = data.get("data_source")
        asset = data.get("asset")
        timeframe = data.get("timeframe")
        redirect_to_output = data.get("redirect_to_output", False)

        if data_source != "live":
            return jsonify({"error": "Only live Yahoo Finance data is supported."})

        # Live Yahoo Finance data only
        start_date = data.get("start_date")
        start_time = data.get("start_time", "00:00")
        end_date = data.get("end_date")
        end_time = data.get("end_time", "23:59")
        use_current_time = data.get("use_current_time", False)

        # Create datetime objects for validation
        if start_date:
            start_datetime_str = f"{start_date} {start_time}"
            try:
                start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                return jsonify({"error": "Invalid start date/time format."})

            if start_dt > datetime.now():
                return jsonify({"error": "Start date/time cannot be in the future."})

        if end_date:
            if use_current_time:
                end_dt = datetime.now()
            else:
                end_datetime_str = f"{end_date} {end_time}"
                try:
                    end_dt = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return jsonify({"error": "Invalid end date/time format."})

                if end_dt > datetime.now():
                    return jsonify({"error": "End date/time cannot be in the future."})

            if start_date and start_dt and end_dt and end_dt < start_dt:
                return jsonify(
                    {"error": "End date/time cannot be earlier than start date/time."}
                )

        # Fetch data with datetime objects
        df = analyzer.fetch_yfinance_data_with_datetime(
            asset, timeframe, start_dt, end_dt
        )
        if df.empty:
            return jsonify({"error": "No data available for the specified parameters"})

        display_name = analyzer.asset_mapping.get(asset, asset)
        if display_name is None:
            display_name = asset
        results = analyzer.run_analysis(df, display_name, timeframe)
        formatted_results = analyzer.extract_analysis_results(results)

        # If redirect is requested, return redirect URL with results
        if redirect_to_output:
            if formatted_results.get("success", False):
                # Create a version without base64 images for URL encoding
                # Base64 images are too large for URL parameters
                url_safe_results = formatted_results.copy()
                url_safe_results["pattern_chart"] = ""  # Remove base64 data
                url_safe_results["trend_chart"] = ""  # Remove base64 data

                # Encode results for URL
                results_json = json.dumps(url_safe_results)
                encoded_results = urllib.parse.quote(results_json)
                redirect_url = f"/output results={encoded_results}"

                # Store full results (with images) in session or temporary storage
                # For now, we'll pass them back in the response for the frontend to handle
                return jsonify(
                    {
                        "redirect": redirect_url,
                        "full_results": formatted_results,  # Include images in response body
                    }
                )
            else:
                return jsonify(
                    {"error": formatted_results.get("error", "Analysis failed")}
                )

        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/files/<asset>/<timeframe>")
def get_files(asset, timeframe):
    """API endpoint to get available files for an asset/timeframe."""
    try:
        files = analyzer.get_available_files(asset, timeframe)
        file_list = []

        for i, file_path in enumerate(files):
            match = re.search(r"_(\d+)\.csv$", file_path.name)
            file_number = match.group(1) if match else "N/A"
            file_list.append(
                {"index": i, "number": file_number, "name": file_path.name}
            )

        return jsonify({"files": file_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/save-custom-asset", methods=["POST"])
def save_custom_asset():
    """Save a custom asset symbol server-side for persistence."""
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "Symbol required"}), 400

        ok = analyzer.save_custom_asset(symbol)
        if not ok:
            return jsonify({"success": False, "error": "Failed to save symbol"}), 500

        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/custom-assets", methods=["GET"])
def custom_assets():
    """Return server-persisted custom assets."""
    try:
        return jsonify({"custom_assets": analyzer.custom_assets or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets")
def get_assets():
    """Return full asset catalogue with market classification (NSE / US)."""
    try:
        # Catalogue assets (includes market label)
        asset_list = dp.list_assets()

        # Append server-persisted custom assets
        for custom in analyzer.custom_assets:
            yf_sym, name = dp.resolve_symbol(custom)
            market = dp.get_market(yf_sym)
            asset_list.append({"code": custom, "yf_symbol": yf_sym, "name": name, "market": market})

        return jsonify({"assets": asset_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/timeframe-limits/<timeframe>")
def get_timeframe_limits(timeframe):
    """API endpoint to get date range limits for a timeframe."""
    try:
        limits = analyzer.get_timeframe_date_limits(timeframe)
        return jsonify(limits)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-date-range", methods=["POST"])
def validate_date_range():
    """API endpoint to validate date and time range for a timeframe."""
    try:
        data = request.get_json()
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        timeframe = data.get("timeframe")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")

        if not all([start_date, end_date, timeframe]):
            return jsonify({"error": "Missing required parameters"})

        validation = analyzer.validate_date_range(
            start_date, end_date, timeframe, start_time, end_time
        )
        return jsonify(validation)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """API endpoint to update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "azure")

        if provider not in ["azure", "anthropic", "qwen"]:
            return jsonify({"error": "Provider must be 'azure', 'anthropic', or 'qwen'"})

        # Update model defaults per provider
        if provider == "anthropic":
            model = data.get("model", "claude-sonnet-4-5")
        elif provider == "qwen":
            model = data.get("model", "qwen3-max")
        else:  # azure
            model = data.get("model") or analyzer.config.get("azure_deployment", "gpt-4o")

        analyzer.config["agent_llm_provider"] = provider
        analyzer.config["graph_llm_provider"] = provider
        analyzer.config["agent_llm_model"] = model
        analyzer.config["graph_llm_model"] = model
        analyzer.trading_graph.config.update(analyzer.config)
        analyzer.trading_graph.refresh_llms()
        analyzer.live_sim.trading_graph = analyzer.trading_graph

        return jsonify({"success": True, "message": f"Provider updated to {provider} ({model})"})

    except Exception as e:
        log.error(f"update_provider: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    """API endpoint to update API key for OpenAI or Anthropic."""
    try:
        data = request.get_json()
        new_api_key = data.get("api_key")
        provider = data.get("provider", "openai")  # Default to "openai" for backward compatibility

        if not new_api_key:
            return jsonify({"error": "API key is required"})

        if provider not in ["azure", "anthropic", "qwen"]:
            return jsonify({"error": "Provider must be 'azure', 'anthropic', or 'qwen'"})

        analyzer.trading_graph.update_api_key(new_api_key, provider=provider)
        return jsonify({"success": True, "message": f"{provider} API key updated successfully"})

    except Exception as e:
        log.error(f"update_api_key: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/get-api-key-status")
def get_api_key_status():
    """API endpoint to check if API key is set for a provider."""
    try:
        provider = request.args.get("provider", "azure")

        if provider == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY") or analyzer.config.get("azure_api_key", "")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY") or analyzer.config.get("anthropic_api_key", "")
        elif provider == "qwen":
            api_key = os.environ.get("DASHSCOPE_API_KEY") or analyzer.config.get("qwen_api_key", "")
        else:
            api_key = ""

        placeholders = {"AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DASHSCOPE_API_KEY", ""}
        if api_key and api_key not in placeholders:
            # Return masked version for security
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            return jsonify({"has_key": False})
    except Exception as e:
        log.error(f"get_api_key_status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "has_key": False})


@app.route("/api/images/<image_type>")
def get_image(image_type):
    """API endpoint to serve generated images."""
    try:
        if image_type == "pattern":
            image_path = "kline_chart.png"
        elif image_type == "trend":
            image_path = "trend_graph.png"
        elif image_type == "pattern_chart":
            image_path = "pattern_chart.png"
        elif image_type == "trend_chart":
            image_path = "trend_chart.png"
        else:
            return jsonify({"error": "Invalid image type"})

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"})

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-api-key", methods=["POST"])
def validate_api_key():
    """API endpoint to validate the current API key."""
    try:
        data = request.get_json() or {}
        provider = data.get("provider") or analyzer.config.get("agent_llm_provider", "openai")
        validation = analyzer.validate_api_key(provider=provider)
        return jsonify(validation)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static assets from the assets folder."""
    try:
        return send_file(f"assets/{filename}")
    except FileNotFoundError:
        return jsonify({"error": "Asset not found"}), 404


# ---------------------------------------------------------------------------
# Resolve symbol
# ---------------------------------------------------------------------------

@app.route("/api/resolve-symbol", methods=["POST"])
def resolve_symbol():
    try:
        data   = request.get_json()
        ticker = (data.get("ticker") or "").strip()
        if not ticker:
            return jsonify({"error": "ticker required"}), 400
        yf_sym, name = dp.resolve_symbol(ticker)
        market = dp.get_market(yf_sym)
        return jsonify({"yf_symbol": yf_sym, "name": name, "market": market})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Azure config
# ---------------------------------------------------------------------------

@app.route("/api/update-azure-config", methods=["POST"])
def update_azure_config():
    try:
        data        = request.get_json()
        endpoint    = data.get("endpoint")
        api_version = data.get("api_version")
        deployment  = data.get("deployment")
        if endpoint:
            analyzer.config["azure_endpoint"]   = endpoint
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        if api_version:
            analyzer.config["azure_api_version"]   = api_version
            os.environ["AZURE_OPENAI_API_VERSION"] = api_version
        if deployment:
            analyzer.config["azure_deployment"]  = deployment
            analyzer.config["agent_llm_model"]   = deployment
            analyzer.config["graph_llm_model"]   = deployment
        analyzer.trading_graph.config.update(analyzer.config)
        analyzer.trading_graph.refresh_llms()
        return jsonify({"success": True, "message": "Azure config updated"})
    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# Live simulation
# ---------------------------------------------------------------------------

@app.route("/api/live/start", methods=["POST"])
def live_start():
    try:
        data     = request.get_json()
        ticker   = (data.get("ticker") or "").strip()
        interval = data.get("interval", "1h")
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400
        log.ok(f"Live sim starting: {ticker} @ {interval}")
        result = analyzer.live_sim.start(ticker=ticker, interval=interval)

        return jsonify(result)
    except Exception as e:
        log.error(f"live/start error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    try:
        return jsonify(analyzer.live_sim.stop())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/status")
def live_status():
    try:
        return jsonify(analyzer.live_sim.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/stream")
def live_stream():
    def generate():
        yield from analyzer.live_sim.subscribe()
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.route("/api/live/history")
def live_history():
    try:
        return jsonify({"history": analyzer.live_sim.get_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Prediction layer
# ---------------------------------------------------------------------------

@app.route("/api/prediction/train", methods=["POST"])
def train_prediction_model():
    try:
        data     = request.get_json()
        filename = (data.get("filename") or "").strip()
        if not filename:
            return jsonify({"error": "filename required"}), 400
        result = bt.load_backtest_result(filename)
        if result is None:
            return jsonify({"error": f"Backtest file not found: {filename}"}), 404
        return jsonify(analyzer.prediction_layer.train(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prediction/status")
def prediction_model_status():
    try:
        status = analyzer.prediction_layer.status()
        log.debug(f"prediction/status: {status}")
        return jsonify(status)
    except Exception as e:
        log.error(f"prediction/status error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@app.route("/api/backtest/run", methods=["POST"])
def run_backtest():
    try:
        data       = request.get_json()
        ticker     = (data.get("ticker") or "").strip()
        interval   = data.get("interval", "1h")
        start_date = data.get("start_date")
        end_date   = data.get("end_date")
        step       = int(data.get("step",     bt.MIN_STEP))
        max_runs   = int(data.get("max_runs", 30))
        if not ticker or not start_date or not end_date:
            return jsonify({"error": "ticker, start_date, end_date are required"}), 400
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
        result = analyzer.backtest_engine.run(
            ticker=ticker, interval=interval,
            start=start_dt, end=end_dt, step=step, max_runs=max_runs,
        )
        if "error" in result:
            log.error(f"backtest/run returned error: {result}")
            return jsonify(result), 400
        acc = result.get("agent_stats", {}).get("accuracy", 0)
        log.ok(f"Backtest complete   accuracy: {acc:.1%}")
        return jsonify({
            "meta":             result["meta"],
            "agent_stats":      result["agent_stats"],
            "baseline_stats":   result["baseline_stats"],
            "pnl_curve":        result["pnl_curve"],
            "rolling_accuracy": result["rolling_accuracy"],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/list")
def list_backtests():
    try:
        results = bt.list_backtest_results()
        log.debug(f"backtest/list returning {len(results)} results")
        return jsonify({"results": results})
    except Exception as e:
        log.error(f"backtest/list error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/load/<path:filename>")
def load_backtest(filename):
    try:
        result = bt.load_backtest_result(filename)
        if result is None:
            return jsonify({"error": "File not found"}), 404
        return jsonify({
            "meta":             result.get("meta"),
            "agent_stats":      result.get("agent_stats"),
            "baseline_stats":   result.get("baseline_stats"),
            "pnl_curve":        result.get("pnl_curve"),
            "rolling_accuracy": result.get("rolling_accuracy"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/records/<path:filename>")
def load_backtest_records(filename):
    try:
        result = bt.load_backtest_result(filename)
        if result is None:
            return jsonify({"error": "File not found"}), 404
        return jsonify({"records": result.get("records", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

@app.route("/api/ablation/run", methods=["POST"])
def run_ablation():
    try:
        data     = request.get_json()
        filename = (data.get("filename") or "").strip()
        if not filename:
            return jsonify({"error": "filename required"}), 400
        result = abl.run_ablation(
            backtest_filename=filename,
            prediction_layer=analyzer.prediction_layer,
        )
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ablation/list")
def list_ablations():
    try:
        return jsonify({"results": abl.list_ablation_results()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ablation/load/<path:filename>")
def load_ablation(filename):
    try:
        result = abl.load_ablation_result(filename)
        if result is None:
            return jsonify({"error": "File not found"}), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    app.run(debug=True, host="127.0.0.1", port=5000)
