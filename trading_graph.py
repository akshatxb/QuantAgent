"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Supports Azure OpenAI (primary), Anthropic, and Qwen as LLM providers.
"""

import os
from typing import Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_qwq import ChatQwen

from default_config import DEFAULT_CONFIG
from graph_setup import SetGraph
from graph_util import TechnicalTools

SUPPORTED_PROVIDERS = ("azure", "anthropic", "qwen")


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Supports Azure OpenAI (primary), Anthropic, and Qwen.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.toolkit = TechnicalTools()

        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "azure"),
            model=self.config.get("agent_llm_model", "gpt-4o"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "azure"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )

        self.graph_setup = SetGraph(self.agent_llm, self.graph_llm, self.toolkit)
        self.graph = self.graph_setup.set_graph()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_api_credentials(self, provider: str) -> dict:
        """
        Resolve credentials for the given provider.
        Checks config dict first, then falls back to environment variables.
        """
        if provider == "azure":
            api_key    = self.config.get("azure_api_key")    or os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint   = self.config.get("azure_endpoint")   or os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = self.config.get("azure_api_version") or os.environ.get("AZURE_OPENAI_API_VERSION")
            deployment  = (
                self.config.get("azure_deployment")
                or self.config.get("agent_llm_model")
                or "gpt-4o"
            )

            missing = []
            if not api_key or api_key == "AZURE_OPENAI_API_KEY":
                missing.append("azure_api_key  /  env: AZURE_OPENAI_API_KEY")
            if not endpoint or endpoint == "AZURE_OPENAI_ENDPOINT":
                missing.append("azure_endpoint  /  env: AZURE_OPENAI_ENDPOINT")
            if not api_version or api_version == "AZURE_OPENAI_API_VERSION":
                missing.append("azure_api_version  /  env: AZURE_OPENAI_API_VERSION")
            if missing:
                raise ValueError(
                    "Azure OpenAI credentials are missing or still set to placeholder values.\n"
                    "Fill these in default_config.py or set as environment variables:\n"
                    + "\n".join(f"  - {m}" for m in missing)
                )

            return {
                "api_key": api_key,
                "azure_endpoint": endpoint,
                "api_version": api_version,
                "azure_deployment": deployment,
            }

        elif provider == "anthropic":
            api_key = self.config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key or api_key == "ANTHROPIC_API_KEY":
                raise ValueError(
                    "Anthropic API key not found.\n"
                    "Set anthropic_api_key in default_config.py or export ANTHROPIC_API_KEY."
                )
            return {"api_key": api_key}

        elif provider == "qwen":
            api_key = self.config.get("qwen_api_key") or os.environ.get("DASHSCOPE_API_KEY")
            if not api_key or api_key == "DASHSCOPE_API_KEY":
                raise ValueError(
                    "Qwen API key not found.\n"
                    "Set qwen_api_key in default_config.py or export DASHSCOPE_API_KEY."
                )
            return {"api_key": api_key}

        else:
            raise ValueError(
                f"Unsupported provider: '{provider}'. Must be one of: {SUPPORTED_PROVIDERS}"
            )

    def _create_llm(self, provider: str, model: str, temperature: float) -> BaseChatModel:
        """Instantiate the correct LangChain LLM for the given provider."""
        creds = self._get_api_credentials(provider)

        if provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=creds["azure_endpoint"],
                azure_deployment=creds["azure_deployment"],
                api_version=creds["api_version"],
                api_key=creds["api_key"],
                temperature=temperature,
            )

        elif provider == "anthropic":
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=creds["api_key"],
            )

        elif provider == "qwen":
            return ChatQwen(
                model=model,
                temperature=temperature,
                api_key=creds["api_key"],
                max_retries=4,
            )

    # ------------------------------------------------------------------
    # Public methods (called by web_interface.py)
    # ------------------------------------------------------------------

    def refresh_llms(self):
        """Re-instantiate LLMs from current config. Called after any config change."""
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "azure"),
            model=self.config.get("agent_llm_model", "gpt-4o"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "azure"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )
        self.graph_setup = SetGraph(self.agent_llm, self.graph_llm, self.toolkit)
        self.graph = self.graph_setup.set_graph()

    def update_api_key(self, api_key: str, provider: str = "azure"):
        """Update a provider's API key at runtime and refresh LLMs."""
        if provider == "azure":
            self.config["azure_api_key"] = api_key
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            self.config["anthropic_api_key"] = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "qwen":
            self.config["qwen_api_key"] = api_key
            os.environ["DASHSCOPE_API_KEY"] = api_key
        else:
            raise ValueError(f"Unsupported provider: '{provider}'")
        self.refresh_llms()

    def update_azure_config(self, endpoint: str = None, api_version: str = None, deployment: str = None):
        """Update Azure-specific connection settings at runtime."""
        if endpoint:
            self.config["azure_endpoint"] = endpoint
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        if api_version:
            self.config["azure_api_version"] = api_version
            os.environ["AZURE_OPENAI_API_VERSION"] = api_version
        if deployment:
            self.config["azure_deployment"] = deployment
            self.config["agent_llm_model"] = deployment
            self.config["graph_llm_model"] = deployment
        self.refresh_llms()
