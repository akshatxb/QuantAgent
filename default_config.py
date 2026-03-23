DEFAULT_CONFIG = {
    # --- Primary provider (Azure OpenAI) ---
    "agent_llm_provider": "azure",        # "azure", "anthropic", or "qwen"
    "graph_llm_provider": "azure",        # "azure", "anthropic", or "qwen"
    "agent_llm_model": "gpt-4o",          # Azure deployment name
    "graph_llm_model": "gpt-4o",          # Azure deployment name (same model for all)
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,

    # --- Azure OpenAI credentials (fill from Azure AI Foundry portal) ---
    "azure_api_key": "api_key",
    "azure_endpoint": "https://quantagent.cognitiveservices.azure.com/",       # e.g. https://YOUR-RESOURCE.openai.azure.com/
    "azure_deployment": "gpt-4o",                    # your deployment name in Azure
    "azure_api_version": "2024-12-01-preview", # e.g. 2024-02-15-preview

    # --- Anthropic (optional alternative) ---
    "anthropic_api_key": "ANTHROPIC_API_KEY",

    # --- Qwen / DashScope (optional alternative) ---
    "qwen_api_key": "DASHSCOPE_API_KEY",
}
