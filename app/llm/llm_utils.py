# Environment configuration and dependency imports.
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

# Internal project dependencies.
from app.config.llm_config import llm_config
from app.core.logger import logger

# Global cache: key is (model name, JSON output mode), value is a ChatOpenAI instance.
# Purpose: avoid repeated client initialization, improve performance, and manage instances consistently.
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    Get a globally cached LangChain ChatOpenAI client instance.

    Supports OpenAI-compatible APIs such as OpenAI, Qwen, and Jimeng AI.
    Allows custom models and standardized JSON output.
    Main features: caching, centralized configuration loading, precise exception handling,
    and compatibility with provider-specific model parameters.

    :param model: Model name. Priority: function argument > llm_config.llm_model > built-in default qwen3-32b.
    :param json_mode: Whether to enable JSON output mode. When enabled, the model returns json_object format for structured parsing.
    :return: Initialized ChatOpenAI instance. Returns from global cache first, otherwise creates and caches a new client.
    :raise ValueError: Required configuration such as API key or base URL is missing.
    :raise Exception: Model initialization failed in the LangChain wrapper layer.
    """
    # 1. Resolve the target model using descending priority and ensure it is not empty.
    target_model = model or llm_config.llm_model
    # Cache key: model name plus JSON mode uniquely identifies a client configuration.
    cache_key = (target_model, json_mode)

    # 2. Cache hit: return the initialized instance directly to avoid duplicate creation.
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM Client] Cache hit. Returning existing instance: model={target_model}, json_mode={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. Validate required API configuration early and fail with clear errors.
    if not llm_config.api_key:
        raise ValueError("[LLM Client] Missing configuration: set OPENAI_API_KEY in .env.")
    if not llm_config.base_url:
        raise ValueError("[LLM Client] Missing configuration: set OPENAI_API_BASE in .env.")
    logger.info(f"[LLM Client] Initializing new instance: model={target_model}, json_mode={json_mode}")

    # 4. Assemble model parameters.
    # extra_body is for provider-specific parameters used by models such as Qwen or Jimeng AI.
    # LangChain passes these parameters through to the API.
    # extra_body = {"enable_thinking": False}  # Qwen-specific: disable reasoning-chain output to reduce redundant content.
    # model_kwargs contains OpenAI-compatible parameters supported by all compatible APIs.

    model_kwargs = {}
    if json_mode:
        # Enable standard JSON output mode so the model returns parseable json_object content.
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM Client] JSON output mode enabled. The model will return standard JSON structure.")

    # 5. Initialize the client and convert LangChain wrapper errors into clearer messages.
    try:
        llm_client = ChatOpenAI(
            model=target_model,  # Target model name.
            temperature=llm_config.llm_temperature or 0.1,  # Low temperature keeps output more deterministic (0 to 1).
            api_key=llm_config.api_key,  # API key.
            base_url=llm_config.base_url,  # API base URL, including proxy URLs for compatible providers.
            # extra_body=extra_body,  # Pass through provider-specific parameters.
            model_kwargs=model_kwargs,  # OpenAI-compatible parameters.
        )
    except LangChainException as e:
        raise Exception(f"[LLM Client] Model '{target_model}' initialization failed in LangChain layer: {str(e)}") from e

    # 6. Store the new instance in the global cache for reuse.
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM Client] Instance initialized and cached: model={target_model}, json_mode={json_mode}")

    return llm_client


# Test example: verify client creation, cache behavior, and logging output.
if __name__ == "__main__":
    logger.info("===== Starting LLM client utility test =====")
    try:
        # Test 1: default configuration with the default model and normal mode.
        client1 = get_llm_client()
        logger.info("✅ Test 1 passed: default configuration client created successfully.")

        # Test 2: specified multimodal model with normal mode.
        client2 = get_llm_client(model="gpt-4o-mini")
        logger.info("✅ Test 2 passed: specified multimodal model client created successfully.")

        # Test 3: same model and mode to verify cache hit.
        client3 = get_llm_client(model="gpt-4o-mini")
        logger.info(f"✅ Test 3 passed: cache behavior verified, client2 and client3 are the same instance: {client2 is client3}")

        # Test 4: enable JSON output mode.
        client4 = get_llm_client(model="gpt-5-nano", json_mode=True)
        logger.info("✅ Test 4 passed: JSON output mode client created successfully.")

    except Exception as e:
        logger.error(f"❌ LLM client utility test failed: {str(e)}", exc_info=True)
    finally:
        logger.info("===== Finished LLM client utility test =====")
