from typing import Optional

from ..providers import ProviderFactory
from .base import BaseJudge
from .generic_llm_judge import GenericLLMJudge
from .openai_judge import OpenAIJudge


class JudgesFactory:
  """Factory for creating judge instances.

  Supports:
  - "openai:<model>" -> OpenAIJudge directly
  - "<provider>:<model>" -> GenericLLMJudge using ProviderFactory
  - bare model -> uses provider from config.default_provider (if available)
  """

  @staticmethod
  def create(identifier: str, config=None, **kwargs) -> BaseJudge:
    provider_name: Optional[str] = None
    model: str = identifier

    if ":" in identifier:
      provider_name, model = identifier.split(":", 1)

    # Fast-path OpenAI native judge
    if provider_name is not None and provider_name.lower() == "openai":
      judge_kwargs = {}
      if config and hasattr(config, "providers"):
        judge_kwargs.update(config.providers.get("openai", {}))
      return OpenAIJudge(model=model, **judge_kwargs)

    # Otherwise, construct a provider-backed judge respecting the requested model
    # If identifier is bare (no provider prefix), reuse the vendor from config.default_provider
    # but keep the model from `identifier`.
    if ":" not in identifier:
      # Infer vendor from model name when possible; otherwise take from default_provider
      model_lower = identifier.lower()
      vendor: str
      if model_lower.startswith("gpt-") or model_lower in {
        "gpt4",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4",
      }:
        vendor = "openai"
      elif model_lower.startswith("claude"):
        vendor = "anthropic"
      elif model_lower.startswith("gemini"):
        vendor = "google"
      elif "/" in model_lower:
        vendor = "huggingface"
      else:
        default_provider = (
          getattr(config, "default_provider", None) if config else None
        )
        try:
          vendor, _ = (
            ProviderFactory.parse_provider_string(default_provider)
            if default_provider
            else ("openai", "gpt-4.1")
          )
        except Exception:
          vendor = "openai"

      # Prefer native OpenAI judge when vendor is openai
      if vendor.lower() == "openai":
        judge_kwargs = {}
        if config and hasattr(config, "providers"):
          judge_kwargs.update(config.providers.get("openai", {}))
        return OpenAIJudge(model=model, **judge_kwargs)

      provider = ProviderFactory.create(f"{vendor}:{model}", config=config)
      return GenericLLMJudge(provider=provider)

    # If identifier already has provider prefix, create via that provider
    provider = ProviderFactory.create(identifier, config=config)
    return GenericLLMJudge(provider=provider)


# convenience export
create_judge = JudgesFactory.create
