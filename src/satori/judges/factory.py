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

    # Otherwise, construct a provider-backed generic judge
    provider_str = (
      identifier
      if ":" in identifier
      else (getattr(config, "default_provider", None) or "openai:gpt-4.1")
    )
    provider = ProviderFactory.create(provider_str, config=config)
    return GenericLLMJudge(provider=provider)


# convenience export
create_judge = JudgesFactory.create
