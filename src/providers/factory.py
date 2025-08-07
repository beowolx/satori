"""Provider factory for creating LLM provider instances."""

from typing import Any, Dict, Optional, Type

from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider
from .google_provider import GoogleProvider
from .huggingface_provider import HuggingFaceProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider


class ProviderFactory:
  """Factory class for creating LLM provider instances."""

  # Registry of available providers
  _providers: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,  # Alias for huggingface
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
  }

  # Provider aliases for common shortcuts
  _aliases: Dict[str, str] = {
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "hf": "huggingface",
    "local": "ollama",
  }

  @classmethod
  def register_provider(
    cls,
    name: str,
    provider_class: Type[BaseLLMProvider],
    aliases: Optional[list[str]] = None,
  ) -> None:
    """Register a new provider with the factory.

    Args:
        name: The name of the provider
        provider_class: The provider class to register
        aliases: Optional list of aliases for the provider

    Raises:
        ValueError: If the provider name is already registered
    """
    if name.lower() in cls._providers:
      raise ValueError(f"Provider '{name}' is already registered")

    cls._providers[name.lower()] = provider_class

    # Register aliases if provided
    if aliases:
      for alias in aliases:
        cls._aliases[alias.lower()] = name.lower()

  @classmethod
  def parse_provider_string(cls, provider_str: str) -> tuple[str, str]:
    """Parse provider string format 'provider:model'.

    Args:
        provider_str: String in format 'provider:model'

    Returns:
        Tuple of (provider_name, model_name)

    Raises:
        ValueError: If format is invalid
    """
    if ":" not in provider_str:
      raise ValueError(
        f"Invalid provider format: {provider_str}. "
        f"Expected format: 'provider:model' (e.g., 'openai:gpt-4', 'anthropic:claude-3')"
      )

    parts = provider_str.split(":", 1)
    provider_name = parts[0].lower()
    model_name = parts[1]

    # Resolve aliases
    if provider_name in cls._aliases:
      provider_name = cls._aliases[provider_name]

    return provider_name, model_name

  @classmethod
  def create(
    cls, provider_str: str, config=None, **kwargs: Any
  ) -> BaseLLMProvider:
    """Create a provider instance from a provider string.

    Args:
        provider_str: Provider string (e.g., 'openai:gpt-4o', 'anthropic:claude-3-opus')
        config: Configuration object containing provider settings
        **kwargs: Additional configuration options to pass to the provider

    Returns:
        Provider instance (BaseLLMProvider)

    Raises:
        ValueError: If provider is not supported
    """
    provider_name, model_name = cls.parse_provider_string(provider_str)

    if provider_name not in cls._providers:
      available = ", ".join(sorted(cls._providers.keys()))
      raise ValueError(
        f"Unsupported provider: '{provider_name}'. "
        f"Available providers: {available}"
      )

    provider_class = cls._providers[provider_name]

    # Merge config and kwargs
    provider_kwargs = kwargs.copy()
    if config and hasattr(config, "providers"):
      provider_config = config.providers.get(provider_name, {})
      # Config values are overridden by explicit kwargs
      for key, value in provider_config.items():
        if key not in provider_kwargs:
          provider_kwargs[key] = value

    # Create provider instance with model and configuration
    return provider_class(model=model_name, **provider_kwargs)

  @classmethod
  def list_providers(cls) -> list[str]:
    """List all available provider names.

    Returns:
        List of available provider names
    """
    return sorted(cls._providers.keys())

  @classmethod
  def list_aliases(cls) -> Dict[str, str]:
    """List all provider aliases.

    Returns:
        Dictionary mapping aliases to provider names
    """
    return cls._aliases.copy()

  @classmethod
  def get_provider_class(cls, provider_name: str) -> Type[BaseLLMProvider]:
    """Get the provider class for a given provider name.

    Args:
        provider_name: Name of the provider

    Returns:
        Provider class

    Raises:
        ValueError: If provider is not found
    """
    # Resolve alias if needed
    if provider_name.lower() in cls._aliases:
      provider_name = cls._aliases[provider_name.lower()]

    if provider_name.lower() not in cls._providers:
      raise ValueError(f"Provider '{provider_name}' not found")

    return cls._providers[provider_name.lower()]

  @classmethod
  def validate_provider_string(cls, provider_str: str) -> bool:
    """Validate if a provider string is valid.

    Args:
        provider_str: Provider string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
      provider_name, _ = cls.parse_provider_string(provider_str)
      return provider_name in cls._providers
    except ValueError:
      return False

  @classmethod
  def get_provider_info(cls, provider_name: str) -> Dict[str, Any]:
    """Get information about a specific provider.

    Args:
        provider_name: Name of the provider

    Returns:
        Dictionary with provider information

    Raises:
        ValueError: If provider is not found
    """
    # Resolve alias if needed
    if provider_name.lower() in cls._aliases:
      provider_name = cls._aliases[provider_name.lower()]

    if provider_name.lower() not in cls._providers:
      raise ValueError(f"Provider '{provider_name}' not found")

    provider_class = cls._providers[provider_name.lower()]

    return {
      "name": provider_name,
      "class": provider_class.__name__,
      "module": provider_class.__module__,
      "docstring": provider_class.__doc__,
      "aliases": [
        alias
        for alias, target in cls._aliases.items()
        if target == provider_name.lower()
      ],
    }


# Export the main factory methods for convenience
create_provider = ProviderFactory.create
parse_provider_string = ProviderFactory.parse_provider_string
list_providers = ProviderFactory.list_providers
register_provider = ProviderFactory.register_provider
validate_provider_string = ProviderFactory.validate_provider_string
