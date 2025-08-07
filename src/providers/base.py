from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
  """Base class for all LLM providers."""

  def __init__(self, model: str, **kwargs):
    """Initialize provider with model name and configuration.

    Args:
        model: The model identifier (e.g., "gpt-4o", "claude-3-opus")
        **kwargs: Provider-specific configuration options
    """
    self.model = model
    self.config = kwargs

  @abstractmethod
  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from the LLM asynchronously.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    pass

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous wrapper for generate method.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    import asyncio

    return asyncio.run(self.generate(prompt, **kwargs))

  def _handle_error(self, error: Exception) -> Exception:
    """Handle and transform provider-specific errors.

    Args:
        error: The original error from the provider

    Returns:
        Transformed error with consistent interface
    """
    return ProviderError(
      f"Provider {self.__class__.__name__} failed: {str(error)}"
    )


class ProviderError(Exception):
  """Exception raised when a provider operation fails."""

  pass
