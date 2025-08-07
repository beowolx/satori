import os
from typing import Any, Dict, Optional

from .base import ProviderError
from .openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
  """OpenRouter provider that uses OpenAI-compatible API.

  OpenRouter provides access to multiple models through a unified API
  that is compatible with the OpenAI API format.
  """

  # OpenRouter base URL
  OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

  def __init__(
    self,
    model: str,
    api_key: Optional[str] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
    **kwargs,
  ):
    """Initialize OpenRouter provider.

    Args:
        model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-2", "meta-llama/llama-2-70b-chat")
        api_key: OpenRouter API key. If not provided, will use OPENROUTER_API_KEY env var
        site_url: Optional URL of your site/app for OpenRouter analytics
        site_name: Optional name of your site/app for OpenRouter analytics
        **kwargs: Additional configuration options passed to OpenAI client

    Raises:
        ProviderError: If API key is not provided or found in environment
    """
    # Get OpenRouter API key from parameter or environment
    openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
      raise ProviderError(
        "OpenRouter API key not found. Please provide it as a parameter "
        "or set the OPENROUTER_API_KEY environment variable."
      )

    # Set the base_url to OpenRouter
    kwargs["base_url"] = self.OPENROUTER_BASE_URL

    # OpenRouter requires the API key to be passed as the OpenAI API key
    # Initialize the parent OpenAIProvider with OpenRouter settings
    super().__init__(model=model, api_key=openrouter_key, **kwargs)

    # Store OpenRouter-specific metadata
    self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL")
    self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME")

  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from OpenRouter asynchronously.

    OpenRouter supports additional headers for analytics and routing.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    # Add OpenRouter-specific headers if provided
    if self.site_url or self.site_name:
      # OpenRouter uses custom headers for site identification
      # These need to be passed through the client configuration
      # Note: OpenAI client doesn't directly support custom headers in generate,
      # so we'll rely on the base implementation
      pass

    try:
      # Use the parent class implementation
      return await super().generate(prompt, **kwargs)
    except ProviderError as e:
      # Transform OpenRouter-specific errors
      error_msg = str(e)
      if "credits" in error_msg.lower() or "balance" in error_msg.lower():
        raise ProviderError(
          "OpenRouter credits exhausted. Please add credits to your account."
        )
      elif (
        "model" in error_msg.lower() and "not available" in error_msg.lower()
      ):
        raise ProviderError(
          f"Model '{self.model}' is not available on OpenRouter. "
          "Check https://openrouter.ai/models for available models."
        )
      else:
        # Re-raise the original error
        raise

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous wrapper for generation.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    try:
      # Use the parent class implementation
      return super().generate_sync(prompt, **kwargs)
    except ProviderError as e:
      # Transform OpenRouter-specific errors
      error_msg = str(e)
      if "credits" in error_msg.lower() or "balance" in error_msg.lower():
        raise ProviderError(
          "OpenRouter credits exhausted. Please add credits to your account."
        )
      elif (
        "model" in error_msg.lower() and "not available" in error_msg.lower()
      ):
        raise ProviderError(
          f"Model '{self.model}' is not available on OpenRouter. "
          "Check https://openrouter.ai/models for available models."
        )
      else:
        # Re-raise the original error
        raise

  def _prepare_messages(
    self, prompt: str, system_message: Optional[str] = None
  ) -> list:
    """Prepare messages for the OpenRouter API.

    OpenRouter supports the same message format as OpenAI.

    Args:
        prompt: User prompt
        system_message: Optional system message

    Returns:
        List of message dictionaries
    """
    # Use the parent class implementation
    return super()._prepare_messages(prompt, system_message)

  def _extract_generation_params(
    self, kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Extract valid generation parameters for OpenRouter.

    OpenRouter supports most OpenAI parameters plus some additional ones.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        Dictionary of valid parameters
    """
    # Start with OpenAI parameters
    params = super()._extract_generation_params(kwargs)

    # Add OpenRouter-specific parameters if present
    openrouter_params = [
      "transforms",  # OpenRouter-specific transformations
      "route",  # Routing preference (e.g., "fallback")
      "models",  # List of models for fallback routing
    ]

    for param in openrouter_params:
      if param in kwargs:
        params[param] = kwargs[param]

    return params

  def __repr__(self) -> str:
    """String representation of the provider."""
    return f"OpenRouterProvider(model='{self.model}')"
