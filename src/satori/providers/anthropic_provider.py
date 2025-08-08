import os
from typing import Any, Dict, Optional

from anthropic import Anthropic, AsyncAnthropic

from .base import BaseLLMProvider, ProviderError


class AnthropicProvider(BaseLLMProvider):
  """Anthropic API provider for Claude models."""

  def __init__(
    self,
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    **kwargs,
  ):
    """Initialize Anthropic provider.

    Args:
        model: Model identifier (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
        api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var
        **kwargs: Additional configuration options

    Raises:
        ProviderError: If API key is not provided or found in environment
    """
    super().__init__(model, **kwargs)

    # Get API key from parameter or environment
    self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not self.api_key:
      raise ProviderError(
        "Anthropic API key not found. Please provide it as a parameter "
        "or set the ANTHROPIC_API_KEY environment variable."
      )

    # Extract Anthropic-specific configuration
    self.max_retries = kwargs.get("max_retries", 3)
    self.timeout = kwargs.get("timeout", 60.0)

    # Initialize async client (sync client created on demand)
    self.async_client = AsyncAnthropic(
      api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout
    )
    self._sync_client: Optional[Anthropic] = None

  @property
  def sync_client(self) -> Anthropic:
    """Lazy initialization of sync client."""
    if self._sync_client is None:
      self._sync_client = Anthropic(
        api_key=self.api_key, max_retries=self.max_retries, timeout=self.timeout
      )
    return self._sync_client

  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from Anthropic asynchronously.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters:
            - temperature (float): Sampling temperature (0-1)
            - max_tokens (int): Maximum tokens to generate
            - top_p (float): Nucleus sampling parameter
            - top_k (int): Top-k sampling parameter
            - stop_sequences (list): Stop sequences
            - system_message (str): System message to prepend

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    try:
      # Prepare messages
      messages = self._prepare_messages(prompt)

      # Extract generation parameters
      generation_params = self._extract_generation_params(kwargs)

      # Handle system message separately for Anthropic
      system_message = kwargs.get("system_message", None)

      # Build API call parameters
      api_params = {
        "model": self.model,
        "messages": messages,
        **generation_params,
      }

      # Only add system if it's provided (Anthropic doesn't accept None for system)
      if system_message:
        api_params["system"] = system_message

      # Make API call
      response = await self.async_client.messages.create(**api_params)

      # Extract and return the response text
      if response.content and len(response.content) > 0:
        # Anthropic returns a list of content blocks
        return (
          response.content[0].text
          if hasattr(response.content[0], "text")
          else str(response.content[0])
        )
      else:
        raise ProviderError("Empty response from Anthropic API")

    except Exception as e:
      # Transform provider-specific errors
      if "api_key" in str(e).lower() or "authentication" in str(e).lower():
        raise ProviderError("Invalid Anthropic API key")
      elif "rate" in str(e).lower() and "limit" in str(e).lower():
        raise ProviderError(
          "Anthropic rate limit exceeded. Please wait and retry."
        )
      elif "model" in str(e).lower() and (
        "not found" in str(e).lower() or "does not exist" in str(e).lower()
      ):
        raise ProviderError(f"Model '{self.model}' not found or not accessible")
      elif "credit" in str(e).lower() or "balance" in str(e).lower():
        raise ProviderError("Insufficient Anthropic API credits")
      else:
        raise self._handle_error(e)

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous wrapper using the sync Anthropic client.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    try:
      # Prepare messages
      messages = self._prepare_messages(prompt)

      # Extract generation parameters
      generation_params = self._extract_generation_params(kwargs)

      # Handle system message separately for Anthropic
      system_message = kwargs.get("system_message", None)

      # Build API call parameters
      api_params = {
        "model": self.model,
        "messages": messages,
        **generation_params,
      }

      # Only add system if it's provided (Anthropic doesn't accept None for system)
      if system_message:
        api_params["system"] = system_message

      # Make API call with sync client
      response = self.sync_client.messages.create(**api_params)

      # Extract and return the response text
      if response.content and len(response.content) > 0:
        # Anthropic returns a list of content blocks
        return (
          response.content[0].text
          if hasattr(response.content[0], "text")
          else str(response.content[0])
        )
      else:
        raise ProviderError("Empty response from Anthropic API")

    except Exception as e:
      # Transform provider-specific errors
      if "api_key" in str(e).lower() or "authentication" in str(e).lower():
        raise ProviderError("Invalid Anthropic API key")
      elif "rate" in str(e).lower() and "limit" in str(e).lower():
        raise ProviderError(
          "Anthropic rate limit exceeded. Please wait and retry."
        )
      elif "model" in str(e).lower() and (
        "not found" in str(e).lower() or "does not exist" in str(e).lower()
      ):
        raise ProviderError(f"Model '{self.model}' not found or not accessible")
      elif "credit" in str(e).lower() or "balance" in str(e).lower():
        raise ProviderError("Insufficient Anthropic API credits")
      else:
        raise self._handle_error(e)

  def _prepare_messages(self, prompt: str) -> list:
    """Prepare messages for the Anthropic API.

    Args:
        prompt: User prompt

    Returns:
        List of message dictionaries
    """
    # Anthropic doesn't use system messages in the messages array
    # System message is passed separately
    return [{"role": "user", "content": prompt}]

  def _extract_generation_params(
    self, kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Extract valid Anthropic generation parameters from kwargs.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        Dictionary of valid Anthropic parameters
    """
    # Map common parameter names to Anthropic's expected names
    param_mapping = {
      "temperature": "temperature",
      "max_tokens": "max_tokens",
      "top_p": "top_p",
      "top_k": "top_k",
      "stop": "stop_sequences",
      "stop_sequences": "stop_sequences",
    }

    params = {}
    for key, anthropic_key in param_mapping.items():
      if key in kwargs:
        params[anthropic_key] = kwargs[key]

    # Set defaults for required parameters
    if "max_tokens" not in params:
      params["max_tokens"] = 8192  # Anthropic requires max_tokens

    # Ensure temperature is within Anthropic's range (0-1)
    if "temperature" in params:
      params["temperature"] = min(1.0, max(0.0, params["temperature"]))
    else:
      params["temperature"] = 0.7

    return params

  def __repr__(self) -> str:
    """String representation of the provider."""
    return f"AnthropicProvider(model='{self.model}')"
