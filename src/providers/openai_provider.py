import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from .base import BaseLLMProvider, ProviderError


class OpenAIProvider(BaseLLMProvider):
  """OpenAI API provider for GPT models."""

  def __init__(
    self, model: str = "gpt-4.1", api_key: Optional[str] = None, **kwargs
  ):
    """Initialize OpenAI provider.

    Args:
        model: Model identifier (e.g., "gpt-4.1", "gpt-4o")
        api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var
        **kwargs: Additional configuration options (base_url, organization, etc.)

    Raises:
        ProviderError: If API key is not provided or found in environment
    """
    super().__init__(model, **kwargs)

    self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not self.api_key:
      raise ProviderError(
        "OpenAI API key not found. Please provide it as a parameter "
        "or set the OPENAI_API_KEY environment variable."
      )

    self.base_url = kwargs.get("base_url")
    self.organization = kwargs.get("organization")
    self.timeout = kwargs.get("timeout", self.config.get("timeout", 60.0))

    # Initialize async client (sync client created on demand)
    self.async_client = AsyncOpenAI(
      api_key=self.api_key,
      base_url=self.base_url,
      organization=self.organization,
      timeout=self.timeout,
    )
    self._sync_client: Optional[OpenAI] = None

  @property
  def sync_client(self) -> OpenAI:
    """Lazy initialization of sync client."""
    if self._sync_client is None:
      self._sync_client = OpenAI(
        api_key=self.api_key,
        base_url=self.base_url,
        organization=self.organization,
        timeout=self.timeout,
      )
    return self._sync_client

  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from OpenAI asynchronously.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters:
            - temperature (float): Sampling temperature (0-2)
            - max_tokens (int): Maximum tokens to generate
            - top_p (float): Nucleus sampling parameter
            - frequency_penalty (float): Frequency penalty (-2 to 2)
            - presence_penalty (float): Presence penalty (-2 to 2)
            - stop (list): Stop sequences
            - system_message (str): System message to prepend

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    try:
      # Prepare messages
      messages = self._prepare_messages(prompt, kwargs.get("system_message"))

      # Extract generation parameters
      generation_params = self._extract_generation_params(kwargs)

      # Make API call
      response: ChatCompletion = (
        await self.async_client.chat.completions.create(
          model=self.model, messages=messages, **generation_params
        )
      )

      # Extract and return the response text
      if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content or ""
      else:
        raise ProviderError("Empty response from OpenAI API")

    except Exception as e:
      # Try adaptive remediation for known 400s
      adapted = self._adapt_params_on_400(str(e), kwargs)
      if adapted is not None:
        try:
          messages = self._prepare_messages(
            prompt, adapted.get("system_message")
          )
          generation_params = self._extract_generation_params(adapted)
          retry_resp: ChatCompletion = (
            await self.async_client.chat.completions.create(
              model=self.model, messages=messages, **generation_params
            )
          )
          if retry_resp.choices and len(retry_resp.choices) > 0:
            return retry_resp.choices[0].message.content or ""
        except Exception:
          pass

      # Transform provider-specific errors
      if "api_key" in str(e).lower():
        raise ProviderError("Invalid OpenAI API key")
      elif "rate_limit" in str(e).lower() or "429" in str(e):
        raise ProviderError(
          "OpenAI rate limit exceeded. Please wait and retry."
        )
      elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
        raise ProviderError("Request timed out.")
      elif "model" in str(e).lower() and "not found" in str(e).lower():
        raise ProviderError(f"Model '{self.model}' not found or not accessible")
      else:
        raise self._handle_error(e)

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous wrapper using the sync OpenAI client.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    try:
      # Prepare messages
      messages = self._prepare_messages(prompt, kwargs.get("system_message"))

      # Extract generation parameters
      generation_params = self._extract_generation_params(kwargs)

      # Make API call with sync client
      response: ChatCompletion = self.sync_client.chat.completions.create(
        model=self.model, messages=messages, **generation_params
      )

      # Extract and return the response text
      if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content or ""
      else:
        raise ProviderError("Empty response from OpenAI API")

    except Exception as e:
      # Transform provider-specific errors
      if "api_key" in str(e).lower():
        raise ProviderError("Invalid OpenAI API key")
      elif "rate_limit" in str(e).lower() or "429" in str(e):
        raise ProviderError(
          "OpenAI rate limit exceeded. Please wait and retry."
        )
      elif "model" in str(e).lower() and "not found" in str(e).lower():
        raise ProviderError(f"Model '{self.model}' not found or not accessible")
      else:
        raise self._handle_error(e)

  def _prepare_messages(
    self, prompt: str, system_message: Optional[str] = None
  ) -> list:
    """Prepare messages for the chat completion API.

    Args:
        prompt: User prompt
        system_message: Optional system message

    Returns:
        List of message dictionaries
    """
    messages = []

    if system_message:
      messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": prompt})

    return messages

  def _extract_generation_params(
    self, kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Extract valid OpenAI generation parameters from kwargs.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        Dictionary of valid OpenAI parameters
    """
    valid_params = [
      "temperature",
      "max_tokens",
      "max_completion_tokens",
      "top_p",
      "frequency_penalty",
      "presence_penalty",
      "stop",
      "n",
      "stream",
      "logprobs",
      "echo",
      "best_of",
      "logit_bias",
      "user",
    ]

    params: Dict[str, Any] = {}
    for param in valid_params:
      if param in kwargs:
        params[param] = kwargs[param]

    # Do not set defaults that may be rejected by some models.
    # Map generic max_tokens to max_completion_tokens when newer models demand it.
    if "max_tokens" in params and "max_completion_tokens" not in params:
      params["max_completion_tokens"] = params.pop("max_tokens")

    return params

  def _adapt_params_on_400(
    self, error_str: str, original_kwargs: Dict[str, Any]
  ) -> Optional[Dict[str, Any]]:
    """Try to adapt parameters based on OpenAI 400 messages to maximize compatibility.

    Returns a new kwargs dict if adaptation is attempted, else None.
    """
    lower = error_str.lower()
    adapted = dict(original_kwargs)

    if "unsupported parameter" in lower and "max_tokens" in lower:
      if "max_tokens" in adapted and "max_completion_tokens" not in adapted:
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
        return adapted

    if "unsupported value" in lower and "temperature" in lower:
      if "temperature" in adapted:
        adapted.pop("temperature", None)
        return adapted

    return None

  def __repr__(self) -> str:
    """String representation of the provider."""
    return f"OpenAIProvider(model='{self.model}', base_url='{self.base_url or 'default'}')"
