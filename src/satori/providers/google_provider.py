import asyncio
import os
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

from .base import BaseLLMProvider, ProviderError


class GoogleProvider(BaseLLMProvider):
  """Google Generative AI provider for Gemini models."""

  def __init__(
    self,
    model: str = "gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    **kwargs,
  ):
    """Initialize Google provider.

    Args:
        model: Model identifier (e.g., "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro")
        api_key: Google API key. If not provided, will use GEMINI_API_KEY env var
        **kwargs: Additional configuration options

    Raises:
        ProviderError: If API key is not provided or found in environment
    """
    super().__init__(model, **kwargs)

    # Get API key from parameter or environment (check both GEMINI_API_KEY and GOOGLE_API_KEY)
    self.api_key = (
      api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    if not self.api_key:
      raise ProviderError(
        "Google API key not found. Please provide it as a parameter "
        "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
      )

    # Initialize the Google GenAI client
    try:
      self.client = genai.Client(api_key=self.api_key)
    except Exception as e:
      raise ProviderError(f"Failed to initialize Google client: {e}")

  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from Google Gemini asynchronously.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters:
            - temperature (float): Sampling temperature (0-2)
            - max_tokens (int): Maximum tokens to generate (max_output_tokens in Google)
            - top_p (float): Nucleus sampling parameter
            - top_k (int): Top-k sampling parameter
            - stop_sequences (list): Stop sequences
            - system_message (str): System message (used as system_instruction)

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    try:
      # Prepare contents with system message if provided
      contents = prompt

      # Extract generation parameters
      config = self._extract_generation_params(kwargs)

      # Use asyncio to run the synchronous generation in a thread pool
      # Google's SDK has async support but we'll use sync in executor for consistency
      loop = asyncio.get_event_loop()
      response = await loop.run_in_executor(
        None,
        lambda: self.client.models.generate_content(
          model=self.model, contents=contents, config=config
        ),
      )

      # Extract and return the response text
      if response and response.text:
        return response.text
      else:
        raise ProviderError("Empty response from Google API")

    except Exception as e:
      # Transform provider-specific errors
      if "api_key" in str(e).lower() or "invalid" in str(e).lower():
        raise ProviderError("Invalid Google API key")
      elif "quota" in str(e).lower() or "rate" in str(e).lower():
        raise ProviderError(
          "Google API rate limit or quota exceeded. Please wait and retry."
        )
      elif "model" in str(e).lower() and "not found" in str(e).lower():
        raise ProviderError(f"Model '{self.model}' not found or not accessible")
      elif "blocked" in str(e).lower() or "safety" in str(e).lower():
        raise ProviderError(
          "Content was blocked by Google's safety filters. Try adjusting your prompt."
        )
      else:
        raise self._handle_error(e)

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous generation using Google Generative AI.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    try:
      # Prepare contents
      contents = prompt

      # Extract generation parameters
      config = self._extract_generation_params(kwargs)

      # Make API call
      response = self.client.models.generate_content(
        model=self.model, contents=contents, config=config
      )

      # Extract and return the response text
      if response and response.text:
        return response.text
      else:
        raise ProviderError("Empty response from Google API")

    except Exception as e:
      # Transform provider-specific errors
      if "api_key" in str(e).lower() or "invalid" in str(e).lower():
        raise ProviderError("Invalid Google API key")
      elif "quota" in str(e).lower() or "rate" in str(e).lower():
        raise ProviderError(
          "Google API rate limit or quota exceeded. Please wait and retry."
        )
      elif "model" in str(e).lower() and "not found" in str(e).lower():
        raise ProviderError(f"Model '{self.model}' not found or not accessible")
      elif "blocked" in str(e).lower() or "safety" in str(e).lower():
        raise ProviderError(
          "Content was blocked by Google's safety filters. Try adjusting your prompt."
        )
      else:
        raise self._handle_error(e)

  def _extract_generation_params(
    self, kwargs: Dict[str, Any]
  ) -> types.GenerateContentConfig:
    """Extract valid Google generation parameters from kwargs.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        GenerateContentConfig object for Google API
    """
    config_dict = {}

    # Handle system message as system_instruction
    if "system_message" in kwargs:
      config_dict["system_instruction"] = kwargs["system_message"]

    # Map common parameters to Google's naming
    if "temperature" in kwargs:
      config_dict["temperature"] = kwargs["temperature"]
    else:
      config_dict["temperature"] = 0.7

    if "max_tokens" in kwargs:
      # Google uses max_output_tokens
      config_dict["max_output_tokens"] = kwargs["max_tokens"]
    else:
      # Use a higher default for better completions
      config_dict["max_output_tokens"] = 8192

    if "top_p" in kwargs:
      config_dict["top_p"] = kwargs["top_p"]

    if "top_k" in kwargs:
      config_dict["top_k"] = kwargs["top_k"]

    if "stop_sequences" in kwargs:
      config_dict["stop_sequences"] = kwargs["stop_sequences"]
    elif "stop" in kwargs:
      # Support OpenAI-style stop parameter
      config_dict["stop_sequences"] = kwargs["stop"]

    return types.GenerateContentConfig(**config_dict)

  def __repr__(self) -> str:
    """String representation of the provider."""
    return f"GoogleProvider(model='{self.model}')"
