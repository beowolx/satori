import os
from typing import Optional

from openai import AsyncOpenAI, OpenAI


class OpenAIClientMixin:
  """Mixin for OpenAI client management."""

  def __init__(self, api_key: Optional[str] = None, **kwargs):
    self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not self.api_key:
      raise ValueError(
        "OpenAI API key not found. Please provide it as a parameter "
        "or set the OPENAI_API_KEY environment variable."
      )

    self.base_url = kwargs.get("base_url")
    self.organization = kwargs.get("organization")
    self.timeout = kwargs.get("timeout", 60.0)

    self.async_client = AsyncOpenAI(
      api_key=self.api_key,
      base_url=self.base_url,
      organization=self.organization,
      timeout=self.timeout,
    )
    self._sync_client: Optional[OpenAI] = None

  @property
  def sync_client(self) -> OpenAI:
    if self._sync_client is None:
      self._sync_client = OpenAI(
        api_key=self.api_key,
        base_url=self.base_url,
        organization=self.organization,
        timeout=self.timeout,
      )
    return self._sync_client

  def _handle_openai_error(self, e: Exception) -> Exception:
    """Transform OpenAI-specific errors into more user-friendly ones."""
    error_str = str(e).lower()
    if "api_key" in error_str:
      return ValueError("Invalid OpenAI API key")
    elif "rate_limit" in error_str or "429" in str(e):
      return ValueError("OpenAI rate limit exceeded. Please wait and retry.")
    elif "model" in error_str and "not found" in error_str:
      return ValueError("Model not found or not accessible")
    return e
