import asyncio
import os
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseLLMProvider, ProviderError


class OllamaProvider(BaseLLMProvider):
  """Ollama provider for local LLM models."""

  def __init__(
    self, model: str = "llama2", host: Optional[str] = None, **kwargs
  ):
    """Initialize Ollama provider.

    Args:
        model: Model identifier (e.g., "llama2", "mistral", "codellama", "phi")
        host: Ollama API host. If not provided, will use OLLAMA_HOST env var or default
        **kwargs: Additional configuration options

    Raises:
        ProviderError: If Ollama is not accessible
    """
    super().__init__(model, **kwargs)

    # Get host from parameter, environment, or use default
    self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Ensure host doesn't have trailing slash
    self.host = self.host.rstrip("/")

    # API endpoints
    self.generate_endpoint = f"{self.host}/api/generate"
    self.chat_endpoint = f"{self.host}/api/chat"
    self.tags_endpoint = f"{self.host}/api/tags"

    # Configuration
    self.timeout = kwargs.get(
      "timeout", 120.0
    )  # Longer timeout for local models
    self.stream = kwargs.get("stream", False)  # Disable streaming by default

    # Create session for connection pooling
    self._session: Optional[aiohttp.ClientSession] = None
    self._sync_session: Optional[aiohttp.ClientSession] = None

  async def _ensure_session(self) -> aiohttp.ClientSession:
    """Ensure aiohttp session exists."""
    if self._session is None or self._session.closed:
      timeout = aiohttp.ClientTimeout(total=self.timeout)
      self._session = aiohttp.ClientSession(timeout=timeout)
    return self._session

  async def _check_model_available(self) -> bool:
    """Check if the model is available in Ollama.

    Returns:
        True if model is available, False otherwise
    """
    try:
      session = await self._ensure_session()
      async with session.get(self.tags_endpoint) as response:
        if response.status == 200:
          data = await response.json()
          models = data.get("models", [])
          # Check both full model names and base names
          model_names = [m.get("name", "") for m in models]

          # Check if exact model name exists
          if self.model in model_names:
            return True

          # Also check base names (without tags) for compatibility
          base_names = [name.split(":")[0] for name in model_names]
          if self.model in base_names:
            return True

          return False
        return False
    except Exception:
      return False

  async def _pull_model(self) -> None:
    """Pull the model if it's not available locally.

    Raises:
        ProviderError: If model pull fails
    """
    try:
      session = await self._ensure_session()
      pull_endpoint = f"{self.host}/api/pull"

      # Pull the model
      async with session.post(
        pull_endpoint, json={"name": self.model, "stream": False}
      ) as response:
        if response.status != 200:
          raise ProviderError(
            f"Failed to pull model {self.model}: {response.status}"
          )

        # Wait for pull to complete (non-streaming)
        result = await response.json()
        if "error" in result:
          raise ProviderError(f"Error pulling model: {result['error']}")

    except aiohttp.ClientError as e:
      raise ProviderError(
        f"Failed to connect to Ollama at {self.host}: {str(e)}"
      )
    except Exception as e:
      raise ProviderError(f"Failed to pull model {self.model}: {str(e)}")

  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from Ollama asynchronously.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters:
            - temperature (float): Sampling temperature
            - max_tokens (int): Maximum tokens to generate (num_predict in Ollama)
            - top_p (float): Nucleus sampling parameter
            - top_k (int): Top-k sampling parameter
            - stop (list): Stop sequences
            - system_message (str): System message to prepend

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    try:
      # Ensure session exists
      session = await self._ensure_session()

      # Check if model is available, pull if not
      if not await self._check_model_available():
        print(
          f"Model {self.model} not found locally. Pulling from Ollama registry..."
        )
        await self._pull_model()

      # Prepare the request
      system_message = kwargs.get("system_message", "")

      # Use chat endpoint for better compatibility
      messages = []
      if system_message:
        messages.append({"role": "system", "content": system_message})
      messages.append({"role": "user", "content": prompt})

      # Extract generation parameters
      generation_params = self._extract_generation_params(kwargs)

      # Build request payload
      payload = {
        "model": self.model,
        "messages": messages,
        "stream": False,  # Disable streaming for simplicity
        "options": generation_params,
      }

      # Make API call
      async with session.post(self.chat_endpoint, json=payload) as response:
        if response.status == 200:
          data = await response.json()

          # Extract response text from the message
          if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
          else:
            raise ProviderError("Invalid response format from Ollama")
        else:
          error_text = await response.text()
          raise ProviderError(
            f"Ollama API error (status {response.status}): {error_text}"
          )

    except aiohttp.ClientError as e:
      raise ProviderError(
        f"Failed to connect to Ollama at {self.host}. "
        f"Make sure Ollama is running (ollama serve). Error: {str(e)}"
      )
    except ProviderError:
      raise
    except Exception as e:
      raise self._handle_error(e)

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous wrapper for generate method.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    # Run the async method in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
      result = loop.run_until_complete(self.generate(prompt, **kwargs))
      # Clean up session after sync use
      loop.run_until_complete(self.close())
      return result
    finally:
      loop.close()

  def _extract_generation_params(
    self, kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Extract valid Ollama generation parameters from kwargs.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        Dictionary of valid Ollama parameters in the options format
    """
    # Map common parameter names to Ollama's expected names
    param_mapping = {
      "temperature": "temperature",
      "max_tokens": "num_predict",  # Ollama uses num_predict
      "top_p": "top_p",
      "top_k": "top_k",
      "stop": "stop",
      "seed": "seed",
      "repeat_penalty": "repeat_penalty",
    }

    params = {}
    for key, ollama_key in param_mapping.items():
      if key in kwargs:
        value = kwargs[key]
        # Convert stop sequences to proper format
        if key == "stop" and isinstance(value, list):
          params[ollama_key] = value
        else:
          params[ollama_key] = value

    # Set defaults for common parameters if not specified
    if "temperature" not in params:
      params["temperature"] = 0.7
    if "num_predict" not in params:
      params["num_predict"] = 8192  # Default max tokens

    return params

  async def close(self):
    """Close the aiohttp session."""
    if self._session and not self._session.closed:
      await self._session.close()

  def __repr__(self) -> str:
    """String representation of the provider."""
    return f"OllamaProvider(model='{self.model}', host='{self.host}')"
