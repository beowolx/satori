import os
from typing import Any, Dict, Optional

from huggingface_hub import AsyncInferenceClient, InferenceClient

from .base import BaseLLMProvider, ProviderError


class HuggingFaceProvider(BaseLLMProvider):
  """HuggingFace Inference API provider for models hosted on HuggingFace Hub.

  Supports both the free Inference API and dedicated Inference Endpoints.
  Can use either serverless API or dedicated endpoints.
  """

  def __init__(
    self,
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
  ):
    """Initialize HuggingFace provider.

    Args:
        model: Model identifier on HuggingFace Hub (e.g., "meta-llama/Meta-Llama-3-8B-Instruct",
               "mistralai/Mistral-7B-Instruct-v0.2", "google/flan-t5-xxl")
        api_key: HuggingFace API token. If not provided, will use HF_TOKEN or HUGGINGFACE_TOKEN env var
        base_url: Optional custom endpoint URL for dedicated Inference Endpoints
        **kwargs: Additional configuration options

    Raises:
        ProviderError: If API key is not provided or found in environment (for private models)
    """
    super().__init__(model, **kwargs)

    # Get HuggingFace API token from parameter or environment
    self.api_key = (
      api_key
      or os.getenv("HF_TOKEN")
      or os.getenv("HUGGINGFACE_TOKEN")
      or os.getenv("HUGGINGFACE_API_KEY")
    )

    # Note: API key is optional for public models but required for private models
    # and for higher rate limits

    # Custom endpoint URL (for dedicated Inference Endpoints)
    self.base_url = base_url or kwargs.get("endpoint_url")

    # Extract HuggingFace-specific configuration
    self.timeout = kwargs.get(
      "timeout", 120.0
    )  # HF can be slower, especially for first calls

    # Initialize clients
    client_kwargs = {
      "timeout": self.timeout,
    }

    if self.api_key:
      client_kwargs["token"] = self.api_key

    if self.base_url:
      # Using a dedicated endpoint
      self.async_client = AsyncInferenceClient(
        base_url=self.base_url, **client_kwargs
      )
      self._sync_client = InferenceClient(
        base_url=self.base_url, **client_kwargs
      )
    else:
      # Using the serverless API with model name
      self.async_client = AsyncInferenceClient(
        model=self.model, **client_kwargs
      )
      self._sync_client = InferenceClient(model=self.model, **client_kwargs)

  async def generate(self, prompt: str, **kwargs) -> str:
    """Generate a response from HuggingFace model asynchronously.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters:
            - temperature (float): Sampling temperature (0-2)
            - max_tokens (int): Maximum tokens to generate
            - top_p (float): Nucleus sampling parameter
            - top_k (int): Top-k sampling parameter
            - repetition_penalty (float): Penalty for repetition
            - stop_sequences (list): Stop sequences
            - system_message (str): System message (for chat models)
            - use_chat_completion (bool): Use chat completion API (default: True for chat models)

    Returns:
        The generated response as a string

    Raises:
        ProviderError: If the API call fails
    """
    try:
      # Determine if we should use chat completion or text generation
      use_chat = kwargs.get("use_chat_completion", self._is_chat_model())

      if use_chat:
        # Use chat completion API (OpenAI-compatible)
        messages = self._prepare_messages(prompt, kwargs.get("system_message"))

        # Extract generation parameters for chat
        generation_params = self._extract_chat_params(kwargs)

        # Make async chat completion call
        response = await self.async_client.chat.completions.create(
          messages=messages, **generation_params
        )

        # Extract text from chat response
        if response.choices and len(response.choices) > 0:
          return response.choices[0].message.content or ""
        else:
          raise ProviderError("Empty response from HuggingFace API")
      else:
        # Use text generation API for non-chat models
        full_prompt = self._prepare_prompt(prompt, kwargs.get("system_message"))

        # Extract generation parameters for text generation
        generation_params = self._extract_generation_params(kwargs)

        # Make async text generation call
        response = await self.async_client.text_generation(
          prompt=full_prompt, **generation_params
        )

        # Response is directly a string for text_generation
        if response:
          return response
        else:
          raise ProviderError("Empty response from HuggingFace API")

    except Exception as e:
      # Transform provider-specific errors
      error_msg = str(e).lower()
      if "authorization" in error_msg or "token" in error_msg:
        raise ProviderError(
          "Invalid or missing HuggingFace API token. "
          "Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable."
        )
      elif "rate limit" in error_msg or "429" in str(e):
        raise ProviderError(
          "HuggingFace rate limit exceeded. Please wait and retry, "
          "or use a HuggingFace API token for higher limits."
        )
      elif "model" in error_msg and (
        "not found" in error_msg or "does not exist" in error_msg
      ):
        raise ProviderError(
          f"Model '{self.model}' not found on HuggingFace Hub or not accessible"
        )
      elif "loading" in error_msg or "starting" in error_msg:
        raise ProviderError(
          f"Model '{self.model}' is still loading. Please wait a moment and retry."
        )
      else:
        raise self._handle_error(e)

  def generate_sync(self, prompt: str, **kwargs) -> str:
    """Synchronous generation using HuggingFace Inference API.

    Args:
        prompt: The input prompt to send to the model
        **kwargs: Additional generation parameters

    Returns:
        The generated response as a string
    """
    try:
      # Determine if we should use chat completion or text generation
      use_chat = kwargs.get("use_chat_completion", self._is_chat_model())

      if use_chat:
        # Use chat completion API (OpenAI-compatible)
        messages = self._prepare_messages(prompt, kwargs.get("system_message"))

        # Extract generation parameters for chat
        generation_params = self._extract_chat_params(kwargs)

        # Make sync chat completion call
        response = self._sync_client.chat.completions.create(
          messages=messages, **generation_params
        )

        # Extract text from chat response
        if response.choices and len(response.choices) > 0:
          return response.choices[0].message.content or ""
        else:
          raise ProviderError("Empty response from HuggingFace API")
      else:
        # Use text generation API for non-chat models
        full_prompt = self._prepare_prompt(prompt, kwargs.get("system_message"))

        # Extract generation parameters for text generation
        generation_params = self._extract_generation_params(kwargs)

        # Make sync text generation call
        response = self._sync_client.text_generation(
          prompt=full_prompt, **generation_params
        )

        # Response is directly a string for text_generation
        if response:
          return response
        else:
          raise ProviderError("Empty response from HuggingFace API")

    except Exception as e:
      # Transform provider-specific errors (same as async)
      error_msg = str(e).lower()
      if "authorization" in error_msg or "token" in error_msg:
        raise ProviderError(
          "Invalid or missing HuggingFace API token. "
          "Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable."
        )
      elif "rate limit" in error_msg or "429" in str(e):
        raise ProviderError(
          "HuggingFace rate limit exceeded. Please wait and retry, "
          "or use a HuggingFace API token for higher limits."
        )
      elif "model" in error_msg and (
        "not found" in error_msg or "does not exist" in error_msg
      ):
        raise ProviderError(
          f"Model '{self.model}' not found on HuggingFace Hub or not accessible"
        )
      elif "loading" in error_msg or "starting" in error_msg:
        raise ProviderError(
          f"Model '{self.model}' is still loading. Please wait a moment and retry."
        )
      else:
        raise self._handle_error(e)

  def _is_chat_model(self) -> bool:
    """Determine if the model is a chat/instruct model based on its name.

    Returns:
        True if the model appears to be a chat model, False otherwise
    """
    chat_indicators = [
      "instruct",
      "chat",
      "conversation",
      "dialogue",
      "Instruct",
      "Chat",
      "INST",
      "chat-",
    ]
    return any(indicator in self.model for indicator in chat_indicators)

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

  def _prepare_prompt(
    self, prompt: str, system_message: Optional[str] = None
  ) -> str:
    """Prepare the prompt for text generation (non-chat models).

    Args:
        prompt: User prompt
        system_message: Optional system message

    Returns:
        Combined prompt string
    """
    if system_message:
      return f"{system_message}\n\n{prompt}"
    return prompt

  def _extract_chat_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract valid parameters for chat completion API.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        Dictionary of valid chat completion parameters
    """
    params = {}

    # Model is already set in the client initialization
    # but we can override it here if needed
    if "model" in kwargs:
      params["model"] = kwargs["model"]

    # Map common parameters
    if "temperature" in kwargs:
      params["temperature"] = kwargs["temperature"]

    if "max_tokens" in kwargs:
      params["max_tokens"] = kwargs["max_tokens"]
    elif "max_new_tokens" in kwargs:
      # HuggingFace sometimes uses max_new_tokens
      params["max_tokens"] = kwargs["max_new_tokens"]
    else:
      params["max_tokens"] = 4096

    if "top_p" in kwargs:
      params["top_p"] = kwargs["top_p"]

    if "stop" in kwargs:
      params["stop"] = kwargs["stop"]
    elif "stop_sequences" in kwargs:
      params["stop"] = kwargs["stop_sequences"]

    if "stream" in kwargs:
      params["stream"] = kwargs["stream"]

    return params

  def _extract_generation_params(
    self, kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Extract valid parameters for text generation API.

    Args:
        kwargs: Keyword arguments that may contain generation parameters

    Returns:
        Dictionary of valid text generation parameters
    """
    params = {}

    # HuggingFace text_generation parameters
    if "temperature" in kwargs:
      params["temperature"] = kwargs["temperature"]

    if "max_tokens" in kwargs:
      params["max_new_tokens"] = kwargs["max_tokens"]
    elif "max_new_tokens" in kwargs:
      params["max_new_tokens"] = kwargs["max_new_tokens"]
    else:
      params["max_new_tokens"] = 1000

    if "top_p" in kwargs:
      params["top_p"] = kwargs["top_p"]

    if "top_k" in kwargs:
      params["top_k"] = kwargs["top_k"]

    if "repetition_penalty" in kwargs:
      params["repetition_penalty"] = kwargs["repetition_penalty"]

    if "stop_sequences" in kwargs:
      params["stop_sequences"] = kwargs["stop_sequences"]
    elif "stop" in kwargs:
      params["stop_sequences"] = kwargs["stop"]

    # HuggingFace specific: return full text or just generated
    params["return_full_text"] = kwargs.get("return_full_text", False)

    return params

  def __repr__(self) -> str:
    """String representation of the provider."""
    if self.base_url:
      return f"HuggingFaceProvider(endpoint='{self.base_url}')"
    return f"HuggingFaceProvider(model='{self.model}')"
