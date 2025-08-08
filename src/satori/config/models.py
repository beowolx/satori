from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
  """Configuration for a specific LLM provider."""

  api_key: Optional[str] = None
  base_url: Optional[str] = None
  organization: Optional[str] = None
  timeout: float = 60.0
  max_retries: int = 3


class SatoriConfig(BaseModel):
  """Main configuration model for Satori."""

  # Default providers and models
  default_provider: str = "openai:gpt-4o"
  default_judge: str = "gpt-4.1"

  # Execution settings
  concurrency: int = 5
  rate_limit_delay: float = 1.0

  # Provider configurations
  providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

  # Output settings
  verbose: bool = False
  output_format: str = "csv"

  class Config:
    extra = "allow"  # Allow additional fields for flexibility
