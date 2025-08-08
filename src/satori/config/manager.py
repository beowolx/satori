import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .models import SatoriConfig


class ConfigManager:
  """Manages configuration from multiple sources with priority hierarchy."""

  DEFAULT_CONFIG_DIR = Path.home() / ".config" / "satori"
  DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

  def __init__(self, config_path: Optional[Path] = None):
    """Initialize config manager.

    Args:
        config_path: Custom config file path (optional)
    """
    self.config_path = config_path or self.DEFAULT_CONFIG_FILE
    self._config_cache: Optional[SatoriConfig] = None

  def get_config(
    self, cli_args: Optional[Dict[str, Any]] = None
  ) -> SatoriConfig:
    """Get merged configuration from all sources.

    Priority order:
    1. CLI arguments (highest)
    2. Environment variables
    3. User config file (lowest)

    Args:
        cli_args: Command-line arguments dict

    Returns:
        Merged configuration object
    """
    # Start with defaults
    config_dict = SatoriConfig().model_dump()

    # Layer 3: User config file (lowest priority)
    file_config = self._load_config_file()
    if file_config:
      config_dict.update(file_config)

    # Layer 2: Environment variables
    env_config = self._load_env_config()
    config_dict.update(env_config)

    # Layer 1: CLI arguments (highest priority)
    if cli_args:
      cli_config = self._process_cli_args(cli_args)
      config_dict.update(cli_config)

    return SatoriConfig(**config_dict)

  def _load_config_file(self) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not self.config_path.exists():
      return {}

    try:
      with open(self.config_path, "r") as f:
        return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
      print(f"Warning: Failed to load config file {self.config_path}: {e}")
      return {}

  def _load_env_config(self) -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}

    # API Keys
    if openai_key := os.getenv("OPENAI_API_KEY"):
      config.setdefault("providers", {}).setdefault("openai", {})["api_key"] = (
        openai_key
      )

    if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
      config.setdefault("providers", {}).setdefault("anthropic", {})[
        "api_key"
      ] = anthropic_key

    if google_key := os.getenv("GOOGLE_API_KEY"):
      config.setdefault("providers", {}).setdefault("google", {})["api_key"] = (
        google_key
      )

    if hf_token := os.getenv("HUGGINGFACE_TOKEN"):
      config.setdefault("providers", {}).setdefault("huggingface", {})[
        "api_key"
      ] = hf_token

    if openrouter_key := os.getenv("OPENROUTER_API_KEY"):
      config.setdefault("providers", {}).setdefault("openrouter", {})[
        "api_key"
      ] = openrouter_key

    # Default settings
    if default_provider := os.getenv("SATORI_DEFAULT_PROVIDER"):
      config["default_provider"] = default_provider

    if default_judge := os.getenv("SATORI_DEFAULT_JUDGE"):
      config["default_judge"] = default_judge

    if concurrency := os.getenv("SATORI_CONCURRENCY"):
      try:
        config["concurrency"] = int(concurrency)
      except ValueError:
        pass

    if rate_limit := os.getenv("SATORI_RATE_LIMIT"):
      try:
        config["rate_limit_delay"] = float(rate_limit)
      except ValueError:
        pass

    return config

  def _process_cli_args(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Process CLI arguments into config format."""
    config = {}

    # Map CLI args to config structure
    if "provider" in cli_args and cli_args["provider"]:
      config["default_provider"] = cli_args["provider"]

    if "judge_model" in cli_args and cli_args["judge_model"]:
      config["default_judge"] = cli_args["judge_model"]

    if "concurrency" in cli_args and cli_args["concurrency"]:
      config["concurrency"] = cli_args["concurrency"]

    if "rate_limit_delay" in cli_args and cli_args["rate_limit_delay"]:
      config["rate_limit_delay"] = cli_args["rate_limit_delay"]

    return config

  def save_config(self, config: Union[SatoriConfig, Dict[str, Any]]) -> None:
    """Save configuration to file."""
    self.config_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, SatoriConfig):
      config_dict = config.model_dump(exclude_unset=True)
    else:
      config_dict = config

    with open(self.config_path, "w") as f:
      yaml.dump(config_dict, f, default_flow_style=False, indent=2)

  def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
    """Get configuration for a specific provider."""
    config = self.get_config()
    return config.providers.get(provider_name, {})

  def set_provider_config(
    self, provider_name: str, key: str, value: Any
  ) -> None:
    """Set a configuration value for a provider."""
    current_config = self._load_config_file()

    # Ensure nested structure exists
    if "providers" not in current_config:
      current_config["providers"] = {}
    if provider_name not in current_config["providers"]:
      current_config["providers"][provider_name] = {}

    current_config["providers"][provider_name][key] = value
    self.save_config(current_config)

  def set_config(self, key: str, value: Any) -> None:
    """Set a top-level configuration value."""
    current_config = self._load_config_file()
    current_config[key] = value
    self.save_config(current_config)

  def get_config_value(self, key: str, default: Any = None) -> Any:
    """Get a specific configuration value."""
    config = self.get_config()
    return getattr(config, key, default)

  def list_config(self) -> Dict[str, Any]:
    """List all configuration values (masking sensitive data)."""
    config = self.get_config().model_dump()
    return self._mask_sensitive_data(config)

  def _mask_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive configuration data for display."""
    masked = config.copy()

    if "providers" in masked:
      for provider, settings in masked["providers"].items():
        if isinstance(settings, dict) and "api_key" in settings:
          key = settings["api_key"]
          if key and len(key) > 8:
            # Show first 4 and last 4 chars with asterisks in middle
            masked["providers"][provider]["api_key"] = f"{key[:4]}...{key[-4:]}"

    return masked


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
  """Get the global config manager instance."""
  global _config_manager
  if _config_manager is None:
    _config_manager = ConfigManager(config_path)
  return _config_manager


def get_config(cli_args: Optional[Dict[str, Any]] = None) -> SatoriConfig:
  """Convenience function to get merged configuration."""
  return get_config_manager().get_config(cli_args)
