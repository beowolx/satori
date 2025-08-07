from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .manager import get_config_manager

config_app = typer.Typer(
  name="config",
  help="Manage Satori configuration",
  no_args_is_help=True,
)
console = Console()


@config_app.command()
def init(
  force: bool = typer.Option(
    False, "--force", "-f", help="Overwrite existing configuration"
  ),
):
  """Initialize Satori configuration with an interactive setup wizard.

  This command creates a new configuration file with API keys, default providers,
  and other settings through an interactive prompt. Perfect for first-time setup.

  Examples:
    # First time setup
    satori config init

    # Force recreate existing configuration
    satori config init --force
  """
  config_manager = get_config_manager()
  config_path = config_manager.config_path

  if config_path.exists() and not force:
    if not Confirm.ask(
      f"Configuration already exists at {config_path}. Overwrite?"
    ):
      console.print("Configuration initialization cancelled.")
      return

  console.print("[bold cyan]Satori Configuration Setup[/bold cyan]\n")
  console.print("This will help you set up API keys and default settings.\n")

  config = {}

  # Provider setup
  console.print("[bold]Provider API Keys[/bold]")
  console.print(
    "Enter API keys for the providers you want to use (press Enter to skip):\n"
  )

  providers = {
    "openai": "OpenAI API Key",
    "anthropic": "Anthropic API Key",
    "google": "Google AI API Key",
    "huggingface": "Hugging Face Token",
    "openrouter": "OpenRouter API Key",
  }

  config["providers"] = {}
  for provider, description in providers.items():
    api_key = Prompt.ask(f"{description}", password=True, default="")
    if api_key:
      config["providers"][provider] = {"api_key": api_key}

  # Default settings
  console.print("\n[bold]Default Settings[/bold]")

  default_provider = Prompt.ask(
    "Default provider:model", default="openai:gpt-4o", show_default=True
  )
  config["default_provider"] = default_provider

  default_judge = Prompt.ask(
    "Default judge model", default="gpt-4.1", show_default=True
  )
  config["default_judge"] = default_judge

  concurrency = typer.prompt(
    "Default concurrency", type=int, default=5, show_default=True
  )
  config["concurrency"] = concurrency

  rate_limit = typer.prompt(
    "Default rate limit delay (seconds)",
    type=float,
    default=1.0,
    show_default=True,
  )
  config["rate_limit_delay"] = rate_limit

  # Save configuration
  config_manager.save_config(config)
  console.print(f"\n[green]✓ Configuration saved to {config_path}[/green]")


@config_app.command()
def set(
  key: str = typer.Argument(
    ..., help="Configuration key (use dot notation for nested)"
  ),
  value: str = typer.Argument(..., help="Configuration value"),
  provider: Optional[str] = typer.Option(
    None, "--provider", "-p", help="Set provider-specific config"
  ),
):
  """Set a specific configuration value.

  Allows you to set individual configuration values without going through
  the interactive setup. Supports dot notation for nested values.

  Examples:
    # Set default provider
    satori config set default_provider openai:gpt-4o

    # Set provider-specific API key
    satori config set api_key sk-... --provider openai

    # Set concurrency limit
    satori config set concurrency 10
  """
  config_manager = get_config_manager()

  try:
    if provider:
      # Setting provider-specific config
      config_manager.set_provider_config(provider, key, value)
      console.print(f"[green]✓ Set {provider}.{key} = {value}[/green]")
    else:
      # Setting top-level config
      # Handle type conversion
      processed_value = _process_value(value)
      config_manager.set_config(key, processed_value)
      console.print(f"[green]✓ Set {key} = {processed_value}[/green]")

  except Exception as e:
    console.print(f"[red]Error setting configuration: {e}[/red]")
    raise typer.Exit(1)


@config_app.command()
def get(
  key: Optional[str] = typer.Argument(
    None, help="Configuration key to retrieve"
  ),
  provider: Optional[str] = typer.Option(
    None, "--provider", "-p", help="Get provider-specific config"
  ),
):
  """Get a specific configuration value or display all settings.

  Retrieve individual configuration values or display the entire configuration.
  API keys are automatically masked for security.

  Examples:
    # Show all configuration
    satori config get

    # Get default provider
    satori config get default_provider

    # Get provider-specific API key (masked)
    satori config get api_key --provider openai
  """
  config_manager = get_config_manager()

  try:
    if key and provider:
      # Get specific provider config
      provider_config = config_manager.get_provider_config(provider)
      value = provider_config.get(key, "Not set")

      # Mask sensitive data
      if key == "api_key" and value and value != "Not set":
        if len(value) > 8:
          value = f"{value[:4]}...{value[-4:]}"

      console.print(f"{provider}.{key}: {value}")

    elif key:
      # Get specific top-level config
      value = config_manager.get_config_value(key)
      console.print(f"{key}: {value}")

    else:
      # Show all configuration
      config_dict = config_manager.list_config()
      _display_config(config_dict)

  except Exception as e:
    console.print(f"[red]Error getting configuration: {e}[/red]")
    raise typer.Exit(1)


@config_app.command()
def list():
  """Display all configuration values in a formatted table.

  Shows the complete configuration including default settings and
  provider configurations with API keys masked for security.

  This is equivalent to running 'satori config get' without arguments.
  """
  config_manager = get_config_manager()

  try:
    config_dict = config_manager.list_config()
    _display_config(config_dict)
  except Exception as e:
    console.print(f"[red]Error listing configuration: {e}[/red]")
    raise typer.Exit(1)


@config_app.command()
def path():
  """Display the path to the configuration file.

  Shows where Satori stores its configuration file and whether it exists.
  Useful for troubleshooting configuration issues or manual editing.

  The configuration file is typically located at:
  - Linux/Mac: ~/.config/satori/config.json
  - Windows: %APPDATA%/satori/config.json
  """
  config_manager = get_config_manager()
  console.print(f"Configuration file: {config_manager.config_path}")

  if config_manager.config_path.exists():
    console.print("[green]✓ File exists[/green]")
  else:
    console.print(
      "[yellow]⚠ File does not exist (run 'satori config init' to create)[/yellow]"
    )


def _process_value(value: str) -> any:
  """Process string value into appropriate type."""
  # Try to convert to appropriate type
  if value.lower() in ("true", "false"):
    return value.lower() == "true"

  try:
    # Try int first
    return int(value)
  except ValueError:
    try:
      # Try float
      return float(value)
    except ValueError:
      # Keep as string
      return value


def _display_config(config_dict: dict):
  """Display configuration in a nice table format."""

  table = Table(title="Satori Configuration")
  table.add_column("Key", style="cyan")
  table.add_column("Value", style="green")

  # Show main settings
  main_settings = [
    "default_provider",
    "default_judge",
    "concurrency",
    "rate_limit_delay",
    "verbose",
    "output_format",
  ]

  for key in main_settings:
    if key in config_dict:
      table.add_row(key, str(config_dict[key]))

  console.print(table)

  # Show providers in a separate table
  if "providers" in config_dict and config_dict["providers"]:
    console.print("\n")

    provider_table = Table(title="Provider Configuration")
    provider_table.add_column("Provider", style="cyan")
    provider_table.add_column("API Key", style="green")
    provider_table.add_column("Other Settings", style="yellow")

    for provider, settings in config_dict["providers"].items():
      if isinstance(settings, dict):
        api_key = settings.get("api_key", "Not set")
        other_settings = {k: v for k, v in settings.items() if k != "api_key"}
        other_str = (
          ", ".join([f"{k}={v}" for k, v in other_settings.items()])
          if other_settings
          else "-"
        )

        provider_table.add_row(provider, api_key, other_str)

    console.print(provider_table)
