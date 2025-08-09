import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
  BarColumn,
  Progress,
  SpinnerColumn,
  TaskProgressColumn,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)
from rich.table import Table
from rich.traceback import install as rich_traceback_install

from .config import get_config
from .config.cli import config_app
from .core.run_manager import RunManager
from .io.result_writer import ResultWriter
from .judges.factory import JudgesFactory
from .providers import ProviderFactory, create_provider

app = typer.Typer(
  name="satori",
  help="Vendor-agnostic CLI tool for evaluating LLM responses using LLM-as-a-Judge approach",
  add_completion=False,
)
app.add_typer(config_app)
console = Console()


def create_judge(judge_model: str, config=None):
  """Create a judge instance.

  Supports identifiers like:
  - "openai:gpt-4o" -> OpenAIJudge
  - "anthropic:claude-3-5-sonnet-20241022" -> GenericLLMJudge via provider
  - "gpt-4.1" -> uses default provider from config (falls back to openai)
  """
  return JudgesFactory.create(judge_model, config=config)


@app.command()
def run(
  data: Path = typer.Argument(
    ...,
    help="Path to CSV file with 'input' and 'expected_output' columns",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
  ),
  provider: Optional[str] = typer.Option(
    None,
    "--provider",
    "-p",
    help="Provider and model in format 'provider:model' (e.g., 'openai:gpt-4o', 'anthropic:claude-3-opus')",
  ),
  judge_model: Optional[str] = typer.Option(
    None,
    "--judge-model",
    "-j",
    help="Model to use as judge (defaults to gpt-4.1 if not specified)",
  ),
  output: Optional[Path] = typer.Option(
    None,
    "--output",
    "-o",
    help="Output file path for results. Format detected by extension (.json, .jsonl, .csv)",
  ),
  gen: List[str] = typer.Option(
    [],
    "--gen",
    help=(
      "Arbitrary generation params as key=value; repeatable. "
      "Examples: --gen temperature=0.2 --gen max_tokens=1024 --gen stop=a,b"
    ),
  ),
  temperature: Optional[float] = typer.Option(
    None,
    "--temperature",
    help="Sampling temperature (provider-specific constraints apply)",
  ),
  max_tokens: Optional[int] = typer.Option(
    None,
    "--max-tokens",
    help=(
      "Max tokens for completion. Mapped as needed (e.g., OpenAI newer "
      "models may use max_completion_tokens)."
    ),
  ),
  provider_timeout: Optional[float] = typer.Option(
    None,
    "--provider-timeout",
    help="Timeout (seconds) for provider generation (applies to HTTP client and per-call wait)",
  ),
  concurrency: int = typer.Option(
    5,
    "--concurrency",
    "-c",
    help="Number of concurrent evaluation requests (1-50, higher values increase speed but may hit rate limits)",
    min=1,
    max=50,
  ),
  rate_limit_delay: float = typer.Option(
    1.0,
    "--rate-limit-delay",
    "-r",
    help="Delay in seconds between requests to avoid API rate limiting",
  ),
  verbose: bool = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Show detailed progress and error information during evaluation",
  ),
):
  """Run LLM evaluation on a dataset using the LLM-as-a-Judge approach.

  This command loads test cases from a CSV file (which must contain 'input' and
  'expected_output' columns), generates responses using the specified LLM provider,
  and evaluates the quality of those responses using an LLM judge (GPT-4.1 by default).

  Examples:
    # Basic evaluation with OpenAI
    satori run dataset.csv --provider openai:gpt-4o

    # Evaluation with custom judge and output file
    satori run data.csv --provider anthropic:claude-3-opus --judge-model gpt-4.1 --output results.json

    # High-throughput evaluation with rate limiting
    satori run large_dataset.csv --provider openai:gpt-4o --concurrency 10 --rate-limit-delay 0.5

  The CSV file should have the following format:
    input,expected_output
    "What is 2+2?","4"
    "Capital of France?","Paris"
  """
  try:
    # Enable rich tracebacks when running in verbose mode
    if verbose:
      rich_traceback_install(show_locals=False)
    # Get configuration with CLI args taking precedence
    cli_args = {
      "provider": provider,
      "judge_model": judge_model,
      "concurrency": concurrency,
      "rate_limit_delay": rate_limit_delay,
      "verbose": verbose,
    }
    config = get_config(cli_args)

    # Use selected provider; judge defaults to GPT-4.1 ALWAYS unless explicitly provided via -j
    actual_provider = provider or config.default_provider
    actual_judge = judge_model or "gpt-4.1"
    actual_concurrency = (
      concurrency if "concurrency" in locals() else config.concurrency
    )
    actual_rate_limit = (
      rate_limit_delay
      if "rate_limit_delay" in locals()
      else config.rate_limit_delay
    )

    console.print(
      Panel.fit(
        "[bold cyan]Satori LLM Evaluation[/bold cyan]\n"
        f"Provider: [green]{actual_provider}[/green]\n"
        f"Judge: [green]{actual_judge}[/green]\n"
        f"Dataset: [yellow]{data}[/yellow]",
        title="Configuration",
      )
    )

    with console.status("[bold green]Initializing provider and judge..."):
      try:
        provider_kwargs: Dict[str, Any] = {}
        if provider_timeout is not None:
          provider_kwargs["timeout"] = provider_timeout

        llm_provider = create_provider(
          actual_provider, config=config, **provider_kwargs
        )
        llm_judge = create_judge(actual_judge, config=config)
      except Exception as e:
        console.print(f"[red]Error initializing components: {e}[/red]")
        raise typer.Exit(1)

    # Build generation params: config defaults < --gen < explicit flags
    generation_params: Dict[str, Any] = {}
    try:
      provider_name, _ = ProviderFactory.parse_provider_string(actual_provider)
    except Exception:
      provider_name = (actual_provider or "").split(":", 1)[0]

    provider_defaults = (
      (config.providers.get(provider_name, {}) or {}).get(
        "generation_defaults", {}
      )
      if hasattr(config, "providers")
      else {}
    )

    # Start with provider defaults
    if isinstance(provider_defaults, dict):
      generation_params.update(provider_defaults)

    # Merge --gen key=value overrides
    cli_gen_params = _parse_gen_list(gen)
    generation_params.update(cli_gen_params)

    # Convenience flags have highest precedence
    if temperature is not None:
      generation_params["temperature"] = temperature
    if max_tokens is not None:
      generation_params["max_tokens"] = max_tokens

    run_manager = RunManager(
      provider=llm_provider,
      judge=llm_judge,
      max_concurrent=actual_concurrency,
      rate_limit_delay=actual_rate_limit,
      fail_fast=True,
      generation_params=generation_params,
      provider_call_timeout=provider_timeout,
    )

    console.print("\n[bold]Starting evaluation...[/bold]")
    from .io.csv_loader import CSVLoader

    loader = CSVLoader(str(data))
    loader.load()
    total_cases = len(loader)

    progress = Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      BarColumn(),
      TaskProgressColumn(),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      console=console,
    )

    task_ref = {}
    successes = 0
    failures = 0

    def update_progress(current: int, total: int, result):
      nonlocal successes, failures
      if result and getattr(result, "error", None):
        failures += 1
        if verbose:
          console.print(
            f"[yellow]Warning: Test {current}/{total} failed: {result.error}[/yellow]"
          )
      else:
        successes += 1
      if "task" in task_ref:
        progress.update(
          task_ref["task"],
          advance=1,
          description=f"[cyan]Evaluating... (ok: {successes}, fail: {failures})",
        )

    with progress:
      task = progress.add_task(
        f"[cyan]Evaluating... (ok: {successes}, fail: {failures})",
        total=total_cases,
      )
      task_ref["task"] = task

      batch_results = asyncio.run(
        run_manager.run_batch(
          csv_path=str(data),
          progress_callback=update_progress,
        )
      )

    console.print("\n[bold green]Evaluation Complete![/bold green]\n")

    table = Table(title="Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Cases", str(len(batch_results.results)))
    table.add_row("Successful", str(batch_results.success_count))
    table.add_row("Failed", str(batch_results.failure_count))
    table.add_row("Average Score", f"{batch_results.average_score:.2f}")
    table.add_row("Median Score", f"{batch_results.median_score:.2f}")
    table.add_row("Std Deviation", f"{batch_results.std_score:.2f}")
    table.add_row("Total Time", f"{batch_results.total_time:.2f}s")
    throughput = (
      len(batch_results.results) / batch_results.total_time
      if batch_results.total_time > 0
      else 0.0
    )
    table.add_row("Throughput (cases/s)", f"{throughput:.2f}")

    console.print(table)

    if output:
      with console.status(f"[bold green]Saving results to {output}..."):
        try:
          output_str = str(output)
          if output_str.endswith(".json"):
            format = "json"
          elif output_str.endswith(".jsonl"):
            format = "jsonl"
          else:
            format = "csv"

          writer = ResultWriter(output_str)
          writer.write(batch_results, format=format)
          console.print(
            f"[green]✓ Results saved to {output} (format: {format})[/green]"
          )

        except Exception as e:
          console.print(f"[red]Error saving results: {e}[/red]")

    writer = ResultWriter()
    writer.print_summary(batch_results)

  except KeyboardInterrupt:
    console.print("\n[red]Evaluation interrupted by user[/red]")
    raise typer.Exit(1)
  except Exception as e:
    error_msg = str(e)
    if (
      "ProviderError" in error_msg
      or "Provider" in error_msg
      and "failed" in error_msg
    ):
      console.print(f"\n[red bold]Provider Error:[/red bold] [red]{e}[/red]")
      console.print(
        "\n[yellow]Tip: Check your API key and model name are correct.[/yellow]"
      )
    elif (
      "JudgeError" in error_msg
      or "Judge" in error_msg
      and "failed" in error_msg
    ):
      console.print(f"\n[red bold]Judge Error:[/red bold] [red]{e}[/red]")
      console.print(
        "\n[yellow]Tip: Check your judge API key and model are configured correctly.[/yellow]"
      )
    else:
      console.print(f"\n[red]Error during evaluation: {e}[/red]")

    if verbose:
      console.print("\n[dim]Full traceback:[/dim]")
      console.print_exception(show_locals=False)
    raise typer.Exit(1)


@app.command()
def version():
  """Display version and system information.

  Shows the current version of Satori along with a brief description
  of the tool's capabilities.
  """
  console.print("[bold cyan]Satori[/bold cyan] version [green]0.1.0[/green]")
  console.print(
    "Vendor-agnostic LLM evaluation tool using LLM-as-a-Judge approach"
  )


@app.command("list-providers")
def list_providers():
  """List all available LLM providers and their supported models.

  Displays a comprehensive table showing:
  - Provider names (openai, anthropic, google, etc.)
  - Example model names for each provider
  - The format required for the --provider argument
  - Available aliases for convenience

  Use this command to discover which providers you can use with the 'run' command.

  Examples:
    # List all providers
    satori list-providers

    # Then use a provider from the list
    satori run data.csv --provider openai:gpt-4o
  """
  table = Table(title="Available Providers")
  table.add_column("Provider", style="cyan")
  table.add_column("Example Models", style="green")
  table.add_column("Format", style="yellow")
  table.add_column("Aliases", style="blue")

  provider_info = {
    "openai": {
      "models": "gpt-4.1, gpt-4o, gpt-4, gpt-3.5-turbo",
      "format": "openai:model",
    },
    "anthropic": {
      "models": "claude-3-5-sonnet-20241022, claude-3-opus-20240229",
      "format": "anthropic:model",
    },
    "google": {
      "models": "gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro",
      "format": "google:model",
    },
    "huggingface": {
      "models": "meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2",
      "format": "huggingface:org/model",
    },
    "ollama": {
      "models": "llama2, mistral, gpt-oss:20b (local)",
      "format": "ollama:model or ollama:model:tag",
    },
    "openrouter": {
      "models": "openai/gpt-4, anthropic/claude-2, meta-llama/llama-2-70b-chat",
      "format": "openrouter:provider/model",
    },
  }

  available_providers = ProviderFactory.list_providers()
  aliases_map = ProviderFactory.list_aliases()

  provider_aliases = {}
  for alias, provider in aliases_map.items():
    if provider not in provider_aliases:
      provider_aliases[provider] = []
    provider_aliases[provider].append(alias)

  for provider in available_providers:
    if provider in provider_info:
      info = provider_info[provider]
      aliases = provider_aliases.get(provider, [])
      alias_str = ", ".join(aliases) if aliases else "-"
      table.add_row(provider, info["models"], info["format"], alias_str)

  console.print(table)
  console.print(
    "\n[dim]Note: Some providers may require additional setup[/dim]"
  )
  console.print(
    f"\n[green]Total providers available: {len(available_providers)}[/green]"
  )


def main():
  """Main entry point for the CLI."""
  app()


if __name__ == "__main__":
  main()


def _parse_gen_list(items: List[str]) -> Dict[str, Any]:
  """Parse repeatable --gen key=value items with basic type coercion.

  Coercions:
  - true/false -> bool
  - ints/floats -> numeric
  - comma-separated -> list[str]
  """
  parsed: Dict[str, Any] = {}
  for item in items:
    if "=" not in item:
      continue
    key, value = item.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
      continue

    lower = value.lower()
    if lower in ("true", "false"):
      parsed[key] = lower == "true"
      continue

    # numeric
    try:
      if "." in value:
        parsed[key] = float(value)
      else:
        parsed[key] = int(value)
      continue
    except ValueError:
      pass

    # list
    if "," in value:
      parsed[key] = [v.strip() for v in value.split(",") if v.strip()]
      continue

    parsed[key] = value
  return parsed
