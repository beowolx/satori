# Satori

A vendor-agnostic CLI tool for evaluating LLM responses using an "LLM-as-a-Judge" approach. Compare outputs from different language models (OpenAI, Anthropic, Ollama, etc.) against expected responses with automated scoring.

## Features

- **Multi-Provider Support**: Evaluate responses from Mistral, OpenAI, Anthropic, Google Gemini, OpenRouter, Ollama (local), HuggingFace, and more.
- **LLM-as-Judge**: Uses GPT-4.1 (or other models) to score responses automatically.
- **Flexible Configuration**: Manage settings via CLI commands, environment variables, or a config file (`~/.config/satori/config.yaml`).
- **Comprehensive Metrics**: Get detailed statistics including mean, median, and score distribution.
- **Async Processing**: Fast, concurrent evaluation with configurable rate limiting.
- **Multiple Output Formats**: Save results as CSV, JSON, or JSONL.
- **Local Model Support**: Evaluate local models via Ollama.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Dataset Format](#dataset-format)
- [Supported Providers](#supported-providers)
- [Usage Examples](#usage-examples)
- [Generation Parameters](#generation-parameters)
- [Output Formats](#output-formats)
- [Scoring System](#scoring-system)
- [Development](#development)

## Installation

### Via Homebrew (Recommended)

```bash
brew install satori
```

### From Source (Development)

Clone the repository and install using [uv](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/yourusername/satori.git
cd satori

# Install dependencies using uv
uv sync

# Run the CLI tool in development mode
uv run satori --help
```

_Requires Python 3.12 or higher._

## Quick Start

1.  **Initialize Satori:**
    This command starts an interactive wizard to help you configure API keys and set default models.

    ```bash
    satori config init
    ```

2.  **Run an evaluation:**
    Use the `run` command with a dataset file and a provider.

    ```bash
    satori run examples/sample_dataset.csv --provider openai:gpt-4o
    ```

3.  **View results:**
    A summary table is displayed in the terminal. To save results to a file:

    ```bash
    satori run examples/sample_dataset.csv --provider openai:gpt-4o --output results.json
    ```

## Configuration

Satori uses a flexible, three-tier configuration system that prioritizes settings in the following order:

1.  **Command-line arguments** (highest priority)
2.  **Environment variables**
3.  **User config file** (`~/.config/satori/config.yaml`, lowest priority)

This system allows you to set global defaults and override them for specific commands.

### Configuration Commands

Manage your settings easily with the `satori config` command:

| Command                           | Description                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `satori config init`              | Runs an interactive setup wizard to configure API keys and defaults.         |
| `satori config set <key> <value>` | Sets a configuration value. Use `--provider` for provider-specific settings. |
| `satori config get [key]`         | Retrieves a specific key or all configuration values.                        |
| `satori config list`              | Lists all configuration, automatically masking API keys.                     |
| `satori config path`              | Shows the location of the configuration file.                                |

**Examples:**

```bash
# Start the interactive setup
satori config init

# Set API keys for different providers
satori config set api_key "sk-..." --provider openai
satori config set api_key "sk-ant-..." --provider anthropic

# Set a default provider and judge model
satori config set default_provider "anthropic:claude-3-5-sonnet-20241022"
satori config set default_judge "openai:gpt-4o"

# Get a specific value
satori config get default_provider

# List all current settings
satori config list
```

### Environment Variables

You can also configure Satori using environment variables. These will override settings from the config file.

- **API Keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.
- **Defaults**: `SATORI_DEFAULT_PROVIDER`, `SATORI_DEFAULT_JUDGE`
- **Performance**: `SATORI_CONCURRENCY`, `SATORI_RATE_LIMIT`

### Command-Line Arguments

For maximum flexibility, you can override any setting on a per-command basis using command-line arguments.

```bash
# Run with defaults saved in your config (e.g. provider is set)
satori run data.csv

# Override the provider and judge for a single run
satori run data.csv --provider openai:gpt-4o --judge-model google:gemini-1.5-pro
```

## Dataset Format

Satori expects a CSV file with the following required columns:

| Column            | Description                            |
| ----------------- | -------------------------------------- |
| `input`           | The prompt/question to send to the LLM |
| `expected_output` | The reference answer for comparison    |

### Example Dataset

```csv
input,expected_output
"What is the capital of France?","The capital of France is Paris."
"Explain photosynthesis in simple terms.","Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
```

## Supported Providers

| Provider    | Type           | Example Models                                | API Key Required |
| ----------- | -------------- | --------------------------------------------- | ---------------- |
| OpenAI      | Cloud          | GPT-4.1, GPT-4o                               | Yes              |
| Anthropic   | Cloud          | Claude 4 Sonnet, Claude 3 Opus                | Yes              |
| Google      | Cloud          | Gemini 2.5, Gemini 2.5 Flash                  | Yes              |
| HuggingFace | Cloud          | Llama 4, Mistral, Falcon, FLAN-T5, Qwen, etc. | Optional\*       |
| OpenRouter  | Cloud (Router) | 100+ models from various providers            | Yes              |
| Ollama      | Local          | Llama 3, Mistral, CodeLlama                   | No               |

\*HuggingFace API key is optional for public models but recommended for higher rate limits.

## Usage Examples

### Basic Evaluation

```bash
# Evaluate with a specific provider (if no default is set)
satori run data.csv --provider openai:gpt-4.1
```

### Custom Judge Model

```bash
# Use a different model as judge
satori run data.csv --provider ollama:llama2 --judge-model gpt-4o
```

### Save Results

```bash
# Save as CSV
satori run data.csv --provider openai:gpt-4.1 --output results.csv

# Save as JSON
satori run data.csv --provider openai:gpt-4.1 --output results.json
```

### Performance Tuning

```bash
# Adjust concurrency (default: 5)
satori run data.csv --provider openai:gpt-4.1 --concurrency 10

# Add rate limiting delay (seconds between requests)
satori run data.csv --provider openai:gpt-4.1 --rate-limit-delay 2.0
```

### Compare Multiple Providers

```bash
satori run data.csv --provider openai:gpt-4.1 --output gpt4_results.json
satori run data.csv --provider anthropic:claude-3-5-sonnet-20241022 --output claude_results.json
satori run data.csv --provider google:gemini-1.5-pro --output gemini_results.json
```

## Commands

### List Available Providers

```bash
satori list-providers
```

### Show Version

```bash
satori version
```

### Get Help

```bash
satori --help
satori run --help
satori config --help
```

## Generation Parameters

Satori lets you pass generation parameters from the CLI. These are forwarded to the provider as-is. Some providers/models have different parameter names or constraints.

Common flags:

```bash
# Temperature (float)
satori run data.csv --provider openai:gpt-4o --temperature 0.7

# Token limits (OpenAI newer models use max_completion_tokens)
satori run data.csv --provider openai:gpt-4o --max-completion-tokens 2048

# For older OpenAI models, you can use max-tokens; Satori will adapt when possible
satori run data.csv --provider openai:gpt-4.1 --max-tokens 2048

# Arbitrary params via --gen key=value (repeatable)
satori run data.csv --provider huggingface:mistralai/Mistral-7B-Instruct-v0.2 \
  --gen top_p=0.9 --gen stop=END --gen stop=a,b,c

# Lists: comma-separated values are parsed into arrays
satori run data.csv --provider openai:gpt-4o --gen stop=</s>,<|eot|>
```

Notes:

- Parameters are forwarded to all providers. Provider-specific naming still applies. For example:
  - OpenAI: some models reject `temperature` values other than default 1; Satori will automatically retry without `temperature` if a 400 indicates an unsupported value.
  - OpenAI: if a 400 error indicates `max_tokens` is unsupported, Satori remaps to `max_completion_tokens` and retries.
- For non-OpenAI providers, params are passed through unchanged. Use `--gen` to pass provider-specific options (e.g., `--gen max_new_tokens=512` for HuggingFace text generation, or `--gen stop=a,b`).
- Because models vary, prefer the provider’s canonical names when known.

## Output Formats

### CSV Format

```csv
input,expected,candidate,score,explanation,provider,judge,execution_time,error,timestamp
"What is 2+2?","4","4",5.0,"Perfect answer","OpenAIProvider(model='gpt-4.1')","OpenAIJudge(model='gpt-4.1')",0.52,"","2024-01-15T10:30:00"
```

### JSON Format

```json
{
  "metadata": { ... },
  "statistics": { ... },
  "results": [ ... ]
}
```

## Scoring System

The judge uses a 0-5 integer scale:

| Score | Description                                     |
| ----- | ----------------------------------------------- |
| **5** | Fully correct, comprehensive, and well-reasoned |
| **4** | Almost perfect with minor issues                |
| **3** | Mostly correct with some notable issues         |
| **2** | Partially correct but missing key information   |
| **1** | Substantially incorrect with major errors       |
| **0** | Completely wrong, off-topic, or harmful         |

## Troubleshooting

**"Provider Error: Invalid API key"**

- Run `satori config init` to check and re-enter your API keys.
- You can also set keys directly: `satori config set api_key "sk-..." --provider openai`.

**"Failed to connect to Ollama"**

- Make sure Ollama is running: `ollama serve`.
- Check if Ollama is accessible at `http://localhost:11434`.

**"Model not found"**

- For Ollama: Pull the model first with `ollama pull model-name`.
- For cloud providers, check the model name is correct.

**Rate Limiting Errors**

- Increase `--rate-limit-delay` parameter.
- Reduce `--concurrency` parameter.

## Development

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/satori.git
   cd satori
   ```

2. **Install dependencies using uv:**

   ```bash
   uv sync
   ```

3. **Run the CLI in development mode:**
   ```bash
   # All commands should be prefixed with 'uv run' during development
   uv run satori --help
   uv run satori config init
   uv run satori run examples/sample_dataset.csv --provider openai:gpt-4o
   ```

### Project Structure

```
satori/
├── src/satori/
│   ├── providers/       # LLM provider implementations
│   ├── judges/          # Judge implementations
│   ├── core/            # Core logic (run manager, retry)
│   ├── io/              # Data loading and result writing
│   ├── config/          # Configuration management
│   └── cli.py           # CLI interface
├── examples/            # Example datasets
├── pyproject.toml      # Project configuration
├── uv.lock             # Locked dependencies
└── README.md           # This file
```

### Adding a New Provider

1. Create a new provider class in `src/satori/providers/`
2. Inherit from `BaseLLMProvider`
3. Implement the `generate()` method
4. Update the provider factory in `src/satori/providers/factory.py`

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
