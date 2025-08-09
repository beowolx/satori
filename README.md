# Satori

Run your own LLM benchmarks on your data — quickly. Satori is a vendor‑agnostic CLI that scores model outputs using an “LLM‑as‑a‑judge” approach, so you can see how different models perform on your exact prompts.

## Features

- Simple: point at a CSV, pick a model, get scores.
- Fair: uses a judge model (default: GPT‑4.1) to grade outputs.
- Flexible: works with OpenAI, Anthropic, Google, OpenRouter, HuggingFace, Ollama (local), and more.
- Fast: async processing with progress, ETA, and throughput.
- Portable: save results as CSV/JSON/JSONL and compare runs.

## Quick Start

```bash
# Install
uv tool install "git+https://github.com/beowolx/satori@v0.1.0"

# Configure keys (interactive)
satori config init

# Run on the example dataset
satori run examples/sample_dataset.csv --provider openai:gpt-4o

# Save results to a file
satori run examples/sample_dataset.csv \
  --provider openai:gpt-4o --output results.json

# Compare two runs
satori compare results_a.json results_b.json

# Help
satori --help
satori run --help
satori compare --help
```

## Dataset Format

Satori expects a CSV with two columns (defaults shown):

```csv
input,expected_output
"What is the capital of France?","Paris"
"Explain photosynthesis in simple terms.","Plants use sunlight to make energy."
```

Use custom column names with `--input-col` and `--expected-col`:

```bash
satori run my_data.csv \
  --provider openai:gpt-4o \
  --input-col question \
  --expected-col answer
```

## How It Scores (LLM‑as‑a‑Judge)

- For each row, Satori asks a candidate model to answer `input`.
- A judge model compares the answer to `expected_output` and assigns a score (0–5) with a short explanation.
- Summary statistics (mean, median, std), success/failure counts, and timing are reported.
- You can switch the judge (e.g., `--judge-model anthropic:claude-3-5-sonnet-20241022`).

Notes and caveats:

- Judge models can be biased; treat scores as guidance, not ground truth.
- For subjective tasks, include a clear rubric in your `expected_output` or prompt.

## Providers & Keys

- Configure credentials via `satori config init` or environment variables like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.
- Provider strings follow `provider:model` (e.g., `openai:gpt-4o`, `anthropic:claude-3-5-sonnet-20241022`).
- List available providers: `satori list-providers`.

## Outputs & Comparison

- Results: CSV, JSON, or JSONL via `--output`.
- Compare two runs: `satori compare <file_a> <file_b>`
  - Joins by `--key-col` (default: `input`) and shows deltas
  - Optional `--pass-threshold` to compute accuracy

## Contributing & License

- Dev loop: `uv sync` → `uv run satori …`
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- License: MIT
