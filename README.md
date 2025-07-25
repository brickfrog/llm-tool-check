# LLM Tool Support Tracker

Automated tracking of function calling (tool) support across different models and providers on OpenRouter, Hugging Face, and io.net.

## Usage

### Quick Start

```bash
make setup              # Install dependencies and create .env
# Edit .env and add your API keys (OPENROUTER_API_KEY, HF_TOKEN, IO_API_KEY)
make test-all           # Run all tests and generate website
```

### Manual Commands

```bash
# Setup
uv sync
cp .env.example .env

# Run individual tests
make test-openrouter    # or: uv run src/checkers/http/openrouter.py
make test-huggingface   # or: uv run src/checkers/http/huggingface.py
make test-ionet         # or: uv run src/checkers/http/ionet.py

# Run all tests in parallel
make test               # or run the individual commands with & and wait

# Generate website
make generate-website   # or: uv run src/generator/website.py
```

### Configuration

Edit the model list by changing `config/models.json`.

Sometimes, the model (or the provider) does not properly call tools. Thats why every call is made **three times**.
