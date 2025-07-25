.PHONY: help setup test test-openrouter test-huggingface test-ionet test-ionet-library generate-website test-all clean

help:
	@echo "Available commands:"
	@echo "  make setup              - Install dependencies and create .env file"
	@echo "  make test              - Run all HTTP API tests in parallel"
	@echo "  make test-openrouter   - Test OpenRouter models only"
	@echo "  make test-huggingface  - Test Hugging Face models only"
	@echo "  make test-ionet        - Test io.net models via HTTP API"
	@echo "  make test-ionet-library - Test io.net models via iointel library"
	@echo "  make generate-website  - Generate static website from results"
	@echo "  make test-all          - Run all tests and generate website"
	@echo "  make clean             - Remove generated files"

setup:
	uv sync --extra huggingface --extra iointel
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env file - please add your API keys"; fi

test-openrouter:
	uv run src/checkers/http/openrouter.py

test-huggingface:
	uv run src/checkers/http/huggingface.py

test-ionet:
	uv run src/checkers/http/ionet.py

test-ionet-library:
	uv run src/checkers/iointel/ionet.py

generate-website:
	uv run src/generator/website_generator.py

test:
	@echo "Running all tests in parallel..."
	@uv run src/checkers/http/openrouter.py & \
	uv run src/checkers/http/huggingface.py & \
	uv run src/checkers/http/ionet.py & \
	wait
	@echo "All tests completed!"

test-all: test test-ionet-library generate-website
	@echo "All tests completed and website generated!"

clean:
	rm -rf data/*.json
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__
	rm -rf .ruff_cache
