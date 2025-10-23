.PHONY: help install dev test lint format clean build

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	uv sync

dev:  ## Install package with development dependencies
	uv sync --all-extras

test:  ## Run tests
	uv run pytest -v

test-cov:  ## Run tests with coverage
	uv run pytest -v --cov=src/mela_parser --cov-report=term-missing

lint:  ## Run linter (ruff check)
	uv run ruff check src tests scripts

format:  ## Format code with ruff
	uv run ruff format src tests scripts

format-check:  ## Check if code is formatted
	uv run ruff format --check src tests scripts

clean:  ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	rm -rf output/*.melarecipe

build:  ## Build package distributions
	uv build

lock:  ## Update lock file
	uv lock

sync:  ## Sync dependencies from lock file
	uv sync
