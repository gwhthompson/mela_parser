# Contributing to mela-parser

Thank you for your interest in contributing to mela-parser! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gwhthompson/mela_parser.git
   cd mela_parser
   ```

2. Install dependencies:
   ```bash
   make dev
   ```

3. Install pre-commit hooks:
   ```bash
   make pre-commit-install
   ```

## Code Quality Standards

This project maintains high code quality standards. All contributions must pass:

### Linting and Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
make lint        # Run linter
make format      # Format code
make format-check # Check formatting
```

### Type Checking

We use [basedpyright](https://github.com/DetachHead/basedpyright) with strict mode:

```bash
make type-check
```

### Testing

We use [pytest](https://pytest.org/) with a minimum 70% code coverage requirement:

```bash
make test        # Run tests
make test-cov    # Run tests with coverage (enforces 70% minimum)
```

### All Checks

Run all checks before submitting a PR:

```bash
make pre-commit-run
```

## Coding Style

### Python Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use type hints for all function signatures
- Write docstrings for all public modules, classes, methods, and functions
- Use Google-style docstrings

### Example

```python
def extract_recipes(chapter: Chapter, max_concurrent: int = 10) -> list[MelaRecipe]:
    """Extract recipes from a cookbook chapter.

    Args:
        chapter: The chapter to extract recipes from.
        max_concurrent: Maximum concurrent API requests.

    Returns:
        List of extracted recipe objects.

    Raises:
        ExtractionError: If extraction fails after retries.
    """
    ...
```

## Pull Request Process

1. **Create a branch**: Branch from `master` with a descriptive name
2. **Make changes**: Follow the coding standards above
3. **Test locally**: Ensure all checks pass
4. **Write a clear PR description**: Explain what and why
5. **Request review**: Wait for approval before merging

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters

## Reporting Issues

When reporting issues, please include:

- Python version (`python --version`)
- mela-parser version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages or logs

## Questions?

Feel free to open an issue for questions or discussions about contributing.
