# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Strict type checking with basedpyright
- 90% test coverage enforcement in CI
- EditorConfig for consistent coding style
- CONTRIBUTING.md with development guidelines
- SECURITY.md with vulnerability reporting policy
- CHANGELOG.md following Keep a Changelog format

### Changed

- Enabled D100-D103 docstring rules (full docstring coverage)
- Replaced generic exception handlers with specific exception types
- Upgraded type checking from "standard" to "strict" mode

## [0.1.0] - 2024-12-17

### Added

- Initial release
- Chapter-based EPUB to Mela recipe extraction
- Two-stage title-grounded extraction pipeline
- Async parallel processing with configurable concurrency
- OpenAI structured output API integration
- Image extraction and optimization
- Recipe deduplication with fuzzy matching
- Rich CLI with progress display
- Configuration via TOML files and environment variables
- Comprehensive test suite with pytest
- GitHub Actions CI/CD pipeline
- PyPI publishing workflow

### Technical

- Python 3.13+ required
- Built with hatchling
- Managed with uv package manager
- Linted with ruff
- Type-checked with basedpyright

[Unreleased]: https://github.com/gwhthompson/mela_parser/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gwhthompson/mela_parser/releases/tag/v0.1.0
