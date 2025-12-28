"""Configuration management for mela_parser.

This module provides a centralized configuration system that supports:
- Default values for all settings
- Loading from TOML configuration files
- Environment variable overrides
- CLI argument overrides
- Validation of configuration values

Configuration priority (highest to lowest):
1. CLI arguments (passed directly to functions)
2. Environment variables (MELA_PARSER_*)
3. Project config file (.mela-parser.toml)
4. User config file (~/.config/mela-parser/config.toml)
5. Default values

Example:
    >>> config = ExtractionConfig.load()
    >>> config.model = "gpt-5-mini"  # Override
    >>> config.save("~/.config/mela-parser/config.toml")
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


@dataclass
class ExtractionConfig:
    """Configuration for recipe extraction.

    This dataclass holds all configurable parameters for the recipe extraction
    pipeline. Values can be loaded from files, environment variables, or set
    programmatically.

    Attributes:
        Model Settings:
            model: OpenAI model to use for extraction
            temperature: Sampling temperature (0.0 = deterministic)
            max_concurrent: Maximum concurrent API requests

        Extraction Settings:
            retry_attempts: Number of retry attempts for failed extractions
            initial_retry_delay: Initial delay before retry (exponential backoff)

        Image Settings:
            extract_images: Whether to extract and process images
            min_image_area: Minimum image area in pixels (filter small images)
            max_image_width: Maximum image width in pixels (resize larger images)
            use_ai_verification: Use AI to verify image relevance

        Output Settings:
            output_dir: Base directory for output files
            debug_mode: Enable debug output and logging

        Quality Settings:
            min_ingredients: Minimum number of ingredients for valid recipe
            min_instructions: Minimum number of instructions for valid recipe
            similarity_threshold: Threshold for fuzzy deduplication (0.0-1.0)

    Example:
        >>> config = ExtractionConfig()
        >>> config.model = "gpt-5-mini"
        >>> config.max_concurrent = 100
        >>> config.debug_mode = True
    """

    # Model settings
    model: str = "gpt-5-nano"
    temperature: float = 0.0
    max_concurrent: int = 200

    # Extraction settings
    retry_attempts: int = 3
    initial_retry_delay: float = 1.0

    # Grounded extraction settings (two-stage title-based extraction)
    extraction_concurrency_per_chapter: int = 20  # Concurrent extractions per chapter
    extraction_retry_attempts: int = 2  # Stage 2 retry count
    extraction_retry_delay: float = 0.5  # Base delay for retries (seconds)
    title_match_threshold: float = 0.85  # Fuzzy title matching threshold
    max_pagination_pages: int = 10  # Safety limit for legacy pagination

    # Image settings
    extract_images: bool = True
    min_image_area: int = 300000  # 300k pixels (e.g., 500x600)
    max_image_width: int = 600
    use_ai_verification: bool = False

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output"))
    debug_mode: bool = False

    # Quality settings
    min_ingredients: int = 1
    min_instructions: int = 2
    similarity_threshold: float = 0.90  # For fuzzy deduplication

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If any configuration value is invalid
        """
        # Validate model
        valid_models = {"gpt-5-nano", "gpt-5-mini", "gpt-4o", "gpt-4o-mini"}
        if self.model not in valid_models:
            raise ConfigurationError(
                f"Invalid model: {self.model}",
                model=self.model,
                valid_models=", ".join(valid_models),
            )

        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError(
                "Temperature must be between 0.0 and 2.0",
                temperature=self.temperature,
            )

        # Validate concurrency
        if self.max_concurrent < 1:
            raise ConfigurationError(
                "max_concurrent must be at least 1",
                max_concurrent=self.max_concurrent,
            )

        # Validate retry settings
        if self.retry_attempts < 0:
            raise ConfigurationError(
                "retry_attempts must be non-negative",
                retry_attempts=self.retry_attempts,
            )

        if self.initial_retry_delay <= 0:
            raise ConfigurationError(
                "initial_retry_delay must be positive",
                initial_retry_delay=self.initial_retry_delay,
            )

        # Validate image settings
        if self.min_image_area < 0:
            raise ConfigurationError(
                "min_image_area must be non-negative",
                min_image_area=self.min_image_area,
            )

        if self.max_image_width < 1:
            raise ConfigurationError(
                "max_image_width must be at least 1",
                max_image_width=self.max_image_width,
            )

        # Validate quality settings
        if self.min_ingredients < 0:
            raise ConfigurationError(
                "min_ingredients must be non-negative",
                min_ingredients=self.min_ingredients,
            )

        if self.min_instructions < 0:
            raise ConfigurationError(
                "min_instructions must be non-negative",
                min_instructions=self.min_instructions,
            )

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError(
                "similarity_threshold must be between 0.0 and 1.0",
                similarity_threshold=self.similarity_threshold,
            )

        # Validate grounded extraction settings
        if self.extraction_concurrency_per_chapter < 1:
            raise ConfigurationError(
                "extraction_concurrency_per_chapter must be at least 1",
                extraction_concurrency_per_chapter=self.extraction_concurrency_per_chapter,
            )

        if self.extraction_retry_attempts < 0:
            raise ConfigurationError(
                "extraction_retry_attempts must be non-negative",
                extraction_retry_attempts=self.extraction_retry_attempts,
            )

        if self.extraction_retry_delay <= 0:
            raise ConfigurationError(
                "extraction_retry_delay must be positive",
                extraction_retry_delay=self.extraction_retry_delay,
            )

        if not 0.0 < self.title_match_threshold <= 1.0:
            raise ConfigurationError(
                "title_match_threshold must be between 0.0 (exclusive) and 1.0",
                title_match_threshold=self.title_match_threshold,
            )

        if self.max_pagination_pages < 1:
            raise ConfigurationError(
                "max_pagination_pages must be at least 1",
                max_pagination_pages=self.max_pagination_pages,
            )

        # Ensure output_dir is a Path (may receive str from config/env)
        if not isinstance(self.output_dir, Path):  # type: ignore[reportUnnecessaryIsInstance]
            self.output_dir = Path(self.output_dir)

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        load_user_config: bool = True,
        load_env: bool = True,
    ) -> "ExtractionConfig":
        """Load configuration from file(s) and environment variables.

        Configuration is loaded in this order (later overrides earlier):
        1. Default values
        2. User config file (~/.config/mela-parser/config.toml)
        3. Project config file (.mela-parser.toml or specified path)
        4. Environment variables (MELA_PARSER_*)

        Args:
            config_path: Path to project config file (optional)
            load_user_config: Whether to load user config file
            load_env: Whether to load environment variables

        Returns:
            Loaded and validated configuration

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> # Load with defaults
            >>> config = ExtractionConfig.load()
            >>>
            >>> # Load from specific file
            >>> config = ExtractionConfig.load("myproject.toml")
            >>>
            >>> # Load only from environment
            >>> config = ExtractionConfig.load(
            ...     load_user_config=False,
            ...     load_env=True
            ... )
        """
        # Start with defaults
        config_dict: dict[str, Any] = {}

        # Load user config
        if load_user_config:
            user_config_path = Path.home() / ".config" / "mela-parser" / "config.toml"
            if user_config_path.exists():
                config_dict.update(cls._load_toml(user_config_path))

        # Load project config
        if config_path:
            project_path = Path(config_path)
            if project_path.exists():
                config_dict.update(cls._load_toml(project_path))
        else:
            # Try default project config
            default_path = Path(".mela-parser.toml")
            if default_path.exists():
                config_dict.update(cls._load_toml(default_path))

        # Load environment variables
        if load_env:
            config_dict.update(cls._load_env())

        # Create config instance
        return cls(**config_dict)

    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        """Load configuration from TOML file.

        Args:
            path: Path to TOML file

        Returns:
            Dictionary of configuration values

        Raises:
            ConfigurationError: If TOML file is invalid
        """
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            # Extract mela-parser section if present
            if "mela-parser" in data:
                return data["mela-parser"]
            return data

        except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
            raise ConfigurationError(
                f"Failed to load configuration from {path}",
                path=str(path),
                error=str(e),
            ) from e

    @staticmethod
    def _load_env() -> dict[str, Any]:
        """Load configuration from environment variables.

        Environment variables should be prefixed with MELA_PARSER_ and use
        uppercase snake_case. For example:
        - MELA_PARSER_MODEL=gpt-5-mini
        - MELA_PARSER_MAX_CONCURRENT=100
        - MELA_PARSER_DEBUG_MODE=true

        Returns:
            Dictionary of configuration values from environment
        """
        config: dict[str, Any] = {}
        prefix = "MELA_PARSER_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(prefix) :].lower()

            # Type conversion
            if value.lower() in ("true", "1", "yes"):
                config[config_key] = True
            elif value.lower() in ("false", "0", "no"):
                config[config_key] = False
            elif value.isdigit():
                config[config_key] = int(value)
            elif value.replace(".", "", 1).isdigit():  # Float
                config[config_key] = float(value)
            else:
                config[config_key] = value

        return config

    def save(self, path: str | Path) -> None:
        """Save configuration to TOML file.

        Args:
            path: Path to save configuration file

        Raises:
            ConfigurationError: If save fails

        Example:
            >>> config = ExtractionConfig()
            >>> config.model = "gpt-5-mini"
            >>> config.save("~/.config/mela-parser/config.toml")
        """
        try:
            import tomli_w  # For writing TOML

            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict, handling Path objects
            config_dict: dict[str, Any] = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value

            with open(path, "wb") as f:
                tomli_w.dump(config_dict, f)

        except ImportError:
            raise ConfigurationError(
                "tomli_w package required to save configuration. Install with: pip install tomli-w"
            ) from None
        except (OSError, TypeError, ValueError) as e:
            raise ConfigurationError(
                f"Failed to save configuration to {path}",
                path=str(path),
                error=str(e),
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = ExtractionConfig()
            >>> config_dict = config.to_dict()
            >>> print(config_dict["model"])
            gpt-5-nano
        """
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def update(self, **kwargs: Any) -> None:
        """Update configuration values.

        Args:
            **kwargs: Configuration values to update

        Raises:
            ConfigurationError: If updated values are invalid

        Example:
            >>> config = ExtractionConfig()
            >>> config.update(model="gpt-5-mini", max_concurrent=100)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ConfigurationError(
                    f"Unknown configuration key: {key}",
                    key=key,
                    valid_keys=", ".join(self.__dataclass_fields__.keys()),
                )

        # Revalidate after update
        self._validate()
