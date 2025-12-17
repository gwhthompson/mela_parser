"""Unit tests for mela_parser.config module.

Tests ExtractionConfig validation, loading, and serialization.
"""

from pathlib import Path

import pytest

from mela_parser.config import ExtractionConfig
from mela_parser.exceptions import ConfigurationError


class TestExtractionConfigDefaults:
    """Tests for ExtractionConfig default values."""

    def test_default_model(self) -> None:
        """Default model is gpt-5-nano."""
        config = ExtractionConfig()
        assert config.model == "gpt-5-nano"

    def test_default_temperature(self) -> None:
        """Default temperature is 0.0 (deterministic)."""
        config = ExtractionConfig()
        assert config.temperature == 0.0

    def test_default_concurrency(self) -> None:
        """Default max_concurrent is 200."""
        config = ExtractionConfig()
        assert config.max_concurrent == 200

    def test_default_output_dir_is_path(self) -> None:
        """Default output_dir is a Path object."""
        config = ExtractionConfig()
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("output")


class TestExtractionConfigValidation:
    """Tests for ExtractionConfig validation logic."""

    def test_valid_models(self) -> None:
        """All valid models are accepted."""
        valid_models = ["gpt-5-nano", "gpt-5-mini", "gpt-4o", "gpt-4o-mini"]
        for model in valid_models:
            config = ExtractionConfig(model=model)
            assert config.model == model

    def test_invalid_model_raises(self) -> None:
        """Invalid model raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Invalid model"):
            ExtractionConfig(model="invalid-model")

    def test_temperature_bounds(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        # Valid temperatures
        ExtractionConfig(temperature=0.0)
        ExtractionConfig(temperature=1.0)
        ExtractionConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ConfigurationError, match="Temperature"):
            ExtractionConfig(temperature=-0.1)

        with pytest.raises(ConfigurationError, match="Temperature"):
            ExtractionConfig(temperature=2.1)

    def test_max_concurrent_minimum(self) -> None:
        """max_concurrent must be at least 1."""
        ExtractionConfig(max_concurrent=1)

        with pytest.raises(ConfigurationError, match="max_concurrent"):
            ExtractionConfig(max_concurrent=0)

        with pytest.raises(ConfigurationError, match="max_concurrent"):
            ExtractionConfig(max_concurrent=-1)

    def test_retry_attempts_non_negative(self) -> None:
        """retry_attempts must be non-negative."""
        ExtractionConfig(retry_attempts=0)
        ExtractionConfig(retry_attempts=5)

        with pytest.raises(ConfigurationError, match="retry_attempts"):
            ExtractionConfig(retry_attempts=-1)

    def test_initial_retry_delay_positive(self) -> None:
        """initial_retry_delay must be positive."""
        ExtractionConfig(initial_retry_delay=0.1)
        ExtractionConfig(initial_retry_delay=10.0)

        with pytest.raises(ConfigurationError, match="initial_retry_delay"):
            ExtractionConfig(initial_retry_delay=0)

        with pytest.raises(ConfigurationError, match="initial_retry_delay"):
            ExtractionConfig(initial_retry_delay=-1.0)

    def test_min_image_area_non_negative(self) -> None:
        """min_image_area must be non-negative."""
        ExtractionConfig(min_image_area=0)
        ExtractionConfig(min_image_area=1000000)

        with pytest.raises(ConfigurationError, match="min_image_area"):
            ExtractionConfig(min_image_area=-1)

    def test_max_image_width_minimum(self) -> None:
        """max_image_width must be at least 1."""
        ExtractionConfig(max_image_width=1)
        ExtractionConfig(max_image_width=1920)

        with pytest.raises(ConfigurationError, match="max_image_width"):
            ExtractionConfig(max_image_width=0)

    def test_similarity_threshold_bounds(self) -> None:
        """similarity_threshold must be between 0.0 and 1.0."""
        ExtractionConfig(similarity_threshold=0.0)
        ExtractionConfig(similarity_threshold=0.5)
        ExtractionConfig(similarity_threshold=1.0)

        with pytest.raises(ConfigurationError, match="similarity_threshold"):
            ExtractionConfig(similarity_threshold=-0.1)

        with pytest.raises(ConfigurationError, match="similarity_threshold"):
            ExtractionConfig(similarity_threshold=1.1)

    def test_title_match_threshold_bounds(self) -> None:
        """title_match_threshold must be between 0.0 (exclusive) and 1.0."""
        ExtractionConfig(title_match_threshold=0.1)
        ExtractionConfig(title_match_threshold=1.0)

        with pytest.raises(ConfigurationError, match="title_match_threshold"):
            ExtractionConfig(title_match_threshold=0.0)

        with pytest.raises(ConfigurationError, match="title_match_threshold"):
            ExtractionConfig(title_match_threshold=1.1)


class TestExtractionConfigUpdate:
    """Tests for ExtractionConfig.update method."""

    def test_update_single_value(self) -> None:
        """update() modifies single value."""
        config = ExtractionConfig()
        config.update(model="gpt-5-mini")
        assert config.model == "gpt-5-mini"

    def test_update_multiple_values(self) -> None:
        """update() modifies multiple values."""
        config = ExtractionConfig()
        config.update(model="gpt-4o", temperature=0.5, max_concurrent=100)

        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_concurrent == 100

    def test_update_validates(self) -> None:
        """update() validates new values."""
        config = ExtractionConfig()

        with pytest.raises(ConfigurationError, match="Invalid model"):
            config.update(model="invalid")

    def test_update_unknown_key_raises(self) -> None:
        """update() raises on unknown configuration key."""
        config = ExtractionConfig()

        with pytest.raises(ConfigurationError, match="Unknown configuration key"):
            config.update(unknown_key="value")


class TestExtractionConfigToDict:
    """Tests for ExtractionConfig.to_dict method."""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict returns a dictionary."""
        config = ExtractionConfig()
        result = config.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_converts_path(self) -> None:
        """to_dict converts Path to string."""
        config = ExtractionConfig(output_dir=Path("/some/path"))
        result = config.to_dict()
        assert result["output_dir"] == "/some/path"
        assert isinstance(result["output_dir"], str)

    def test_to_dict_includes_all_fields(self) -> None:
        """to_dict includes all configuration fields."""
        config = ExtractionConfig()
        result = config.to_dict()

        # Check key fields are present
        assert "model" in result
        assert "temperature" in result
        assert "max_concurrent" in result
        assert "output_dir" in result


class TestExtractionConfigEnvLoading:
    """Tests for ExtractionConfig environment variable loading."""

    def test_load_model_from_env(self, clean_env, mock_env) -> None:
        """Model can be loaded from environment variable."""
        mock_env["MODEL"] = "gpt-5-mini"
        config = ExtractionConfig.load(load_user_config=False)
        assert config.model == "gpt-5-mini"

    def test_load_integer_from_env(self, clean_env, mock_env) -> None:
        """Integer values are parsed from env."""
        mock_env["MAX_CONCURRENT"] = "50"
        config = ExtractionConfig.load(load_user_config=False)
        assert config.max_concurrent == 50

    def test_load_float_from_env(self, clean_env, mock_env) -> None:
        """Float values are parsed from env."""
        mock_env["TEMPERATURE"] = "0.7"
        config = ExtractionConfig.load(load_user_config=False)
        assert config.temperature == 0.7

    def test_load_bool_true_from_env(self, clean_env, mock_env) -> None:
        """Boolean true values are parsed from env."""
        for value in ["true", "1", "yes"]:
            mock_env["DEBUG_MODE"] = value
            config = ExtractionConfig.load(load_user_config=False)
            assert config.debug_mode is True

    def test_load_bool_false_from_env(self, clean_env, mock_env) -> None:
        """Boolean false values are parsed from env."""
        for value in ["false", "0", "no"]:
            mock_env["EXTRACT_IMAGES"] = value
            config = ExtractionConfig.load(load_user_config=False)
            assert config.extract_images is False

    def test_load_without_env(self, clean_env) -> None:
        """Config loads defaults when env loading disabled."""
        config = ExtractionConfig.load(load_env=False, load_user_config=False)
        assert config.model == "gpt-5-nano"  # Default


class TestExtractionConfigTomlLoading:
    """Tests for ExtractionConfig TOML file loading."""

    def test_load_from_toml_file(self, tmp_path: Path, clean_env) -> None:
        """Config can be loaded from TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
            model = "gpt-5-mini"
            temperature = 0.5
            max_concurrent = 100
            """
        )

        config = ExtractionConfig.load(
            config_path=config_file,
            load_user_config=False,
            load_env=False,
        )

        assert config.model == "gpt-5-mini"
        assert config.temperature == 0.5
        assert config.max_concurrent == 100

    def test_load_from_mela_parser_section(self, tmp_path: Path, clean_env) -> None:
        """Config loads from [mela-parser] section if present."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
            [other-section]
            foo = "bar"

            [mela-parser]
            model = "gpt-4o"
            """
        )

        config = ExtractionConfig.load(
            config_path=config_file,
            load_user_config=False,
            load_env=False,
        )

        assert config.model == "gpt-4o"

    def test_invalid_toml_raises(self, tmp_path: Path, clean_env) -> None:
        """Invalid TOML file raises ConfigurationError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("invalid [ toml content")

        with pytest.raises(ConfigurationError, match="Failed to load configuration"):
            ExtractionConfig.load(
                config_path=config_file,
                load_user_config=False,
                load_env=False,
            )


class TestExtractionConfigSave:
    """Tests for ExtractionConfig.save method."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save() creates TOML file."""
        config = ExtractionConfig(model="gpt-5-mini")
        save_path = tmp_path / "saved_config.toml"

        config.save(save_path)

        assert save_path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() creates parent directories if needed."""
        config = ExtractionConfig()
        save_path = tmp_path / "nested" / "dir" / "config.toml"

        config.save(save_path)

        assert save_path.exists()

    def test_saved_config_can_be_loaded(self, tmp_path: Path, clean_env) -> None:
        """Saved config can be loaded back."""
        original = ExtractionConfig(model="gpt-4o", temperature=0.8)
        save_path = tmp_path / "config.toml"

        original.save(save_path)

        loaded = ExtractionConfig.load(
            config_path=save_path,
            load_user_config=False,
            load_env=False,
        )

        assert loaded.model == original.model
        assert loaded.temperature == original.temperature
