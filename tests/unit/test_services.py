"""Unit tests for mela_parser.services package.

Tests ServiceFactory, dependency injection, and service creation.
"""

from unittest.mock import MagicMock, patch

from mela_parser.config import ExtractionConfig
from mela_parser.services import ServiceFactory


class TestServiceFactoryInit:
    """Tests for ServiceFactory initialization."""

    def test_init_with_config(self) -> None:
        """ServiceFactory stores config."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)
        assert factory.config is config

    def test_init_with_custom_config(self) -> None:
        """ServiceFactory accepts custom config values."""
        config = ExtractionConfig(model="gpt-4o", max_concurrent=50)
        factory = ServiceFactory(config=config)
        assert factory.config.model == "gpt-4o"
        assert factory.config.max_concurrent == 50


@patch("mela_parser.services.factory.AsyncOpenAI")
class TestServiceFactoryClient:
    """Tests for ServiceFactory.client property."""

    def test_client_is_created(self, mock_openai: MagicMock) -> None:
        """Client property creates AsyncOpenAI instance."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)
        client = factory.client
        # Verify mock was called
        mock_openai.assert_called_once()
        assert client is mock_openai.return_value

    def test_client_is_cached(self, mock_openai: MagicMock) -> None:
        """Client property returns same instance on repeated calls."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)

        client1 = factory.client
        client2 = factory.client

        assert client1 is client2


@patch("mela_parser.services.factory.AsyncOpenAI")
class TestServiceFactoryCreateExtractor:
    """Tests for ServiceFactory.create_extractor method."""

    def test_creates_extractor(self, mock_openai: MagicMock) -> None:
        """create_extractor returns AsyncChapterExtractor."""
        config = ExtractionConfig(model="gpt-5-nano")
        factory = ServiceFactory(config=config)

        extractor = factory.create_extractor()

        # Import here to verify type
        from mela_parser.chapter_extractor import AsyncChapterExtractor

        assert isinstance(extractor, AsyncChapterExtractor)

    def test_extractor_uses_config_model(self, mock_openai: MagicMock) -> None:
        """Extractor is configured with factory's model."""
        config = ExtractionConfig(model="gpt-4o")
        factory = ServiceFactory(config=config)

        extractor = factory.create_extractor()

        assert extractor.model == "gpt-4o"

    def test_extractor_uses_config_retry_settings(self, mock_openai: MagicMock) -> None:
        """Extractor is configured with factory's retry settings."""
        config = ExtractionConfig(retry_attempts=5, initial_retry_delay=2.0)
        factory = ServiceFactory(config=config)

        extractor = factory.create_extractor()

        assert extractor.max_retries == 5
        assert extractor.initial_retry_delay == 2.0

    def test_extractor_shares_client(self, mock_openai: MagicMock) -> None:
        """Extractor uses factory's shared client."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)

        extractor = factory.create_extractor()

        assert extractor.client is factory.client


class TestServiceFactoryCreateValidator:
    """Tests for ServiceFactory.create_validator method."""

    def test_creates_validator(self) -> None:
        """create_validator returns RecipeValidator."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)

        validator = factory.create_validator()

        from mela_parser.validator import RecipeValidator

        assert isinstance(validator, RecipeValidator)

    def test_validator_uses_config_settings(self) -> None:
        """Validator is configured with factory's thresholds."""
        config = ExtractionConfig(min_ingredients=3, min_instructions=4)
        factory = ServiceFactory(config=config)

        validator = factory.create_validator()

        assert validator.min_ingredients == 3
        assert validator.min_instructions == 4


class TestServiceFactoryCreateRepository:
    """Tests for ServiceFactory.create_repository method."""

    def test_creates_repository(self) -> None:
        """create_repository returns FileRecipeRepository."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)

        repository = factory.create_repository()

        from mela_parser.repository import FileRecipeRepository

        assert isinstance(repository, FileRecipeRepository)

    def test_repository_has_validator(self) -> None:
        """Repository is created with a validator."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)

        repository = factory.create_repository()

        # The repository should have the validator
        assert repository.validator is not None


class TestServiceFactoryCreateImageService:
    """Tests for ServiceFactory.create_image_service method."""

    def test_creates_image_service(self) -> None:
        """create_image_service returns ImageService."""
        config = ExtractionConfig()
        factory = ServiceFactory(config=config)

        # Create a mock EpubBook
        mock_book = MagicMock()

        image_service = factory.create_image_service(mock_book)

        from mela_parser.services.images import ImageService

        assert isinstance(image_service, ImageService)

    def test_image_service_uses_config(self) -> None:
        """ImageService is configured with factory's image settings."""
        config = ExtractionConfig(min_image_area=50000, max_image_width=800)
        factory = ServiceFactory(config=config)
        mock_book = MagicMock()

        image_service = factory.create_image_service(mock_book)

        assert image_service.config.min_area == 50000
        assert image_service.config.max_width == 800

    @patch("mela_parser.services.factory.AsyncOpenAI")
    def test_image_service_with_ai_verification(self, mock_openai: MagicMock) -> None:
        """ImageService gets client when AI verification enabled."""
        config = ExtractionConfig(use_ai_verification=True)
        factory = ServiceFactory(config=config)
        mock_book = MagicMock()

        image_service = factory.create_image_service(mock_book)

        assert image_service.client is factory.client

    def test_image_service_without_ai_verification(self) -> None:
        """ImageService gets no client when AI verification disabled."""
        config = ExtractionConfig(use_ai_verification=False)
        factory = ServiceFactory(config=config)
        mock_book = MagicMock()

        image_service = factory.create_image_service(mock_book)

        assert image_service.client is None


class TestServicesPackageExports:
    """Tests for services package exports."""

    def test_exports_service_factory(self) -> None:
        """ServiceFactory is exported from services package."""
        from mela_parser.services import ServiceFactory as ServicesFactory

        assert ServicesFactory is ServiceFactory

    def test_exports_image_service(self) -> None:
        """ImageService is exported from services package."""
        from mela_parser.services import ImageService

        assert ImageService is not None

    def test_exports_image_config(self) -> None:
        """ImageConfig is exported from services package."""
        from mela_parser.services import ImageConfig

        assert ImageConfig is not None

    def test_exports_selection_strategy(self) -> None:
        """SelectionStrategy is exported from services package."""
        from mela_parser.services import SelectionStrategy

        assert SelectionStrategy is not None
