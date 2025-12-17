"""Unit tests for mela_parser.protocols module.

Tests Protocol definitions and runtime checkability.
"""

from pathlib import Path
from typing import Any

from mela_parser.protocols import (
    ContentConverter,
    ImageOptimizer,
    ImageSelector,
    RecipeExtractor,
    RecipeRepository,
    RecipeValidator,
)


class TestContentConverterProtocol:
    """Tests for ContentConverter protocol."""

    def test_is_runtime_checkable(self) -> None:
        """ContentConverter can be used with isinstance."""

        class MockConverter:
            def convert(self, epub_path: Path) -> tuple[Any, list[Any]]:
                return (None, [])

        converter = MockConverter()
        assert isinstance(converter, ContentConverter)

    def test_missing_method_fails_check(self) -> None:
        """Class without convert method fails isinstance check."""

        class BadConverter:
            pass

        converter = BadConverter()
        assert not isinstance(converter, ContentConverter)


class TestRecipeExtractorProtocol:
    """Tests for RecipeExtractor protocol."""

    def test_is_runtime_checkable(self) -> None:
        """RecipeExtractor can be used with isinstance."""

        class MockExtractor:
            async def extract(self, chapters: list[Any]) -> list[Any]:
                return []

        extractor = MockExtractor()
        assert isinstance(extractor, RecipeExtractor)

    def test_missing_method_fails_check(self) -> None:
        """Class without extract method fails isinstance check."""

        class BadExtractor:
            def wrong_method(self) -> None:
                pass

        extractor = BadExtractor()
        assert not isinstance(extractor, RecipeExtractor)


class TestRecipeValidatorProtocol:
    """Tests for RecipeValidator protocol."""

    def test_is_runtime_checkable(self) -> None:
        """RecipeValidator can be used with isinstance."""

        class MockValidator:
            def validate(self, recipe: Any) -> Any:
                return None

        validator = MockValidator()
        assert isinstance(validator, RecipeValidator)

    def test_missing_method_fails_check(self) -> None:
        """Class without validate method fails isinstance check."""

        class BadValidator:
            pass

        validator = BadValidator()
        assert not isinstance(validator, RecipeValidator)


class TestRecipeRepositoryProtocol:
    """Tests for RecipeRepository protocol."""

    def test_is_runtime_checkable(self) -> None:
        """RecipeRepository can be used with isinstance."""

        class MockRepository:
            def save(self, recipe: Any, output_dir: Path) -> Path:
                return output_dir / "test.melarecipe"

            def deduplicate(self, recipes: list[Any]) -> list[Any]:
                return recipes

        repo = MockRepository()
        assert isinstance(repo, RecipeRepository)

    def test_missing_save_fails_check(self) -> None:
        """Class without save method fails isinstance check."""

        class BadRepository:
            def deduplicate(self, recipes: list[Any]) -> list[Any]:
                return recipes

        repo = BadRepository()
        assert not isinstance(repo, RecipeRepository)

    def test_missing_deduplicate_fails_check(self) -> None:
        """Class without deduplicate method fails isinstance check."""

        class BadRepository:
            def save(self, recipe: Any, output_dir: Path) -> Path:
                return output_dir / "test.melarecipe"

        repo = BadRepository()
        assert not isinstance(repo, RecipeRepository)


class TestImageSelectorProtocol:
    """Tests for ImageSelector protocol."""

    def test_is_runtime_checkable(self) -> None:
        """ImageSelector can be used with isinstance."""

        class MockSelector:
            def select(self, recipe: Any, book: Any) -> str | None:
                return None

        selector = MockSelector()
        assert isinstance(selector, ImageSelector)


class TestImageOptimizerProtocol:
    """Tests for ImageOptimizer protocol."""

    def test_is_runtime_checkable(self) -> None:
        """ImageOptimizer can be used with isinstance."""

        class MockOptimizer:
            def optimize(
                self,
                image_data: bytes,
                max_width: int = 600,
                quality: int = 85,
            ) -> bytes:
                return image_data

        optimizer = MockOptimizer()
        assert isinstance(optimizer, ImageOptimizer)


class TestProtocolComposition:
    """Tests for composing multiple protocols."""

    def test_class_can_satisfy_multiple_protocols(self) -> None:
        """A class can satisfy multiple protocols."""

        class ComboService:
            def validate(self, recipe: Any) -> Any:
                return None

            def save(self, recipe: Any, output_dir: Path) -> Path:
                return output_dir / "combo.melarecipe"

            def deduplicate(self, recipes: list[Any]) -> list[Any]:
                return recipes

        service = ComboService()
        assert isinstance(service, RecipeValidator)
        assert isinstance(service, RecipeRepository)
