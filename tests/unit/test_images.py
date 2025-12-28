"""Unit tests for mela_parser.services.images module.

Tests ImageConfig, ImageCandidate, SelectionStrategy, and ImageService.
"""

from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

from mela_parser.parse import IngredientGroup, MelaRecipe
from mela_parser.services.images import (
    ImageCandidate,
    ImageConfig,
    ImageService,
    SelectionStrategy,
)


class TestSelectionStrategyEnum:
    """Tests for SelectionStrategy enum."""

    def test_largest_value(self) -> None:
        """LARGEST strategy has correct value."""
        assert SelectionStrategy.LARGEST.value == "largest"

    def test_ai_verified_value(self) -> None:
        """AI_VERIFIED strategy has correct value."""
        assert SelectionStrategy.AI_VERIFIED.value == "ai_verified"

    def test_all_strategies_exist(self) -> None:
        """All expected strategies are defined."""
        assert hasattr(SelectionStrategy, "LARGEST")
        assert hasattr(SelectionStrategy, "AI_VERIFIED")
        assert len(SelectionStrategy) == 2


class TestImageConfig:
    """Tests for ImageConfig dataclass."""

    def test_default_values(self) -> None:
        """ImageConfig has sensible defaults."""
        config = ImageConfig()
        assert config.min_area == 300_000  # ~550x550
        assert config.max_width == 600
        assert config.quality == 85
        assert config.strategy == SelectionStrategy.LARGEST
        assert config.ai_threshold == 0.6

    def test_custom_values(self) -> None:
        """ImageConfig accepts custom values."""
        config = ImageConfig(
            min_area=500_000,
            max_width=800,
            quality=90,
            strategy=SelectionStrategy.AI_VERIFIED,
            ai_threshold=0.8,
        )
        assert config.min_area == 500_000
        assert config.max_width == 800
        assert config.quality == 90
        assert config.strategy == SelectionStrategy.AI_VERIFIED
        assert config.ai_threshold == 0.8

    def test_partial_override(self) -> None:
        """ImageConfig allows partial customization."""
        config = ImageConfig(max_width=1024)
        # Overridden value
        assert config.max_width == 1024
        # Default values preserved
        assert config.min_area == 300_000
        assert config.quality == 85


class TestImageCandidate:
    """Tests for ImageCandidate dataclass."""

    def test_create_candidate(self) -> None:
        """ImageCandidate can be created with required fields."""
        candidate = ImageCandidate(
            path="images/recipe.jpg",
            data=b"fake image data",
            width=800,
            height=600,
            area=480_000,
        )
        assert candidate.path == "images/recipe.jpg"
        assert candidate.data == b"fake image data"
        assert candidate.width == 800
        assert candidate.height == 600
        assert candidate.area == 480_000

    def test_default_ai_fields(self) -> None:
        """ImageCandidate has sensible AI defaults."""
        candidate = ImageCandidate(
            path="test.jpg",
            data=b"data",
            width=100,
            height=100,
            area=10_000,
        )
        assert candidate.ai_confidence == 0.0
        assert candidate.ai_matches is False

    def test_with_ai_fields(self) -> None:
        """ImageCandidate can store AI verification results."""
        candidate = ImageCandidate(
            path="test.jpg",
            data=b"data",
            width=100,
            height=100,
            area=10_000,
            ai_confidence=0.95,
            ai_matches=True,
        )
        assert candidate.ai_confidence == 0.95
        assert candidate.ai_matches is True


class TestImageServiceInit:
    """Tests for ImageService initialization."""

    def test_init_with_book_and_config(self) -> None:
        """ImageService stores book and config."""
        mock_book = MagicMock()
        config = ImageConfig()

        service = ImageService(book=mock_book, config=config)

        assert service.book is mock_book
        assert service.config is config
        assert service.client is None

    def test_init_with_client(self) -> None:
        """ImageService can receive optional client."""
        mock_book = MagicMock()
        mock_client = MagicMock()
        config = ImageConfig()

        service = ImageService(book=mock_book, config=config, client=mock_client)

        assert service.client is mock_client


class TestImageServiceExtractPaths:
    """Tests for ImageService._extract_image_paths method."""

    @pytest.fixture
    def service(self) -> ImageService:
        """Create ImageService with mock book."""
        return ImageService(book=MagicMock(), config=ImageConfig())

    def test_extracts_markdown_image_paths(self, service: ImageService) -> None:
        """Extracts image paths from markdown syntax."""
        recipe = MelaRecipe(
            title="Test Recipe",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup flour"])],
            instructions=[
                "Step 1 with image ![Recipe photo](../images/recipe.jpg)",
                "Step 2 continues.",
            ],
        )

        paths = service._extract_image_paths(recipe)

        assert "../images/recipe.jpg" in paths

    def test_extracts_png_images(self, service: ImageService) -> None:
        """Extracts PNG image paths."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["![alt](../images/photo.PNG)", "Done."],
        )

        paths = service._extract_image_paths(recipe)

        assert "../images/photo.PNG" in paths

    def test_extracts_gif_images(self, service: ImageService) -> None:
        """Extracts GIF image paths."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["![](path/to/image.gif)", "Done."],
        )

        paths = service._extract_image_paths(recipe)

        assert "path/to/image.gif" in paths

    def test_extracts_jpeg_variant(self, service: ImageService) -> None:
        """Extracts JPEG image paths (both .jpg and .jpeg)."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=[
                "![](photo.jpeg) ![](other.jpg)",
                "Done.",
            ],
        )

        paths = service._extract_image_paths(recipe)

        assert "photo.jpeg" in paths
        assert "other.jpg" in paths

    def test_extracts_src_attribute_paths(self, service: ImageService) -> None:
        """Extracts image paths from src attributes."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=['<img src="images/test.jpg">', "Done."],
        )

        paths = service._extract_image_paths(recipe)

        assert "images/test.jpg" in paths

    def test_deduplicates_paths(self, service: ImageService) -> None:
        """Deduplicates image paths preserving order."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=[
                "![](image.jpg) ![](other.jpg) ![](IMAGE.jpg)",
                "Done.",
            ],
        )

        paths = service._extract_image_paths(recipe)

        # image.jpg and IMAGE.jpg should be deduplicated (case-insensitive)
        assert len([p for p in paths if p.lower() == "image.jpg"]) == 1

    def test_returns_empty_for_no_images(self, service: ImageService) -> None:
        """Returns empty list when no images found."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1 with no images.", "Step 2."],
        )

        paths = service._extract_image_paths(recipe)

        assert paths == []

    def test_handles_complex_alt_text(self, service: ImageService) -> None:
        """Handles images with complex alt text."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=[
                "![A delicious chocolate cake with frosting](../images/cake.jpg)",
                "Done.",
            ],
        )

        paths = service._extract_image_paths(recipe)

        assert "../images/cake.jpg" in paths

    def test_case_insensitive_extension(self, service: ImageService) -> None:
        """Handles image extensions case-insensitively."""
        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=[
                "![](test.JPG) ![](other.Png) ![](third.GIF)",
                "Done.",
            ],
        )

        paths = service._extract_image_paths(recipe)

        assert len(paths) == 3


class TestImageServiceLoadCandidates:
    """Tests for ImageService._load_candidates method."""

    def test_loads_valid_images(self) -> None:
        """Loads valid images from EPUB book."""
        from io import BytesIO

        from PIL import Image

        # Create a valid test image
        img = Image.new("RGB", (800, 600), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_data = img_bytes.getvalue()

        # Mock EPUB item
        mock_item = MagicMock()
        mock_item.get_content.return_value = img_data

        # Mock book
        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = mock_item

        config = ImageConfig(min_area=100_000)  # 800*600=480000 > 100000
        service = ImageService(book=mock_book, config=config)

        candidates = service._load_candidates(["images/test.jpg"])

        assert len(candidates) == 1
        assert candidates[0].path == "images/test.jpg"
        assert candidates[0].width == 800
        assert candidates[0].height == 600
        assert candidates[0].area == 480_000

    def test_filters_small_images(self) -> None:
        """Filters out images below minimum area."""
        from io import BytesIO

        from PIL import Image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_data = img_bytes.getvalue()

        mock_item = MagicMock()
        mock_item.get_content.return_value = img_data

        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = mock_item

        # min_area=300000, image area=10000
        config = ImageConfig(min_area=300_000)
        service = ImageService(book=mock_book, config=config)

        candidates = service._load_candidates(["images/small.jpg"])

        assert len(candidates) == 0

    def test_handles_missing_images(self) -> None:
        """Handles images not found in EPUB."""
        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = None

        service = ImageService(book=mock_book, config=ImageConfig())

        candidates = service._load_candidates(["missing.jpg"])

        assert len(candidates) == 0

    def test_normalizes_relative_paths(self) -> None:
        """Normalizes ../ prefix in paths."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGB", (800, 600))
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")

        mock_item = MagicMock()
        mock_item.get_content.return_value = img_bytes.getvalue()

        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = mock_item

        config = ImageConfig(min_area=100_000)
        service = ImageService(book=mock_book, config=config)

        # Path starts with ../
        service._load_candidates(["../images/test.jpg"])

        # Should call with normalized path
        mock_book.get_item_with_href.assert_called_with("images/test.jpg")

    def test_handles_invalid_image_data(self) -> None:
        """Handles invalid/corrupt image data gracefully."""
        mock_item = MagicMock()
        mock_item.get_content.return_value = b"not valid image data"

        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = mock_item

        service = ImageService(book=mock_book, config=ImageConfig())

        candidates = service._load_candidates(["corrupt.jpg"])

        assert len(candidates) == 0


class TestImageServiceSelectBySize:
    """Tests for ImageService._select_by_size method."""

    def test_selects_largest_image(self) -> None:
        """Selects image with largest area."""
        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig())

        candidates = [
            ImageCandidate(path="small.jpg", data=b"small", width=100, height=100, area=10_000),
            ImageCandidate(
                path="large.jpg", data=b"large", width=1000, height=1000, area=1_000_000
            ),
            ImageCandidate(path="medium.jpg", data=b"medium", width=500, height=500, area=250_000),
        ]

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        result = service._select_by_size(candidates, recipe)

        # Should return base64-encoded large image
        import base64

        assert result == base64.b64encode(b"large").decode("utf-8")

    def test_returns_base64_encoded(self) -> None:
        """Returns base64-encoded image data."""
        import base64

        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig())

        test_data = b"test image data"
        candidates = [
            ImageCandidate(path="test.jpg", data=test_data, width=800, height=600, area=480_000),
        ]

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        result = service._select_by_size(candidates, recipe)

        assert result == base64.b64encode(test_data).decode("utf-8")


class TestImageServiceSelectBestImage:
    """Tests for ImageService.select_best_image method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_images(self) -> None:
        """Returns None when recipe has no images."""
        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig())

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["No images here", "Just text"],
        )

        result = await service.select_best_image(recipe)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_valid_candidates(self) -> None:
        """Returns None when no valid image candidates found."""
        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = None  # Image not found

        service = ImageService(book=mock_book, config=ImageConfig())

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["![](missing.jpg)", "Step 2"],
        )

        result = await service.select_best_image(recipe)

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_size_strategy_by_default(self) -> None:
        """Uses LARGEST strategy by default."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGB", (800, 600))
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")

        mock_item = MagicMock()
        mock_item.get_content.return_value = img_bytes.getvalue()

        mock_book = MagicMock()
        mock_book.get_item_with_href.return_value = mock_item

        config = ImageConfig(min_area=100_000, strategy=SelectionStrategy.LARGEST)
        service = ImageService(book=mock_book, config=config)

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["![](images/test.jpg)", "Step 2"],
        )

        result = await service.select_best_image(recipe)

        assert result is not None
        # Should be base64 encoded
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0


class TestImageServiceOptimize:
    """Tests for ImageService.optimize method."""

    def test_resizes_wide_images(self) -> None:
        """Resizes images wider than max_width."""
        from io import BytesIO

        from PIL import Image

        # Create a wide image
        img = Image.new("RGB", (1200, 800), color="green")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        original_data = img_bytes.getvalue()

        mock_book = MagicMock()
        config = ImageConfig(max_width=600, quality=85)
        service = ImageService(book=mock_book, config=config)

        result = service.optimize(original_data)

        # Result should be smaller
        assert len(result) < len(original_data)

        # Check dimensions
        with Image.open(BytesIO(result)) as optimized:
            assert optimized.width == 600
            # Height should be proportionally reduced
            assert optimized.height == 400  # 800 * (600/1200)

    def test_preserves_small_images(self) -> None:
        """Does not resize images smaller than max_width."""
        from io import BytesIO

        from PIL import Image

        # Create a small image
        img = Image.new("RGB", (400, 300), color="blue")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG", quality=95)
        original_data = img_bytes.getvalue()

        mock_book = MagicMock()
        config = ImageConfig(max_width=600, quality=85)
        service = ImageService(book=mock_book, config=config)

        result = service.optimize(original_data)

        # Check dimensions unchanged
        with Image.open(BytesIO(result)) as optimized:
            assert optimized.width == 400
            assert optimized.height == 300

    def test_converts_rgba_to_rgb(self) -> None:
        """Converts RGBA images to RGB."""
        from io import BytesIO

        from PIL import Image

        # Create RGBA image
        img = Image.new("RGBA", (400, 300), color=(255, 0, 0, 128))
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        rgba_data = img_bytes.getvalue()

        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig())

        result = service.optimize(rgba_data)

        # Result should be RGB JPEG
        with Image.open(BytesIO(result)) as optimized:
            assert optimized.mode == "RGB"

    def test_handles_invalid_image(self) -> None:
        """Returns original data for invalid images."""
        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig())

        invalid_data = b"not an image"
        result = service.optimize(invalid_data)

        assert result == invalid_data

    def test_uses_custom_quality(self) -> None:
        """Uses custom quality parameter."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGB", (400, 300), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG", quality=100)
        original_data = img_bytes.getvalue()

        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig())

        # Low quality should produce smaller file
        low_quality = service.optimize(original_data, quality=20)
        high_quality = service.optimize(original_data, quality=95)

        assert len(low_quality) < len(high_quality)


class TestImageServiceAIVerification:
    """Tests for AI verification methods."""

    @pytest.mark.asyncio
    async def test_select_with_ai_uses_client(self) -> None:
        """_select_with_ai uses the client for verification."""
        from io import BytesIO
        from unittest.mock import AsyncMock

        from PIL import Image

        mock_book = MagicMock()

        # Create mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "YES - this looks like the dish"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        config = ImageConfig(strategy=SelectionStrategy.AI_VERIFIED, ai_threshold=0.6)
        service = ImageService(book=mock_book, config=config, client=mock_client)

        # Create test candidates
        img = Image.new("RGB", (800, 600))
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        test_data = img_bytes.getvalue()

        candidates = [
            ImageCandidate(path="test.jpg", data=test_data, width=800, height=600, area=480_000),
        ]

        recipe = MelaRecipe(
            title="Chocolate Cake",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup flour", "2 cups sugar"])],
            instructions=["Mix ingredients", "Bake at 350F"],
        )

        result = await service._select_with_ai(candidates, recipe)

        assert result is not None
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_image_returns_confidence(self) -> None:
        """_verify_image returns matches and confidence."""
        from unittest.mock import AsyncMock

        mock_book = MagicMock()

        # Mock YES response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "YES"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = ImageService(
            book=mock_book, config=ImageConfig(ai_threshold=0.6), client=mock_client
        )

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        matches, confidence = await service._verify_image(b"fake_image", recipe)

        assert matches is True
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_verify_image_maybe_response(self) -> None:
        """_verify_image handles MAYBE response."""
        from unittest.mock import AsyncMock

        mock_book = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "MAYBE - could be the dish"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = ImageService(
            book=mock_book, config=ImageConfig(ai_threshold=0.6), client=mock_client
        )

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        matches, confidence = await service._verify_image(b"fake_image", recipe)

        assert matches is False  # 0.5 < 0.6 threshold
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_verify_image_no_response(self) -> None:
        """_verify_image handles NO response."""
        from unittest.mock import AsyncMock

        mock_book = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "NO - different dish"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = ImageService(
            book=mock_book, config=ImageConfig(ai_threshold=0.6), client=mock_client
        )

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        matches, confidence = await service._verify_image(b"fake_image", recipe)

        assert matches is False
        assert confidence == 0.1

    @pytest.mark.asyncio
    async def test_verify_image_without_client(self) -> None:
        """_verify_image returns default when no client."""
        mock_book = MagicMock()
        service = ImageService(book=mock_book, config=ImageConfig(), client=None)

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        matches, confidence = await service._verify_image(b"fake_image", recipe)

        assert matches is True
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_verify_image_handles_api_error(self) -> None:
        """_verify_image handles API errors gracefully."""
        from unittest.mock import AsyncMock

        mock_book = MagicMock()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=OpenAIError("API error"))

        service = ImageService(
            book=mock_book, config=ImageConfig(ai_threshold=0.6), client=mock_client
        )

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        matches, confidence = await service._verify_image(b"fake_image", recipe)

        # Should return default on error
        assert matches is True
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_verify_image_handles_empty_response(self) -> None:
        """_verify_image handles empty response content."""
        from unittest.mock import AsyncMock

        mock_book = MagicMock()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        service = ImageService(
            book=mock_book, config=ImageConfig(ai_threshold=0.6), client=mock_client
        )

        recipe = MelaRecipe(
            title="Test",
            ingredients=[IngredientGroup(title="", ingredients=["1 cup"])],
            instructions=["Step 1", "Step 2"],
        )

        matches, confidence = await service._verify_image(b"fake_image", recipe)

        assert matches is True
        assert confidence == 0.5
