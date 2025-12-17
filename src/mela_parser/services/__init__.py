"""Services package for mela_parser.

This package contains service classes that implement the core business logic
with proper dependency injection support.

Modules:
    factory: ServiceFactory for centralized dependency management
    images: ImageService for unified image processing
"""

from .factory import ServiceFactory
from .images import ImageConfig, ImageService, SelectionStrategy

__all__ = ["ImageConfig", "ImageService", "SelectionStrategy", "ServiceFactory"]
