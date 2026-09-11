"""
Strategy Factory — detector selection only 
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from .interfaces import DetectorStrategy
from .detectors import StatisticalDetector

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    Factory for creating detector strategies by name.

    Example:
        >>> detector = StrategyFactory.get_detector("statistical")
        >>> detector.name
        'StatisticalDetector'
    """

    _detectors: Dict[str, Type[DetectorStrategy]] = {
        "statistical": StatisticalDetector,
        # "ml": MLDetector,   # future extension point
    }

    @classmethod
    def get_detector(cls, name: str) -> DetectorStrategy:
        """
        Create detector by name.

        Raises:
            ValueError: Unknown name (fail-fast — no silent fallback)
        """
        if name not in cls._detectors:
            raise ValueError(
                f"Unknown detector strategy '{name}'. "
                f"Available: {sorted(cls._detectors.keys())}"
            )
        instance = cls._detectors[name]()
        logger.debug(f"Created detector: {name} → {instance.__class__.__name__}")
        return instance

    @classmethod
    def register_detector(cls, name: str, klass: Type[DetectorStrategy]) -> None:
        """Register a new detector strategy at runtime (e.g., ML detector)."""
        cls._detectors[name] = klass
        logger.info(f"Registered detector '{name}' → {klass.__name__}")

    @classmethod
    def list_available(cls) -> Dict[str, list]:
        """List all registered detector names."""
        return {"detectors": sorted(cls._detectors.keys())}
