"""
Strategy Interfaces - Abstract Base Classes for DETECT pipeline strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pandas as pd


# =============================================================================
# DETECTOR STRATEGY — the only surviving interface
# =============================================================================

class DetectorStrategy(ABC):
    """
    Abstract interface for anomaly detection strategies.

    Detector is responsible for:
    - Running the detection algorithm on an aggregated DataFrame
    - Returning the annotated result (anomaly type, deviation, ...)

    Concrete implementations:
    - StatisticalDetector: thin wrapper around core.models.detect_axes
    - (future) MLDetector: could replace it without changing
      GenericPipeline, as long as it implements this same contract
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (for logging/debugging)."""
        pass

    @abstractmethod
    def detect(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Run detection on an aggregated DataFrame.

        Args:
            df: Aggregated DataFrame (output of Stage 2 + scope filter)
            params: Detection parameters (thresholds, mappings, ...)

        Returns:
            DataFrame with detection results
        """
        pass

    def get_required_params(self) -> List[str]:
        """
        List of required keys in `params` for this detector.

        Note:
            Default implementation returns an empty list.
            Override to declare required params — used for fail-fast
            validation before calling detect().
        """
        return []
