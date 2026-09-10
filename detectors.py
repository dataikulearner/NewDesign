"""
Concrete Detector Strategy.
Classes:
    StatisticalDetector: thin wrapper around core.models.detect_axes
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

import pandas as pd

from .interfaces import DetectorStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# STATISTICAL DETECTOR
# =============================================================================

class StatisticalDetector(DetectorStrategy):
    """
    Statistical anomaly detection — thin wrapper around core.models.detect_axes.

    Called exactly once per run_modeling() call — detect_axes handles
    every target variable / axes combination present in the input
    DataFrame internally, no per-target looping needed here.

    Real signature (core.models.detect_axes), confirmed via screenshots:
        detect_axes(df, thresholds_segment_anomaly, thresholds_negligibility,
                    target_detection_mapping, seg_agg_materiality,
                    nb_periods, nb_last, s_window, nbs_last) -> pd.DataFrame

    Params contract (from flow params JSON — ccirc_params.json, all 8
    keys confirmed present):
        {
            "thresholds_segment_anomaly": {...},
            "thresholds_negligibility": {...},
            "target_detection_mapping": {...},
            "seg_agg_materiality": "agg4",
            "nb_periods": 10,
            "nb_last": 4,
            "s_window": 4,
            "nbs_last": 2,
        }

    Example:
        >>> detector = StatisticalDetector()
        >>> result = detector.detect(df_agg_focus_model, params)
    """

    #: Required params — matches detect_axes' 8 arguments exactly (df not
    #: included, passed separately to detect()).
    REQUIRED_PARAMS: List[str] = [
        "thresholds_segment_anomaly",
        "thresholds_negligibility",
        "target_detection_mapping",
        "seg_agg_materiality",
        "nb_periods",
        "nb_last",
        "s_window",
        "nbs_last",
    ]

    def __init__(self, detect_axes_func=None):
        """
        Args:
            detect_axes_func: Injectable core.models.detect_axes function
                (for testing without a full legacy import chain). Defaults
                to a lazy import of the real function in production.
        """
        self._detect_axes = detect_axes_func

    @property
    def name(self) -> str:
        """Strategy name."""
        return "StatisticalDetector"

    # -------------------------------------------------------------------------
    # DETECTION
    # -------------------------------------------------------------------------

    def detect(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Run statistical detection — single call to detect_axes.

        Args:
            df: Aggregated DataFrame — the real recipe passes
                SP_agg_focus_model (output of an intermediate
                compute_SP_agg_focus_model step, NOT the raw SP_agg
                from pre-processing directly — GenericPipeline callers
                must supply this already-computed DataFrame).
            params: Flow params, must contain the 8 required keys
                (REQUIRED_PARAMS)

        Returns:
            Whatever core.models.detect_axes returns — a DataFrame
            covering every target variable / axes combination present
            in `df`, already labeled (anomaly, deviation, model_type, ...)

        Raises:
            ValueError: If required params missing (fail-fast, before
                        calling into legacy code)
        """
        self._validate_params(params)

        detect_axes = self._resolve_detect_axes()

        logger.info(
            f"[{self.name}] Calling detect_axes once on {len(df)} rows "
            f"(nb_periods={params['nb_periods']}, nb_last={params['nb_last']})"
        )

        result = detect_axes(
            df,
            params["thresholds_segment_anomaly"],
            params["thresholds_negligibility"],
            params["target_detection_mapping"],
            params["seg_agg_materiality"],
            params["nb_periods"],
            params["nb_last"],
            params["s_window"],
            params["nbs_last"],
        )

        logger.info(f"[{self.name}] Done: {len(result)} rows")
        return result

    # -------------------------------------------------------------------------
    # INTERNAL
    # -------------------------------------------------------------------------

    def get_required_params(self) -> List[str]:
        """Required parameters — matches detect_axes' required arguments exactly."""
        return list(self.REQUIRED_PARAMS)

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate params. FAIL FAST with explicit message, before touching
        legacy code — a missing key here would otherwise surface as a
        confusing TypeError deep inside detect_axes.

        Raises:
            ValueError: If any required param is missing
        """
        errors = []

        for key in self.get_required_params():
            if key not in params:
                errors.append(f"missing required param '{key}'")

        if errors:
            raise ValueError(
                f"[{self.name}] Invalid params:\n" +
                "\n".join(f"  ❌ {e}" for e in errors)
            )

    def _resolve_detect_axes(self):
        """
        Resolve detect_axes function.

        Injected function (tests) > core.models import (production).
        """
        if self._detect_axes is not None:
            return self._detect_axes

        try:
            from core.models import detect_axes
            return detect_axes
        except ImportError as e:
            raise ImportError(
                f"[{self.name}] Cannot import detect_axes from core.models. "
                f"Inject a function via constructor for testing. Error: {e}"
            )
