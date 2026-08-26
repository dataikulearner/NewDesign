from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import pandas as pd

from generics.params_loader import load_and_validate_params
from generics.registries.column_registry import ColumnRegistry
from generics.registries.scope_registry import ScopeRegistry

logger = logging.getLogger(__name__)


class GenericPipeline:
    """
    Orchestrator connecting config (generics/) to legacy code (unchanged).

    Example:
        >>> pipeline = GenericPipeline(spark_sess, config_root="config")
        >>> df_agg = pipeline.run_pre_processing(
        ...     flow_type="ccirc",
        ...     axes_str_par="entity___accounting_site_code_post_acc",  # from axes_periods_par dataset
        ...     table_path="hdfs://.../FEM.orc", period="2026Q1",
        ...     period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
        ...     scope="risk_corp",
        ... )
    """

    def __init__(
        self,
        spark_sess,
        config_root: str = "config",
        prepare_agg_period_axe_func=None,
        generate_business_axes_func=None,
        compute_scope_func=None,
        period_const: str = None,
        table_path_const: str = None,
    ):
        """
        Args:
            spark_sess: SparkSession
            config_root: Root folder containing column_registry.json and
                flows/<flow_type>_params.json
            prepare_agg_period_axe_func: Injectable legacy
                core.pre_processing.prepare_agg_period_axe (testing).
                Defaults to lazy import in production.
            generate_business_axes_func: Injectable legacy
                utils.py.generate_business_axes (testing). Defaults to
                lazy import in production.
            compute_scope_func: Injectable utils.py compute_scope
                (testing). Defaults to lazy import in production.
            period_const, table_path_const: Injectable values for
                core.constants.PERIOD / TABLE_PATH (testing — these are
                literal column-name strings, e.g. "period"/"table_path").
                Defaults to lazy import in production.
        """
        self.spark = spark_sess
        self.config_root = config_root
        self.column_registry = ColumnRegistry.load_from_file(
            f"{config_root}/column_registry.json"
        )
        self._prepare_agg_period_axe = prepare_agg_period_axe_func
        self._generate_business_axes = generate_business_axes_func
        self._compute_scope = compute_scope_func
        self._period_const = period_const
        self._table_path_const = table_path_const

    # -------------------------------------------------------------------------
    # PARAMS LOADING
    # -------------------------------------------------------------------------

    def _load_params(self, flow_type: str) -> Dict[str, Any]:
        """Load and validate this flow's own params file."""
        path = f"{self.config_root}/flows/{flow_type}_params.json"
        return load_and_validate_params(path)

    # -------------------------------------------------------------------------
    # RELEVANT COLS — matches real recipe formula exactly
    # -------------------------------------------------------------------------

    def _build_relevant_cols(self, flow_type: str, params: Dict[str, Any]) -> List[str]:
        """
        relevant_cols = id_cols + target_vars + business_axes
                         + [PERIOD, TABLE_PATH] + column_registry.get_for_flow(flow_type)

        Base formula matches compute_SP_agg (DETECT_CCIRC_PROD) exactly
        — see module docstring point 2. ColumnRegistry's columns are
        ADDED ON TOP (additive, not a replacement of the base formula).

        WHY THIS ACHIEVES GENERICITY WITHOUT TOUCHING LEGACY:
        Inside enrich_data_period_ccirc (frozen, legacy), the final
        column selection does:
            for scope_col in SCOPE_SOURCE_COLMNS:   # hardcoded list
                if scope_col not in relevant_cols and scope_col in df.columns:
                    relevant_cols.append(scope_col)
        This is APPEND-ONLY — it only adds columns MISSING from the
        `relevant_cols` we pass in, never removes any. So if we already
        include a new scope column here (via ColumnRegistry, itself
        driven by column_registry.json — one file per flow's project),
        that column reaches enrich_data_period_ccirc ALREADY present,
        and the hardcoded SCOPE_SOURCE_COLMNS loop simply skips it
        (condition `not in relevant_cols` is False) — a no-op, not a
        conflict. Legacy code is not modified, not even touched at
        runtime for that column. Adding a new scope-filter column
        therefore requires editing ONLY column_registry.json.
        """
        id_cols = params["line_id_cols"]
        target_vars = params["target_variables"]

        generate_business_axes = self._resolve_generate_business_axes()
        axes = [combo.split("___") for combo in params["axes_str"].split("/")]
        business_axes = generate_business_axes(axes)

        period_const, table_path_const = self._resolve_constants()

        registry_cols = self.column_registry.get_for_flow(flow_type)

        # Preserve order for the base formula, dedup across all sources
        seen = set()
        relevant_cols = []
        for col in (
            list(id_cols) + list(target_vars) + list(business_axes)
            + [period_const, table_path_const] + list(registry_cols)
        ):
            if col not in seen:
                seen.add(col)
                relevant_cols.append(col)

        return relevant_cols

    # -------------------------------------------------------------------------
    # STAGE: PRE-PROCESSING (enrich + aggregate + scope filter)
    # -------------------------------------------------------------------------

    def run_pre_processing(
        self,
        flow_type: str,
        axes_str_par: str,
        table_path: str,
        period: str,
        period_run: str,
        scope_run: str,
        id_set_params: str,
        scope: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run Stage 1+2: enrich (legacy) + aggregate (legacy) + scope
        filter (legacy utils.py, if this flow has scopes).

        Args:
            flow_type: e.g. "ccirc"
            axes_str_par: The SPECIFIC axis combination for this
                partition (e.g. "entity___stage") — comes from the
                upstream "axes_periods_par" dataset in the real DSS
                recipe, NOT from params["axes_str"] (which is only
                used to compute business_axes for relevant_cols).
            table_path: HDFS path to the source table (FEM)
            period: Period being enriched (e.g. "2026Q1")
            period_run: Running period for partitioning
            scope_run: Running scope label (legacy tag)
            id_set_params: Running parameter-set id
            scope: Scope id to filter by. None = no filtering.

        Returns:
            Aggregated (and, if requested, scope-filtered) DataFrame
        """
        params = self._load_params(flow_type)

        scope_registry = None
        if params.get("scopes"):
            scope_registry = ScopeRegistry(params, self.column_registry)
            scope_registry.validate_and_fail_fast()
        elif scope:
            logger.warning(
                f"[GenericPipeline] scope='{scope}' requested but flow "
                f"'{flow_type}' has no 'scopes' declared in its params — ignored"
            )
            scope = None

        relevant_cols = self._build_relevant_cols(flow_type, params)

        prepare_agg = self._resolve_prepare_agg_period_axe()

        logger.info(
            f"[GenericPipeline] Stage 1+2: flow='{flow_type}' "
            f"axes_str_par='{axes_str_par}' period={period}"
        )

        df_agg = prepare_agg(
            self.spark,
            axes_str_par,
            params["target_variables"],
            params["target_agg_functions"],
            params["segment_vars"],
            params["weight_vars"],
            table_path,
            period,
            period_run,
            scope_run,
            id_set_params,
            flow_type,
            pma_mapping_path=params["pma_mapping_path"],
            pma_col=params["pma_col"],
            pma_mapping_col=params["pma_mapping_col"],
            entity_mapping_col=params["entity_mapping_col"],
            entity_title=params["entity_title"],
            relevant_cols=relevant_cols,
        )

        if scope:
            compute_scope = self._resolve_compute_scope()
            logger.info(f"[GenericPipeline] Applying scope filter: '{scope}'")
            mask = compute_scope(df_agg, scope, params["scopes"][scope]["filters"])
            rows_before = len(df_agg)
            df_agg = df_agg[mask]
            logger.info(
                f"[GenericPipeline] Scope '{scope}': {rows_before} -> {len(df_agg)} rows"
            )

        return df_agg

    # -------------------------------------------------------------------------
    # INJECTABLE RESOLUTION (lazy import in production, injectable in tests)
    # -------------------------------------------------------------------------

    def _resolve_prepare_agg_period_axe(self):
        if self._prepare_agg_period_axe is not None:
            return self._prepare_agg_period_axe
        from core.pre_processing import prepare_agg_period_axe
        return prepare_agg_period_axe

    def _resolve_generate_business_axes(self):
        if self._generate_business_axes is not None:
            return self._generate_business_axes
        from utils.py import generate_business_axes
        return generate_business_axes

    def _resolve_compute_scope(self):
        if self._compute_scope is not None:
            return self._compute_scope
        from utils.py import compute_scope
        return compute_scope

    def _resolve_constants(self):
        if self._period_const is not None and self._table_path_const is not None:
            return self._period_const, self._table_path_const
        from core.constants import PERIOD, TABLE_PATH
        return PERIOD, TABLE_PATH
