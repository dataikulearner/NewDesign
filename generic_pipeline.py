from __future__ import annotations

import ast
import logging
from typing import Optional, List, Dict, Any

import pandas as pd

from generics.params_loader import load_and_validate_params
from generics.registries.column_registry import ColumnRegistry
from generics.registries.scope_registry import ScopeRegistry

logger = logging.getLogger(__name__)

#: Keys in DSS custom variables that are STRING-ENCODED Python literals
#: (list/dict) and must go through ast.literal_eval() — confirmed from
#: the real compute_SP_agg recipe (DETECT_CCIRC_PROD). Any key NOT in
#: this set is used as-is (already a plain string in dss_vars, e.g.
#: axes_str, pma_col, entity_title...).
DSS_VARS_LITERAL_KEYS = [
    "target_variables",
    "line_id_cols",
    "target_agg_functions",
    "segment_vars",
    "weight_vars",
]


class GenericPipeline:
    """
    Orchestrator connecting config (generics/) to legacy code (unchanged).

    Example:
        >>> pipeline = GenericPipeline(spark_sess, config_root="config")
        >>> df_agg = pipeline.run_pre_processing(
        ...     flow_type="ccirc",
        ...     axes_str_par="entity___accounting_site_code_post_acc",
        ...     table_path="hdfs://.../FEM.orc", period="2026Q1",
        ...     period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
        ... )
        >>> # ... modeling stage in between ...
        >>> df_final = pipeline.run_post_processing(df_final, flow_type="ccirc")
        >>> # df_final now has a "perimeters_list" column, SAME row count
    """

    def __init__(
        self,
        spark_sess,
        config_root: str = "config",
        prepare_agg_period_axe_func=None,
        generate_business_axes_func=None,
        add_perimeters_list_func=None,
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
            add_perimeters_list_func: Injectable legacy
                utils.scopes.add_perimeters_list (testing). Defaults to
                lazy import in production.
            period_const, table_path_const: Injectable values for
                core.constants.PERIOD / TABLE_PATH (testing).
        """
        self.spark = spark_sess
        self.config_root = config_root
        self.column_registry = ColumnRegistry.load_from_file(
            f"{config_root}/column_registry.json"
        )
        self._prepare_agg_period_axe = prepare_agg_period_axe_func
        self._generate_business_axes = generate_business_axes_func
        self._add_perimeters_list = add_perimeters_list_func
        self._period_const = period_const
        self._table_path_const = table_path_const

    # -------------------------------------------------------------------------
    # PARAMS LOADING
    # -------------------------------------------------------------------------

    def _load_params(self, flow_type: str) -> Dict[str, Any]:
        """Load and validate this flow's own params file."""
        path = f"{self.config_root}/flows/{flow_type}_params.json"
        return load_and_validate_params(path)

    @staticmethod
    def params_from_dss_vars(dss_vars: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DSS Project Variables (dataiku.get_custom_variables())
        into the clean params dict GenericPipeline methods expect.

        DSS custom variables are a flat dict where list/dict-typed
        values are STRING-ENCODED (Python literal syntax) — the real
        recipe parses them with ast.literal_eval() before use. This
        does the same, only for the confirmed keys (DSS_VARS_LITERAL_KEYS).
        All other keys (axes_str, pma_col, pma_mapping_col,
        pma_mapping_path, entity_mapping_col, entity_title, flow_type,
        scopes if present...) are passed through as-is.

        NOTE: unlike load_and_validate_params() (file-based path), this
        does NOT enforce source_tables/detector as required — those are
        not consumed by any code yet (see params_loader.REQUIRED_KEYS
        note). Only "flow_type" matters, and it's used as-is (no key
        renaming needed — ScopeRegistry reads "flow_type" directly).

        Args:
            dss_vars: dict from dataiku.get_custom_variables()

        Returns:
            Params dict, ready to pass as `params=` to run_pre_processing()
        """
        params = dict(dss_vars)  # shallow copy, don't mutate caller's dict

        for key in DSS_VARS_LITERAL_KEYS:
            if key in params and isinstance(params[key], str):
                params[key] = ast.literal_eval(params[key])

        if "scopes" in params and isinstance(params["scopes"], str):
            params["scopes"] = ast.literal_eval(params["scopes"])

        return params

    # -------------------------------------------------------------------------
    # RELEVANT COLS
    # -------------------------------------------------------------------------

    def _build_relevant_cols(self, flow_type: str, params: Dict[str, Any]) -> List[str]:
        """
        relevant_cols = id_cols + target_vars + business_axes
                         + [PERIOD, TABLE_PATH] + column_registry.get_for_flow(flow_type)

        Base formula matches compute_SP_agg (DETECT_CCIRC_PROD) exactly.
        ColumnRegistry's columns are ADDED ON TOP (additive) — see
        ARCHITECTURE_GENERICS_FR_V2.md for why this achieves genericity
        without ever touching enrich_data_period_ccirc.py (its own
        SCOPE_SOURCE_COLMNS append loop is append-only, so a column we
        already supplied is simply skipped there, not conflicted with).
        """
        id_cols = params["line_id_cols"]
        target_vars = params["target_variables"]

        generate_business_axes = self._resolve_generate_business_axes()
        axes = [combo.split("___") for combo in params["axes_str"].split("/")]
        business_axes = generate_business_axes(axes)

        period_const, table_path_const = self._resolve_constants()

        registry_cols = self.column_registry.get_for_flow(flow_type)

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
    # STAGE 1+2: PRE-PROCESSING (enrich + aggregate)
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
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Run Stage 1+2: enrich (legacy) + aggregate (legacy).

        NOTE: no scope filtering happens here (v3 correction) — scope
        labeling (add_perimeters_list) happens in run_post_processing()
        instead, and does not remove any row. This stage still runs
        ScopeRegistry.validate_and_fail_fast() when the flow has scopes
        declared — config validation is independent of where the
        labeling itself happens, and must fail fast, before Spark work.

        Args:
            flow_type: e.g. "ccirc"
            axes_str_par: The SPECIFIC axis combination for this
                partition (from the upstream "axes_periods_par" dataset
                in the real DSS recipe) — NOT params["axes_str"].
            table_path: HDFS path to the source table (FEM)
            period: Period being enriched (e.g. "2026Q1")
            period_run: Running period for partitioning
            scope_run: Running scope label (legacy tag, unrelated to
                row filtering — see module docstring)
            id_set_params: Running parameter-set id
            params: Pre-loaded params dict (e.g. from
                GenericPipeline.params_from_dss_vars(dss_vars)). If
                None (default), params are loaded from
                config_root/flows/<flow_type>_params.json instead.

        Returns:
            Aggregated DataFrame (same row semantics as legacy output)
        """
        if params is None:
            params = self._load_params(flow_type)

        if params.get("scopes"):
            scope_registry = ScopeRegistry(params, self.column_registry)
            scope_registry.validate_and_fail_fast()

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

        return df_agg

    # -------------------------------------------------------------------------
    # STAGE 5: POST-PROCESSING (scope labeling — perimeters_list)
    # -------------------------------------------------------------------------

    def run_post_processing(
        self,
        df: pd.DataFrame,
        flow_type: str,
    ) -> pd.DataFrame:
        """
        Run the scope-labeling step: adds a "perimeters_list" column
        (legacy utils.scopes.add_perimeters_list), called directly,
        never re-implemented.

        Does NOT filter any row — row count in == row count out.
        If this flow has no "scopes" declared (e.g. IFRS9),
        add_perimeters_list itself returns a DataFrame with
        perimeters_list = NaN for every row (legacy behavior,
        unchanged here).

        Args:
            df: DataFrame to label (post-modeling / post-detection)
            flow_type: e.g. "ccirc"

        Returns:
            Same DataFrame with a "perimeters_list" column added,
            no scope_* intermediate columns, same row count.
        """
        params = self._load_params(flow_type)
        add_perimeters_list = self._resolve_add_perimeters_list()

        rows_before = len(df)
        result = add_perimeters_list(df, params.get("scopes"), flow_type)

        logger.info(
            f"[GenericPipeline] run_post_processing: {rows_before} -> "
            f"{len(result)} rows (must be unchanged)"
        )

        return result

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

    def _resolve_add_perimeters_list(self):
        if self._add_perimeters_list is not None:
            return self._add_perimeters_list
        from utils.scopes import add_perimeters_list
        return add_perimeters_list

    def _resolve_constants(self):
        if self._period_const is not None and self._table_path_const is not None:
            return self._period_const, self._table_path_const
        from core.constants import PERIOD, TABLE_PATH
        return PERIOD, TABLE_PATH
