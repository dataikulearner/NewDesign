"""
Unit tests for GenericPipeline 
"""

import json

import pytest
import pandas as pd

from generics.pipeline.generic_pipeline import GenericPipeline
from generics.registries.column_registry import ColumnRegistry


# =============================================================================
# FIXTURES
# =============================================================================

CCIRC_PARAMS = {
    "target_agg_functions": {"seg_agg1": "count", "t_agg1": "avg", "t_agg4": "sum"},
    "weight_vars": {"t_agg5": "mt_cal_ead_tot_b"},
    "target_variables": ["lgd_before_flex", "mt_cal_ead_tot_b", "eir"],
    "segment_vars": {"seg_agg2": "mt_cal_ead_tot_b", "seg_agg3": "provision_uni"},
    "axes_str": "entity/entity___accounting_site_code_post_acc/entity___asset_class",
    "entity_mapping_col": "entity",
    "entity_title": "entity",
    "flow_type": "ccirc",
    "pma_col": "cd_uds_pmas_metier_ap_acc",
    "pma_mapping_col": "cd_metier_pmas",
    "pma_mapping_path": "/Projects/.../PMA_CCIRC_Mapping",
    "line_id_cols": ["id_technique", "facility_id", "ctpr_id", "lb_raison_sociale"],
    "flow_id": "ccirc",
    "source_tables": ["FEM", "TIERS"],
    "detector": "statistical",
    "scopes": {
        "risk_corp": {
            "label": "Risk Corp",
            "filters": {
                "asset_class": {"operator": "EQUALS", "value": "CORP_L"}
            }
        }
    }
}

IFRS9_PARAMS_NO_SCOPES = {
    "target_agg_functions": {"t_agg1": "avg"},
    "weight_vars": {},
    "target_variables": ["eir"],
    "segment_vars": {},
    "axes_str": "entity",
    "entity_mapping_col": "entity",
    "entity_title": "entity",
    "flow_type": "ifrs9",
    "pma_col": "pmas_code_post_acc",
    "pma_mapping_col": "",
    "pma_mapping_path": "",
    "line_id_cols": ["id_technique", "facility_id", "original_ctpr_id", "lb_raison_sociale"],
    "flow_id": "ifrs9",
    "source_tables": ["FEM", "TIERS"],
    "detector": "statistical",
}

COLUMN_REGISTRY_DATA = {
    "columns": {
        "asset_class": {"source_table": "FEM"},
        "accounting_site_code_post_acc": {"source_table": "FEM"},
    }
}


def real_generate_business_axes(technical_axes: list) -> list:
    """
    Faithful copy of the REAL utils.py function (confirmed via
    screenshot) — used as the injected default in tests, since it's a
    pure function with no external dependency, safe to mirror exactly
    for test purposes without importing the real utils.py module.
    """
    res_all = []
    for axes in technical_axes:
        res_all += axes
    return list(set(res_all))


@pytest.fixture(autouse=True)
def reset_column_registry():
    ColumnRegistry.reset()
    yield
    ColumnRegistry.reset()


@pytest.fixture
def config_root(tmp_path):
    (tmp_path / "flows").mkdir()
    with open(tmp_path / "column_registry.json", "w") as f:
        json.dump(COLUMN_REGISTRY_DATA, f)
    with open(tmp_path / "flows" / "ccirc_params.json", "w") as f:
        json.dump(CCIRC_PARAMS, f)
    with open(tmp_path / "flows" / "ifrs9_params.json", "w") as f:
        json.dump(IFRS9_PARAMS_NO_SCOPES, f)
    return str(tmp_path)


@pytest.fixture
def fake_agg_df():
    return pd.DataFrame({
        "entity": ["A", "B", "C"],
        "asset_class": ["CORP_L", "CORP_C", "CORP_L"],
        "t_agg1": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def mock_prepare_agg(fake_agg_df):
    calls = []

    def fn(spark, axes_str, t_vars, t_agg_funcs_mapping, seg_vars_mapping,
           weight_vars_mapping, table_path, period, period_run, scope_run,
           id_set_params, flow_type, **kwargs):
        calls.append(dict(
            axes_str=axes_str, t_vars=t_vars, t_agg_funcs_mapping=t_agg_funcs_mapping,
            seg_vars_mapping=seg_vars_mapping, weight_vars_mapping=weight_vars_mapping,
            table_path=table_path, period=period, period_run=period_run,
            scope_run=scope_run, id_set_params=id_set_params, flow_type=flow_type,
            **kwargs
        ))
        return fake_agg_df

    fn.calls = calls
    return fn


@pytest.fixture
def mock_compute_scope():
    calls = []

    def fn(df, scope_id, filters):
        calls.append((scope_id, filters))
        return df["asset_class"] == filters["asset_class"]["value"]

    fn.calls = calls
    return fn


@pytest.fixture
def pipeline(config_root, mock_prepare_agg, mock_compute_scope):
    return GenericPipeline(
        spark_sess=object(),
        config_root=config_root,
        prepare_agg_period_axe_func=mock_prepare_agg,
        generate_business_axes_func=real_generate_business_axes,
        compute_scope_func=mock_compute_scope,
        period_const="period",
        table_path_const="table_path",
    )


# =============================================================================
# TEST: _build_relevant_cols — matches real formula exactly
# =============================================================================

class TestBuildRelevantCols:

    def test_formula_matches_real_recipe_base(self, pipeline):
        """
        Base formula (id_cols + target_vars + business_axes + [PERIOD,
        TABLE_PATH]) must still be present, in the documented order,
        BEFORE ColumnRegistry's additive contribution.
        """
        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        assert result[:4] == CCIRC_PARAMS["line_id_cols"]
        assert result[4:7] == CCIRC_PARAMS["target_variables"]
        assert "entity" in result
        assert "accounting_site_code_post_acc" in result
        assert "asset_class" in result

    def test_column_registry_columns_added_additively(self, pipeline):
        """
        ColumnRegistry's declared columns (this project's own
        column_registry.json) must ALSO be present — added on top of
        the base formula, not instead of it. This is what makes a new
        scope-filter column reach relevant_cols WITHOUT ever touching
        enrich_data_period_ccirc.py (legacy) — see _build_relevant_cols
        docstring for the append-only mechanism this relies on.
        """
        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        for col in COLUMN_REGISTRY_DATA["columns"]:
            assert col in result

    def test_no_duplicates_when_column_appears_in_multiple_sources(self, pipeline):
        """
        'asset_class' and 'accounting_site_code_post_acc' appear BOTH
        via business_axes (axes_str) AND via ColumnRegistry — must
        appear exactly once in the final list.
        """
        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        assert result.count("asset_class") == 1
        assert result.count("accounting_site_code_post_acc") == 1

    def test_new_column_reaches_relevant_cols_via_config_only(self, config_root, mock_prepare_agg, mock_compute_scope):
        """
        GENERICITY PROOF (config-level, mirrors ScopeRegistry's own
        TestGenericityRegression): declare a brand-new column ONLY in
        column_registry.json — not referenced anywhere in axes_str,
        target_variables, or line_id_cols — and confirm it reaches
        relevant_cols with zero Python code touched.
        """
        import os
        new_column_registry = {
            "columns": dict(COLUMN_REGISTRY_DATA["columns"], residual_maturity={"source_table": "FEM"})
        }
        with open(os.path.join(config_root, "column_registry.json"), "w") as f:
            json.dump(new_column_registry, f)

        ColumnRegistry.reset()
        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            compute_scope_func=mock_compute_scope,
            period_const="period", table_path_const="table_path",
        )

        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        assert "residual_maturity" in result


# =============================================================================
# TEST: run_pre_processing — CCIRC (has scopes)
# =============================================================================

class TestRunPreProcessingCcirc:

    def test_passes_axes_str_par_not_static_axes_str(self, pipeline, mock_prepare_agg):
        """
        CRITICAL: the axes_str argument to legacy must be axes_str_par
        (per-partition, caller-supplied), NOT params["axes_str"].
        """
        pipeline.run_pre_processing(
            flow_type="ccirc",
            axes_str_par="entity___stage",  # different from params["axes_str"]!
            table_path="hdfs://fem.orc", period="2026Q1",
            period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
        )

        call = mock_prepare_agg.calls[0]
        assert call["axes_str"] == "entity___stage"
        assert call["axes_str"] != CCIRC_PARAMS["axes_str"]

    def test_forwards_5_pma_kwargs(self, pipeline, mock_prepare_agg):
        pipeline.run_pre_processing(
            flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
        )

        call = mock_prepare_agg.calls[0]
        assert call["pma_col"] == CCIRC_PARAMS["pma_col"]
        assert call["pma_mapping_col"] == CCIRC_PARAMS["pma_mapping_col"]
        assert call["pma_mapping_path"] == CCIRC_PARAMS["pma_mapping_path"]
        assert call["entity_mapping_col"] == CCIRC_PARAMS["entity_mapping_col"]
        assert call["entity_title"] == CCIRC_PARAMS["entity_title"]

    def test_passes_flow_type_positionally(self, pipeline, mock_prepare_agg):
        pipeline.run_pre_processing(
            flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
        )

        call = mock_prepare_agg.calls[0]
        assert call["flow_type"] == "ccirc"

    def test_no_scope_requested_no_filtering(self, pipeline, mock_compute_scope, fake_agg_df):
        result = pipeline.run_pre_processing(
            flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
            scope=None,
        )

        assert len(mock_compute_scope.calls) == 0
        assert len(result) == len(fake_agg_df)

    def test_scope_requested_applies_real_filter(self, pipeline, mock_compute_scope):
        result = pipeline.run_pre_processing(
            flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
            scope="risk_corp",
        )

        assert len(mock_compute_scope.calls) == 1
        scope_id, filters = mock_compute_scope.calls[0]
        assert scope_id == "risk_corp"
        assert filters == CCIRC_PARAMS["scopes"]["risk_corp"]["filters"]
        assert len(result) == 2  # 2 rows with asset_class == CORP_L

    def test_invalid_scope_config_fails_fast_before_spark(self, config_root, mock_prepare_agg, mock_compute_scope):
        import os
        bad_params = dict(CCIRC_PARAMS)
        bad_params["scopes"] = {
            "bad_scope": {"filters": {"totally_unknown_column": {"operator": "EQUALS", "value": "X"}}}
        }
        with open(os.path.join(config_root, "flows", "ccirc_params.json"), "w") as f:
            json.dump(bad_params, f)

        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            compute_scope_func=mock_compute_scope,
            period_const="period", table_path_const="table_path",
        )

        with pytest.raises(ValueError, match="totally_unknown_column"):
            pipeline.run_pre_processing(
                flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
                period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
            )

        assert len(mock_prepare_agg.calls) == 0


# =============================================================================
# TEST: run_pre_processing — IFRS9 (no scopes)
# =============================================================================

class TestRunPreProcessingIfrs9NoScopes:

    def test_runs_without_scope_registry(self, config_root, mock_prepare_agg, mock_compute_scope):
        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            compute_scope_func=mock_compute_scope,
            period_const="period", table_path_const="table_path",
        )

        pipeline.run_pre_processing(
            flow_type="ifrs9", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="none", id_set_params="v1",
        )

        assert len(mock_prepare_agg.calls) == 1

    def test_scope_requested_on_scopeless_flow_warns_and_ignores(
        self, config_root, mock_prepare_agg, mock_compute_scope, fake_agg_df
    ):
        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            compute_scope_func=mock_compute_scope,
            period_const="period", table_path_const="table_path",
        )

        result = pipeline.run_pre_processing(
            flow_type="ifrs9", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="none", id_set_params="v1",
            scope="nonexistent_scope",
        )

        assert len(mock_compute_scope.calls) == 0
        assert len(result) == len(fake_agg_df)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
