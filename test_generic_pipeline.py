import json

import pytest
import pandas as pd
import numpy as np

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
    "scopes": {
        "us": {"label": "US", "filters": {"accounting_site_code_post_acc": {"operator": "IN", "values": [12309]}}},
        "risk_corp": {
            "label": "Risk Corp",
            "filters": {"asset_class": {"operator": "EQUALS", "value": "CORP_L"}}
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
    # no "scopes" key
}

COLUMN_REGISTRY_DATA = {
    "columns": {
        "asset_class": {"source_table": "FEM"},
        "accounting_site_code_post_acc": {"source_table": "FEM"},
    }
}


def real_generate_business_axes(technical_axes: list) -> list:
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
        "accounting_site_code_post_acc": [12309, 40043, 12309],
        "t_agg1": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def mock_prepare_agg(fake_agg_df):
    calls = []

    def fn(spark, axes_str, t_vars, t_agg_funcs_mapping, seg_vars_mapping,
           weight_vars_mapping, table_path, period, period_run, scope_run,
           id_set_params, flow_type, **kwargs):
        calls.append(dict(
            axes_str=axes_str, t_vars=t_vars, table_path=table_path,
            period=period, period_run=period_run, scope_run=scope_run,
            id_set_params=id_set_params, flow_type=flow_type, **kwargs
        ))
        return fake_agg_df

    fn.calls = calls
    return fn


@pytest.fixture
def mock_add_perimeters_list():
    """
    Mimics real utils.scopes.add_perimeters_list: adds a
    "perimeters_list" column, drops nothing row-wise, never filters.
    """
    calls = []

    def fn(df, scope_definitions, flow_type):
        calls.append((scope_definitions, flow_type))
        df = df.copy()
        if not scope_definitions:
            df["perimeters_list"] = np.nan
            return df
        # simplistic real-ish simulation: label by asset_class match
        labels = []
        for _, row in df.iterrows():
            row_labels = []
            if "us" in scope_definitions and row.get("accounting_site_code_post_acc") == 12309:
                row_labels.append("US")
            if "risk_corp" in scope_definitions and row.get("asset_class") == "CORP_L":
                row_labels.append("Risk Corp")
            labels.append(",".join(row_labels) if row_labels else "N/A")
        df["perimeters_list"] = labels
        return df

    fn.calls = calls
    return fn


@pytest.fixture
def pipeline(config_root, mock_prepare_agg, mock_add_perimeters_list):
    return GenericPipeline(
        spark_sess=object(),
        config_root=config_root,
        prepare_agg_period_axe_func=mock_prepare_agg,
        generate_business_axes_func=real_generate_business_axes,
        add_perimeters_list_func=mock_add_perimeters_list,
        period_const="period",
        table_path_const="table_path",
    )


# =============================================================================
# TEST: run_pre_processing — NO scope filtering (v3 correction)
# =============================================================================

class TestRunPreProcessing:

    def test_no_longer_accepts_scope_parameter(self, pipeline):
        """
        Regression guard: run_pre_processing() must NOT have a `scope`
        parameter anymore — scope labeling moved to post-processing.
        """
        import inspect
        sig = inspect.signature(pipeline.run_pre_processing)
        assert "scope" not in sig.parameters

    def test_returns_full_unfiltered_dataframe(self, pipeline, fake_agg_df):
        result = pipeline.run_pre_processing(
            flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
        )

        assert len(result) == len(fake_agg_df)  # no row removed

    def test_validates_scope_config_fail_fast_even_though_not_filtering(
        self, config_root, mock_prepare_agg, mock_add_perimeters_list
    ):
        """
        ScopeRegistry.validate_and_fail_fast() STILL runs here — config
        correctness is validated independently of where scope labeling
        happens later.
        """
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
            add_perimeters_list_func=mock_add_perimeters_list,
            period_const="period", table_path_const="table_path",
        )

        with pytest.raises(ValueError, match="totally_unknown_column"):
            pipeline.run_pre_processing(
                flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
                period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
            )

        assert len(mock_prepare_agg.calls) == 0  # Spark never touched

    def test_ifrs9_no_scopes_runs_fine_without_validation(self, config_root, mock_prepare_agg, mock_add_perimeters_list):
        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            add_perimeters_list_func=mock_add_perimeters_list,
            period_const="period", table_path_const="table_path",
        )

        pipeline.run_pre_processing(
            flow_type="ifrs9", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="none", id_set_params="v1",
        )

        assert len(mock_prepare_agg.calls) == 1


# =============================================================================
# TEST: run_post_processing — scope LABELING, never filtering
# =============================================================================

class TestRunPostProcessing:

    def test_adds_perimeters_list_column(self, pipeline, fake_agg_df):
        result = pipeline.run_post_processing(fake_agg_df, flow_type="ccirc")

        assert "perimeters_list" in result.columns

    def test_row_count_unchanged(self, pipeline, fake_agg_df):
        """CRITICAL: post-processing must NEVER remove rows."""
        result = pipeline.run_post_processing(fake_agg_df, flow_type="ccirc")

        assert len(result) == len(fake_agg_df)

    def test_row_can_belong_to_multiple_scopes(self, pipeline, fake_agg_df):
        result = pipeline.run_post_processing(fake_agg_df, flow_type="ccirc")

        # row 0: site=12309 (US) AND asset_class=CORP_L (Risk Corp)
        assert result["perimeters_list"].iloc[0] == "US,Risk Corp"

    def test_row_matching_no_scope_gets_na(self, pipeline, fake_agg_df):
        result = pipeline.run_post_processing(fake_agg_df, flow_type="ccirc")

        # row 1: site=40043 (not US), asset_class=CORP_C (not Risk Corp)
        assert result["perimeters_list"].iloc[1] == "N/A"

    def test_passes_scope_definitions_from_params(self, pipeline, fake_agg_df, mock_add_perimeters_list):
        pipeline.run_post_processing(fake_agg_df, flow_type="ccirc")

        scope_definitions, flow_type = mock_add_perimeters_list.calls[0]
        assert scope_definitions == CCIRC_PARAMS["scopes"]
        assert flow_type == "ccirc"

    def test_ifrs9_no_scopes_gets_null_column_no_row_loss(
        self, config_root, mock_prepare_agg, mock_add_perimeters_list, fake_agg_df
    ):
        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            add_perimeters_list_func=mock_add_perimeters_list,
            period_const="period", table_path_const="table_path",
        )

        result = pipeline.run_post_processing(fake_agg_df, flow_type="ifrs9")

        assert len(result) == len(fake_agg_df)
        assert result["perimeters_list"].isna().all()


# =============================================================================
# TEST: _build_relevant_cols — unchanged from v2 (still correct)
# =============================================================================

class TestBuildRelevantCols:

    def test_formula_matches_real_recipe_base(self, pipeline):
        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        assert result[:4] == CCIRC_PARAMS["line_id_cols"]
        assert result[4:7] == CCIRC_PARAMS["target_variables"]
        assert "entity" in result
        assert "accounting_site_code_post_acc" in result
        assert "asset_class" in result

    def test_column_registry_columns_added_additively(self, pipeline):
        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        for col in COLUMN_REGISTRY_DATA["columns"]:
            assert col in result

    def test_no_duplicates_when_column_appears_in_multiple_sources(self, pipeline):
        result = pipeline._build_relevant_cols("ccirc", CCIRC_PARAMS)

        assert result.count("asset_class") == 1
        assert result.count("accounting_site_code_post_acc") == 1


# =============================================================================
# TEST: params_from_dss_vars — DSS Project Variables adapter
# =============================================================================

class TestParamsFromDssVars:

    def test_parses_literal_encoded_list_keys(self):
        dss_vars = {
            "flow_type": "ccirc",
            "axes_str": "entity/entity___asset_class",
            "target_variables": "['lgd_before_flex', 'mt_cal_ead_tot_b', 'eir']",
            "line_id_cols": "['id_technique', 'facility_id']",
            "target_agg_functions": "{'t_agg1': 'avg'}",
            "segment_vars": "{'seg_agg2': 'mt_cal_ead_tot_b'}",
            "weight_vars": "{'t_agg5': 'mt_cal_ead_tot_b'}",
            "pma_col": "cd_uds_pmas_metier_ap_acc",
        }

        params = GenericPipeline.params_from_dss_vars(dss_vars)

        assert params["target_variables"] == ["lgd_before_flex", "mt_cal_ead_tot_b", "eir"]
        assert params["line_id_cols"] == ["id_technique", "facility_id"]
        assert params["target_agg_functions"] == {"t_agg1": "avg"}
        assert params["segment_vars"] == {"seg_agg2": "mt_cal_ead_tot_b"}
        assert params["weight_vars"] == {"t_agg5": "mt_cal_ead_tot_b"}

    def test_plain_string_keys_left_unchanged(self):
        dss_vars = {
            "axes_str": "entity/entity___asset_class",
            "pma_col": "cd_uds_pmas_metier_ap_acc",
            "entity_title": "entity",
        }

        params = GenericPipeline.params_from_dss_vars(dss_vars)

        assert params["axes_str"] == "entity/entity___asset_class"
        assert params["pma_col"] == "cd_uds_pmas_metier_ap_acc"
        assert params["entity_title"] == "entity"

    def test_scopes_key_parsed_if_string_encoded(self):
        dss_vars = {
            "scopes": "{'us': {'label': 'US', 'filters': {}}}",
        }

        params = GenericPipeline.params_from_dss_vars(dss_vars)

        assert params["scopes"] == {"us": {"label": "US", "filters": {}}}

    def test_scopes_key_left_alone_if_already_dict(self):
        """If caller already parsed scopes (not a string), don't double-parse."""
        dss_vars = {"scopes": {"us": {"label": "US", "filters": {}}}}

        params = GenericPipeline.params_from_dss_vars(dss_vars)

        assert params["scopes"] == {"us": {"label": "US", "filters": {}}}

    def test_flow_type_passed_through_unchanged(self):
        """flow_type is used as-is everywhere — no key renaming needed."""
        dss_vars = {"flow_type": "ccirc"}

        params = GenericPipeline.params_from_dss_vars(dss_vars)

        assert params["flow_type"] == "ccirc"

    def test_does_not_mutate_input_dict(self):
        dss_vars = {"target_variables": "['eir']"}

        GenericPipeline.params_from_dss_vars(dss_vars)

        assert dss_vars["target_variables"] == "['eir']"  # original untouched


# =============================================================================
# TEST: run_pre_processing accepts pre-loaded params (params_from_dss_vars path)
# =============================================================================

class TestRunPreProcessingWithInjectedParams:

    def test_uses_injected_params_instead_of_file(self, config_root, mock_prepare_agg, mock_add_perimeters_list):
        """
        When params= is provided, _load_params()/file reading must NOT
        be triggered — the DSS-vars-sourced params are used directly.
        """
        pipeline = GenericPipeline(
            spark_sess=object(), config_root=config_root,
            prepare_agg_period_axe_func=mock_prepare_agg,
            generate_business_axes_func=real_generate_business_axes,
            add_perimeters_list_func=mock_add_perimeters_list,
            period_const="period", table_path_const="table_path",
        )

        dss_vars = dict(CCIRC_PARAMS)
        dss_vars["target_variables"] = str(CCIRC_PARAMS["target_variables"])  # simulate DSS string-encoding
        injected_params = GenericPipeline.params_from_dss_vars(dss_vars)

        pipeline.run_pre_processing(
            flow_type="ccirc", axes_str_par="entity", table_path="hdfs://fem.orc",
            period="2026Q1", period_run="2026Q1", scope_run="risk_corp", id_set_params="v1",
            params=injected_params,
        )

        call = mock_prepare_agg.calls[0]
        assert call["t_vars"] == CCIRC_PARAMS["target_variables"]


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
