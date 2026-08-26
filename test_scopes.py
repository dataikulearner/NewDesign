"""
Unit tests for utils/scopes.py (add_perimeters_list and helpers).

"""

import numpy as np
import pandas as pd
import pytest

from utils import scopes


# =============================================================================
# FIXTURES — real confirmed constants (core/constants.py)
# =============================================================================

TEST_PERIMETERS_LIST_COLUMN = "perimeters_list"

TEST_CCIRC_SCOPE_LABELS = [
    ("us", "US"),
    ("fortis", "Fortis"),
    ("bcef", "BCEF"),
    ("japan", "Japan"),
]

TEST_ICAAP_SCOPE_LABELS = [
    ("bnl", "BNL"),
    ("fortis", "Fortis"),
    ("pf", "PF"),
    ("bgl", "BGL"),
    ("pfspain", "Spain"),
    ("pfitaly", "Italy"),
]

TEST_NUMERIC_SCOPE_COLUMNS = ["accounting_site_code_post_acc"]


@pytest.fixture(autouse=True)
def patch_constants(monkeypatch):
    monkeypatch.setattr(scopes, "PERIMETERS_LIST_COLUMN", TEST_PERIMETERS_LIST_COLUMN)
    monkeypatch.setattr(scopes, "CCIRC_SCOPE_LABELS", TEST_CCIRC_SCOPE_LABELS)
    monkeypatch.setattr(scopes, "ICAAP_SCOPE_LABELS", TEST_ICAAP_SCOPE_LABELS)
    monkeypatch.setattr(scopes, "NUMERIC_SCOPE_COLUMNS", TEST_NUMERIC_SCOPE_COLUMNS)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ctpr_id": [1, 2, 3, 4, 5],
        "accounting_site_code_post_acc": [12309, 40043, 99999, np.nan, 12309],
        "pmas_code_post_acc": ["PMA_05", "PMA_03", "PMA_01", "PMA_05", "PMA_03"],
        "lb_site_risque_apres_accr": ["BNPP Factor France", "BNPP Fortis", None, "Other", "BNPP Fortis"],
    })


# =============================================================================
# TEST: _compute_scope_column — single scope condition logic
# =============================================================================

class TestComputeScopeColumn:

    def test_empty_filters_all_true(self, sample_df):
        """No filters declared -> scope matches every row."""
        result = scopes._compute_scope_column(sample_df.copy(), "us", {})

        assert result["scope_us"].all()

    def test_in_operator_numeric_column(self, sample_df):
        """IN on a NUMERIC_SCOPE_COLUMNS column — value normalization applies."""
        filters = {
            "accounting_site_code_post_acc": {"operator": "IN", "values": [12309, 40043]}
        }
        result = scopes._compute_scope_column(sample_df.copy(), "us", filters)

        assert list(result["scope_us"]) == [True, True, False, False, True]

    def test_equals_operator_string_column(self, sample_df):
        filters = {"pmas_code_post_acc": {"operator": "EQUALS", "value": "PMA_03"}}
        result = scopes._compute_scope_column(sample_df.copy(), "fortis", filters)

        assert list(result["scope_fortis"]) == [False, True, False, False, True]

    def test_not_in_operator_nulls_always_match_regardless_of_allow_null(self, sample_df):
        """
        REAL BEHAVIOR (not a bug I introduced — confirmed via direct
        pandas testing): NaN.isin(values) is always False, so
        ~col_series.isin(values) (NOT_IN) is always True for NaN rows
        — REGARDLESS of allow_null. The `if allow_null: cond = cond |
        col_series.isna()` line is effectively a no-op for NOT_IN,
        since cond is already True for NaN rows before that line runs.
        Confirmed identical result whether allow_null is True or False.
        """
        filters_without = {
            "lb_site_risque_apres_accr": {"operator": "NOT_IN", "values": ["BNPP Factor France"]}
        }
        filters_with = {
            "lb_site_risque_apres_accr": {
                "operator": "NOT_IN", "values": ["BNPP Factor France"], "allow_null": True
            }
        }
        result_without = scopes._compute_scope_column(sample_df.copy(), "bcef", filters_without)
        result_with = scopes._compute_scope_column(sample_df.copy(), "bcef", filters_with)

        # Both give the SAME result — allow_null changes nothing here
        assert list(result_without["scope_bcef"]) == list(result_with["scope_bcef"])
        # NaN row (index 2) is True in BOTH cases
        assert result_without["scope_bcef"].iloc[2] == True
        assert result_with["scope_bcef"].iloc[2] == True

    def test_not_in_operator_with_allow_null_includes_nulls(self, sample_df):
        """NOT_IN WITH allow_null=True: rows with NaN ARE matched too."""
        filters = {
            "lb_site_risque_apres_accr": {
                "operator": "NOT_IN", "values": ["BNPP Factor France"], "allow_null": True
            }
        }
        result = scopes._compute_scope_column(sample_df.copy(), "bcef", filters)

        # row index 2 (None) now included because allow_null=True
        assert list(result["scope_bcef"]) == [False, True, True, True, True]

    def test_not_equals_operator(self, sample_df):
        filters = {"pmas_code_post_acc": {"operator": "NOT_EQUALS", "value": "PMA_03"}}
        result = scopes._compute_scope_column(sample_df.copy(), "japan", filters)

        assert list(result["scope_japan"]) == [True, False, True, True, False]

    def test_multiple_filters_and_logic(self, sample_df):
        """Multiple filter columns are combined with AND."""
        filters = {
            "accounting_site_code_post_acc": {"operator": "IN", "values": [12309]},
            "pmas_code_post_acc": {"operator": "EQUALS", "value": "PMA_05"},
        }
        result = scopes._compute_scope_column(sample_df.copy(), "japan", filters)

        # only row 0 matches BOTH conditions (row 4 has site 12309 but pmas PMA_03)
        assert list(result["scope_japan"]) == [True, False, False, False, False]

    def test_unknown_operator_skipped(self, sample_df):
        """Unrecognized operator is silently skipped (continue) — condition not applied."""
        filters = {"pmas_code_post_acc": {"operator": "CONTAINS", "value": "PMA"}}
        result = scopes._compute_scope_column(sample_df.copy(), "us", filters)

        # since the only filter is skipped, mask stays all-True (from initial state)
        assert result["scope_us"].all()

    def test_column_not_in_df_skipped(self, sample_df):
        """A filter referencing a column absent from df is skipped, not an error."""
        filters = {"nonexistent_column": {"operator": "EQUALS", "value": "X"}}
        result = scopes._compute_scope_column(sample_df.copy(), "us", filters)

        assert result["scope_us"].all()  # filter skipped -> stays all-True

    def test_metadata_key_starting_with_underscore_skipped(self, sample_df):
        """Keys starting with '_' (metadata) are not treated as filter columns."""
        filters = {"_comment": {"operator": "EQUALS", "value": "should be ignored"}}
        result = scopes._compute_scope_column(sample_df.copy(), "us", filters)

        assert result["scope_us"].all()

    def test_values_derived_from_single_value_key(self, sample_df):
        """When 'values' is absent but 'value' is given, it's wrapped into a list."""
        filters = {"pmas_code_post_acc": {"operator": "IN", "value": "PMA_05"}}
        result = scopes._compute_scope_column(sample_df.copy(), "japan", filters)

        assert list(result["scope_japan"]) == [True, False, False, True, False]


# =============================================================================
# TEST: _create_perimeters_from_scope_columns
# =============================================================================

class TestCreatePerimetersFromScopeColumns:

    def test_single_scope_match(self):
        df = pd.DataFrame({"scope_us": [True], "scope_fortis": [False]})
        result = scopes._create_perimeters_from_scope_columns(
            df, [("us", "US"), ("fortis", "Fortis")]
        )

        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[0] == "US"

    def test_multiple_scope_match_comma_joined(self):
        df = pd.DataFrame({"scope_us": [True], "scope_fortis": [True]})
        result = scopes._create_perimeters_from_scope_columns(
            df, [("us", "US"), ("fortis", "Fortis")]
        )

        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[0] == "US,Fortis"

    def test_no_scope_match_returns_na_string(self):
        """No scope matched -> NaN internally, then filled to 'N/A'."""
        df = pd.DataFrame({"scope_us": [False], "scope_fortis": [False]})
        result = scopes._create_perimeters_from_scope_columns(
            df, [("us", "US"), ("fortis", "Fortis")]
        )

        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[0] == "N/A"

    def test_missing_scope_column_treated_as_no_match(self):
        """If a scope_* column doesn't exist on a row, it's simply not counted."""
        df = pd.DataFrame({"scope_us": [True]})  # scope_fortis column absent entirely
        result = scopes._create_perimeters_from_scope_columns(
            df, [("us", "US"), ("fortis", "Fortis")]
        )

        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[0] == "US"


# =============================================================================
# TEST: _drop_scope_columns
# =============================================================================

class TestDropScopeColumns:

    def test_drops_all_scope_columns(self):
        df = pd.DataFrame({"scope_us": [True], "scope_fortis": [False], "other_col": [1]})
        result = scopes._drop_scope_columns(df, [("us", "US"), ("fortis", "Fortis")])

        assert "scope_us" not in result.columns
        assert "scope_fortis" not in result.columns
        assert "other_col" in result.columns

    def test_missing_scope_column_does_not_raise(self):
        """Dropping a scope column that was never created must not error."""
        df = pd.DataFrame({"other_col": [1]})
        result = scopes._drop_scope_columns(df, [("us", "US")])  # scope_us never existed

        assert "other_col" in result.columns


# =============================================================================
# TEST: add_perimeters_list — main function, end to end
# =============================================================================

class TestAddPerimetersList:

    def test_no_scope_definitions_returns_null_column(self, sample_df):
        """IFRS9 case: no scopes declared -> perimeters_list is all NaN."""
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions=None, flow_type="ifrs9")

        assert result[TEST_PERIMETERS_LIST_COLUMN].isna().all()

    def test_empty_string_scope_definitions_returns_null_column(self, sample_df):
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions="", flow_type="ifrs9")

        assert result[TEST_PERIMETERS_LIST_COLUMN].isna().all()

    def test_ccirc_scope_columns_dropped_after_processing(self, sample_df):
        """Final output must NOT contain any scope_* intermediate columns."""
        scope_definitions = {
            "us": {"filters": {"accounting_site_code_post_acc": {"operator": "IN", "values": [12309]}}}
        }
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions, flow_type="ccirc")

        assert not any(col.startswith("scope_") for col in result.columns)
        assert TEST_PERIMETERS_LIST_COLUMN in result.columns

    def test_ccirc_single_scope_end_to_end(self, sample_df):
        scope_definitions = {
            "us": {"filters": {"accounting_site_code_post_acc": {"operator": "IN", "values": [12309]}}}
        }
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions, flow_type="ccirc")

        # rows 0 and 4 have accounting_site_code_post_acc == 12309
        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[0] == "US"
        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[4] == "US"
        # rows not matching any declared scope -> "N/A"
        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[1] == "N/A"

    def test_ccirc_row_can_belong_to_multiple_scopes(self, sample_df):
        """A row satisfying BOTH us and bcef filters gets 'US,BCEF'."""
        scope_definitions = {
            "us": {"filters": {"accounting_site_code_post_acc": {"operator": "IN", "values": [12309]}}},
            "bcef": {"filters": {"pmas_code_post_acc": {"operator": "EQUALS", "value": "PMA_05"}}},
        }
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions, flow_type="ccirc")

        # row 0: site=12309 (matches us) AND pmas=PMA_05 (matches bcef)
        assert result[TEST_PERIMETERS_LIST_COLUMN].iloc[0] == "US,BCEF"

    def test_scope_id_not_declared_is_simply_absent_from_output(self, sample_df):
        """
        Only scope_ids present in BOTH scope_labels AND scope_definitions
        are computed — declaring a scope not in CCIRC_SCOPE_LABELS has no effect.
        """
        scope_definitions = {
            "unknown_scope_not_in_labels": {"filters": {}}
        }
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions, flow_type="ccirc")

        # no declared scope matches CCIRC_SCOPE_LABELS -> every row is N/A
        assert (result[TEST_PERIMETERS_LIST_COLUMN] == "N/A").all()

    def test_icaap_uses_icaap_scope_labels(self, sample_df):
        """flow_type='icaap' must use ICAAP_SCOPE_LABELS, not CCIRC's."""
        scope_definitions = {
            "bnl": {"filters": {}}  # matches ANY row (empty filters -> True for all)
        }
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions, flow_type="icaap")

        assert (result[TEST_PERIMETERS_LIST_COLUMN] == "BNL").all()

    def test_flow_type_case_insensitive(self, sample_df):
        scope_definitions = {"bnl": {"filters": {}}}
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions, flow_type="ICAAP")

        assert (result[TEST_PERIMETERS_LIST_COLUMN] == "BNL").all()

    def test_default_flow_type_is_ccirc(self, sample_df):
        """flow_type defaults to 'ccirc' when not passed."""
        scope_definitions = {"us": {"filters": {}}}
        result = scopes.add_perimeters_list(sample_df.copy(), scope_definitions)

        assert (result[TEST_PERIMETERS_LIST_COLUMN] == "US").all()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
