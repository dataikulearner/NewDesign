"""
Unit tests for ColumnRegistry 
"""

import pytest

from generics.registries.column_registry import (
    ColumnRegistry,
    ValidationResult,
    ConfigurationError,
    RegistryError,
    startup_validation,
)


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_REGISTRY = {
    "version": "2.0.0",
    "source_tables": {
        "FEM": {"join_key": "id_contrat"},
        "TIERS": {"join_key": "id_contrat"},
    },
    "columns": {
        "accounting_site_code_post_acc": {
            "source_table": "FEM",
            "data_type": "string",
            "nullable": False,
        },
        "pmas_code_post_acc": {
            "source_table": "FEM",
            "data_type": "string",
            "nullable": True,
        },
        "asset_class": {
            "source_table": "FEM",
            "data_type": "string",
            "nullable": True,
        },
        "business_country": {
            "source_table": "TIERS",
            "data_type": "string",
            "nullable": True,
        },
    },
    "operators": {
        "EQUALS": {"supported_types": ["string", "integer"], "requires_array": False},
        "IN": {"supported_types": ["string", "integer"], "requires_array": True},
    },
}

VALID_SCOPE_CONFIG = {
    "scopes": {
        "risk_corp": {
            "filters": {
                "pmas_code_post_acc": {"operator": "EQUALS", "value": "PMA_05"},
                "asset_class": {"operator": "EQUALS", "value": "CORP_L"},
            }
        }
    }
}

INVALID_SCOPE_CONFIG_UNKNOWN_COLUMN = {
    "scopes": {
        "test_scope": {
            "filters": {"unknown_column": {"operator": "EQUALS", "value": "X"}}
        }
    }
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset Singleton before/after each test — avoids cross-test leakage."""
    ColumnRegistry.reset()
    yield
    ColumnRegistry.reset()


@pytest.fixture
def registry():
    return ColumnRegistry.load_from_dict(SAMPLE_REGISTRY)


# =============================================================================
# TEST: VALIDATION RESULT
# =============================================================================

class TestValidationResult:

    def test_initial_state_is_valid(self):
        result = ValidationResult()
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_marks_invalid(self):
        result = ValidationResult()
        result.add_error("Test error")

        assert not result.is_valid
        assert result.errors == ["Test error"]

    def test_add_warning_keeps_valid(self):
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid
        assert len(result.warnings) == 1

    def test_raise_if_invalid_raises(self):
        result = ValidationResult()
        result.add_error("Test error")

        with pytest.raises(ConfigurationError):
            result.raise_if_invalid()

    def test_raise_if_invalid_passes_when_valid(self):
        result = ValidationResult()
        result.add_warning("Just a warning")

        result.raise_if_invalid()  # should not raise

    def test_merge_results(self):
        result1 = ValidationResult()
        result1.add_error("Error 1")

        result2 = ValidationResult()
        result2.add_warning("Warning 1")

        result1.merge(result2)

        assert not result1.is_valid
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1


# =============================================================================
# TEST: COLUMN REGISTRY
# =============================================================================

class TestColumnRegistry:

    def test_load_from_dict(self, registry):
        assert len(registry.all()) == 4

    def test_singleton_pattern(self, registry):
        registry2 = ColumnRegistry()
        assert registry is registry2

    def test_get_existing_column(self, registry):
        meta = registry.get("asset_class")

        assert meta is not None
        assert meta["source_table"] == "FEM"
        assert meta["data_type"] == "string"

    def test_get_nonexistent_column(self, registry):
        assert registry.get("unknown_column") is None

    def test_has_column(self, registry):
        assert registry.has("asset_class")
        assert not registry.has("unknown_column")

    def test_get_for_flow_returns_all_declared_columns(self, registry):
        """
        v3.2: get_for_flow() ignores its flow argument entirely —
        this registry belongs to exactly one flow (one file per
        project), so every declared column IS for that flow.
        """
        cols_a = registry.get_for_flow("ccirc")
        cols_b = registry.get_for_flow("anything_else")

        assert "asset_class" in cols_a
        assert cols_a == cols_b  # flow argument has no effect
        assert len(cols_a) == 4  # all columns in the registry

    def test_get_by_source(self, registry):
        fem_cols = registry.get_by_source("FEM")
        tiers_cols = registry.get_by_source("TIERS")

        assert "asset_class" in fem_cols
        assert "business_country" in tiers_cols
        assert "business_country" not in fem_cols

    def test_validate_valid_config(self, registry):
        result = registry.validate_scope_config(VALID_SCOPE_CONFIG, "ccirc")

        assert result.is_valid

    def test_validate_unknown_column(self, registry):
        result = registry.validate_scope_config(
            INVALID_SCOPE_CONFIG_UNKNOWN_COLUMN, "ccirc"
        )

        assert not result.is_valid
        assert any("not found" in e for e in result.errors)


# =============================================================================
# TEST: STARTUP VALIDATION (module-level fail-fast helper)
# =============================================================================

class TestStartupValidation:

    def test_valid_config_passes(self, registry):
        startup_validation(registry, VALID_SCOPE_CONFIG, "ccirc")  # should not raise

    def test_invalid_config_raises(self, registry):
        with pytest.raises(ConfigurationError):
            startup_validation(registry, INVALID_SCOPE_CONFIG_UNKNOWN_COLUMN, "ccirc")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
