import dataiku
import pandas as pd

# =============================================================================
# CẤU HÌNH — điều chỉnh cho đúng giá trị thật
# =============================================================================

TARGET_VAR = "provision3"
PERIOD_RUN = "2025Q4"        # đúng period_run thật (theo variables['local'])
SCOPE_RUN = "SP1"            # đúng scope_run thật (đã quan sát scope_clone = SP1)
MODEL_TYPE = "basic"         # top_segment_par lọc theo model_type trước tiên
T_AGG_NAME = "agg4"          # t_agg_name_mapping["provision3"] = "agg4" (theo config)

ENTITY_CHECK = "IPS_PI"

# =============================================================================
# LOAD ĐÚNG NGUỒN — SP_anomalies (có sẵn cột materiality), KHÔNG phải SP_agg
# =============================================================================

df_ano_old = dataiku.Dataset("SP_anomalies").get_dataframe()
df_ano_new = dataiku.Dataset("SP_generic_anomalies").get_dataframe()


def get_reference_candidates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Tái hiện đúng các bước filter đầu tiên của top_segment_par trước khi
    tìm tab_values/candidates cho nhánh df_ref (reference theo entity)."""
    d = df[
        (df["model_type"] == MODEL_TYPE) &
        (df["period_clone"] == PERIOD_RUN) &
        (df["scope_clone"] == SCOPE_RUN) &
        (df["t_variable_clone"] == TARGET_VAR)
    ]

    # Điều kiện query_reference thật: "seg_axes == 'entity'" — dùng để lấy
    # danh sách tab_values, KHÔNG áp dụng ở đây vì ta muốn xem TẤT CẢ ứng
    # viên (mọi seg_axes) chứa entity này, giống đúng logic vòng lặp gốc:
    #   df[df[SEG_AXES_VALUE].apply(lambda s: tab_value in s.split(pipe))]
    cand = d[d["seg_axes_value"].apply(
        lambda s: ENTITY_CHECK in str(s).split("|")
    )][["seg_axes", "seg_axes_value", "materiality"]].sort_values(
        "materiality", ascending=False
    ).reset_index(drop=True)

    print(f"\n=== TOP ứng viên reference cho entity '{ENTITY_CHECK}' — {label} ===")
    print(f"Tổng số ứng viên: {len(cand)}")
    cand_display = cand.head(15).copy()
    cand_display.index = cand_display.index + 1  # rank 1-based cho dễ đọc
    print(cand_display.to_string())
    return cand


cand_old = get_reference_candidates(df_ano_old, "ANCIEN")
cand_new = get_reference_candidates(df_ano_new, "NOUVEAU")

# =============================================================================
# SO SÁNH VÙNG RANH GIỚI TOP-10 (thresholds_segment_materiality.reference.top)
# =============================================================================

TOP_N_REFERENCE = 10  # theo "reference": {"top": 10} trong ccirc_params

print(f"\n=== Vùng ranh giới top-{TOP_N_REFERENCE} (hạng {TOP_N_REFERENCE - 2} "
      f"đến {TOP_N_REFERENCE + 3}) ===")

print("\nANCIEN:")
boundary_old = cand_old.iloc[TOP_N_REFERENCE - 3: TOP_N_REFERENCE + 3].copy()
boundary_old.index = boundary_old.index + 1
print(boundary_old.to_string())

print("\nNOUVEAU:")
boundary_new = cand_new.iloc[TOP_N_REFERENCE - 3: TOP_N_REFERENCE + 3].copy()
boundary_new.index = boundary_new.index + 1
print(boundary_new.to_string())

# Kiểm tra: segment đang điều tra (entity|migration_matrix=IPS_PI|CIB_EUROPE_DFT)
# nằm ở vị trí nào trong danh sách của mỗi bên
SEG_AXES_CHECK = "entity|migration_matrix"
SEG_AXES_VALUE_CHECK = "IPS_PI|CIB_EUROPE_DFT"

for label, cand in [("ANCIEN", cand_old), ("NOUVEAU", cand_new)]:
    match = cand[(cand["seg_axes"] == SEG_AXES_CHECK) &
                (cand["seg_axes_value"] == SEG_AXES_VALUE_CHECK)]
    if not match.empty:
        pos = match.index[0] + 1
        print(f"\n{label}: segment '{SEG_AXES_VALUE_CHECK}' ở vị trí #{pos} "
              f"trong danh sách candidates (top-{TOP_N_REFERENCE} = {'OUI' if pos <= TOP_N_REFERENCE else 'NON'})")
    else:
        print(f"\n{label}: segment '{SEG_AXES_VALUE_CHECK}' KHÔNG có trong danh sách candidates")
