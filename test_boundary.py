import dataiku
import pandas as pd
import numpy as np

# =============================================================================
# CẤU HÌNH — chỉnh lại tên dataset và target_variable cho đúng dự án
# =============================================================================

# Dataset ở cấp SEGMENT (output của top_segment_par, có cột rank/materiality)
# -> thường là SP_top_seg trong pipeline cũ, và tương đương trong pipeline generic
SEG_DATASET_OLD = "SP_top_seg"
SEG_DATASET_NEW = "SP_top_seg_generic"

# Dataset ở cấp LINE (output cuối, dùng để đếm số dòng như trong bảng bạn đã thấy)
LINE_DATASET_OLD = "SP_display_toplines"
LINE_DATASET_NEW = "SP_display_toplines_generic"

# Danh sách 4 target_variable bị lệch số dòng theo quan sát thực tế
AFFECTED_TARGET_VARS = ["provision3", "el_ratio", "eir", "ead_lgd"]

# Ngưỡng rank cắt (lấy từ threshols_segment_materiality[target_variable]["rank"]
# trong ccirc_params.json, ví dụ "(rank <= 30) & (rank_axes <= 10)")
RANK_CUTOFF = 30
RANK_AXES_CUTOFF = 10

# Cửa sổ quanh ranh giới để soi các dòng "cận biên" (rank 25 -> 35)
BOUNDARY_WINDOW = 5

# Ngưỡng coi là "gần bằng nhau" giữa 2 giá trị materiality (tương đối)
NEAR_TIE_TOLERANCE = 1e-6


# =============================================================================
# PHẦN A — Kiểm tra hiện tượng lật ranh giới ở cấp SEGMENT
# =============================================================================

def load_segment_rank_data(dataset_name: str, target_var: str) -> pd.DataFrame:
    """Đọc dataset cấp segment, lọc theo target_variable, giữ các cột cần thiết."""
    df = dataiku.Dataset(dataset_name).get_dataframe()
    df = df[df["target_variable_clone"] == target_var].copy()
    return df[["segment_id", "seg_axes", "seg_axes_value", "rank", "rank_axes", "materiality"]]


def find_boundary_flips(df_old: pd.DataFrame, df_new: pd.DataFrame, target_var: str) -> pd.DataFrame:
    """So sánh trạng thái 'được giữ hay bị loại' (theo RANK_CUTOFF) của từng segment_id
    giữa 2 pipeline, chỉ ra các segment bị lật trạng thái + mức chênh lệch materiality
    tại thời điểm lật (nếu materiality gần bằng nhau -> xác nhận boundary-flip).
    """
    df_old = df_old.copy()
    df_new = df_new.copy()
    df_old["kept_old"] = (df_old["rank"] <= RANK_CUTOFF) & (df_old["rank_axes"] <= RANK_AXES_CUTOFF)
    df_new["kept_new"] = (df_new["rank"] <= RANK_CUTOFF) & (df_new["rank_axes"] <= RANK_AXES_CUTOFF)

    merged = pd.merge(
        df_old[["segment_id", "rank", "materiality", "kept_old"]],
        df_new[["segment_id", "rank", "materiality", "kept_new"]],
        on="segment_id", how="outer", suffixes=("_old", "_new")
    )

    # Segment nào bị lật trạng thái giữa 2 pipeline (được giữ ở bên này nhưng bị loại ở bên kia)
    flips = merged[merged["kept_old"].fillna(False) != merged["kept_new"].fillna(False)].copy()

    if flips.empty:
        print("  -> Không có segment nào bị lật trạng thái cho '{}'".format(target_var))
        return flips

    # Tính độ lệch tương đối materiality để xác nhận có phải "cận biên" không
    flips["materiality_rel_diff"] = (
        (flips["materiality_old"] - flips["materiality_new"]).abs() /
        flips[["materiality_old", "materiality_new"]].abs().max(axis=1).replace(0, np.nan)
    )
    flips["is_near_tie"] = flips["materiality_rel_diff"] < NEAR_TIE_TOLERANCE

    print("  -> {} segment(s) bị lật trạng thái cho '{}', trong đó {} là near-tie (materiality lệch < {})".format(
        len(flips), target_var, flips["is_near_tie"].sum(), NEAR_TIE_TOLERANCE))
    return flips.sort_values("materiality_rel_diff")


print("=" * 70)
print("PHẦN A — Kiểm tra lật ranh giới rank ở cấp SEGMENT")
print("=" * 70)

all_flips = {}
for target_var in AFFECTED_TARGET_VARS:
    print("\n--- Target variable: {} ---".format(target_var))
    df_seg_old = load_segment_rank_data(SEG_DATASET_OLD, target_var)
    df_seg_new = load_segment_rank_data(SEG_DATASET_NEW, target_var)
    flips = find_boundary_flips(df_seg_old, df_seg_new, target_var)
    all_flips[target_var] = flips
    if not flips.empty:
        print(flips[["segment_id", "rank_old", "rank_new", "materiality_old",
                      "materiality_new", "materiality_rel_diff", "is_near_tie"]].head(10).to_string(index=False))


# =============================================================================
# PHẦN B — Soi trực tiếp các dòng nằm quanh ranh giới rank (rank 25-35)
# dù chưa bị lật hẳn, để thấy mức độ "mong manh" của các segment cận biên
# =============================================================================

print("\n" + "=" * 70)
print("PHẦN B — Các segment nằm trong cửa sổ cận biên (rank {}-{})".format(
    RANK_CUTOFF - BOUNDARY_WINDOW, RANK_CUTOFF + BOUNDARY_WINDOW))
print("=" * 70)

for target_var in AFFECTED_TARGET_VARS:
    print("\n--- Target variable: {} ---".format(target_var))
    df_seg_old = load_segment_rank_data(SEG_DATASET_OLD, target_var)
    df_seg_new = load_segment_rank_data(SEG_DATASET_NEW, target_var)

    boundary_old = df_seg_old[
        (df_seg_old["rank"] >= RANK_CUTOFF - BOUNDARY_WINDOW) &
        (df_seg_old["rank"] <= RANK_CUTOFF + BOUNDARY_WINDOW)
    ].sort_values("rank")
    boundary_new = df_seg_new[
        (df_seg_new["rank"] >= RANK_CUTOFF - BOUNDARY_WINDOW) &
        (df_seg_new["rank"] <= RANK_CUTOFF + BOUNDARY_WINDOW)
    ].sort_values("rank")

    print("  Ancien (autour de rank={}):".format(RANK_CUTOFF))
    print(boundary_old[["segment_id", "rank", "materiality"]].to_string(index=False))
    print("  Nouveau (autour de rank={}):".format(RANK_CUTOFF))
    print(boundary_new[["segment_id", "rank", "materiality"]].to_string(index=False))


# =============================================================================
# PHẦN C — Đối chiếu số dòng ở cấp LINE (SP_display_toplines), giống bảng
# bạn đã thấy (4026 vs 4039 v.v.), rồi liệt kê chính xác dòng nào bị thêm/bớt
# =============================================================================

print("\n" + "=" * 70)
print("PHẦN C — Đối chiếu chi tiết dòng bị thêm/bớt ở cấp LINE")
print("=" * 70)

df_line_old_full = dataiku.Dataset(LINE_DATASET_OLD).get_dataframe()
df_line_new_full = dataiku.Dataset(LINE_DATASET_NEW).get_dataframe()

# Cần một khóa định danh duy nhất cho từng dòng để so sánh (điều chỉnh theo
# line_id_cols thực tế trong config, ví dụ ["id_technique", "facility_id", "ctpr_id"])
LINE_KEY_COLS = ["target_variable_clone", "segment_id", "ctpr_id_generic", "period_clone"]

for target_var in AFFECTED_TARGET_VARS:
    df_old = df_line_old_full[df_line_old_full["target_variable_clone"] == target_var]
    df_new = df_line_new_full[df_line_new_full["target_variable_clone"] == target_var]

    print("\n--- {} : ancien={} lignes, nouveau={} lignes ---".format(
        target_var, len(df_old), len(df_new)))

    key_old = df_old[LINE_KEY_COLS].drop_duplicates()
    key_new = df_new[LINE_KEY_COLS].drop_duplicates()

    only_in_old = pd.merge(key_old, key_new, on=LINE_KEY_COLS, how="left", indicator=True)
    only_in_old = only_in_old[only_in_old["_merge"] == "left_only"].drop(columns="_merge")

    only_in_new = pd.merge(key_new, key_old, on=LINE_KEY_COLS, how="left", indicator=True)
    only_in_new = only_in_new[only_in_new["_merge"] == "left_only"].drop(columns="_merge")

    print("  Dòng CHỈ có ở ancien (bị mất ở nouveau): {}".format(len(only_in_old)))
    if not only_in_old.empty:
        print(only_in_old.head(10).to_string(index=False))

    print("  Dòng CHỈ có ở nouveau (mới xuất hiện): {}".format(len(only_in_new)))
    if not only_in_new.empty:
        print(only_in_new.head(10).to_string(index=False))


print("\n" + "=" * 70)
print("KẾT LUẬN CẦN RÚT RA TỪ 3 PHẦN TRÊN")
print("=" * 70)
print("""
- Nếu Phần A cho thấy các segment bị lật đều có materiality_rel_diff < 1e-6
  (is_near_tie = True) -> xác nhận đúng giả thuyết boundary-flip do sai số
  rounding của Spark SUM(), KHÔNG phải lỗi logic.
- Nếu có segment bị lật với materiality_rel_diff LỚN (không near-tie) ->
  đây là dấu hiệu của một vấn đề khác (không phải rounding), cần điều tra
  riêng logic tính materiality/rank giữa 2 pipeline.
- Phần C giúp xác định chính xác facility/dòng nào biến mất hoặc xuất hiện
  thêm, để đối chiếu ngược lại với Phần A/B xem chúng có nằm trong cùng
  segment bị lật hay không.
""")
