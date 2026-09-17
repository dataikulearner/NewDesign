"""
Kiểm tra: 2 tổ hợp axes khác nhau (seg_axes A vs B) có thực sự cùng một tập
facility hay không — không chỉ cùng SỐ LƯỢNG, mà cùng đúng facility_id.

Đây là cách xác nhận chính xác hiện tượng "refinement suy biến" đã phân
tích trong tài liệu (§7.2/7.6): khi axe bổ sung không tách được facility
nào ra, materiality/deviation của 2 segment sẽ bằng nhau tuyệt đối.

Chạy ở cấp LINE (facility), không phải cấp SEG_AGG đã tổng hợp sẵn — vì
SP_agg/SP_anomalies chỉ có SUM/COUNT theo segment, không có danh sách
facility_id gốc.
"""

import dataiku
import pandas as pd

# =============================================================================
# CẤU HÌNH — điều chỉnh cho đúng
# =============================================================================

# Dataset cấp LINE (facility), trước khi bị gộp theo seg_axes — thường là
# input của compute_agg_by_period_axe (bước PRE), có 1 dòng/facility/period
LINE_DATASET = "SP_toplines_cumul"          # hoặc dataset PRE tương đương
FACILITY_ID_COL = "ctpr_id_generic"          # theo line_fac_col trong config
PERIOD_COL = "period_clone"
PERIOD_RUN = "2025Q4"

# 2 tổ hợp axes muốn so sánh — mỗi tổ hợp là 1 dict {tên cột: giá trị lọc}
AXES_A = {
    "entity": "IPS_PI",
    "migration_matrix": "CIB_EUROPE_DFT",
}
AXES_B = {
    "entity": "IPS_PI",
    "basel_approach_type_arc": "AS",
    "migration_matrix": "CIB_EUROPE_DFT",
}

# =============================================================================
# LOAD DỮ LIỆU CẤP FACILITY
# =============================================================================

df_line = dataiku.Dataset(LINE_DATASET).get_dataframe()
df_period = df_line[df_line[PERIOD_COL] == PERIOD_RUN]


def get_facility_set(df: pd.DataFrame, axes_filter: dict) -> set:
    """Lọc dataframe theo axes_filter (dict cột->giá trị), trả về tập
    facility_id thỏa mãn tất cả điều kiện."""
    mask = pd.Series(True, index=df.index)
    for col, val in axes_filter.items():
        mask &= (df[col] == val)
    return set(df[mask][FACILITY_ID_COL].unique())


fac_a = get_facility_set(df_period, AXES_A)
fac_b = get_facility_set(df_period, AXES_B)

print(f"Tổ hợp A ({AXES_A}): {len(fac_a)} facility")
print(f"Tổ hợp B ({AXES_B}): {len(fac_b)} facility")

# =============================================================================
# SO SÁNH — không chỉ số lượng, mà đúng từng facility_id
# =============================================================================

only_in_a = fac_a - fac_b
only_in_b = fac_b - fac_a
common = fac_a & fac_b

print(f"\nChỉ có trong A (không có trong B): {len(only_in_a)}")
print(f"Chỉ có trong B (không có trong A): {len(only_in_b)}")
print(f"Chung cả 2: {len(common)}")

if fac_a == fac_b:
    print("\n>>> XÁC NHẬN: 2 tổ hợp axes có ĐÚNG CÙNG một tập facility "
          "(refinement suy biến — axe bổ sung không tách được gì) <<<")
elif len(fac_a) == len(fac_b) and fac_a != fac_b:
    print("\n>>> CẢNH BÁO: 2 tổ hợp CÙNG SỐ LƯỢNG nhưng KHÁC facility_id — "
          "đây KHÔNG phải refinement suy biến thật, mà là trùng hợp ngẫu "
          "nhiên về đếm số lượng. Materiality có thể bằng nhau tình cờ "
          "(hoặc khác nhau), cần kiểm tra SUM(EAD) riêng, không suy ra "
          "tie từ số lượng facility. <<<")
    if only_in_a:
        print("  Ví dụ facility chỉ ở A:", list(only_in_a)[:5])
    if only_in_b:
        print("  Ví dụ facility chỉ ở B:", list(only_in_b)[:5])
else:
    print("\n>>> Số lượng khác nhau — 2 tổ hợp axes thực sự tách biệt "
          "facility, không phải refinement suy biến. <<<")

# =============================================================================
# BONUS — quét TỰ ĐỘNG toàn bộ cặp axes "cha-con" (A ⊂ B theo số cột) để
# tìm tất cả trường hợp refinement suy biến trong 1 entity, không cần chỉ
# định thủ công từng cặp
# =============================================================================

print("\n" + "=" * 70)
print("QUÉT TỰ ĐỘNG — tìm mọi cặp seg_axes suy biến cho 1 entity")
print("=" * 70)

ENTITY_CHECK = "IPS_PI"
# Yêu cầu: df_agg/SP_anomalies đã có seg_axes/seg_axes_value/COUNT facility
# (nếu có sẵn cột đếm distinct facility theo segment, ví dụ seg_agg1 = COUNT)
try:
    df_ano = dataiku.Dataset("SP_anomalies").get_dataframe()
    cand = df_ano[
        (df_ano[PERIOD_COL] == PERIOD_RUN) &
        (df_ano["seg_axes_value"].apply(lambda s: ENTITY_CHECK in str(s).split("|")))
    ][["seg_axes", "seg_axes_value", "seg_agg1"]].drop_duplicates()  # seg_agg1 giả định = COUNT facility

    # Với mỗi materiality-bucket giống nhau, kiểm tra facility set thật
    dup_count = cand.groupby("seg_agg1").filter(lambda g: len(g) > 1)
    print(f"Số nhóm có cùng COUNT facility (seg_agg1) nhưng seg_axes khác nhau: "
          f"{dup_count['seg_agg1'].nunique()}")
    print(dup_count.sort_values("seg_agg1").to_string(index=False))
    print("\n(Dùng get_facility_set ở trên để xác nhận từng cặp trong danh "
          "sách này có đúng facility set giống hệt nhau hay chỉ trùng số lượng)")
except Exception as e:
    print(f"Bỏ qua bước quét tự động (điều chỉnh tên dataset/cột nếu cần): {e}")
