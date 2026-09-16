import dataiku
import pandas as pd

# =============================================================================
# CẤU HÌNH
# =============================================================================

AFFECTED_TARGET_VARS = ["provision3", "el_ratio", "eir", "ead_lgd"]
RANK_CUTOFF = 30
RANK_AXES_CUTOFF = 10
NEAR_TIE_TOLERANCE = 1e-6

LINE_KEY_COLS = ["target_variable_clone", "segment_id", "ctpr_id_generic", "period_clone"]


# =============================================================================
# BƯỚC 1 — Xác nhận lại tổng quan: không còn duplicate ở cấp segment
# =============================================================================

df_seg_old = dataiku.Dataset("SP_top_seg").get_dataframe()
df_seg_new = dataiku.Dataset("SP_top_seg_generic").get_dataframe()

for tv in AFFECTED_TARGET_VARS:
    n_dup_old = df_seg_old[df_seg_old["target_variable_clone"] == tv].duplicated(
        subset=["seg_axes", "seg_axes_value"], keep=False).sum()
    n_dup_new = df_seg_new[df_seg_new["target_variable_clone"] == tv].duplicated(
        subset=["seg_axes", "seg_axes_value"], keep=False).sum()
    print("{}: duplicate ancien={}, nouveau={}".format(tv, n_dup_old, n_dup_new))

print()

# =============================================================================
# BƯỚC 2 — Đối chiếu lại số dòng ở cấp LINE (SP_display_toplines), tìm chính
# xác dòng nào khác biệt SAU KHI đã fix duplicate
# =============================================================================

df_line_old_full = dataiku.Dataset("SP_display_toplines").get_dataframe()
df_line_new_full = dataiku.Dataset("SP_display_toplines_generic").get_dataframe()

print("Tổng số dòng: ancien={}, nouveau={}".format(len(df_line_old_full), len(df_line_new_full)))

all_only_old = []
all_only_new = []

for tv in AFFECTED_TARGET_VARS:
    df_old = df_line_old_full[df_line_old_full["target_variable_clone"] == tv]
    df_new = df_line_new_full[df_line_new_full["target_variable_clone"] == tv]

    print("\n--- {} : ancien={} lignes, nouveau={} lignes (diff={}) ---".format(
        tv, len(df_old), len(df_new), len(df_new) - len(df_old)))

    key_old = df_old[LINE_KEY_COLS].drop_duplicates()
    key_new = df_new[LINE_KEY_COLS].drop_duplicates()

    only_in_old = pd.merge(key_old, key_new, on=LINE_KEY_COLS, how="left", indicator=True)
    only_in_old = only_in_old[only_in_old["_merge"] == "left_only"].drop(columns="_merge")
    only_in_old["target_variable_clone"] = tv

    only_in_new = pd.merge(key_new, key_old, on=LINE_KEY_COLS, how="left", indicator=True)
    only_in_new = only_in_new[only_in_new["_merge"] == "left_only"].drop(columns="_merge")
    only_in_new["target_variable_clone"] = tv

    print("  Dòng CHỈ có ở ancien (mất ở nouveau):", len(only_in_old))
    print("  Dòng CHỈ có ở nouveau (mới xuất hiện):", len(only_in_new))

    all_only_old.append(only_in_old)
    all_only_new.append(only_in_new)

df_only_old = pd.concat(all_only_old) if all_only_old else pd.DataFrame()
df_only_new = pd.concat(all_only_new) if all_only_new else pd.DataFrame()


# =============================================================================
# BƯỚC 3 — Với các segment_id có dòng khác biệt, truy ngược lên cấp SEGMENT
# để kiểm tra: dòng đó có nằm SÁT ranh giới rank_cutoff không (boundary-flip)
# =============================================================================

print("\n" + "=" * 70)
print("TRUY NGƯỢC LÊN CẤP SEGMENT — kiểm tra boundary-flip cho các dòng khác biệt")
print("=" * 70)

affected_segment_ids = pd.concat([
    df_only_old[["target_variable_clone", "segment_id"]] if not df_only_old.empty else pd.DataFrame(columns=["target_variable_clone", "segment_id"]),
    df_only_new[["target_variable_clone", "segment_id"]] if not df_only_new.empty else pd.DataFrame(columns=["target_variable_clone", "segment_id"]),
]).drop_duplicates()

print("Số segment_id duy nhất liên quan đến các dòng khác biệt:", len(affected_segment_ids))

for _, row in affected_segment_ids.iterrows():
    tv, seg_id = row["target_variable_clone"], row["segment_id"]

    seg_old = df_seg_old[
        (df_seg_old["target_variable_clone"] == tv) & (df_seg_old["segment_id"] == seg_id)
    ][["rank", "rank_axes", "materiality"]]
    seg_new = df_seg_new[
        (df_seg_new["target_variable_clone"] == tv) & (df_seg_new["segment_id"] == seg_id)
    ][["rank", "rank_axes", "materiality"]]

    print("\nSegment: {} | {}".format(tv, seg_id))
    print("  Ancien:", seg_old.to_dict("records"))
    print("  Nouveau:", seg_new.to_dict("records"))

    if not seg_old.empty and not seg_new.empty:
        r_old, r_new = seg_old.iloc[0], seg_new.iloc[0]
        kept_old = (r_old["rank"] <= RANK_CUTOFF) and (r_old["rank_axes"] <= RANK_AXES_CUTOFF)
        kept_new = (r_new["rank"] <= RANK_CUTOFF) and (r_new["rank_axes"] <= RANK_AXES_CUTOFF)
        if kept_old != kept_new:
            rel_diff = abs(r_old["materiality"] - r_new["materiality"]) / max(
                abs(r_old["materiality"]), abs(r_new["materiality"]), 1e-12)
            near_tie = rel_diff < NEAR_TIE_TOLERANCE
            print("  >>> LẬT TRẠNG THÁI: kept_old={}, kept_new={}, materiality_rel_diff={:.2e}, near_tie={} <<<".format(
                kept_old, kept_new, rel_diff, near_tie))
    elif seg_old.empty and not seg_new.empty:
        print("  >>> Segment CHỈ tồn tại ở nouveau (không có ở ancien) <<<")
    elif not seg_old.empty and seg_new.empty:
        print("  >>> Segment CHỈ tồn tại ở ancien (không có ở nouveau) <<<")


print("\n" + "=" * 70)
print("KẾT LUẬN CẦN RÚT RA")
print("=" * 70)
print("""
- Nếu phần lớn segment bị lật đều có near_tie=True -> xác nhận ĐÚNG giả
  thuyết boundary-flip do sai số rounding của Spark SUM(), vẫn CHẤP NHẬN
  ĐƯỢC về nghiệp vụ (không phải lỗi logic).
- Nếu segment "CHỈ tồn tại ở nouveau/ancien" (không phải lật rank mà là
  segment hoàn toàn mới/biến mất) -> cần kiểm tra thêm liệu segment đó có
  tồn tại ở CẤP INPUT (SP_agg_generic vs SP_agg) hay không, để loại trừ khả
  năng khác biệt xuất phát từ tầng aggregation, không phải post-processing.
""")
