import dataiku
import pandas as pd

AFFECTED_TARGET_VARS = ["provision3", "el_ratio", "eir", "ead_lgd"]
MATERIALITY_REL_TOLERANCE = 1e-6

df_seg_old = dataiku.Dataset("SP_top_seg").get_dataframe()
df_seg_new = dataiku.Dataset("SP_top_seg_generic").get_dataframe()

for tv in AFFECTED_TARGET_VARS:
    print("\n" + "=" * 70)
    print("TARGET VARIABLE: {}".format(tv))
    print("=" * 70)

    so = df_seg_old[df_seg_old["target_variable_clone"] == tv]
    sn = df_seg_new[df_seg_new["target_variable_clone"] == tv]

    key_old = so[["seg_axes", "seg_axes_value"]].drop_duplicates()
    key_new = sn[["seg_axes", "seg_axes_value"]].drop_duplicates()

    merged = pd.merge(key_old, key_new, on=["seg_axes", "seg_axes_value"], how="outer", indicator=True)
    only_old_keys = merged[merged["_merge"] == "left_only"][["seg_axes", "seg_axes_value"]]
    only_new_keys = merged[merged["_merge"] == "right_only"][["seg_axes", "seg_axes_value"]]

    only_old = pd.merge(only_old_keys, so, on=["seg_axes", "seg_axes_value"])
    only_new = pd.merge(only_new_keys, sn, on=["seg_axes", "seg_axes_value"])

    print("Số segment CHỈ có ở ancien: {} | CHỈ có ở nouveau: {}".format(len(only_old), len(only_new)))

    if only_old.empty and only_new.empty:
        print(">>> Không có khác biệt segment nào cho target_variable này <<<")
        continue

    # -----------------------------------------------------------------
    # Khớp cặp theo materiality GẦN BẰNG NHAU (không theo segment_id)
    # -----------------------------------------------------------------
    matched_pairs = []
    used_new_idx = set()

    for _, row_old in only_old.iterrows():
        m_old = row_old["materiality"]
        candidates = only_new[~only_new.index.isin(used_new_idx)].copy()
        if candidates.empty:
            continue
        candidates["rel_diff"] = (candidates["materiality"] - m_old).abs() / max(abs(m_old), 1e-12)
        best = candidates.nsmallest(1, "rel_diff")
        if not best.empty and best.iloc[0]["rel_diff"] < MATERIALITY_REL_TOLERANCE:
            matched_pairs.append({
                "seg_axes_old": row_old["seg_axes"], "seg_axes_value_old": row_old["seg_axes_value"],
                "rank_old": row_old["rank"], "materiality_old": m_old,
                "seg_axes_new": best.iloc[0]["seg_axes"], "seg_axes_value_new": best.iloc[0]["seg_axes_value"],
                "rank_new": best.iloc[0]["rank"], "materiality_new": best.iloc[0]["materiality"],
                "rel_diff": best.iloc[0]["rel_diff"],
            })
            used_new_idx.add(best.index[0])

    df_matched = pd.DataFrame(matched_pairs)
    n_matched = len(df_matched)
    pct_old = (n_matched / len(only_old) * 100) if len(only_old) > 0 else 0
    pct_new = (n_matched / len(only_new) * 100) if len(only_new) > 0 else 0

    print("\nSố cặp khớp được theo materiality (tie-swap): {}".format(n_matched))
    print("  -> chiếm {:.1f}% số dòng 'chỉ ancien', {:.1f}% số dòng 'chỉ nouveau'".format(pct_old, pct_new))

    if n_matched > 0:
        print("\nVí dụ các cặp khớp (xác nhận hoán đổi do đồng hạng materiality):")
        print(df_matched[["seg_axes_old", "seg_axes_value_old", "rank_old",
                           "seg_axes_new", "seg_axes_value_new", "rank_new", "rel_diff"]].head(10).to_string(index=False))

    # Phần KHÔNG khớp được -> cần điều tra riêng (không phải tie-swap)
    unmatched_old = only_old[~only_old.set_index(["seg_axes", "seg_axes_value"]).index.isin(
        pd.MultiIndex.from_frame(df_matched[["seg_axes_old", "seg_axes_value_old"]].rename(
            columns={"seg_axes_old": "seg_axes", "seg_axes_value_old": "seg_axes_value"})) if n_matched > 0 else [])]
    print("\nSố dòng 'chỉ ancien' KHÔNG khớp được (cần điều tra riêng):", len(unmatched_old))
    if not unmatched_old.empty:
        print(unmatched_old[["seg_axes", "seg_axes_value", "rank", "materiality"]].head(5).to_string(index=False))


print("\n" + "=" * 70)
print("KẾT LUẬN")
print("=" * 70)
print("""
- Nếu tỷ lệ khớp cặp (matched) chiếm PHẦN LỚN (VD >80%) số dòng khác biệt ở
  mỗi target_variable -> xác nhận đúng giả thuyết TIE-SWAP: hiện tượng đồng
  hạng (materiality bằng nhau do refinement suy biến giữa các seg_axes khác
  nhau) khiến thứ tự sort không ổn định giữa 2 lần chạy Spark, làm 1 trong 2
  segment đồng hạng bị đẩy qua/vào top-N. Đây là hệ quả CHẤP NHẬN ĐƯỢC của
  cùng nguyên nhân gốc (non-deterministic SUM), không phải lỗi logic mới.
- Phần "KHÔNG khớp được" cần điều tra riêng, có thể là khác biệt thật sự
  (không liên quan tie) hoặc do materiality không đủ gần để lọt tolerance.
""")
