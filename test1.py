import dataiku
import pandas as pd

df_new = dataiku.Dataset("SP_top_seg_generic").get_dataframe()
df_new_p3 = df_new[df_new["target_variable_clone"] == "provision3"]

# Tìm các segment_id bị duplicate
dup_ids = df_new_p3[df_new_p3.duplicated(subset=["segment_id"], keep=False)]["segment_id"].unique()
print("Số segment bị duplicate:", len(dup_ids))

# Với 1 segment bị duplicate, so sánh TOÀN BỘ cột giữa 2 bản sao để tìm cột nào khác nhau
sample_id = dup_ids[0]
rows = df_new_p3[df_new_p3["segment_id"] == sample_id]
print(rows.T)  # transpose để dễ soi từng cột theo hàng dọc

# Tự động liệt kê các cột có giá trị khác nhau giữa 2 dòng duplicate
if len(rows) == 2:
    r1, r2 = rows.iloc[0], rows.iloc[1]
    diff_cols = [c for c in rows.columns if not (r1[c] == r2[c] or (pd.isna(r1[c]) and pd.isna(r2[c])))]
    print("Các cột khác nhau giữa 2 bản duplicate:", diff_cols)
    for c in diff_cols:
        print(f"  {c}: {r1[c]!r} vs {r2[c]!r}")
