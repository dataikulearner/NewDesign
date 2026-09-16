import dataiku
import pandas as pd

df_agg_new = dataiku.Dataset("SP_agg_generic").get_dataframe()

# Lấy đúng 1 segment_id bị duplicate để dò ngược, ví dụ "entity|stage=ARVAL|2"
# Cần đối chiếu seg_axes / seg_axes_value gốc (không qua segment_id friendly-text)
check = df_agg_new[
    (df_agg_new["seg_axes"] == "entity|stage") &
    (df_agg_new["seg_axes_value"] == "ARVAL|2") &
    (df_agg_new["target_variable_clone"] == "provision3")
]
print("Số dòng tại SP_agg_generic cho segment này:", len(check))
print(check)

# Kiểm tra tổng quát: có bao nhiêu segment (seg_axes, seg_axes_value, target_variable_clone,
# target_variable_agg_name, period_clone) bị lặp NGAY TỪ SP_agg_generic
key_cols = ["seg_axes", "seg_axes_value", "target_variable_clone",
            "target_variable_agg_name", "period_clone"]
dup_at_source = df_agg_new[df_agg_new.duplicated(subset=key_cols, keep=False)]
print("\nSố dòng bị duplicate NGAY TỪ SP_agg_generic:", len(dup_at_source))
if len(dup_at_source) > 0:
    print(dup_at_source.sort_values(key_cols).head(10))
