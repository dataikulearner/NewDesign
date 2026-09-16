import dataiku
import pandas as pd

df_new = dataiku.Dataset("SP_top_seg_generic").get_dataframe()

# 1. Kiểm tra whitespace ẩn trong seg_axes_value khi seg_axes == "entity"
entity_rows = df_new[df_new["seg_axes"] == "entity"]
tab_values = entity_rows["seg_axes_value"].unique()
stripped_values = set(v.strip() for v in tab_values)

print("Số tab_value distinct (chưa strip):", len(tab_values))
print("Số tab_value distinct (đã strip):", len(stripped_values))

if len(tab_values) != len(stripped_values):
    print("\n>>> XÁC NHẬN: có whitespace ẩn gây trùng logic <<<")
    # In ra repr() để thấy rõ khoảng trắng thừa
    for v in tab_values:
        if v.strip() in stripped_values and v != v.strip():
            print("  Lệch whitespace:", repr(v))

# 2. Đối chiếu: các segment_id bị duplicate có nằm trong nhóm entity bị lệch whitespace không?
dup_ids = df_new[df_new.duplicated(subset=["segment_id"], keep=False)]["segment_id"].unique()
print("\nSố segment bị duplicate:", len(dup_ids))
print("Ví dụ 5 segment_id bị duplicate:", dup_ids[:5])
