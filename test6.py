import dataiku
import pandas as pd

df = dataiku.Dataset("SP_top_seg_generic").get_dataframe()
df_p3 = df[df["target_variable_clone"] == "provision3"]

# Lấy 1 segment bị duplicate bất kỳ
dup_segs = df_p3[df_p3.duplicated(keep=False)]
sample_seg = dup_segs.iloc[0]
seg_axes_val = sample_seg["seg_axes_value"]
seg_axes = sample_seg["seg_axes"]

print("Segment mẫu:", seg_axes, "=", seg_axes_val)

# Đếm CHÍNH XÁC có bao nhiêu dòng cho segment này
rows = df_p3[(df_p3["seg_axes"] == seg_axes) & (df_p3["seg_axes_value"] == seg_axes_val)]
print("Số dòng:", len(rows))
print(rows)

# Kiểm tra: các segment bị duplicate có đặc điểm CHUNG nào không?
# VD: cùng nằm trong seg_axes nào, cùng rank range nào, cùng period nào...
print("\nPhân phối seg_axes của các segment bị duplicate:")
print(dup_segs["seg_axes"].value_counts())

print("\nPhân phối rank của các segment bị duplicate (nếu có cột rank):")
if "rank" in dup_segs.columns:
    print(dup_segs["rank"].describe())
