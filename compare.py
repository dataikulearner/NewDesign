import dataiku
import pandas as pd

df_old = dataiku.Dataset("SP_agg").get_dataframe()
df_new = dataiku.Dataset("SP_agg_generic").get_dataframe()

print("=== 1. So sánh shape ===")
print(f"SP_agg (cũ):        {df_old.shape}")
print(f"SP_agg_generic (mới): {df_new.shape}")

print("\n=== 2. So sánh tập cột ===")
cols_old, cols_new = set(df_old.columns), set(df_new.columns)
print(f"Cột chỉ có ở bản CŨ:  {cols_old - cols_new}")
print(f"Cột chỉ có ở bản MỚI: {cols_new - cols_old}")

print("\n=== 3. So sánh dòng — key để join ===")
# Dùng id_technique + period + axe làm key join (điều chỉnh nếu key khác)
key_cols = ["id_technique", "period", "axe_clone"]  # ⚠️ chỉnh lại đúng key thật của SP_agg

df_old_sorted = df_old.sort_values(key_cols).reset_index(drop=True)
df_new_sorted = df_new.sort_values(key_cols).reset_index(drop=True)

common_cols = sorted(cols_old & cols_new)
merged = df_old_sorted[key_cols].merge(
    df_new_sorted[key_cols], on=key_cols, how="outer", indicator=True
)
print(f"Dòng chỉ có ở bản CŨ:  {(merged['_merge']=='left_only').sum()}")
print(f"Dòng chỉ có ở bản MỚI: {(merged['_merge']=='right_only').sum()}")
print(f"Dòng khớp cả 2 bên:    {(merged['_merge']=='both').sum()}")

print("\n=== 4. So sánh giá trị số — cho các cột số chung ===")
numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_old[c])]

df_old_indexed = df_old.set_index(key_cols)[numeric_cols]
df_new_indexed = df_new.set_index(key_cols)[numeric_cols]

diff = (df_old_indexed - df_new_indexed).abs()
tolerance = 1e-6   # sai số float chấp nhận được

for col in numeric_cols:
    max_diff = diff[col].max()
    status = "✅ KHỚP" if max_diff < tolerance else f"❌ LỆCH (max diff = {max_diff})"
    print(f"  {col}: {status}")
