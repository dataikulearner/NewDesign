import dataiku
import pandas as pd

df_agg_old = dataiku.Dataset("SP_agg").get_dataframe()
df_agg_new = dataiku.Dataset("SP_agg_generic").get_dataframe()

PERIOD_RUN = "2025Q4"  # điều chỉnh đúng period_run thật
TARGET_VAR = "provision3"

# Lấy TOÀN BỘ segment liên quan đến entity IPS_PI (mọi seg_axes), period_run,
# sắp xếp theo materiality giảm dần — đây chính là "ứng viên" tranh top-10
def get_reference_candidates(df, label):
    cand = df[
        (df["t_variable_clone"] == TARGET_VAR) &
        (df["period_clone"] == PERIOD_RUN) &
        (df["seg_axes_value"].apply(lambda s: "IPS_PI" in s.split("|")))
    ][["seg_axes", "seg_axes_value", "materiality"]].sort_values("materiality", ascending=False)
    print(f"\n=== TOP ứng viên reference — {label} ===")
    print(cand.head(15).to_string(index=False))
    return cand

cand_old = get_reference_candidates(df_agg_old, "ANCIEN")
cand_new = get_reference_candidates(df_agg_new, "NOUVEAU")
