import dataiku
import pandas as pd

# SP_anomalies là output của zone MODELING (sau compute_SP_agg_focus_model +
# compute_SP_anomalies) — đây mới là nơi có cột anomaly/deviation, không phải
# SP_agg (chỉ là output PRE, thuần aggregation).
df_ano_old = dataiku.Dataset("SP_anomalies").get_dataframe()
df_ano_new = dataiku.Dataset("SP_anomalies_generic").get_dataframe()

SEG_AXES_CHECK = "entity|migration_matrix"
SEG_AXES_VALUE_CHECK = "IPS_PI|CIB_EUROPE_DFT"
TARGET_VAR = "provision3"

cols_to_check = ["period_clone", "model_type", "anomaly", "deviation",
                  "observed_value", "expected_value", "materiality"]

seg_old = df_ano_old[
    (df_ano_old["seg_axes"] == SEG_AXES_CHECK) &
    (df_ano_old["seg_axes_value"] == SEG_AXES_VALUE_CHECK) &
    (df_ano_old["t_variable_clone"] == TARGET_VAR)
]
cols_present_old = [c for c in cols_to_check if c in seg_old.columns]
print("=== ANCIEN (SP_anomalies) ===")
print(seg_old[cols_present_old].sort_values("period_clone").to_string(index=False))

seg_new = df_ano_new[
    (df_ano_new["seg_axes"] == SEG_AXES_CHECK) &
    (df_ano_new["seg_axes_value"] == SEG_AXES_VALUE_CHECK) &
    (df_ano_new["t_variable_clone"] == TARGET_VAR)
]
cols_present_new = [c for c in cols_to_check if c in seg_new.columns]
print("\n=== NOUVEAU (SP_anomalies_generic) ===")
print(seg_new[cols_present_new].sort_values("period_clone").to_string(index=False))

# So sánh trực tiếp theo period_clone để thấy khác biệt cụ thể (nếu có)
print("\n=== SO SÁNH THEO period_clone ===")
merged = pd.merge(
    seg_old[["period_clone"] + [c for c in cols_present_old if c != "period_clone"]],
    seg_new[["period_clone"] + [c for c in cols_present_new if c != "period_clone"]],
    on="period_clone", how="outer", suffixes=("_old", "_new"), indicator=True
)
print(merged.to_string(index=False))

print("\nKỳ chỉ có ở ancien:", merged[merged["_merge"] == "left_only"]["period_clone"].tolist())
print("Kỳ chỉ có ở nouveau:", merged[merged["_merge"] == "right_only"]["period_clone"].tolist())
print("Kỳ có ở cả 2 nhưng anomaly/deviation khác nhau:")
both = merged[merged["_merge"] == "both"]
if "anomaly_old" in both.columns and "anomaly_new" in both.columns:
    diff_anomaly = both[both["anomaly_old"] != both["anomaly_new"]]
    print(diff_anomaly[["period_clone", "anomaly_old", "anomaly_new"]].to_string(index=False)
          if not diff_anomaly.empty else "  (không có khác biệt anomaly)")
