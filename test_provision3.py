import dataiku

df_agg_old = dataiku.Dataset("SP_agg").get_dataframe()
df_agg_new = dataiku.Dataset("SP_agg_generic").get_dataframe()

seg = df_agg_old[
    (df_agg_old["seg_axes"] == "entity|migration_matrix") &
    (df_agg_old["seg_axes_value"] == "IPS_PI|CIB_EUROPE_DFT") &
    (df_agg_old["target_variable_clone"] == "provision3")
][["period_clone", "anomaly", "t_agg4", "deviation"]]  # cột agg dùng cho provision3

print("Ancien:\n", seg)

seg_new = df_agg_new[
    (df_agg_new["seg_axes"] == "entity|migration_matrix") &
    (df_agg_new["seg_axes_value"] == "IPS_PI|CIB_EUROPE_DFT") &
    (df_agg_new["target_variable_clone"] == "provision3")
][["period_clone", "anomaly", "t_agg4", "deviation"]]

print("Nouveau:\n", seg_new)
