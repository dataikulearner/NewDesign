df_p3 = df[df["t_variable_clone"] == "provision3"]
total_unique_segments = df_p3[["seg_axes", "seg_axes_value"]].drop_duplicates().shape[0]
total_rows = len(df_p3)
print("Tổng số segment DUY NHẤT:", total_unique_segments)
print("Tổng số DÒNG (bao gồm duplicate):", total_rows)
print("Tỷ lệ dòng dư ra:", (total_rows - total_unique_segments) / total_unique_segments * 100, "%")
