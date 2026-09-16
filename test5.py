import dataiku
import pandas as pd

# =============================================================================
# 0. LOAD DATASET ĐÃ CÓ SẴN — output thật của top_segment_par
# =============================================================================

df = dataiku.Dataset("SP_top_seg_generic").get_dataframe()

SEG_AXES = "seg_axes"
SEG_AXES_VALUE = "seg_axes_value"
TARGET_VARIABLE_CLONE = "target_variable_clone"
PIPE = "|"

TARGET_VARIABLES_TO_CHECK = ["provision3", "el_ratio", "eir", "ead_lgd"]


def analyze_target_variable(df_all: pd.DataFrame, target_var: str) -> None:
    print("\n" + "=" * 70)
    print("TARGET VARIABLE: {}".format(target_var))
    print("=" * 70)

    df_tv = df_all[df_all[TARGET_VARIABLE_CLONE] == target_var].copy()

    # ---------------------------------------------------------------
    # 1. Danh sách "tab_value" (entity code) — chính là các dòng có
    #    seg_axes == "entity" đã tồn tại sẵn trong dataset này
    # ---------------------------------------------------------------
    entity_codes = set(df_tv[df_tv[SEG_AXES] == "entity"][SEG_AXES_VALUE].unique())
    print("Số entity code tìm được (seg_axes == 'entity'):", len(entity_codes))

    # ---------------------------------------------------------------
    # 2. Xác định các dòng bị duplicate (toàn bộ cột giống hệt nhau)
    # ---------------------------------------------------------------
    dup_mask = df_tv.duplicated(keep=False)
    df_dup = df_tv[dup_mask]
    df_unique = df_tv[~dup_mask]

    n_dup_segments = df_dup[[SEG_AXES, SEG_AXES_VALUE]].drop_duplicates().shape[0]
    print("Số segment_id bị duplicate (giá trị 100% giống hệt nhau):", n_dup_segments)

    # ---------------------------------------------------------------
    # 3. Hàm đếm số entity_code khớp với các thành phần của seg_axes_value
    # ---------------------------------------------------------------
    def count_entity_matches(seg_axes_value: str) -> int:
        components = seg_axes_value.split(PIPE)
        return sum(1 for c in components if c in entity_codes)

    # ---------------------------------------------------------------
    # 4. Áp dụng cho NHÓM BỊ DUPLICATE
    # ---------------------------------------------------------------
    df_dup_unique_segs = df_dup[[SEG_AXES, SEG_AXES_VALUE]].drop_duplicates().copy()
    df_dup_unique_segs["n_entity_match"] = df_dup_unique_segs[SEG_AXES_VALUE].apply(count_entity_matches)

    n_multi_match_in_dup = (df_dup_unique_segs["n_entity_match"] > 1).sum()
    pct_multi_match_in_dup = (n_multi_match_in_dup / len(df_dup_unique_segs) * 100
                               if len(df_dup_unique_segs) > 0 else 0)
    print("\n[NHÓM BỊ DUPLICATE] Số segment có >1 entity_match: {} / {} ({:.1f}%)".format(
        n_multi_match_in_dup, len(df_dup_unique_segs), pct_multi_match_in_dup))

    # ---------------------------------------------------------------
    # 5. NHÓM ĐỐI CHỨNG — áp dụng cho segment KHÔNG bị duplicate, để
    #    xem tỷ lệ multi-match có thật sự đặc trưng cho nhóm bị lỗi hay
    #    xảy ra ngẫu nhiên ở cả segment bình thường (tránh false positive)
    # ---------------------------------------------------------------
    df_unique_segs = df_unique[[SEG_AXES, SEG_AXES_VALUE]].drop_duplicates().copy()
    df_unique_segs["n_entity_match"] = df_unique_segs[SEG_AXES_VALUE].apply(count_entity_matches)

    n_multi_match_in_normal = (df_unique_segs["n_entity_match"] > 1).sum()
    pct_multi_match_in_normal = (n_multi_match_in_normal / len(df_unique_segs) * 100
                                  if len(df_unique_segs) > 0 else 0)
    print("[NHÓM ĐỐI CHỨNG - không duplicate] Số segment có >1 entity_match: {} / {} ({:.1f}%)".format(
        n_multi_match_in_normal, len(df_unique_segs), pct_multi_match_in_normal))

    # ---------------------------------------------------------------
    # 6. Kết luận cho target_variable này
    # ---------------------------------------------------------------
    if pct_multi_match_in_dup > pct_multi_match_in_normal * 2 and n_multi_match_in_dup > 0:
        print("\n>>> GIẢ THUYẾT ĐƯỢC CỦNG CỐ: tỷ lệ multi-match ở nhóm duplicate "
              "cao hơn hẳn nhóm bình thường ({:.1f}% vs {:.1f}%) <<<".format(
                  pct_multi_match_in_dup, pct_multi_match_in_normal))
    elif n_dup_segments == 0:
        print("\n>>> Không có segment nào bị duplicate cho target_variable này <<<")
    else:
        print("\n>>> Tỷ lệ multi-match KHÔNG khác biệt rõ rệt giữa 2 nhóm — "
              "giả thuyết multi-match có thể KHÔNG phải nguyên nhân chính, "
              "cần điều tra hướng khác <<<")

    # In vài ví dụ cụ thể để xem thủ công
    if n_multi_match_in_dup > 0:
        print("\nVí dụ segment bị duplicate VÀ có multi-match entity:")
        print(df_dup_unique_segs[df_dup_unique_segs["n_entity_match"] > 1].head(5).to_string(index=False))


for tv in TARGET_VARIABLES_TO_CHECK:
    analyze_target_variable(df, tv)
