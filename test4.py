import dataiku, json
import pandas as pd
from core.post_processing import top_segment_par
from core.constants import (SEG_AXES, SEG_AXES_VALUE, TARGET_VARIABLE_CLONE,
                            TARGET_VARIABLE_AGG_NAME, PERIOD_CLONE, PAR_SCOPE_CLONE,
                            MODEL_TYPE, SepType, PP_REF, PP_REF_TAB, PP_REF_MEASURE, PP_REF_TOP)

# =============================================================================
# 0. LOAD ĐÚNG NHƯ RECIPE THẬT — SP_anomalies + project variables
# =============================================================================

df_seg_full = dataiku.Dataset("SP_anomalies").get_dataframe()

dss_client = dataiku.api_client()
project = dss_client.get_default_project()
variables = project.get_variables()

period_run = variables['local']["period_run"]
scope_run = variables['local']["scope_run"]
id_set_parameters = variables['local']["id_set_params"]

thresholds_segment_materiality = variables['standard']['thresholds_segment_materiality']
if isinstance(thresholds_segment_materiality, str):
    thresholds_segment_materiality = json.loads(thresholds_segment_materiality)

target_detection_mapping = variables['standard']['target_detection_mapping']
if isinstance(target_detection_mapping, str):
    target_detection_mapping = json.loads(target_detection_mapping)
t_agg_name_mapping = {k: v[0] for k, v in target_detection_mapping.items()}

segment_filters = variables['standard']['segment_filters']
segment_conditions = variables['standard']['segment_conditions']
text_mapping = {}  # điền đúng text_mapping thật dùng trong recipe nếu cần tái hiện 100%

TARGET_VARIABLE = "provision3"
SEG_AXES_CHECK = "entity|stage"
SEG_AXES_VALUE_CHECK = "ARVAL|2"


# =============================================================================
# 1. TÁI HIỆN ĐÚNG BƯỚC ĐẦU CỦA top_anomalies_line_par: filter model_type
#    rồi gọi thẳng top_segment_par (import trực tiếp, không viết lại logic)
# =============================================================================

df_seg = df_seg_full[df_seg_full[MODEL_TYPE] == "basic"]
target_variable = TARGET_VARIABLE  # bình thường lấy từ df_seg[TARGET_VARIABLE_CLONE].values[0]
                                     # nhưng ta cố định để test đúng 1 target_variable

df_seg_topn = top_segment_par(
    df_seg, period_run, scope_run, thresholds_segment_materiality, id_set_parameters,
    target_variable, t_agg_name_mapping, segment_filters, segment_conditions, text_mapping
)

print("Tổng số dòng df_seg_topn (output top_segment_par):", len(df_seg_topn))

dup_mask = df_seg_topn.duplicated(subset=[SEG_AXES, SEG_AXES_VALUE], keep=False)
print("Số dòng bị duplicate trong df_seg_topn (theo seg_axes+seg_axes_value):", dup_mask.sum())

check = df_seg_topn[
    (df_seg_topn[SEG_AXES] == SEG_AXES_CHECK) &
    (df_seg_topn[SEG_AXES_VALUE] == SEG_AXES_VALUE_CHECK)
]
print("\nSố dòng cho segment '{}={}' trong df_seg_topn:".format(SEG_AXES_CHECK, SEG_AXES_VALUE_CHECK), len(check))
if len(check) > 1:
    print(">>> XÁC NHẬN: duplicate đã tồn tại NGAY TRONG output top_segment_par <<<")


# =============================================================================
# 2. KIỂM TRA GIẢ THUYẾT CHÍNH — MULTI-MATCH TRONG VÒNG LẶP tab_value
#    (df[SEG_AXES_VALUE] chứa nhiều "thành phần" bị trùng với >1 tab_value)
# =============================================================================

df_period_scope = df_seg[
    (df_seg[PERIOD_CLONE] == period_run) &
    (df_seg[PAR_SCOPE_CLONE] == scope_run) &
    (df_seg[TARGET_VARIABLE_AGG_NAME] == t_agg_name_mapping[TARGET_VARIABLE])
]

query_reference = "(" + thresholds_segment_materiality[PP_REF][PP_REF_TAB] + ")"
tab_values = df_period_scope.query(query_reference)[SEG_AXES_VALUE].unique()

# Với segment đang điều tra, tách các "thành phần" trong seg_axes_value
components = SEG_AXES_VALUE_CHECK.split(SepType.PIPE)
print("\nCác thành phần của seg_axes_value '{}': {}".format(SEG_AXES_VALUE_CHECK, components))

matching_tab_values = [tv for tv in tab_values if tv in components]
print("Số tab_value KHỚP với segment này (should be 1, nếu >1 la nguyên nhân duplicate):",
      len(matching_tab_values))
print("Danh sách tab_value khớp:", matching_tab_values)

if len(matching_tab_values) > 1:
    print("\n>>> XÁC NHẬN GIẢ THUYẾT: segment '{}' bị chọn {} lần trong vòng lặp "
          "tab_value vì nhiều hơn 1 thành phần của seg_axes_value trùng với "
          "một entity code hợp lệ khác trong tab_values. <<<".format(
              SEG_AXES_VALUE_CHECK, len(matching_tab_values)))
else:
    print("\n>>> Giả thuyết multi-match KHÔNG đúng cho segment cụ thể này — "
          "cần tìm nguyên nhân khác (xem phần 3). <<<")


# =============================================================================
# 3. QUÉT TOÀN BỘ — tìm TẤT CẢ segment nào bị multi-match, để xác nhận đây có
#    phải nguyên nhân CHUNG cho toàn bộ 632 segment bị duplicate hay không
# =============================================================================

print("\n" + "=" * 70)
print("QUÉT TOÀN BỘ df_period_scope — đếm số tab_value khớp cho MỌI segment")
print("=" * 70)

def count_matching_tab_values(seg_axes_value: str) -> int:
    comps = seg_axes_value.split(SepType.PIPE)
    return sum(1 for tv in tab_values if tv in comps)

df_period_scope = df_period_scope.copy()
df_period_scope["n_tab_value_match"] = df_period_scope[SEG_AXES_VALUE].apply(count_matching_tab_values)

multi_match_segments = df_period_scope[df_period_scope["n_tab_value_match"] > 1]
print("Tổng số dòng (segment_id) có n_tab_value_match > 1:", len(multi_match_segments))
print("Số seg_axes_value DUY NHẤT bị multi-match:",
      multi_match_segments[SEG_AXES_VALUE].nunique())

if not multi_match_segments.empty:
    print("\nVí dụ 10 segment bị multi-match:")
    print(multi_match_segments[[SEG_AXES, SEG_AXES_VALUE, "n_tab_value_match"]].drop_duplicates().head(10))

print("\n" + "=" * 70)
print("SO SÁNH với số segment_id bị duplicate đã biết trước đó (632) để kiểm "
      "chứng đây có phải nguyên nhân bao trùm hay chỉ một phần")
print("=" * 70)
