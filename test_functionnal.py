import dataiku
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession

from admin.paths import get_table_path_period
from generics.pipeline.generic_pipeline import GenericPipeline

project = dataiku.api_client().get_default_project()
dss_vars = dataiku.get_custom_variables()
dss_client = dataiku.api_client()

variables = project.get_variables()
period_run = variables['local']["period_run"]
scope_run = variables['local']["scope_run"]
id_set_params = variables['local']["id_set_params"]

df_axes_periods = dataiku.Dataset("axes_periods_par").get_dataframe()
type_run = variables["local"]["type_run"]
period = df_axes_periods["period_clone"].values[0]
axes_str_par = df_axes_periods["axe_clone"].values[0]
table_path = get_table_path_period(dss_client, dss_vars, scope_run, period, type_run)

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
dataset_out = dataiku.Dataset("SP_agg_generic")   # ⚠️ TÊN DATASET MỚI — không đè lên SP_agg gốc

spark = SparkSession.builder.config("spark.driver.maxResultSize", "4g").enableHiveSupport().getOrCreate()

config_folder = dataiku.Folder("parameters_folder")
pipeline = GenericPipeline(spark_sess=spark, config_root=config_folder.get_path())
params = GenericPipeline.params_from_dss_vars(dss_vars)

df_out = pipeline.run_pre_processing(
    flow_type=dss_vars["flow_type"],
    axes_str_par=axes_str_par,
    table_path=table_path,
    period=period,
    period_run=period_run,
    scope_run=scope_run,
    id_set_params=id_set_params,
    params=params,
)

dataset_out.write_with_schema(df_out)
