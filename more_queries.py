# more_queries.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from models.train import train_all
from aqe.adaptive_runner import AdaptiveQueryRunner

spark = SparkSession.builder \
    .appName("MoreQueries") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

results   = train_all(verbose=False)
pipelines = {t: r["pipeline"] for t, r in results.items()}
runner    = AdaptiveQueryRunner(spark, pipelines, verbose=True)

# --- write new queries here, each structurally different ---

def q_three_way_join(spark):
    a = spark.range(500_000).withColumnRenamed("id", "key")
    b = spark.range(200_000).withColumnRenamed("id", "key")
    c = spark.range(100_000).withColumnRenamed("id", "key")
    return a.join(b, "key").join(c, "key").groupBy("key").count()

def q_window_function(spark):
    df = spark.range(300_000) \
              .withColumn("grp", F.col("id") % 500) \
              .withColumn("val", (F.col("id") % 999).cast("double"))
    from pyspark.sql.window import Window
    w = Window.partitionBy("grp").orderBy("val")
    return df.withColumn("rank", F.rank().over(w))

def q_self_join(spark):
    df = spark.range(200_000).withColumnRenamed("id", "key")
    return df.alias("a").join(df.alias("b"), on="key")

def q_multi_agg_wide(spark):
    df = spark.range(1_000_000) \
              .withColumn("grp",  F.col("id") % 200) \
              .withColumn("val1", (F.col("id") % 777).cast("double")) \
              .withColumn("val2", (F.col("id") % 333).cast("double"))
    return df.groupBy("grp").agg(
        F.count("*"), F.sum("val1"), F.avg("val1"),
        F.stddev("val1"), F.max("val2"), F.min("val2")
    )

def q_filter_then_join(spark):
    a = spark.range(2_000_000) \
             .withColumn("key", F.col("id") % 500_000) \
             .filter(F.col("id") % 3 == 0)
    b = spark.range(500_000).withColumnRenamed("id", "key")
    return a.join(b, "key")

def q_broadcast_small(spark):
    big   = spark.range(1_500_000).withColumnRenamed("id", "key")
    small = spark.range(500).withColumnRenamed("id", "key")
    return big.join(broadcast(small), "key").groupBy("key").count()

def q_heavy_sort_then_agg(spark):
    df = spark.range(800_000) \
              .withColumn("val", (F.col("id") % 9999).cast("double")) \
              .withColumn("grp", F.col("id") % 50)
    return df.orderBy(F.col("val").desc()) \
             .groupBy("grp").agg(F.first("val"), F.last("val"))

for fn, name in [
    (q_three_way_join,    "three_way_join"),
    (q_window_function,   "window_function"),
    (q_self_join,         "self_join"),
    (q_multi_agg_wide,    "multi_agg_wide"),
    (q_filter_then_join,  "filter_then_join"),
    (q_broadcast_small,   "broadcast_small"),
    (q_heavy_sort_then_agg, "heavy_sort_agg"),
]:
    runner.run(fn, query_name=name)

spark.stop()