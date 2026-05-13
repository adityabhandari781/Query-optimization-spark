# main.py

from pyspark.sql import SparkSession
from profiler.listener      import QueryProfilerListener
from profiler.plan_parser   import extract_plan_features
from profiler.feature_store import save_records
import uuid


def run_profiled_query(spark, query_fn):
    query_id = str(uuid.uuid4())[:8]
    listener = QueryProfilerListener(spark.sparkContext)

    df = query_fn(spark)
    plan_features = extract_plan_features(spark, df)

    listener.start()
    df.collect()          # trigger execution
    listener.stop()

    for r in listener.records:
        r.update(plan_features)

    save_records(listener.records, query_id)
    return df, listener.records


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("SparkMLOptimizer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    def q1_small_join(spark):
        df1 = spark.range(100_000).withColumnRenamed("id", "key")
        df2 = spark.range(50_000).withColumnRenamed("id", "key")
        return df1.join(df2, on="key")

    def q2_large_join(spark):
        df1 = spark.range(2_000_000).withColumnRenamed("id", "key")
        df2 = spark.range(1_000_000).withColumnRenamed("id", "key")
        return df1.join(df2, on="key").groupBy("key").count()

    def q3_aggregation_heavy(spark):
        from pyspark.sql import functions as F
        df = spark.range(1_000_000)
        return df.withColumn("grp", (F.col("id") % 100)) \
                 .groupBy("grp").agg(F.count("*"), F.sum("id"), F.avg("id"))

    def q4_multi_join(spark):
        a = spark.range(500_000).withColumnRenamed("id", "key")
        b = spark.range(300_000).withColumnRenamed("id", "key")
        c = spark.range(200_000).withColumnRenamed("id", "key")
        return a.join(b, "key").join(c, "key")

    def q5_skewed_join(spark):
        from pyspark.sql import functions as F
        # artificially skewed: most rows share key=0
        df1 = spark.range(1_000_000).withColumn(
            "key", F.when(F.col("id") < 900_000, 0).otherwise(F.col("id"))
        )
        df2 = spark.range(100_000).withColumnRenamed("id", "key")
        return df1.join(df2, on="key")

    def q6_wide_transform(spark):
        from pyspark.sql import functions as F
        df = spark.range(500_000)
        for i in range(10):
            df = df.withColumn(f"col_{i}", F.col("id") * i + F.lit(i))
        return df.groupBy("id").sum()

    def q7_sort_heavy(spark):
        from pyspark.sql import functions as F
        df = spark.range(1_000_000) \
                  .withColumn("val", (F.col("id") % 9999).cast("double"))
        return df.orderBy("val")

    def q8_broadcast_hint(spark):
        from pyspark.sql import functions as F
        from pyspark.sql.functions import broadcast
        big   = spark.range(1_000_000).withColumnRenamed("id", "key")
        small = spark.range(1_000).withColumnRenamed("id", "key")
        return big.join(broadcast(small), on="key")

    queries = [q1_small_join, q2_large_join, q3_aggregation_heavy,
               q4_multi_join, q5_skewed_join, q6_wide_transform,
               q7_sort_heavy, q8_broadcast_hint]

    for qfn in queries:
        print(f"\n>>> Profiling: {qfn.__name__}")
        _, records = run_profiled_query(spark, qfn)
        print(f"    Records collected: {len(records)}")

    spark.stop()