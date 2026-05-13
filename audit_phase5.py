from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from models.train          import train_all
from aqe.adaptive_runner   import AdaptiveQueryRunner

# ── Build models ───────────────────────────────────────────────────────────────
print("Training models...")
results   = train_all(verbose=False)
pipelines = {t: r["pipeline"] for t, r in results.items()}

# ── Start Spark ────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("AdaptiveRunnerTest") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

runner = AdaptiveQueryRunner(spark, pipelines, verbose=True)

# ── Three test queries ─────────────────────────────────────────────────────────

def q_join_heavy(spark):
    a = spark.range(1_000_000).withColumnRenamed("id", "key")
    b = spark.range(400_000).withColumnRenamed("id", "key")
    return a.join(b, "key").groupBy("key").count()

# FIX: agg that produces many output rows, not 50
def q_agg_heavy(spark):
    df = spark.range(1_000_000)
    return df.withColumn("grp", F.col("id") % 50_000) \
             .groupBy("grp") \
             .agg(F.count("*"), F.sum("id"), F.avg("id"), F.max("id"))

# FIX: sort that keeps all rows, not 100
# def q_sort_heavy(spark):
#     df = spark.range(800_000) \
#               .withColumn("val", (F.col("id") % 9999).cast("double")) \
#               .withColumn("grp", F.col("id") % 10_000)
#     return df.orderBy("val").groupBy("grp").count()

def q_sort_heavy(spark):
    # orderBy at the END with no subsequent groupBy
    # AQE cannot eliminate this — it's the terminal operation
    df = spark.range(1_000_000) \
              .withColumn("val", (F.col("id") % 9999).cast("double")) \
              .withColumn("grp", F.col("id") % 200)
    return df.groupBy("grp") \
             .agg(F.sum("val").alias("total"), F.count("*").alias("cnt")) \
             .orderBy(F.col("total").desc())

summaries = []
for fn, name in [
    (q_join_heavy,  "join_heavy"),
    (q_agg_heavy,   "agg_heavy"),
    (q_sort_heavy,  "sort_heavy"),
]:
    _, summary = runner.run(fn, query_name=name)
    summaries.append(summary)

# ── Final comparison table ─────────────────────────────────────────────────────
print(f"\n\n{'═'*65}")
print(f"  PHASE 5 SUMMARY")
print(f"{'═'*65}")
print(f"  {'Query':<20} {'Predicted':>12} {'Actual':>12} {'Error':>8}  {'Pareto'}")
print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8}  {'─'*6}")
for s in summaries:
    pred = s["predictions"]["latency_ms"]
    act  = s["actual_exec_ms"]
    err  = abs(act - pred) / max(pred, 1) * 100
    print(f"  {s['query_name']:<20} {pred:>10.0f}ms {act:>10.0f}ms {err:>7.1f}%  {s['pareto_size']}")

print(f"\n✓ Phase 5 complete — optimizer is live on real queries")
spark.stop()