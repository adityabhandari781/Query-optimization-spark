# generate_data.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from main import run_profiled_query
import random, time

spark = SparkSession.builder \
    .appName("DataGenerator") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
random.seed(42)

def make_query(size_a, size_b, size_c,
               do_agg, do_sort, do_broadcast,
               n_extra_cols, skew_factor):
    """
    Dynamically builds a query from parameters so every
    combination produces a structurally different plan.
    """
    def query_fn(spark):
        df_a = spark.range(size_a).withColumnRenamed("id", "key")
        df_b = spark.range(size_b).withColumnRenamed("id", "key")

        # optional skew
        if skew_factor > 0:
            df_a = df_a.withColumn(
                "key",
                F.when(F.col("key") < size_a * skew_factor,
                       F.lit(0)).otherwise(F.col("key"))
            )

        # join strategy
        if do_broadcast and size_b <= 50_000:
            df = df_a.join(broadcast(df_b), on="key")
        else:
            df = df_a.join(df_b, on="key")

        # optional third join
        if size_c > 0:
            df_c = spark.range(size_c).withColumnRenamed("id", "key")
            df = df.join(df_c, on="key")

        # extra columns (widens the plan)
        for i in range(n_extra_cols):
            df = df.withColumn(f"c{i}", F.col("key") * F.lit(i + 1))

        # aggregation
        if do_agg:
            df = df.groupBy("key").agg(
                F.count("*").alias("cnt"),
                F.sum("key").alias("s")
            )

        # sort
        if do_sort:
            df = df.orderBy("key")

        return df

    return query_fn


# ── Parameter grid — 100 combinations ─────────────────────────────────────────

configs = []
sizes_a      = [50_000, 200_000, 500_000, 1_000_000, 2_000_000]
sizes_b      = [10_000, 50_000,  200_000, 500_000]
sizes_c      = [0, 50_000, 200_000]          # 0 = no third join
skew_factors = [0, 0.5, 0.9]
extra_cols   = [0, 3, 8]

for _ in range(120):                          # sample randomly, take first 100 unique
    cfg = {
        "size_a":       random.choice(sizes_a),
        "size_b":       random.choice(sizes_b),
        "size_c":       random.choice(sizes_c),
        "do_agg":       random.choice([True, False]),
        "do_sort":      random.choice([True, False]),
        "do_broadcast": random.choice([True, False]),
        "n_extra_cols": random.choice(extra_cols),
        "skew_factor":  random.choice(skew_factors),
    }
    if cfg not in configs:
        configs.append(cfg)
    if len(configs) == 100:
        break

# ── Run them all ───────────────────────────────────────────────────────────────

print(f"Generating {len(configs)} profiled queries...\n")
for i, cfg in enumerate(configs):
    qfn = make_query(**cfg)
    try:
        _, records = run_profiled_query(spark, qfn)
        print(f"  [{i+1:>3}/100]  stages={len(records):<3}  "
              f"size_a={cfg['size_a']:<10}  "
              f"agg={cfg['do_agg']:<6}  "
              f"skew={cfg['skew_factor']}")
    except Exception as e:
        print(f"  [{i+1:>3}/100]  SKIPPED — {e}")
    time.sleep(0.5)   # brief pause to let Spark settle between queries

spark.stop()
print("\nDone. Run audit_store.py to check total record count.")