# real_queries.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from models.train import train_all
from aqe.adaptive_runner import AdaptiveQueryRunner

TPCH = r"C:\Users\adity\Desktop\code\GitHub\Query-optimization-spark\data\tpch"

spark = SparkSession.builder \
    .appName("RealQueryOptimizer") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# load tables once
orders    = spark.read.parquet(f"{TPCH}/orders.parquet")
lineitem  = spark.read.parquet(f"{TPCH}/lineitem.parquet")
customer  = spark.read.parquet(f"{TPCH}/customer.parquet")
part      = spark.read.parquet(f"{TPCH}/part.parquet")
supplier  = spark.read.parquet(f"{TPCH}/supplier.parquet")
nation    = spark.read.parquet(f"{TPCH}/nation.parquet")

# register as views so queries can use SQL too
for name, df in [("orders",orders),("lineitem",lineitem),
                 ("customer",customer),("part",part),
                 ("supplier",supplier),("nation",nation)]:
    df.createOrReplaceTempView(name)

results   = train_all(verbose=False)
pipelines = {t: r["pipeline"] for t, r in results.items()}
runner    = AdaptiveQueryRunner(spark, pipelines, verbose=True)

# ── TPC-H inspired queries — each tests a different plan shape ────────────────

def q_order_totals(spark):
    # simple agg on large table
    return orders.groupBy("o_orderstatus") \
                 .agg(F.count("*"), F.sum("o_totalprice"), F.avg("o_totalprice"))

def q_lineitem_join_orders(spark):
    # large-large join — the hardest case
    return lineitem.join(orders, lineitem.l_orderkey == orders.o_orderkey) \
                   .groupBy("o_orderstatus", "l_returnflag") \
                   .agg(F.sum("l_quantity"), F.sum("l_extendedprice"))

def q_customer_nation(spark):
    # broadcast-eligible join (nation is tiny: 25 rows)
    return customer.join(broadcast(nation),
                         customer.c_nationkey == nation.n_nationkey) \
                   .groupBy("n_name") \
                   .agg(F.count("*"), F.sum("c_acctbal"))

def q_supplier_parts(spark):
    # three-way join
    from pyspark.sql import functions as F
    ps = spark.read.parquet(f"{TPCH}/partsupp.parquet")
    return ps.join(supplier, ps.ps_suppkey == supplier.s_suppkey) \
             .join(part,     ps.ps_partkey == part.p_partkey) \
             .groupBy("s_nationkey", "p_type") \
             .agg(F.sum("ps_availqty"), F.avg("ps_supplycost"))

def q_late_orders(spark):
    # filter then join — selectivity matters
    late = lineitem.filter(F.col("l_commitdate") < F.col("l_receiptdate"))
    return late.join(orders, late.l_orderkey == orders.o_orderkey) \
               .groupBy("o_orderpriority") \
               .count()

def q_revenue_by_month(spark):
    # date extraction + agg — tests sort and group
    return lineitem \
        .withColumn("month", F.month("l_shipdate")) \
        .withColumn("year",  F.year("l_shipdate")) \
        .groupBy("year","month") \
        .agg(F.sum("l_extendedprice").alias("revenue"),
             F.sum("l_discount").alias("total_discount")) \
        .orderBy("year","month")

def q_top_customers(spark):
    # join + agg + sort — full pipeline
    return orders.join(customer, orders.o_custkey == customer.c_custkey) \
                 .groupBy("c_name","c_acctbal") \
                 .agg(F.sum("o_totalprice").alias("total_spent")) \
                 .orderBy(F.col("total_spent").desc())

def q_part_supplier_cost(spark):
    # aggregation with filter pushdown
    ps = spark.read.parquet(f"{TPCH}/partsupp.parquet")
    return ps.filter(F.col("ps_availqty") > 100) \
             .groupBy("ps_suppkey") \
             .agg(F.sum("ps_supplycost").alias("total_cost"),
                  F.avg("ps_availqty").alias("avg_qty")) \
             .orderBy(F.col("total_cost").desc())

for fn, name in [
    (q_order_totals,        "order_totals"),
    (q_lineitem_join_orders,"lineitem_join_orders"),
    (q_customer_nation,     "customer_nation"),
    (q_supplier_parts,      "supplier_parts"),
    (q_late_orders,         "late_orders"),
    (q_revenue_by_month,    "revenue_by_month"),
    (q_top_customers,       "top_customers"),
    (q_part_supplier_cost,  "part_supplier_cost"),
]:
    print(f"\n>>> {name}")
    runner.run(fn, query_name=name)

spark.stop()