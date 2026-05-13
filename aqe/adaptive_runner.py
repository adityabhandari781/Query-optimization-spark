import time
import uuid
from pyspark.sql import SparkSession

from profiler.listener         import QueryProfilerListener
from profiler.plan_parser      import extract_plan_features
from profiler.feature_store    import save_records
from features.engineer         import FEATURE_COLS
from optimizer.candidate_space import enumerate_candidates
from optimizer.pareto          import score_candidates, compute_pareto_front
from optimizer.policy          import select_best, config_to_spark_conf

# Spark's built-in defaults — used when a key has never been set
_SPARK_CONF_DEFAULTS = {
    "spark.sql.shuffle.partitions":          "200",
    "spark.sql.autoBroadcastJoinThreshold":  "10485760",
    "spark.sql.join.preferSortMergeJoin":    "true",
}


class AdaptiveQueryRunner:

    def __init__(self, spark: SparkSession, pipelines: dict,
                 weights: dict = None, verbose: bool = True):
        self.spark      = spark
        self.pipelines  = pipelines
        self.weights    = weights
        self.verbose    = verbose
        self.candidates = enumerate_candidates()

    def _snapshot_conf(self, keys: list) -> dict:
        """
        Save current Spark conf values so we can restore them after.
        Falls back to known Spark defaults when a key is unset,
        avoiding the IllegalArgumentException on typed conf keys.
        """
        snapshot = {}
        for k in keys:
            try:
                val = self.spark.conf.get(k)
                snapshot[k] = val if val else _SPARK_CONF_DEFAULTS.get(k, "")
            except Exception:
                snapshot[k] = _SPARK_CONF_DEFAULTS.get(k, "")
        return snapshot

    def _apply_conf(self, conf: dict):
        for k, v in conf.items():
            self.spark.conf.set(k, v)

    def _restore_conf(self, snapshot: dict):
        for k, v in snapshot.items():
            if v:
                self.spark.conf.set(k, v)

    def _plan_to_base_features(self, plan_features: dict, row_count: int = 0) -> dict:
        base = {col: plan_features.get(col, 0) for col in FEATURE_COLS}
        base["row_count_max"] = row_count
        return base

    def run(self, query_fn, query_name: str = None):
        query_id   = str(uuid.uuid4())[:8]
        query_name = query_name or query_id

        if self.verbose:
            print(f"\n{'━'*55}")
            print(f"  AdaptiveRunner: {query_name}  [{query_id}]")
            print(f"{'━'*55}")

        # ── Step 1: build DataFrame + extract plan features ────────────────────
        t0 = time.time()
        df = query_fn(self.spark)

        # explain(extended=True) forces the Spark planner to fully resolve
        # join strategies, exchanges, and aggregations before we read the plan
        plan_features = extract_plan_features(self.spark, df)
        base_features = self._plan_to_base_features(plan_features)

        if self.verbose:
            print(f"  Plan features extracted in {time.time()-t0:.2f}s")
            print(f"    joins     = {plan_features.get('op_hash_join_first',0) + plan_features.get('op_broadcast_join_first',0) + plan_features.get('op_sort_merge_join_first',0)}")
            print(f"    exchanges = {plan_features.get('op_exchange_first',0)}")
            print(f"    aggs      = {plan_features.get('op_aggregate_first',0)}")
            print(f"    sorts     = {plan_features.get('op_sort_first',0)}")
            print(f"    depth     = {plan_features.get('plan_depth_first',0)}")

        # ── Step 2-3: score candidates + Pareto front ──────────────────────────
        scored = score_candidates(base_features, self.pipelines, self.candidates)
        pareto = compute_pareto_front(scored)
        best   = select_best(pareto, weights=self.weights, verbose=self.verbose)

        # ── Step 4: apply recommended config ──────────────────────────────────
        spark_conf = config_to_spark_conf(best)
        snapshot   = self._snapshot_conf(list(spark_conf.keys()))
        self._apply_conf(spark_conf)

        if self.verbose:
            print(f"\n  Applying config:")
            for k, v in spark_conf.items():
                print(f"    {k.replace('spark.sql.','')} = {v}")

        # ── Step 5: execute with profiler attached ─────────────────────────────
        listener = QueryProfilerListener(self.spark.sparkContext)
        listener.start()

        exec_start = time.time()
        rows       = df.collect()
        row_count  = len(rows)
        actual_ms  = (time.time() - exec_start) * 1000

        listener.stop()

        # ── Step 6: restore original config ───────────────────────────────────
        self._restore_conf(snapshot)

        # ── Step 7: save enriched records to feature store ────────────────────
        records = listener.records
        for r in records:
            r.update(plan_features)
            r["chosen_join_strategy"]       = best["join_strategy"]
            r["chosen_partition_count"]     = best["partition_count"]
            r["chosen_broadcast_threshold"] = best["broadcast_threshold"]
            r["actual_exec_ms"]             = actual_ms
            r["predicted_latency_ms"]       = best["target_latency_ms"]
            r["predicted_memory_mb"]        = best["target_memory_mb"]
            r["predicted_shuffle_mb"]       = best["target_shuffle_mb"]
            r["row_count"]                  = row_count

        save_records(records, query_id)

        run_summary = {
            "query_id":        query_id,
            "query_name":      query_name,
            "plan_features":   plan_features,
            "pareto_size":     len(pareto),
            "chosen_config":   spark_conf,
            "predictions": {
                "latency_ms": best["target_latency_ms"],
                "memory_mb":  best["target_memory_mb"],
                "shuffle_mb": best["target_shuffle_mb"],
                "cpu_eff":    best["target_cpu_efficiency"],
            },
            "actual_exec_ms":  actual_ms,
            "row_count":       row_count,
            "stages_recorded": len(records),
        }

        if self.verbose:
            pred_ms = best["target_latency_ms"]
            err_pct = abs(actual_ms - pred_ms) / max(pred_ms, 1) * 100
            print(f"\n  ✓ Execution complete")
            print(f"    rows         = {row_count:,}")
            print(f"    actual ms    = {actual_ms:,.0f}")
            print(f"    predicted ms = {pred_ms:,.0f}")
            print(f"    error        = {err_pct:.1f}%")
            print(f"    stages saved = {len(records)}")

        return df, run_summary