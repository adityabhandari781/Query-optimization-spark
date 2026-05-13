from models.train             import train_all
from optimizer.candidate_space import enumerate_candidates
from optimizer.pareto          import score_candidates, compute_pareto_front
from optimizer.policy          import select_best, config_to_spark_conf
from features.engineer         import build_feature_matrix, FEATURE_COLS

print("=" * 55)
print("  PHASE 4 — PARETO OPTIMISATION")
print("=" * 55)

# ── 1. Train models (reuse Phase 3) ───────────────────────────────────────────
results   = train_all(verbose=False)
pipelines = {t: r["pipeline"] for t, r in results.items()}

# ── 2. Simulate three different incoming queries ───────────────────────────────
# In production, base_features comes from extract_plan_features()
# on the actual query being submitted.

query_profiles = {
    "join_heavy": {
        "op_hash_join_first": 2, "op_sort_merge_join_first": 0,
        "op_broadcast_join_first": 0, "op_aggregate_first": 0,
        "op_sort_first": 0, "op_exchange_first": 2,
        "op_filter_first": 0, "op_scan_first": 6, "plan_depth_first": 5,
        "join_complexity": 4, "shuffle_pressure_ratio": 0.2,
        "task_completion_ratio": 1.0, "had_failures": 0,
        "plan_breadth_ratio": 1.2, "agg_density": 0.0,
        "is_sort_heavy": 0, "uses_broadcast": 0,
    },
    "agg_heavy": {
        "op_hash_join_first": 0, "op_sort_merge_join_first": 0,
        "op_broadcast_join_first": 0, "op_aggregate_first": 3,
        "op_sort_first": 1, "op_exchange_first": 2,
        "op_filter_first": 1, "op_scan_first": 4, "plan_depth_first": 7,
        "join_complexity": 0, "shuffle_pressure_ratio": 0.18,
        "task_completion_ratio": 1.0, "had_failures": 0,
        "plan_breadth_ratio": 0.57, "agg_density": 0.27,
        "is_sort_heavy": 0, "uses_broadcast": 0,
    },
    "skewed_broadcast": {
        "op_hash_join_first": 1, "op_sort_merge_join_first": 0,
        "op_broadcast_join_first": 1, "op_aggregate_first": 1,
        "op_sort_first": 2, "op_exchange_first": 1,
        "op_filter_first": 0, "op_scan_first": 8, "plan_depth_first": 6,
        "join_complexity": 3, "shuffle_pressure_ratio": 0.07,
        "task_completion_ratio": 0.8, "had_failures": 0,
        "plan_breadth_ratio": 1.33, "agg_density": 0.07,
        "is_sort_heavy": 1, "uses_broadcast": 1,
    },
}

candidates = enumerate_candidates()
print(f"\nCandidate configs : {len(candidates)}")

for profile_name, base_features in query_profiles.items():
    print(f"\n\n{'▓'*55}")
    print(f"  Query profile: {profile_name}")
    print(f"{'▓'*55}")

    # score all candidates
    scored   = score_candidates(base_features, pipelines, candidates)

    # compute Pareto front
    pareto   = compute_pareto_front(scored)

    # pick best from front
    best     = select_best(pareto, verbose=True)

    # translate to Spark conf
    spark_conf = config_to_spark_conf(best)
    print(f"\n  Spark conf to apply:")
    for k, v in spark_conf.items():
        print(f"    spark.conf.set('{k}', '{v}')")