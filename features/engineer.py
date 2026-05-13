import pandas as pd
import numpy as np
from profiler.feature_store import load_all_records


# ── 1. Load & filter to stage-level records only ──────────────────────────────

def load_stage_records() -> pd.DataFrame:
    df = load_all_records()
    df = df[df["record_type"] == "stage"].copy()
    df = df.sort_values(["query_id", "stage_id"]).reset_index(drop=True)
    return df


# ── 2. Per-query aggregation ───────────────────────────────────────────────────
# Each query may produce multiple stage records.
# We collapse them into one row per query for ML training.

def aggregate_per_query(df: pd.DataFrame) -> pd.DataFrame:
    plan_cols = [
        "op_hash_join", "op_sort_merge_join", "op_broadcast_join",
        "op_aggregate", "op_sort", "op_exchange", "op_filter",
        "op_scan", "plan_depth",
        "estimated_rows_min", "estimated_rows_max", "estimated_rows_mean",
    ]
    stage_cols = [
        "num_tasks", "num_completed_tasks", "num_failed_tasks",
    ]

    agg_dict = {}
    if "row_count" in df.columns:
        agg_dict["row_count"] = "max"

    # plan features: take first value per query (same across all stages)
    for c in plan_cols:
        agg_dict[c] = "first"

    # stage features: sum and max across stages
    for c in stage_cols:
        agg_dict[f"{c}"] = ["sum", "max"]

    grouped = df.groupby("query_id").agg(agg_dict)

    # flatten multi-level columns
    grouped.columns = [
        f"{a}_{b}" if b else a
        for a, b in grouped.columns
    ]
    grouped = grouped.reset_index()
    feature_cols = [col for col in grouped.columns if col != "query_id"]
    grouped[feature_cols] = grouped[feature_cols].fillna(0)
    return grouped


# ── 3. Derived / engineered features ──────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["phys_sort", "phys_exchange", "phys_broadcast", "row_count_max"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # 3a. Join complexity score
    #     Weighted sum: SortMergeJoin is heavier than HashJoin, BroadcastJoin is cheapest
    if "row_count_max" not in df.columns:
        df["row_count_max"] = 0
    df["row_count_max"] = df["row_count_max"].fillna(0)

    df["join_complexity"] = (
        df["op_sort_merge_join_first"] * 3 +
        df["op_hash_join_first"]       * 2 +
        df["op_broadcast_join_first"]  * 1
    )

    # 3b. Shuffle pressure — exchanges relative to total operators
    total_ops = (
        df["op_hash_join_first"] + df["op_sort_merge_join_first"] +
        df["op_broadcast_join_first"] + df["op_aggregate_first"] +
        df["op_sort_first"] + df["op_exchange_first"] +
        df["op_filter_first"] + df["op_scan_first"]
    ).replace(0, 1)  # avoid div/0
    df["shuffle_pressure_ratio"] = df["op_exchange_first"] / total_ops

    # 3c. Task completion ratio — proxy for how smoothly stages ran
    total_tasks = df["num_tasks_sum"].replace(0, 1)
    df["task_completion_ratio"] = df["num_completed_tasks_sum"] / total_tasks

    # 3d. Task failure flag
    df["had_failures"] = (df["num_failed_tasks_sum"] > 0).astype(int)

    # 3e. Plan breadth — scans per unit depth (wide vs deep plans)
    depth = df["plan_depth_first"].replace(0, 1)
    df["plan_breadth_ratio"] = df["op_scan_first"] / depth

    # 3f. Aggregation density
    df["agg_density"] = df["op_aggregate_first"] / total_ops

    # 3g. Sort overhead flag
    df["is_sort_heavy"] = (df["op_sort_first"] >= 2).astype(int)

    # 3h. Broadcast opportunity flag
    df["uses_broadcast"] = (df["op_broadcast_join_first"] > 0).astype(int)

    return df


# ── 4. Synthetic targets (for local dev without real timing data) ──────────────
# In production these come from actual measured metrics.
# Here we derive plausible proxies from what the StatusTracker gave us.

def add_synthetic_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed=42)

    # latency proxy: more tasks + joins + sorts = higher latency
    df["target_latency_ms"] = (
        df["num_tasks_sum"] * 120 +
        df["join_complexity"] * 800 +
        df["op_sort_first"] * 500 +
        df["op_exchange_first"] * 400 +
        rng.normal(0, 200, len(df))
    ).clip(lower=100)

    # memory proxy: aggregations and sorts are memory-heavy
    df["target_memory_mb"] = (
        df["num_tasks_max"] * 40 +
        df["op_aggregate_first"] * 300 +
        df["op_sort_first"] * 200 +
        rng.normal(0, 50, len(df))
    ).clip(lower=64)

    # shuffle I/O proxy: exchanges drive shuffle bytes
    df["target_shuffle_mb"] = (
        df["op_exchange_first"] * 250 +
        df["join_complexity"] * 150 +
        rng.normal(0, 30, len(df))
    ).clip(lower=0)

    # CPU efficiency proxy: broadcast = good, sort-merge = wasteful
    df["target_cpu_efficiency"] = (
        0.9
        - df["op_sort_merge_join_first"] * 0.12
        + df["uses_broadcast"] * 0.08
        - df["had_failures"] * 0.2
        + rng.normal(0, 0.05, len(df))
    ).clip(0.1, 1.0)

    return df


# ── 5. Final feature matrix builder ───────────────────────────────────────────

# FEATURE_COLS = [
#     # plan structure
#     "op_hash_join_first", "op_sort_merge_join_first", "op_broadcast_join_first",
#     "op_aggregate_first", "op_sort_first", "op_exchange_first",
#     "op_filter_first", "op_scan_first", "plan_depth_first",
#     # derived
#     "join_complexity", "shuffle_pressure_ratio", "task_completion_ratio",
#     "had_failures", "plan_breadth_ratio", "agg_density",
#     "is_sort_heavy", "uses_broadcast",
# ]

# add to FEATURE_COLS list:
FEATURE_COLS = [
    # plan structure
    "op_hash_join_first", "op_sort_merge_join_first", "op_broadcast_join_first",
    "op_aggregate_first", "op_sort_first", "op_exchange_first",
    "op_filter_first", "op_scan_first", "plan_depth_first",
    # derived
    "join_complexity", "shuffle_pressure_ratio", "task_completion_ratio",
    "had_failures", "plan_breadth_ratio", "agg_density",
    "is_sort_heavy", "uses_broadcast",
    # output scale  ← new
    "row_count_max",
    "phys_sort", "phys_exchange", "phys_broadcast",
]

TARGET_COLS = [
    "target_latency_ms",
    "target_memory_mb",
    "target_shuffle_mb",
    "target_cpu_efficiency",
]

def build_feature_matrix(verbose: bool = True):
    raw     = load_stage_records()
    per_q   = aggregate_per_query(raw)
    derived = add_derived_features(per_q)
    final   = add_synthetic_targets(derived)

    X = final[FEATURE_COLS]
    y = final[TARGET_COLS]

    if verbose:
        print(f"Feature matrix : {X.shape[0]} rows × {X.shape[1]} features")
        print(f"Target matrix  : {y.shape[0]} rows × {y.shape[1]} targets")
        print(f"\nFeature columns:\n  {list(X.columns)}")
        print(f"\nTarget stats:\n{y.describe().T[['mean','min','max']].round(2)}")
        print(f"\nFeature correlation with latency (top 5):")
        corr = X.corrwith(y["target_latency_ms"]).abs().sort_values(ascending=False)
        print(corr.head(5).round(3).to_string())

    return X, y, final