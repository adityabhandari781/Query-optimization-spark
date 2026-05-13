import numpy as np
import pandas as pd
from features.engineer import FEATURE_COLS

# ── Spark configuration knobs we're optimising over ───────────────────────────
# Each candidate is a dict of Spark config values.
# The ML models score each candidate by translating config choices
# into plan-feature deltas, then predicting all four objectives.

JOIN_STRATEGIES   = ["hash", "sort_merge", "broadcast"]
PARTITION_COUNTS  = [50, 100, 200, 400]
BROADCAST_THRESH  = [5, 10, 20, 50]        # MB — spark.sql.autoBroadcastJoinThreshold


def enumerate_candidates() -> list[dict]:
    """
    Return the full Cartesian product of tunable knobs.
    48 candidates total — small enough for exhaustive scoring.
    """
    candidates = []
    for js in JOIN_STRATEGIES:
        for pc in PARTITION_COUNTS:
            for bt in BROADCAST_THRESH:
                candidates.append({
                    "join_strategy":        js,
                    "partition_count":      pc,
                    "broadcast_threshold":  bt,
                })
    return candidates


def config_to_feature_delta(base_features: dict, candidate: dict) -> np.ndarray:
    """
    Given a base feature vector (from the current query plan) and a
    candidate config, return a modified feature vector that reflects
    what the plan would look like under that config.

    This is a lightweight simulation — not a full re-plan.
    In production you would re-invoke the Spark planner; here we
    apply heuristic deltas that capture the dominant effects.
    """
    f = base_features.copy()
    js = candidate["join_strategy"]
    pc = candidate["partition_count"]
    bt = candidate["broadcast_threshold"]

    # ── join strategy effects ──────────────────────────────────────────────────
    if js == "broadcast":
        f["op_broadcast_join_first"] = max(f.get("op_broadcast_join_first", 0), 1)
        f["op_hash_join_first"]      = 0
        f["op_sort_merge_join_first"]= 0
        f["uses_broadcast"]          = 1
        f["op_exchange_first"]       = max(0, f.get("op_exchange_first", 1) - 1)

    elif js == "hash":
        f["op_hash_join_first"]       = max(f.get("op_hash_join_first", 0), 1)
        f["op_broadcast_join_first"]  = 0
        f["op_sort_merge_join_first"] = 0
        f["uses_broadcast"]           = 0

    elif js == "sort_merge":
        f["op_sort_merge_join_first"] = max(f.get("op_sort_merge_join_first", 0), 1)
        f["op_hash_join_first"]       = 0
        f["op_broadcast_join_first"]  = 0
        f["uses_broadcast"]           = 0
        f["op_sort_first"]            = f.get("op_sort_first", 0) + 1
        f["is_sort_heavy"]            = int(f["op_sort_first"] >= 2)

    # ── partition count effects ────────────────────────────────────────────────
    # more partitions → more exchanges (shuffle overhead) but better parallelism
    base_pc = 200  # Spark default
    pc_ratio = pc / base_pc
    f["op_exchange_first"] = max(0, round(f.get("op_exchange_first", 1) * pc_ratio))

    # ── broadcast threshold effects ────────────────────────────────────────────
    # higher threshold → more tables qualify for broadcast
    if bt >= 20 and js != "sort_merge":
        f["op_broadcast_join_first"] = max(f.get("op_broadcast_join_first", 0), 1)
        f["uses_broadcast"]          = 1

    # ── recompute derived features ─────────────────────────────────────────────
    total_ops = max(1, sum([
        f.get("op_hash_join_first", 0),
        f.get("op_sort_merge_join_first", 0),
        f.get("op_broadcast_join_first", 0),
        f.get("op_aggregate_first", 0),
        f.get("op_sort_first", 0),
        f.get("op_exchange_first", 0),
        f.get("op_filter_first", 0),
        f.get("op_scan_first", 0),
    ]))

    f["join_complexity"] = (
        f.get("op_sort_merge_join_first", 0) * 3 +
        f.get("op_hash_join_first", 0)       * 2 +
        f.get("op_broadcast_join_first", 0)  * 1
    )
    f["shuffle_pressure_ratio"] = f.get("op_exchange_first", 0) / total_ops
    f["agg_density"]            = f.get("op_aggregate_first", 0) / total_ops
    depth = max(1, f.get("plan_depth_first", 4))
    f["plan_breadth_ratio"]     = f.get("op_scan_first", 4) / depth

    # return as ordered numpy array matching FEATURE_COLS
    return np.array([f.get(col, 0) for col in FEATURE_COLS], dtype=float)