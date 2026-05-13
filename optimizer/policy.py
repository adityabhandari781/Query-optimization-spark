import pandas as pd
import numpy as np


# Weights let the caller express a preference — e.g. a batch job
# cares more about memory, an interactive query cares about latency.
DEFAULT_WEIGHTS = {
    "target_latency_ms":    0.40,
    "target_memory_mb":     0.25,
    "target_shuffle_mb":    0.20,
    "target_cpu_efficiency": 0.15,   # higher is better
}


def select_best(
    pareto_df: pd.DataFrame,
    weights: dict = None,
    verbose: bool = True,
) -> dict:
    """
    From the Pareto front, pick the single best candidate using
    weighted normalised scoring (TOPSIS-lite).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    df = pareto_df.copy()

    obj_cols = list(weights.keys())

    # normalise each objective to [0, 1]
    for col in obj_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        rng = col_max - col_min
        if rng == 0:
            df[f"norm_{col}"] = 0.0
        else:
            df[f"norm_{col}"] = (df[col] - col_min) / rng

    # cpu_efficiency: higher is better → invert after normalisation
    df["norm_target_cpu_efficiency"] = 1 - df["norm_target_cpu_efficiency"]

    # weighted sum score — lower is better
    df["score"] = sum(
        df[f"norm_{col}"] * w
        for col, w in weights.items()
    )

    best_idx = df["score"].idxmin()
    best     = df.loc[best_idx].to_dict()

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Pareto front size : {len(df)} candidates")
        print(f"  Selected config   :")
        print(f"    join_strategy       = {best['join_strategy']}")
        print(f"    partition_count     = {int(best['partition_count'])}")
        print(f"    broadcast_threshold = {int(best['broadcast_threshold'])} MB")
        print(f"\n  Predicted objectives:")
        print(f"    latency     = {best['target_latency_ms']:>10.1f} ms")
        print(f"    memory      = {best['target_memory_mb']:>10.1f} MB")
        print(f"    shuffle I/O = {best['target_shuffle_mb']:>10.1f} MB")
        print(f"    CPU eff.    = {best['target_cpu_efficiency']:>10.3f}")
        print(f"\n  Weighted score    = {best['score']:.4f}  (lower = better)")
        print(f"{'═'*55}")

    return best


def config_to_spark_conf(best: dict) -> dict:
    """
    Translate the chosen config into actual Spark SQL conf key-value pairs
    ready to apply with spark.conf.set().
    """
    import math
    
    js  = best["join_strategy"]
    bt  = int(best["broadcast_threshold"])
    pc  = int(best["partition_count"])
    
    # Validate that values are not NaN or invalid
    if math.isnan(float(bt)) or math.isnan(float(pc)):
        raise ValueError(f"Invalid config values: broadcast_threshold={bt}, partition_count={pc}")
    
    # Ensure positive values
    if pc <= 0:
        pc = 200  # default fallback
    if bt <= 0:
        bt = 10  # default fallback in MB

    conf = {
        "spark.sql.shuffle.partitions":
            str(pc),
        "spark.sql.autoBroadcastJoinThreshold":
            str(bt * 1024 * 1024),           # MB → bytes
        "spark.sql.join.preferSortMergeJoin":
            "true" if js == "sort_merge" else "false",
    }

    if js == "broadcast":
        conf["spark.sql.autoBroadcastJoinThreshold"] = str(bt * 1024 * 1024)
    elif js == "hash":
        # disable broadcast so Spark falls back to hash join
        conf["spark.sql.autoBroadcastJoinThreshold"] = "-1"

    return conf