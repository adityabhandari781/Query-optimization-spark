import re

def extract_plan_features(spark, df):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        df.explain(extended=True)
    plan_str = buf.getvalue()

    # split into logical vs physical sections
    sections  = plan_str.split("== Physical Plan ==")
    phys_plan = sections[1] if len(sections) > 1 else plan_str
    logic_plan = sections[0]

    features = {
        "plan_raw": plan_str,

        # logical plan operator counts (existing)
        "op_hash_join_first":        len(re.findall(r"HashJoin",                 plan_str)),
        "op_sort_merge_join_first":  len(re.findall(r"SortMergeJoin",            plan_str)),
        "op_broadcast_join_first":   len(re.findall(r"BroadcastHashJoin",        plan_str)),
        "op_aggregate_first":        len(re.findall(r"HashAggregate",            plan_str)),
        "op_sort_first":             len(re.findall(r"Sort\b",                   plan_str)),
        "op_exchange_first":         len(re.findall(r"Exchange",                 plan_str)),
        "op_filter_first":           len(re.findall(r"Filter\b",                 plan_str)),
        "op_scan_first":             len(re.findall(r"FileScan|LogicalRDD|Range", plan_str)),

        # physical plan specific — what Spark will ACTUALLY execute
        "phys_sort":      len(re.findall(r"Sort\b",          phys_plan)),
        "phys_exchange":  len(re.findall(r"Exchange",         phys_plan)),
        "phys_broadcast": len(re.findall(r"BroadcastHashJoin",phys_plan)),

        "plan_depth_first": max(
            (len(line) - len(line.lstrip()))
            for line in plan_str.splitlines() if line.strip()
        ) // 3,
    }

    row_counts = re.findall(r"rows=(\d+)", plan_str)
    if row_counts:
        counts = list(map(int, row_counts))
        features["estimated_rows_min"]  = min(counts)
        features["estimated_rows_max"]  = max(counts)
        features["estimated_rows_mean"] = sum(counts) / len(counts)
    else:
        features["estimated_rows_min"]  = -1
        features["estimated_rows_max"]  = -1
        features["estimated_rows_mean"] = -1

    return features