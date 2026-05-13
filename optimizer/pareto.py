import numpy as np
import pandas as pd
from models.registry import load_models
from optimizer.candidate_space import enumerate_candidates, config_to_feature_delta
from features.engineer import FEATURE_COLS


def score_candidates(
    base_feature_dict: dict,
    pipelines: dict,
    candidates: list[dict],
) -> pd.DataFrame:
    """
    Score every candidate config on all 4 objectives using the ML models.
    Returns a DataFrame with one row per candidate.
    """
    rows = []
    for cand in candidates:
        x = config_to_feature_delta(base_feature_dict, cand).reshape(1, -1)
        row = dict(cand)
        for target, pipe in pipelines.items():
            row[target] = float(pipe.predict(x)[0])
        rows.append(row)

    return pd.DataFrame(rows)


def is_dominated(costs: np.ndarray, i: int) -> bool:
    """
    Return True if solution i is dominated by any other solution.
    costs shape: (n_candidates, n_objectives) — all objectives are minimised.
    Note: cpu_efficiency is maximised so we pass it in negated.
    """
    for j in range(len(costs)):
        if j == i:
            continue
        # j dominates i if j is no worse on all objectives and strictly
        # better on at least one
        if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
            return True
    return False


def compute_pareto_front(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the non-dominated (Pareto-optimal) candidates.
    Minimise: latency, memory, shuffle.
    Maximise: cpu_efficiency  (passed in as negative so we minimise).
    """
    obj_cols = [
        "target_latency_ms",
        "target_memory_mb",
        "target_shuffle_mb",
        "target_cpu_efficiency",   # will be negated
    ]

    costs = scored_df[obj_cols].values.copy()
    costs[:, 3] = -costs[:, 3]    # negate cpu_efficiency → minimise

    pareto_mask = [not is_dominated(costs, i) for i in range(len(costs))]
    return scored_df[pareto_mask].reset_index(drop=True)