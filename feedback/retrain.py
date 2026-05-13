import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from profiler.feature_store import load_all_records
from features.engineer      import FEATURE_COLS, aggregate_per_query, add_derived_features
from models.train           import MODEL_CONFIGS, build_pipeline
from models.registry        import log_and_save


def build_live_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    For records that have actual_exec_ms (from AdaptiveRunner),
    use measured values as targets instead of synthetic formulas.
    For records without measurements, fall back to synthetic targets.
    """
    has_live = df["actual_exec_ms"].notna()

    # latency: use actual measurement where available
    df["target_latency_ms"] = np.where(
        has_live,
        df["actual_exec_ms"],
        # synthetic fallback for records without measurements
        (df["num_tasks_sum"] * 120 +
         df["join_complexity"] * 800 +
         df["op_sort_first"] * 500 +
         df["op_exchange_first"] * 400).clip(lower=100)
    )

    # memory: synthetic (StatusTracker doesn't expose real memory)
    df["target_memory_mb"] = (
        df["num_tasks_max"] * 40 +
        df["op_aggregate_first"] * 300 +
        df["op_sort_first"] * 200
    ).clip(lower=64)

    # shuffle: synthetic
    df["target_shuffle_mb"] = (
        df["op_exchange_first"] * 250 +
        df["join_complexity"] * 150
    ).clip(lower=0)

    # cpu efficiency: synthetic
    df["target_cpu_efficiency"] = (
        0.9
        - df["op_sort_merge_join_first"] * 0.12
        + df["uses_broadcast"] * 0.08
        - df["had_failures"] * 0.2
    ).clip(0.1, 1.0)

    return df


TARGET_COLS = [
    "target_latency_ms",
    "target_memory_mb",
    "target_shuffle_mb",
    "target_cpu_efficiency",
]


def retrain(verbose: bool = True) -> dict:
    """
    Full retrain pipeline on the current feature store contents.
    Uses actual measured latency where available, synthetic elsewhere.
    Logs new model versions to MLflow.
    """
    # ── 1. Load all records ────────────────────────────────────────────────────
    raw = load_all_records()
    raw = raw[raw["record_type"] == "stage"].copy()

    # ── 2. Aggregate per query ─────────────────────────────────────────────────
    per_q = aggregate_per_query(raw)

    # bring actual_exec_ms into per-query level (take max across stages)
    if "actual_exec_ms" in raw.columns:
        live_times = raw.groupby("query_id")["actual_exec_ms"].max().reset_index()
        per_q = per_q.merge(live_times, on="query_id", how="left")
    else:
        per_q["actual_exec_ms"] = np.nan

    # ── 3. Derived features + live targets ────────────────────────────────────
    per_q   = add_derived_features(per_q)
    per_q   = build_live_targets(per_q)

    X = per_q[FEATURE_COLS].values
    y = per_q[TARGET_COLS]

    n_live = per_q["actual_exec_ms"].notna().sum()

    if verbose:
        print(f"\n  Training rows      : {len(per_q)}")
        print(f"  Live measurements  : {n_live}  "
              f"({n_live/len(per_q)*100:.0f}% of data)")
        print(f"  Synthetic fallback : {len(per_q) - n_live}")

    # ── 4. Retrain all four models ─────────────────────────────────────────────
    cv      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for target in TARGET_COLS:
        y_arr = y[target].values
        pipe  = build_pipeline(target)

        cv_r2 = cross_val_score(
            pipe, X, y_arr, cv=cv, scoring="r2"
        )
        cv_mae = -cross_val_score(
            pipe, X, y_arr, cv=cv, scoring="neg_mean_absolute_error"
        )

        pipe.fit(X, y_arr)
        train_r2  = r2_score(y_arr, pipe.predict(X))
        train_mae = mean_absolute_error(y_arr, pipe.predict(X))

        importances = pd.Series(
            pipe.named_steps["model"].feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=False)

        results[target] = {
            "pipeline":    pipe,
            "cv_r2":       float(np.nanmean(cv_r2)),
            "cv_mae":      float(np.nanmean(cv_mae)),
            "cv_mae_std":  float(np.nanstd(cv_mae)),
            "train_r2":    float(train_r2),
            "train_mae":   float(train_mae),
            "importances": importances,
        }

        if verbose:
            print(f"\n  {'─'*50}")
            print(f"  {target}")
            print(f"  {'─'*50}")
            print(f"  CV-5 R²  : {results[target]['cv_r2']:>8.3f}")
            print(f"  CV-5 MAE : {results[target]['cv_mae']:>8.3f}  "
                  f"(±{results[target]['cv_mae_std']:.3f})")
            print(f"  Train R² : {train_r2:>8.3f}")
            top3 = importances.head(3)
            for feat, imp in top3.items():
                bar = "█" * int(imp * 30)
                print(f"    {feat:<35} {imp:.3f}  {bar}")

    return results


def run_feedback_cycle(verbose: bool = True) -> dict:
    """
    Full feedback cycle:
      1. Load feature store
      2. Check drift on live queries
      3. Retrain if needed (or forced)
      4. Log new model versions to MLflow
    """
    from feedback.drift import compute_residuals, check_drift

    raw  = load_all_records()
    raw  = raw[raw["record_type"] == "stage"].copy()

    # drift check
    residuals   = compute_residuals(raw)
    drift_state = check_drift(residuals)

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  DRIFT CHECK")
        print(f"{'═'*55}")
        print(f"  Live query samples : {drift_state['n_samples']}")
        if drift_state["mean_abs_residual"] is not None:
            print(f"  Mean abs residual  : {drift_state['mean_abs_residual']:.2%}")
            print(f"  Bias               : {drift_state['bias']:+.2%}")
            print(f"  Std of residuals   : {drift_state['std']:.2%}")
        print(f"  Decision           : {drift_state['reason']}")
        print(f"  Retrain?           : {'YES ✓' if drift_state['should_retrain'] else 'NO'}")

    if drift_state["should_retrain"]:
        if verbose:
            print(f"\n{'═'*55}")
            print(f"  RETRAINING")
            print(f"{'═'*55}")
        results = retrain(verbose=verbose)

        if verbose:
            print(f"\nLogging retrained models to MLflow...")
        run_ids = log_and_save(results)

        return {"drift": drift_state, "retrained": True, "run_ids": run_ids}
    else:
        return {"drift": drift_state, "retrained": False}