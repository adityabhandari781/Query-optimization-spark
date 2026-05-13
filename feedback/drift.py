import numpy as np
import pandas as pd


def compute_residuals(df: pd.DataFrame) -> pd.Series:
    """
    Compute prediction residuals from feature store records
    that contain both predicted and actual execution times.
    Only rows produced by AdaptiveRunner have both columns.
    """
    live = df.dropna(subset=["actual_exec_ms", "predicted_latency_ms"])
    if live.empty:
        return pd.Series(dtype=float)

    residuals = (
        live["actual_exec_ms"] - live["predicted_latency_ms"]
    ) / live["predicted_latency_ms"].clip(lower=1)

    return residuals


def check_drift(residuals: pd.Series,
                mae_threshold: float = 0.40,
                min_samples: int = 5) -> dict:
    """
    Simple drift check: if mean absolute residual exceeds
    mae_threshold (default 40%) across at least min_samples
    live queries, flag for retraining.

    Returns a dict with drift status and supporting stats.
    """
    if len(residuals) < min_samples:
        return {
            "should_retrain": False,
            "reason": f"only {len(residuals)} live samples (need {min_samples})",
            "mean_abs_residual": None,
            "n_samples": len(residuals),
        }

    mar = residuals.abs().mean()
    bias = residuals.mean()      # positive = model underestimates (slow queries)
    std  = residuals.std()

    should_retrain = mar > mae_threshold

    return {
        "should_retrain":    should_retrain,
        "reason":            f"MAR={mar:.2%} {'>' if should_retrain else '<='} threshold={mae_threshold:.0%}",
        "mean_abs_residual": mar,
        "bias":              bias,
        "std":               std,
        "n_samples":         len(residuals),
    }