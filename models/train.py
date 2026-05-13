import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from features.engineer import build_feature_matrix, FEATURE_COLS, TARGET_COLS

MODEL_CONFIGS = {
    "target_latency_ms": dict(
        n_estimators=40, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    "target_memory_mb": dict(
        n_estimators=40, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    "target_shuffle_mb": dict(
        n_estimators=40, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    "target_cpu_efficiency": dict(
        n_estimators=40, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
}

def build_pipeline(target: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBRegressor(**MODEL_CONFIGS[target], verbosity=0)),
    ])

def train_all(verbose: bool = True) -> dict:
    X, y, _ = build_feature_matrix(verbose=False)
    X_arr   = X.values
    results = {}

    # KFold with 5 splits — needs at least 5 samples per fold, works fine with 100+
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in TARGET_COLS:
        y_arr = y[target].values
        pipe  = build_pipeline(target)

        cv_mae = -cross_val_score(
            pipe, X_arr, y_arr,
            cv=cv, scoring="neg_mean_absolute_error"
        )
        cv_r2 = cross_val_score(
            pipe, X_arr, y_arr,
            cv=cv, scoring="r2"
        )

        # fit on full dataset for inference
        pipe.fit(X_arr, y_arr)
        train_preds = pipe.predict(X_arr)
        train_mae   = mean_absolute_error(y_arr, train_preds)
        train_r2    = r2_score(y_arr, train_preds)

        importances = pd.Series(
            pipe.named_steps["model"].feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=False)

        results[target] = {
            "pipeline":    pipe,
            "cv_mae":      float(np.nanmean(cv_mae)),
            "cv_mae_std":  float(np.nanstd(cv_mae)),
            "cv_r2":       float(np.nanmean(cv_r2)),
            "train_mae":   float(train_mae),
            "train_r2":    float(train_r2),
            "importances": importances,
        }

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  {target}")
            print(f"{'─'*55}")
            print(f"  CV-5 MAE : {results[target]['cv_mae']:>10.3f}  (±{results[target]['cv_mae_std']:.3f})")
            print(f"  CV-5 R²  : {results[target]['cv_r2']:>10.3f}")
            print(f"  Train MAE: {train_mae:>10.3f}")
            print(f"  Train R² : {train_r2:>10.3f}")
            print(f"  Top-5 features:")
            for feat, imp in importances.head(5).items():
                bar = "█" * int(imp * 40)
                print(f"    {feat:<35} {imp:.3f}  {bar}")

    return results