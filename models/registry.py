import math, mlflow, mlflow.sklearn
from models.train import train_all

MLFLOW_URI      = "http://localhost:5000"
EXPERIMENT_NAME = "spark-query-optimizer"

def _safe(val):
    """Replace NaN/Inf with 0.0 so SQLite never sees them."""
    if val is None:
        return 0.0
    try:
        return 0.0 if (math.isnan(val) or math.isinf(val)) else float(val)
    except Exception:
        return 0.0

def log_and_save(results: dict) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_ids = {}

    for target, res in results.items():
        with mlflow.start_run(run_name=target):
            mlflow.log_param("target",       target)
            mlflow.log_param("cv_strategy",  "kfold_5")
            mlflow.log_param("n_train_rows", 117)

            mlflow.log_metric("cv_mae",     _safe(res["cv_mae"]))
            mlflow.log_metric("cv_mae_std", _safe(res["cv_mae_std"]))
            mlflow.log_metric("cv_r2",      _safe(res["cv_r2"]))
            mlflow.log_metric("train_mae",  _safe(res["train_mae"]))
            mlflow.log_metric("train_r2",   _safe(res["train_r2"]))

            for feat, imp in res["importances"].head(5).items():
                mlflow.log_param(f"imp_{feat[:30]}", round(float(imp), 4))

            mlflow.sklearn.log_model(
                res["pipeline"],
                name=f"model_{target}",
                registered_model_name=f"spark_optimizer_{target}",
            )
            run_ids[target] = mlflow.active_run().info.run_id
            print(f"[MLflow] Logged {target} → run {run_ids[target][:8]}...")

    return run_ids

def load_models(run_ids: dict) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    pipelines = {}
    for target, run_id in run_ids.items():
        uri = f"runs:/{run_id}/model_{target}"
        pipelines[target] = mlflow.sklearn.load_model(uri)
        print(f"[MLflow] Loaded {target} from run {run_id[:8]}")
    return pipelines