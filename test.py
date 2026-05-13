# test.py
from pyspark.sql import SparkSession
import xgboost as xgb
import mlflow
import numpy as np

spark = SparkSession.builder \
    .appName("SetupTest") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "val"])
print(f"Spark OK — row count: {df.count()}")

X = np.random.rand(100, 5)
y = np.random.rand(100)
model = xgb.XGBRegressor(n_estimators=10).fit(X, y)
print(f"XGBoost OK — prediction: {model.predict(X[:1])}")

mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    mlflow.log_param("test", "setup_check")
    mlflow.log_metric("dummy_metric", 1.0)
print("MLflow OK — check http://localhost:5000")

spark.stop()