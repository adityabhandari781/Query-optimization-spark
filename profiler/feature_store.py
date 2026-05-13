# feature_store.py

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import glob
from datetime import datetime
from config import FEATURE_STORE_PATH


def _partition_value(file_path: str, key: str):
    prefix = f"{key}="
    for part in os.path.normpath(file_path).split(os.sep):
        if part.startswith(prefix):
            return part[len(prefix):]
    return None

def save_records(records: list, query_id: str):
    if not records:
        return

    df = pd.DataFrame(records)
    date_value = datetime.utcnow().strftime("%Y-%m-%d")

    partition_path = os.path.join(
        FEATURE_STORE_PATH,
        f"date={date_value}",
        f"query={query_id}"
    )
    os.makedirs(partition_path, exist_ok=True)

    # Keep partition columns only in folder names to prevent schema merge conflicts.
    table = pa.Table.from_pandas(
        df.drop(columns=["query_id", "date"], errors="ignore"),
        preserve_index=False,
    )

    pq.write_table(
        table,
        os.path.join(partition_path, "data.parquet"),
        compression="snappy"
    )
    print(f"[FeatureStore] Saved {len(records)} records → {partition_path}")

def load_all_records() -> pd.DataFrame:
    """Load the full feature store into a pandas DataFrame for ML training."""
    parquet_files = sorted(
        glob.glob(os.path.join(FEATURE_STORE_PATH, "date=*", "query=*", "*.parquet"))
    )

    if not parquet_files:
        return pd.DataFrame()

    frames = []
    for file_path in parquet_files:
        shard = pd.read_parquet(file_path)
        if shard.empty:
            continue

        date_value = _partition_value(file_path, "date")
        query_value = _partition_value(file_path, "query")

        if "date" not in shard.columns:
            shard["date"] = date_value
        else:
            shard["date"] = shard["date"].astype("string")
            if date_value is not None:
                shard["date"] = shard["date"].fillna(date_value)

        if "query_id" not in shard.columns:
            shard["query_id"] = query_value
        else:
            shard["query_id"] = shard["query_id"].astype("string")
            if query_value is not None:
                shard["query_id"] = shard["query_id"].fillna(query_value)

        frames.append(shard)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = combined["date"].astype("string")
    combined["query_id"] = combined["query_id"].astype("string")
    return combined