from profiler.feature_store import load_all_records
import pandas as pd

df = load_all_records()
print(f"Total records : {len(df)}")
print(f"Unique queries: {df['query_id'].nunique()}")
print(f"Columns       : {list(df.columns)}")
print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nNumeric summary:\n{df.describe().T[['mean','min','max']]}")