from features.engineer import build_feature_matrix
import pandas as pd

X, y, full_df = build_feature_matrix(verbose=True)

# Sanity checks
assert X.isnull().sum().sum() == 0,  "NaNs in feature matrix!"
assert y.isnull().sum().sum() == 0,  "NaNs in target matrix!"
assert len(X) == len(y),             "Row count mismatch!"

print("\n✓ All checks passed — feature matrix is clean")
print(f"\nSample row:\n{X.iloc[0].to_string()}")