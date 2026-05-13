from models.train    import train_all
from models.registry import log_and_save, load_models
from features.engineer import build_feature_matrix

print("=" * 55)
print("  PHASE 3 — MODEL TRAINING")
print("=" * 55)
results = train_all(verbose=True)

print("\n\nLogging to MLflow...")
run_ids = log_and_save(results)

print("\n\nRound-trip load + predict test...")
pipelines = load_models(run_ids)

X, y, _ = build_feature_matrix(verbose=False)

print("\nPredictions on first 3 queries:")
print(f"{'Query':<8}", end="")
for t in pipelines:
    label = t.replace("target_", "").replace("_", " ")[:14]
    print(f"  {label:>16}", end="")
print()

for i in range(min(3, len(X))):
    row = X.values[i:i+1]
    print(f"  Q{i+1:<5}", end="")
    for target, pipe in pipelines.items():
        pred = pipe.predict(row)[0]
        print(f"  {pred:>16.2f}", end="")
    print()

print("\n✓ Phase 3 complete — models trained and registered in MLflow")
print("  Open http://localhost:5000 to inspect runs")