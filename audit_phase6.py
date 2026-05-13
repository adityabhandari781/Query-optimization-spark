from feedback.retrain import run_feedback_cycle, retrain
from feedback.drift   import compute_residuals, check_drift
from profiler.feature_store import load_all_records

print("=" * 55)
print("  PHASE 6 — CONTINUOUS LEARNING FEEDBACK LOOP")
print("=" * 55)

# ── Show current live data state ───────────────────────────────────────────────
raw       = load_all_records()
stage_raw = raw[raw["record_type"] == "stage"].copy()

total_q   = stage_raw["query_id"].nunique()
live_q    = stage_raw["actual_exec_ms"].notna().any()

print(f"\n  Feature store state:")
print(f"    Total queries    : {total_q}")
print(f"    Has live timings : {live_q}")

# ── Drift check ────────────────────────────────────────────────────────────────
residuals   = compute_residuals(stage_raw)
drift_state = check_drift(residuals, min_samples=3)  # low threshold for demo

print(f"\n  Residuals computed from {len(residuals)} live-timed queries")

# ── Run full feedback cycle ────────────────────────────────────────────────────
# Force retrain=True by temporarily patching threshold to 0
# so we can see the full retraining output even with few live samples
print(f"\n  Forcing retrain to demonstrate full cycle...")
results = retrain(verbose=True)

# ── Compare Phase 3 vs Phase 6 CV R² ──────────────────────────────────────────
print(f"\n\n{'═'*55}")
print(f"  MODEL IMPROVEMENT SUMMARY")
print(f"{'═'*55}")
print(f"  {'Target':<30} {'CV R² (Phase 6)':>15}")
print(f"  {'─'*30} {'─'*15}")
for target, res in results.items():
    label = target.replace("target_", "").replace("_", " ")
    print(f"  {label:<30} {res['cv_r2']:>15.3f}")

print(f"\n✓ Phase 6 complete")
print(f"  The feedback loop is now closed.")
print(f"  Each AdaptiveRunner.run() call adds measured data,")
print(f"  which improves model accuracy on the next retrain cycle.")