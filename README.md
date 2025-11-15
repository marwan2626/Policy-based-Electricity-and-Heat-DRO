Overview
--------
v1: Deterministic baseline (no real-time recourse, no DRO).

v2: Migration complete to K-only affine recourse. Legacy λ/χ/ρ budget policy code has been pruned; flags now force it off. Real-time robustness handled via affine K gains and DRCC tightening only.

v3: Out-of-sample (OOS) Monte Carlo analysis harness for a chosen v2 (or v4) solution.

v4: Extended OOS + instrumentation (network loading, voltages, BESS envelopes, clipping diagnostics). Shadow λ/χ comparison path retained behind a feature flag `ENABLE_POLICY_SHADOW_COMPARE` (default False) for archival evaluation.

Scripts
-------
`oos-analysis.py`: Generates plots from v3/v4 simulation outputs.
`v4_consistency_report.py`: Aggregates multi-epsilon divergence / cost metrics (now oriented to K-only; shadow columns appear only if the flag is re-enabled).

Policy Architecture (Post-Cleanup)
----------------------------------
- Active recourse: affine K gains only.
- Removed: λ/χ/ρ budget channels (previous proxy decomposition).
- Optional (disabled): shadow reconstruction for historical comparison.

How to Re-Enable Shadow Comparison (Optional)
---------------------------------------------
Set `ENABLE_POLICY_SHADOW_COMPARE = True` near the top of `dso_model_v4.py` before running simulations. Extra shadow_* columns will appear in the summary CSV.

Next Steps
----------
If you need to further simplify v4, you can delete the guarded shadow block entirely once no longer required for publications or validation.

Reproduction Quick Start
------------------------
```powershell
conda run -n CMESnew python dso_model_v4.py --epsilon 0.10 --max-traj 50
conda run -n CMESnew python v4_consistency_report.py
```

This will produce K-only divergence and cost metrics.