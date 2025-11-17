# Out-of-sample analysis for v4 results
# Builds an overview figure with costs vs epsilon and violation counts vs epsilon.

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    import pyarrow  # noqa: F401 (ensure parquet engine availability if installed)
except Exception:  # soft dependency
    pyarrow = None
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
import matplotlib.colors as mcolors
import glob
import re
import math

# === User config ===
# Distribution toggle: 'gaussian' (default) -> v4_oos; 'uniform' -> v4_oos_uniform; 'contaminated' -> v4_oos_contaminated; 'studentt' -> v4_oos_studentt
# Defaulting to 'gaussian' to match dso_model_v4.py's default OUTDIR ('v4_oos')
DISTRIBUTION: str = os.getenv('V4_SAMPLE_DISTRIBUTION', 'gaussian').strip().lower()
# Enable splitting by RT mode suffix (_rt_on / _rt_off) if present in filenames.
INCLUDE_RT_SPLIT: bool = True  # set False to ignore RT mode suffixes and aggregate silently

# Editable in-file aggregation toggle (no console needed):
#   Set ENABLE_AGGREGATE_GAUSSIAN_STUDENTT = True to automatically build a combined dataset
#   from Gaussian and Student-t results, then run ALL analyses on the combined data.
#   You can optionally set AGGREGATE_GAUSSIAN_STUDENTT_WEIGHTS = (w_gaussian, w_studentt) for the small
#   “overview aggregation” figure (full analyses always use concatenated samples for fairness).
ENABLE_AGGREGATE_GAUSSIAN_STUDENTT: bool = True  # disable by default to avoid legacy aggregation 'label' KeyError with RT variants
AGGREGATE_GAUSSIAN_STUDENTT_WEIGHTS: Tuple[float, float] = (0.5, 0.5)  # editable; only affects overview weighted bars
import sys as _sys
# New: optional aggregation of two distributions, e.g. --aggregate-dists gaussian,studentt --agg-weights 0.5,0.5
AGGREGATE_DISTS: List[str] | None = None
AGGREGATE_WEIGHTS: Tuple[float, float] | None = None
# Allow CLI override: --dist <gaussian|uniform|contaminated|studentt> or --distribution <...>
_argv = list(_sys.argv[1:])
for _flag in ('--dist', '--distribution'):
    if _flag in _argv:
        try:
            _val = _argv[_argv.index(_flag) + 1].strip().lower()
            if _val in {'gaussian','uniform','contaminated','studentt'}:
                DISTRIBUTION = _val
            else:
                raise ValueError()
        except Exception:
            raise SystemExit("Provide --dist/--distribution followed by 'gaussian', 'uniform', 'contaminated', or 'studentt'.")
# Convenience short-hand flags for common dual aggregation (Gaussian + Student-t)
if '--aggregate-gs' in _argv or '--aggregate-gaussian-studentt' in _argv:
    AGGREGATE_DISTS = ['gaussian','studentt']
    # Optional: allow a single weight ratio like --gs-ratio 0.6 meaning 0.6 gaussian / 0.4 studentt
    if '--gs-ratio' in _argv:
        try:
            _rval = float(_argv[_argv.index('--gs-ratio') + 1])
            if _rval <= 0:
                raise ValueError()
            AGGREGATE_WEIGHTS = (_rval, 1.0 - _rval if _rval < 1.0 else 0.0)
        except Exception:
            raise SystemExit("--gs-ratio expects a positive float (<=1 recommended), e.g. 0.6 for 60% Gaussian / 40% Student-t.")
# Parse aggregation flags
if '--aggregate-dists' in _argv:
    try:
        _raw = _argv[_argv.index('--aggregate-dists') + 1]
        parts = [p.strip().lower() for p in _raw.split(',') if p.strip()]
        if len(parts) != 2 or any(p not in {'gaussian','uniform','contaminated','studentt'} for p in parts):
            raise ValueError()
        AGGREGATE_DISTS = parts
    except Exception:
        raise SystemExit("--aggregate-dists expects exactly two comma-separated values from {gaussian,uniform,contaminated,studentt}.")
    
# Apply in-file aggregation toggle if no CLI aggregation provided
if AGGREGATE_DISTS is None and ENABLE_AGGREGATE_GAUSSIAN_STUDENTT:
    AGGREGATE_DISTS = ['gaussian', 'studentt']
    try:
        g, s = AGGREGATE_GAUSSIAN_STUDENTT_WEIGHTS
        total = float(g) + float(s)
        if total > 0:
            AGGREGATE_WEIGHTS = (float(g)/total, float(s)/total)
    except Exception:
        AGGREGATE_WEIGHTS = (0.5, 0.5)
if '--agg-weights' in _argv:
    try:
        _raww = _argv[_argv.index('--agg-weights') + 1]
        w = [float(p.strip()) for p in _raww.split(',') if p.strip()]
        if len(w) != 2:
            raise ValueError()
        s = w[0] + w[1]
        if s <= 0:
            raise ValueError()
        AGGREGATE_WEIGHTS = (w[0]/s, w[1]/s)
    except Exception:
        raise SystemExit("--agg-weights expects two comma-separated numbers, e.g., 0.5,0.5 (they will be normalized).")
if DISTRIBUTION not in {'gaussian', 'uniform', 'contaminated', 'studentt'}:
    DISTRIBUTION = 'gaussian'
RESULTS_DIR = (
    "v4_oos_uniform" if DISTRIBUTION == 'uniform' else (
        "v4_oos_contaminated" if DISTRIBUTION == 'contaminated' else (
            "v4_oos_studentt" if DISTRIBUTION == 'studentt' else "v4_oos"
        )
    )
)
print(f"[config] DISTRIBUTION = {DISTRIBUTION} | RESULTS_DIR = {RESULTS_DIR}")
if AGGREGATE_DISTS is None and not os.path.isdir(RESULTS_DIR):
    raise FileNotFoundError(
        "Results directory '" + RESULTS_DIR + "' not found. "
        "If you intended to analyze uniform runs, set --dist uniform (or V4_SAMPLE_DISTRIBUTION=uniform) and run v4 in uniform mode first; "
        "for contaminated, use --dist contaminated and run v4 in contaminated mode; for studentt, use --dist studentt and run v4 in studentt mode."
    )
# Global Matplotlib style: Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
# Epsilon set: replace 0.20 with 0.05 -> show 0.15, 0.10, 0.05 (descending conservatism)
EPSILONS: List[float] = [0.15, 0.10, 0.05]
# Include baseline (k=1, no network tightening) summary as an extra category.
# We now call this 'stochastic' (it still has RT budgets sized by forecast std but no quantile amplification).
INCLUDE_DETERMINISTIC: bool = True
DETERMINISTIC_LABEL: str = "deterministic"  # displayed label for drcc_false baseline run
OUT_FIG = "oos_overview.png"
OUT_CSV = "oos_overview_summary.csv"
SHOW: bool = False  # set True to display interactively
PLOT_SOC_ENVELOPES: bool = True
SOC_ENV_FIG = "soc_envelopes.png"
SOC_FINAL_FIG = "soc_final_envelope.png"  # final timestep summary (median with 5–95% error bars)
SOC_FINAL_BOXPLOT_FIG = "soc_final_boxplot.png"  # final timestep distribution across cases (boxplot)
SOC_DAILY_BOXPLOT_FIG = "soc_daily_boxplot.png"  # full-day distribution across cases (boxplot)
FRONTIER_CSV = "frontier_summary.csv"
PLOT_FRONTIER_SCATTER: bool = True
FRONTIER_SCATTER_FIG = "frontier_scatter.png"
PLOT_POLICY_HEATMAPS: bool = True
POLICY_HEATMAP_FIG = "policy_heatmaps.png"
PLOT_FRONTIER_TRAJECTORY_SCATTER: bool = True
FRONTIER_TRAJECTORY_SCATTER_FIG = "frontier_trajectory_scatter.png"
FRONTIER_HYBRID_SCATTER_FIG = "frontier_hybrid_scatter.png"  # new hybrid figure (cloud + mean)

# New: per-timestep transformer violation probability plot
PLOT_TRAFO_VIOLATION_TIME_PROFILE: bool = True
TRAFO_VIOLATION_TIME_PROFILE_FIG = "trafo_violation_time_profile.png"
# New: heatmap of per-timestep transformer overload chance (cases x time)
PLOT_TRAFO_VIOLATION_HEATMAP: bool = True
TRAFO_VIOLATION_HEATMAP_FIG = "trafo_violation_heatmap.png"
FRONTIER_HYBRID_SCATTER_FIG = "frontier_hybrid_scatter.png"
# New: time series of K-gains across epsilon cases (replacing legacy lambda/chi focus)
PLOT_POLICY_LAMBDA_TIME_SERIES: bool = False
POLICY_LAMBDA_TIME_SERIES_FIG = "policy_lambda_time_series.png"  # kept for backward compatibility if re-enabled
PLOT_POLICY_K_TIME_SERIES: bool = True
POLICY_K_TIME_SERIES_FIG = "policy_k_time_series.png"
# New: Tail/zoomed overview figure (additional diagnostics; does not replace existing plots)
PLOT_TAIL_OVERVIEW: bool = True
TAIL_OVERVIEW_FIG = "oos_overview_tail.png"

# New: dedicated two-violin comparison (deterministic vs epsilon=0.05)
PLOT_VIOLIN_COMPARE: bool = True
VIOLIN_COMPARE_FIG = "violin_compare_det_vs_005.png"
PLOT_VIOLIN_EPS_010_RT_COMPARE: bool = True
VIOLIN_COMPARE_010_RT_FIG = "violin_compare_010_rt_on_vs_off.png"
PLOT_VIOLIN_ALL_CASES: bool = True  # new: standalone multi-case violin (epsilon × RT variants)
VIOLIN_ALL_CASES_FIG: str = "violin_all_cases.png"
VIOLIN_SPLIT_USE_KDE: bool = True
VIOLIN_SPLIT_KDE_BW_ADJ: float = 2.0  # 1.0 ~ Matplotlib default smoothness; <1 sharper, >1 smoother

# New: deterministic vs ε=0.05 comparison of total transformer overload energy (MVAh)
PLOT_OVERLOAD_ENERGY_COMPARE: bool = True
OVERLOAD_ENERGY_COMPARE_FIG = "overload_energy_compare_det_vs_005.png"
OVERLOAD_ENERGY_COMPARE_CSV = "overload_energy_compare_det_vs_005.csv"
# New: ε=0.10 RT ON vs RT OFF overload energy comparison
PLOT_OVERLOAD_ENERGY_010_RT_COMPARE: bool = True
OVERLOAD_ENERGY_010_RT_FIG = "overload_energy_compare_010_rt_on_vs_off.png"
OVERLOAD_ENERGY_010_RT_CSV = "overload_energy_compare_010_rt_on_vs_off.csv"
# Parameters per user instruction
# Effective violation/overload threshold (pct). Allow env override; default to 80.09% to ignore tiny numerical overshoots.
try:
    OVERLOAD_THRESHOLD_PCT: float = float(os.getenv('V4_VIOL_THRESHOLD_PCT', '80.09'))
except Exception:
    OVERLOAD_THRESHOLD_PCT = 80.09
RATED_TRAFO_MVA: float = 0.5
STEP_HOURS: float = 0.25  # 15-minute steps
OVERLOAD_SAMPLE_COUNT_DEFAULT: int = 1000  # divide by number of samples to get per-sample energy

# New: deterministic vs ε=0.05 comparison of CVaR90 transformer loading (%)
PLOT_CVAR90_COMPARE: bool = True
CVAR90_COMPARE_FIG = "cvar90_loading_compare_det_vs_005.png"
CVAR90_COMPARE_CSV = "cvar90_loading_compare_det_vs_005.csv"

# New: evening transformer sigma decomposition diagnostic (v2 export)
PLOT_EVENING_TRAFO_SIGMA_DECOMP: bool = True
EVENING_TRAFO_SIGMA_DECOMP_FIG: str = "evening_sigma_decomposition.png"
EVENING_TRAFO_SIGMA_DECOMP_CSV: str = "evening_sigma_decomposition.csv"

# New: BESS clipping vs transformer violations correlation
PLOT_BESS_CLIPPING_CORRELATION: bool = True
BESS_CLIP_CORR_FIG: str = "bess_clipping_vs_violations.png"
BESS_CLIP_SUMMARY_PREFIX: str = "bess_clipping_summary_epsilon_"  # + <token>.csv

# Cost model parameters for OOS components
PV_CURT_PRICE_FACTOR = 1.0  # EUR per MWh of curtailed PV is factor * price
BESS_THROUGHPUT_COST_EUR_PER_MWH = 0.0  # cost per MWh of RT BESS throughput (set >0 if you price cycling)

# Shared colormap for heatmaps: white at 0, blue at max
WHITE_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list('white_blue', ['#ffffff', '#1f77b4'])

# --- IO helpers: parquet preferred, CSV fallback ---
def _read_parquet_or_csv(path: str) -> pd.DataFrame | None:
    """Try reading a parquet table; if unavailable or engine missing, fall back to CSV.

    If 'path' doesn't exist and endswith '.parquet', also try the sibling '.csv'.
    Returns a DataFrame or None on failure.
    """
    try:
        if os.path.exists(path):
            # Try parquet first if extension matches
            if path.lower().endswith('.parquet'):
                try:
                    return pd.read_parquet(path)
                except Exception:
                    # fall back to csv with same basename
                    csv_path = path[:-8] + 'csv'
                    if os.path.exists(csv_path):
                        try:
                            return pd.read_csv(csv_path)
                        except Exception:
                            return None
                    return None
            # If caller passed a CSV directly
            if path.lower().endswith('.csv'):
                try:
                    return pd.read_csv(path)
                except Exception:
                    return None
        # If file not present, attempt CSV sibling when parquet was expected
        if path.lower().endswith('.parquet'):
            csv_path = path[:-8] + 'csv'
            if os.path.exists(csv_path):
                try:
                    return pd.read_csv(csv_path)
                except Exception:
                    return None
        return None
    except Exception:
        return None


def epsilon_token(eps: float) -> str:
    return f"{eps:.2f}".replace(".", "_")

def _extract_rt_tag_from_name(name: str) -> str | None:
    """Return 'rt_on' / 'rt_off' / 'rt_unk' if present in filename, else None."""
    for tag in ('rt_on','rt_off','rt_unk'):
        if f"_{tag}" in name:
            return tag
    return None

def _rt_display(tag: str | None) -> str:
    if tag == 'rt_on':
        return 'RT ON'
    if tag == 'rt_off':
        return 'RT OFF'
    if tag == 'rt_unk':
        return 'RT ?'
    return ''

def find_summary_variant_paths(eps: float, results_dir: str) -> List[Tuple[str, str, str]]:
    """Strict RT-only locator: returns (rt_tag, summary_path, meta_path) for existing RT-suffixed variants.

    Unsuffixed legacy files are intentionally ignored.
    """
    token = epsilon_token(eps)
    patterns = [
        f"v4_summary_drcc_true_epsilon_{token}_rt_on.csv",
        f"v4_summary_drcc_true_epsilon_{token}_rt_off.csv",
        f"v4_summary_drcc_true_epsilon_{token}_rt_unk.csv",
    ]
    out: List[Tuple[str,str,str]] = []
    for fname in patterns:
        sp = os.path.join(results_dir, fname)
        if os.path.exists(sp):
            rt_tag = _extract_rt_tag_from_name(fname)  # guaranteed by pattern
            meta_name = f"v4_meta_drcc_true_epsilon_{token}_{rt_tag}.json"
            mp = os.path.join(results_dir, meta_name)
            out.append((rt_tag, sp, mp if os.path.exists(mp) else ''))
    return out

def find_deterministic_variant_paths(results_dir: str) -> List[Tuple[str,str,str]]:
    patterns = [
        "v4_summary_drcc_false_rt_on.csv",
        "v4_summary_drcc_false_rt_off.csv",
        "v4_summary_drcc_false_rt_unk.csv",
    ]
    out: List[Tuple[str,str,str]] = []
    for fname in patterns:
        sp = os.path.join(results_dir, fname)
        if os.path.exists(sp):
            rt_tag = _extract_rt_tag_from_name(fname)
            meta_name = f"v4_meta_drcc_false_{rt_tag}.json"
            mp = os.path.join(results_dir, meta_name)
            out.append((rt_tag, sp, mp if os.path.exists(mp) else ''))
    return out


def load_summary_for_epsilon(eps: float) -> pd.DataFrame:
    token = epsilon_token(eps)
    path = os.path.join(RESULTS_DIR, f"v4_summary_drcc_true_epsilon_{token}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Canonical summary missing for epsilon={eps:.2f}: {os.path.basename(path)}")
    return pd.read_csv(path)

# Dir-parameterized variants for aggregation mode
def _load_summary_for_epsilon_in_dir(results_dir: str, eps: float) -> pd.DataFrame:
    token = epsilon_token(eps)
    path = os.path.join(results_dir, f"v4_summary_drcc_true_epsilon_{token}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{results_dir}] Missing canonical summary for epsilon={eps:.2f}: {os.path.basename(path)}")
    return pd.read_csv(path)


def load_meta_for_epsilon(eps: float) -> Dict:
    token = epsilon_token(eps)
    path = os.path.join(RESULTS_DIR, f"v4_meta_drcc_true_epsilon_{token}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _load_meta_for_epsilon_in_dir(results_dir: str, eps: float) -> Dict:
    token = epsilon_token(eps)
    path = os.path.join(results_dir, f"v4_meta_drcc_true_epsilon_{token}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def compute_avg_price_from_v2(meta: Dict) -> float:
    v2_csv = meta.get("v2_results_csv")
    if v2_csv and os.path.exists(v2_csv):
        try:
            v2 = pd.read_csv(v2_csv)
            if "electricity_price_eur_mwh" in v2.columns:
                return float(pd.to_numeric(v2["electricity_price_eur_mwh"], errors="coerce").dropna().mean())
        except Exception:
            pass
    return 0.0


def compute_v2_base_cost(meta: Dict) -> float:
    """Compute v2 base (pre-RT) total cost to match 'total_cost_base (no RT proxies)'.

    We replicate v2's base by summing electricity import payments across the horizon:
      base ≈ sum_t price[t] * net_grid_power_mw[t] * dt_hours
    Notes:
      - net_grid_power_mw comes directly from v2 CSV; if absent, use import - export.
      - dt_hours taken from 'meta_dt_hours' column if available; else inferred from timestamp delta; else 0.25.
      - Tiny additions (e.g., bess_cost or pv_curtail_cost) are negligible; if needed, extend here.
    Returns NaN if unavailable.
    """
    try:
        v2_csv = meta.get('v2_results_csv') if isinstance(meta, dict) else None
        if not v2_csv:
            return float('nan')
        # Resolve relative path against RESULTS_DIR if needed
        if not os.path.exists(v2_csv):
            cand = os.path.join(RESULTS_DIR, v2_csv)
            v2_csv = cand if os.path.exists(cand) else v2_csv
        if not os.path.exists(v2_csv):
            return float('nan')
        df = pd.read_csv(v2_csv)
        if df is None or df.empty:
            return float('nan')
        # price
        if 'electricity_price_eur_mwh' not in df.columns:
            return float('nan')
        price = pd.to_numeric(df['electricity_price_eur_mwh'], errors='coerce')
        # net grid power
        if 'net_grid_power_mw' in df.columns:
            net_mw = pd.to_numeric(df['net_grid_power_mw'], errors='coerce')
        elif {'ext_grid_import_mw','ext_grid_export_mw'} <= set(df.columns):
            imp = pd.to_numeric(df['ext_grid_import_mw'], errors='coerce').fillna(0.0)
            exp = pd.to_numeric(df['ext_grid_export_mw'], errors='coerce').fillna(0.0)
            net_mw = imp - exp
        elif 'ext_grid_import_mw' in df.columns:
            net_mw = pd.to_numeric(df['ext_grid_import_mw'], errors='coerce')
        else:
            return float('nan')
        # dt_hours
        dt_hours = None
        if 'meta_dt_hours' in df.columns:
            try:
                dt_hours = float(pd.to_numeric(df['meta_dt_hours'], errors='coerce').dropna().iloc[0])
            except Exception:
                dt_hours = None
        if dt_hours is None:
            try:
                if 'timestamp' in df.columns and len(df['timestamp']) >= 2:
                    t0 = pd.to_datetime(df['timestamp'].iloc[0])
                    t1 = pd.to_datetime(df['timestamp'].iloc[1])
                    dt_hours = float((t1 - t0).total_seconds()) / 3600.0
            except Exception:
                dt_hours = None
        if dt_hours is None or not np.isfinite(dt_hours) or dt_hours <= 0:
            dt_hours = 0.25
        base_cost = float(np.nansum(price.to_numpy(dtype=float) * net_mw.to_numpy(dtype=float) * dt_hours))
        return base_cost
    except Exception:
        return float('nan')

def read_v2_da_total_cost(meta: Dict) -> float:
    """Read exported scalar DA total cost (electricity + flex curtailment) from v2 results CSV.

    Expects column 'da_total_cost_eur' with identical value each timestep.
    Returns first non-NaN value if present; else NaN.
    """
    try:
        v2_csv = meta.get('v2_results_csv') if isinstance(meta, dict) else None
        if not v2_csv:
            return float('nan')
        if not os.path.exists(v2_csv):
            cand = os.path.join(RESULTS_DIR, v2_csv)
            v2_csv = cand if os.path.exists(cand) else v2_csv
        if not os.path.exists(v2_csv):
            return float('nan')
        df = pd.read_csv(v2_csv, usecols=lambda c: c == 'da_total_cost_eur' or c == 'period')
        if 'da_total_cost_eur' not in df.columns or df.empty:
            return float('nan')
        vals = pd.to_numeric(df['da_total_cost_eur'], errors='coerce').dropna()
        if vals.empty:
            return float('nan')
        return float(vals.iloc[0])
    except Exception:
        return float('nan')


def aggregate_metrics(df: pd.DataFrame, avg_price_eur_mwh: float) -> Dict[str, float]:
    """Legacy aggregate (kept for compatibility, but plotting now uses direct RT cost columns).

    We still compute basic pieces for reference; line/trafo steps prefer the 80pct columns if present.
    """
    import_cost_mean = float(df.get("energy_cost_eur", pd.Series([0.0] * len(df))).mean())
    pv_curt_mwh_mean = float(df.get("pv_curtail_mwh", pd.Series([0.0] * len(df))).mean())
    pv_rt_curt_cost_mean = pv_curt_mwh_mean * max(avg_price_eur_mwh, 0.0) * float(PV_CURT_PRICE_FACTOR)
    bess_throughput_mwh_mean = float(df.get("bess_rt_energy_throughput_mwh", pd.Series([0.0] * len(df))).mean())
    bess_rt_cycle_cost_mean = bess_throughput_mwh_mean * float(BESS_THROUGHPUT_COST_EUR_PER_MWH)
    v_steps = int(df.get("steps_voltage_violation", pd.Series([0] * len(df))).sum())
    # Prefer 80% threshold columns if available
    if 'steps_line_over_80pct' in df.columns:
        l_steps = int(df['steps_line_over_80pct'].sum())
    elif 'steps_line_over_100pct' in df.columns:
        l_steps = int(df['steps_line_over_100pct'].sum())
    else:
        l_steps = 0
    if 'steps_trafo_over_80pct' in df.columns:
        t_steps = int(df['steps_trafo_over_80pct'].sum())
    elif 'steps_trafo_over_100pct' in df.columns:
        t_steps = int(df['steps_trafo_over_100pct'].sum())
    else:
        t_steps = 0
    return {
        "import_cost_eur_mean": import_cost_mean,
        "pv_rt_curt_cost_eur_mean": pv_rt_curt_cost_mean,
        "bess_rt_cycle_cost_eur_mean": bess_rt_cycle_cost_mean,
        "total_cost_eur_mean": import_cost_mean + pv_rt_curt_cost_mean + bess_rt_cycle_cost_mean,
        "voltage_steps": v_steps,
        "line_steps": l_steps,
        "trafo_steps": t_steps,
    }


def _results_dir_from_dist(name: str) -> str:
    name = name.strip().lower()
    return (
        "v4_oos" if name == 'gaussian' else (
            "v4_oos_uniform" if name == 'uniform' else (
                "v4_oos_contaminated" if name == 'contaminated' else (
                    "v4_oos_studentt" if name == 'studentt' else name
                )
            )
        )
    )


def _build_bundle_for_dir(results_dir: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load a minimal case bundle from a results directory.

    Returns:
      - rt_summary_like: DataFrame with at least columns ['label','epsilon','rt_imbalance_cost_mean','total_rt_cost_mean','trafo_steps','trafo_violation_probability_pct']
      - dist_by_label: dict label -> np.ndarray of per-(sample,t) max trafo loading (%)
    """
    def _build_rt_row_local(df_eps: pd.DataFrame, meta: Dict, eps: float | None, label: str) -> Dict[str, float]:
        horizon = np.nan
        v2_csv = meta.get('v2_results_csv') if isinstance(meta, dict) else None
        if v2_csv and os.path.exists(v2_csv):
            try:
                v2_df = pd.read_csv(v2_csv)
                if 'electricity_price_eur_mwh' in v2_df.columns:
                    horizon = int(pd.to_numeric(v2_df['electricity_price_eur_mwh'], errors='coerce').dropna().shape[0])
                else:
                    horizon = int(v2_df.shape[0])
            except Exception:
                horizon = np.nan
        da_import_cost_mean = float(df_eps.get('da_energy_cost_eur', pd.Series([0.0])).mean())
        # Direct DA total cost (electricity + flex curtailment) from v2 CSV (scalar)
        da_total_cost_eur = read_v2_da_total_cost(meta)
        rt_imb_cost_mean = float(df_eps.get('rt_imbalance_cost_eur', pd.Series([0.0])).mean())
        rt_pv_cost_mean = float(df_eps.get('rt_pv_curtail_cost_eur', pd.Series([0.0])).mean())
        rt_bess_cost_mean = float(df_eps.get('rt_bess_cycle_cost_eur', pd.Series([0.0])).mean())
        if 'steps_trafo_over_80pct' in df_eps.columns:
            trafo_steps = int(df_eps['steps_trafo_over_80pct'].sum())
        elif 'steps_trafo_over_100pct' in df_eps.columns:
            trafo_steps = int(df_eps['steps_trafo_over_100pct'].sum())
        else:
            trafo_steps = 0
        n_traj = len(df_eps)
        if isinstance(horizon, (int, np.integer)) and horizon > 0 and n_traj > 0:
            trafo_violation_probability_pct = (trafo_steps / (n_traj * horizon)) * 100.0
        else:
            trafo_violation_probability_pct = np.nan
        return {
            'epsilon': eps if eps is not None else np.nan,
            'label': label,
            'da_import_cost_mean': da_import_cost_mean,
            'da_total_cost_eur': da_total_cost_eur,
            'rt_imbalance_cost_mean': rt_imb_cost_mean,
            'rt_pv_cost_mean': rt_pv_cost_mean,
            'rt_bess_cost_mean': rt_bess_cost_mean,
            'trafo_steps': trafo_steps,
            'trafo_violation_probability_pct': trafo_violation_probability_pct,
            'horizon_timesteps': horizon,
            'n_trajectories': n_traj,
            'total_rt_cost_mean': rt_imb_cost_mean + rt_pv_cost_mean + rt_bess_cost_mean,
        }

    # summaries (RT-variant aware with legacy fallback)
    rows: List[Dict[str, float]] = []
    # Deterministic variants
    try:
        det_variants = find_deterministic_variant_paths(results_dir)
    except Exception:
        det_variants = []
    if det_variants:
        for rt_tag, summary_path, meta_path in det_variants:
            try:
                det_df = pd.read_csv(summary_path)
            except Exception:
                continue
            det_meta = {}
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path,'r',encoding='utf-8') as f:
                        det_meta = json.load(f)
                except Exception:
                    det_meta = {}
            rows.append(_build_rt_row_local(det_df, det_meta, None, f"{DETERMINISTIC_LABEL} ({_rt_display(rt_tag)})"))
    else:
        # Legacy unsuffixed deterministic
        det_path = os.path.join(results_dir, 'v4_summary_drcc_false.csv')
        if os.path.exists(det_path):
            try:
                det_df = pd.read_csv(det_path)
                det_meta = {}
                det_meta_path = os.path.join(results_dir, 'v4_meta_drcc_false.json')
                if os.path.exists(det_meta_path):
                    with open(det_meta_path,'r',encoding='utf-8') as f:
                        det_meta = json.load(f)
                rows.append(_build_rt_row_local(det_df, det_meta, None, DETERMINISTIC_LABEL))
            except Exception:
                pass
    # DRCC epsilon variants
    for e in EPSILONS:
        try:
            var_paths = find_summary_variant_paths(e, results_dir)
        except Exception:
            var_paths = []
        if var_paths:
            for rt_tag, summary_path, meta_path in var_paths:
                try:
                    df_e = pd.read_csv(summary_path)
                except Exception:
                    continue
                meta_e = {}
                if meta_path and os.path.exists(meta_path):
                    try:
                        with open(meta_path,'r',encoding='utf-8') as f:
                            meta_e = json.load(f)
                    except Exception:
                        meta_e = {}
                rows.append(_build_rt_row_local(df_e, meta_e, e, f"{e:.2f} ({_rt_display(rt_tag)})"))
        else:
            # Legacy unsuffixed per-epsilon
            try:
                df_e = _load_summary_for_epsilon_in_dir(results_dir, e)
            except Exception:
                df_e = None
            if df_e is not None:
                meta_e = _load_meta_for_epsilon_in_dir(results_dir, e)
                rows.append(_build_rt_row_local(df_e, meta_e, e, f"{e:.2f}"))
    rt_df = pd.DataFrame(rows)
    if 'label' not in rt_df.columns:
        rt_df = pd.DataFrame({'label': []})

    # distributions
    def _load_flat_distribution_from_meta(meta: Dict) -> np.ndarray:
        rel = meta.get('trafo_loading_file') if isinstance(meta, dict) else None
        if not rel:
            return np.array([])
        pq = os.path.join(results_dir, rel.replace('/', os.sep))
        if not os.path.exists(pq):
            if os.path.exists(rel):
                pq = rel
            else:
                return np.array([])
        try:
            pdf = _read_parquet_or_csv(pq)
        except Exception:
            pdf = None
        if pdf is None:
            return np.array([])
        must = {'sample_id','t','trafo_index','loading_pct'}
        if not must <= set(pdf.columns):
            return np.array([])
        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
        arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
        return arr[np.isfinite(arr)]

    dist_map: Dict[str, np.ndarray] = {}
    # Deterministic distributions
    if 'det_variants' in locals() and det_variants:
        for rt_tag, _sp, mp in det_variants:
            dmeta = {}
            if mp and os.path.exists(mp):
                try:
                    with open(mp,'r',encoding='utf-8') as f:
                        dmeta = json.load(f)
                except Exception:
                    dmeta = {}
            lab = f"{DETERMINISTIC_LABEL} ({_rt_display(rt_tag)})"
            dist_map[lab] = _load_flat_distribution_from_meta(dmeta)
    else:
        det_meta_p = os.path.join(results_dir, 'v4_meta_drcc_false.json')
        if os.path.exists(det_meta_p):
            try:
                with open(det_meta_p,'r',encoding='utf-8') as f:
                    m = json.load(f)
                dist_map[DETERMINISTIC_LABEL] = _load_flat_distribution_from_meta(m)
            except Exception:
                pass
    # Epsilon distributions
    for e in EPSILONS:
        try:
            var_paths = find_summary_variant_paths(e, results_dir)
        except Exception:
            var_paths = []
        if var_paths:
            for rt_tag, _sp, mp in var_paths:
                m = {}
                if mp and os.path.exists(mp):
                    try:
                        with open(mp,'r',encoding='utf-8') as f:
                            m = json.load(f)
                    except Exception:
                        m = {}
                lab = f"{e:.2f} ({_rt_display(rt_tag)})"
                dist_map[lab] = _load_flat_distribution_from_meta(m)
        else:
            m = _load_meta_for_epsilon_in_dir(results_dir, e)
            lab = f"{e:.2f}"
            dist_map[lab] = _load_flat_distribution_from_meta(m)

    return rt_df, dist_map


def _run_dual_aggregation(dist_a: str, dist_b: str, weights: Tuple[float,float] | None) -> None:
    """Aggregate two distributions and write a compact overview to a new folder.

    The aggregation computes weighted averages of scalar metrics per case label and concatenates
    transformer loading distributions (simple mixture) for violin visualization.
    """
    a_dir = _results_dir_from_dist(dist_a)
    b_dir = _results_dir_from_dist(dist_b)
    if not os.path.isdir(a_dir) or not os.path.isdir(b_dir):
        raise SystemExit(f"Required result directories not found: '{a_dir}' or '{b_dir}'.")
    w = weights if weights is not None else (0.5, 0.5)
    out_dir = f"v4_oos_agg_{dist_a}_{dist_b}"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    print(f"[aggregate] Building dual-distribution aggregation: {dist_a} + {dist_b} -> {out_dir} (weights={w})")

    a_sum, a_dist = _build_bundle_for_dir(a_dir)
    b_sum, b_dist = _build_bundle_for_dir(b_dir)
    # unify labels with RT-aware sorting
    def _base_from_label(lbl: str) -> Tuple[int, float]:
        try:
            if isinstance(lbl, str) and lbl.startswith(DETERMINISTIC_LABEL):
                return (0, 0.0)
            part = str(lbl).split()[0]
            return (1, float(part))
        except Exception:
            return (2, 0.0)
    def _rt_rank(lbl: str) -> int:
        s = str(lbl)
        if '(RT ON' in s:
            return 0
        if '(RT OFF' in s:
            return 1
        if '(RT ?' in s or '(RT UNK' in s:
            return 2
        return 3
    labels = sorted(set(a_sum['label'] if 'label' in a_sum.columns else []).union(set(b_sum['label'] if 'label' in b_sum.columns else [])),
                    key=lambda s: (_base_from_label(s)[0], _base_from_label(s)[1], _rt_rank(s), str(s)))
    rows = []
    for lab in labels:
        a_row = a_sum[a_sum['label']==lab].iloc[0] if (lab in set(a_sum['label'])) else None
        b_row = b_sum[b_sum['label']==lab].iloc[0] if (lab in set(b_sum['label'])) else None
        def _get(row, col):
            try:
                return float(row[col])
            except Exception:
                return float('nan')
        # Weighted average where available
        def _wavg(col):
            va = _get(a_row, col) if a_row is not None else np.nan
            vb = _get(b_row, col) if b_row is not None else np.nan
            if np.isfinite(va) and np.isfinite(vb):
                return w[0]*va + w[1]*vb
            return va if np.isfinite(va) else (vb if np.isfinite(vb) else np.nan)
        # Count-weighted mean using number of trajectories
        def _cwmean(col):
            va = _get(a_row, col) if a_row is not None else np.nan
            vb = _get(b_row, col) if b_row is not None else np.nan
            na = _get(a_row, 'n_trajectories') if a_row is not None else np.nan
            nb = _get(b_row, 'n_trajectories') if b_row is not None else np.nan
            na = na if np.isfinite(na) and na > 0 else 0.0
            nb = nb if np.isfinite(nb) and nb > 0 else 0.0
            if na + nb <= 0:
                return np.nan
            sa = (va * na) if np.isfinite(va) else 0.0
            sb = (vb * nb) if np.isfinite(vb) else 0.0
            return (sa + sb) / (na + nb)
        # epsilon value if available
        eps_val = None
        try:
            eps_val = float(lab) if lab != DETERMINISTIC_LABEL else None
        except Exception:
            eps_val = None
        # Count-based aggregation for transformer violation steps and probability
        a_steps = _get(a_row, 'trafo_steps') if a_row is not None else 0.0
        b_steps = _get(b_row, 'trafo_steps') if b_row is not None else 0.0
        steps_total = 0.0
        if np.isfinite(a_steps):
            steps_total += a_steps
        if np.isfinite(b_steps):
            steps_total += b_steps
        a_h = _get(a_row, 'horizon_timesteps') if a_row is not None else np.nan
        b_h = _get(b_row, 'horizon_timesteps') if b_row is not None else np.nan
        a_n = _get(a_row, 'n_trajectories') if a_row is not None else np.nan
        b_n = _get(b_row, 'n_trajectories') if b_row is not None else np.nan
        denom = 0.0
        if np.isfinite(a_h) and np.isfinite(a_n) and a_h > 0 and a_n > 0:
            denom += a_h * a_n
        if np.isfinite(b_h) and np.isfinite(b_n) and b_h > 0 and b_n > 0:
            denom += b_h * b_n
        agg_prob_pct = (steps_total / denom * 100.0) if denom > 0 else np.nan
        rows.append({
            'label': lab,
            'epsilon': eps_val,
            'rt_imbalance_cost_mean': _cwmean('rt_imbalance_cost_mean'),
            'total_rt_cost_mean': _cwmean('total_rt_cost_mean'),
            'trafo_steps': steps_total,
            'trafo_violation_probability_pct': agg_prob_pct
        })
    agg_df = pd.DataFrame(rows)
    agg_csv = os.path.join(out_dir, 'agg_summary.csv')
    agg_df.to_csv(agg_csv, index=False)
    print(f"✓ Aggregated summary CSV: {agg_csv}")

    # Build compact 3-panel plot: (1) Total RT cost (PCC deviation + recourse) (2) Trafo steps (3) Violin of loading per case (mixture)
    fig, axes = plt.subplots(1, 3, figsize=(22, 4.5), constrained_layout=True)
    x = np.arange(len(labels))

    # 1) Total RT cost bars (show A, B, and weighted)
    def _values_for(col: str, df: pd.DataFrame):
        vals = []
        for lab in labels:
            row = df[df['label']==lab]
            vals.append(float(row[col].iloc[0]) if not row.empty and np.isfinite(row[col].iloc[0]) else np.nan)
        return np.array(vals)
    # augment a_sum/b_sum to ensure necessary columns
    for df in (a_sum, b_sum):
        for c in ['total_rt_cost_mean','rt_imbalance_cost_mean','rt_pv_cost_mean','rt_bess_cost_mean','trafo_steps','trafo_violation_probability_pct','n_trajectories','horizon_timesteps']:
            if c not in df.columns:
                df[c] = np.nan
    # Use count-weighted aggregated total RT cost (imbalance + pv + bess)
    a_vals = _values_for('total_rt_cost_mean', a_sum)
    b_vals = _values_for('total_rt_cost_mean', b_sum)
    # Build count arrays per label
    a_counts = _values_for('n_trajectories', a_sum)
    b_counts = _values_for('n_trajectories', b_sum)
    agg_vals = np.empty_like(a_vals)
    for i in range(len(labels)):
        va = a_vals[i] if np.isfinite(a_vals[i]) else np.nan
        vb = b_vals[i] if np.isfinite(b_vals[i]) else np.nan
        na = a_counts[i] if np.isfinite(a_counts[i]) and a_counts[i] > 0 else 0.0
        nb = b_counts[i] if np.isfinite(b_counts[i]) and b_counts[i] > 0 else 0.0
        if na + nb <= 0:
            agg_vals[i] = np.nan
        else:
            sa = va * na if np.isfinite(va) else 0.0
            sb = vb * nb if np.isfinite(vb) else 0.0
            agg_vals[i] = (sa + sb) / (na + nb)
    width = 0.25
    axes[0].bar(x - width, a_vals, width=width, color='#1f77b4', label=dist_a)
    axes[0].bar(x, agg_vals, width=width, color='#636363', label='combined')
    axes[0].bar(x + width, b_vals, width=width, color='#ff7f0e', label=dist_b)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([('deterministic' if lab==DETERMINISTIC_LABEL else f"DRCC, ε={lab}") for lab in labels])
    axes[0].set_ylabel('EUR (mean across samples)')
    axes[0].set_title('Total RT cost (mean)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)

    # 2) Trafo steps
    a_steps = _values_for('trafo_steps', a_sum)
    b_steps = _values_for('trafo_steps', b_sum)
    agg_steps = np.nan_to_num(a_steps, nan=0.0) + np.nan_to_num(b_steps, nan=0.0)
    axes[1].bar(x - width, a_steps, width=width, color='#1f77b4', label=dist_a)
    axes[1].bar(x, agg_steps, width=width, color='#636363', label='aggregated')
    axes[1].bar(x + width, b_steps, width=width, color='#ff7f0e', label=dist_b)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([('deterministic' if lab==DETERMINISTIC_LABEL else f"DRCC, ε={lab}") for lab in labels])
    axes[1].set_ylabel('Steps (sum across trajectories)')
    axes[1].set_title('Transformer loading violations (> threshold)')
    axes[1].grid(axis='y', alpha=0.3)

    # 3) Violin of mixture distributions per case
    violin_data = []
    violin_pos = []
    violin_labels = []
    for i, lab in enumerate(labels, start=1):
        arr_a = a_dist.get(lab, np.array([]))
        arr_b = b_dist.get(lab, np.array([]))
        if arr_a.size == 0 and arr_b.size == 0:
            continue
        # Simple mixture: concatenate; if both present, optionally subsample larger set to approx equal weight
        if arr_a.size > 0 and arr_b.size > 0:
            na, nb = len(arr_a), len(arr_b)
            if na > 0 and nb > 0:
                # equalize counts roughly to avoid dominance
                k_a = min(na, nb)
                k_b = min(na, nb)
                rng = np.random.default_rng(12345)
                arr_a2 = arr_a if na <= k_a else rng.choice(arr_a, size=k_a, replace=False)
                arr_b2 = arr_b if nb <= k_b else rng.choice(arr_b, size=k_b, replace=False)
                mix = np.concatenate([arr_a2, arr_b2])
            else:
                mix = np.concatenate([arr_a, arr_b])
        else:
            mix = arr_a if arr_a.size else arr_b
        mix = mix[np.isfinite(mix)]
        if mix.size == 0:
            continue
        violin_data.append(mix)
        violin_pos.append(i)
        violin_labels.append(lab)
    if violin_data:
        vp = axes[2].violinplot(violin_data, positions=violin_pos, showmeans=False, showmedians=True, showextrema=False)
        for pc in vp['bodies']:
            pc.set_facecolor('#b2df8a')
            pc.set_edgecolor('#1b7837')
            pc.set_alpha(0.6)
        if 'cmedians' in vp:
            vp['cmedians'].set_color('#1b7837')
        axes[2].set_xticks(violin_pos)
        axes[2].set_xticklabels([('deterministic' if lab==DETERMINISTIC_LABEL else f"DRCC, ε={lab}") for lab in violin_labels])
        axes[2].set_ylabel('Transformer loading %')
        axes[2].set_title('Mixture transformer loading (violin)')
        axes[2].grid(axis='y', alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No transformer loading data', ha='center', va='center', transform=axes[2].transAxes, fontsize=9, color='gray')
    out_fig = os.path.join(out_dir, f"agg_overview_{dist_a}_{dist_b}.png")
    fig.savefig(out_fig, dpi=150)
    print(f"✓ Aggregated overview figure: {out_fig}")

    # Build a complete combined dataset in out_dir so the standard pipeline can generate ALL analyses
    def _combine_summary(a_path: str, b_path: str, out_path: str):
        dfs = []
        for p in (a_path, b_path):
            if p and os.path.exists(p):
                try:
                    dfs.append(pd.read_csv(p))
                except Exception:
                    pass
        if dfs:
            try:
                all_df = pd.concat(dfs, ignore_index=True)
                all_df.to_csv(out_path, index=False)
                return True
            except Exception:
                return False
        return False

    def _combine_parquet_from_meta(a_meta: Dict, b_meta: Dict, out_rel: str) -> bool:
        # Returns True if parquet written and meta updated
        def _extract_parquet(meta: Dict):
            rel = meta.get('trafo_loading_file') if isinstance(meta, dict) else None
            if not rel:
                return None
            a = os.path.join(a_dir, rel)
            b = os.path.join(b_dir, rel)
            # if rel is different between metas, try both absolute constructions
            return a if os.path.exists(a) else (b if os.path.exists(b) else (rel if os.path.exists(rel) else None))

        # Resolve candidate parquet paths for A and B separately
        p_a = None
        p_b = None
        if isinstance(a_meta, dict) and 'trafo_loading_file' in a_meta:
            rel_a = a_meta['trafo_loading_file']
            cand = os.path.join(a_dir, rel_a)
            if os.path.exists(cand):
                p_a = cand
            elif os.path.exists(rel_a):
                p_a = rel_a
        if isinstance(b_meta, dict) and 'trafo_loading_file' in b_meta:
            rel_b = b_meta['trafo_loading_file']
            cand = os.path.join(b_dir, rel_b)
            if os.path.exists(cand):
                p_b = cand
            elif os.path.exists(rel_b):
                p_b = rel_b
        if not p_a and not p_b:
            return False
        try:
            df_a = _read_parquet_or_csv(p_a) if p_a else None
            df_b = _read_parquet_or_csv(p_b) if p_b else None
            if df_a is None and df_b is None:
                return False
            # Ensure required columns exist; otherwise drop that source
            must = {'sample_id','t','trafo_index','loading_pct'}
            if df_a is not None and not must <= set(df_a.columns):
                df_a = None
            if df_b is not None and not must <= set(df_b.columns):
                df_b = None
            if df_a is None and df_b is None:
                return False
            # Normalize dtypes carefully, preserving distinct sample identities
            def _normalize_and_encode_samples(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                # loading and indices
                df['t'] = pd.to_numeric(df['t'], errors='coerce')
                df['trafo_index'] = pd.to_numeric(df['trafo_index'], errors='coerce')
                df['loading_pct'] = pd.to_numeric(df['loading_pct'], errors='coerce')
                # sample id: prefer numeric if fully numeric; else factorize to unique ints
                sid_raw = df['sample_id']
                sid_num = pd.to_numeric(sid_raw, errors='coerce')
                if sid_num.isna().any():
                    # mixed or non-numeric -> factorize on string form
                    sid_codes, _ = pd.factorize(sid_raw.astype(str), sort=False)
                    df['sample_id'] = sid_codes.astype(int)
                else:
                    df['sample_id'] = sid_num.astype(int)
                # finalize index columns as ints
                df['t'] = df['t'].fillna(-1).astype(int)
                df['trafo_index'] = df['trafo_index'].fillna(-1).astype(int)
                df['loading_pct'] = df['loading_pct'].astype(float)
                return df

            if df_a is not None:
                df_a = _normalize_and_encode_samples(df_a)
            if df_b is not None:
                df_b = _normalize_and_encode_samples(df_b)

            # Make sample_id unique across A and B by offsetting df_b codes
            if df_a is not None and df_b is not None:
                try:
                    max_id = pd.to_numeric(df_a['sample_id'], errors='coerce').dropna().astype(int).max()
                    offset = int(max_id) + 1 if np.isfinite(max_id) else 0
                except Exception:
                    offset = 0
                df_b = df_b.copy()
                try:
                    df_b['sample_id'] = pd.to_numeric(df_b['sample_id'], errors='coerce').fillna(0).astype(int) + int(offset)
                except Exception:
                    # As a last resort, factorize again with a global offset via row index
                    df_b['sample_id'] = (pd.factorize(df_b.index, sort=False)[0] + int(offset)).astype(int)
                pdf = pd.concat([df_a, df_b], ignore_index=True)
            else:
                pdf = df_a if df_a is not None else df_b
            out_abs = os.path.join(out_dir, out_rel.replace('/', os.sep))
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)
            pdf.to_parquet(out_abs, index=False)
            return True
        except Exception as e:
            print(f"[WARN] Failed to write combined parquet for {out_rel}: {e}")
            return False

    # Baseline: summaries and meta/parquet
    det_a_sum = os.path.join(a_dir, 'v4_summary_drcc_false.csv')
    det_b_sum = os.path.join(b_dir, 'v4_summary_drcc_false.csv')
    det_out_sum = os.path.join(out_dir, 'v4_summary_drcc_false.csv')
    _combine_summary(det_a_sum, det_b_sum, det_out_sum)
    # RT-suffixed deterministic summaries
    for tag in ('rt_on','rt_off','rt_unk'):
        a_p = os.path.join(a_dir, f'v4_summary_drcc_false_{tag}.csv')
        b_p = os.path.join(b_dir, f'v4_summary_drcc_false_{tag}.csv')
        out_p = os.path.join(out_dir, f'v4_summary_drcc_false_{tag}.csv')
        _combine_summary(a_p, b_p, out_p)
    # meta
    det_a_meta_p = os.path.join(a_dir, 'v4_meta_drcc_false.json')
    det_b_meta_p = os.path.join(b_dir, 'v4_meta_drcc_false.json')
    det_meta_a = {}
    det_meta_b = {}
    if os.path.exists(det_a_meta_p):
        try:
            with open(det_a_meta_p,'r',encoding='utf-8') as f:
                det_meta_a = json.load(f)
        except Exception:
            pass
    if os.path.exists(det_b_meta_p):
        try:
            with open(det_b_meta_p,'r',encoding='utf-8') as f:
                det_meta_b = json.load(f)
        except Exception:
            pass
    det_parquet_rel = 'v4_loading/trafo_loading_raw_drcc_false_combined.parquet'
    if _combine_parquet_from_meta(det_meta_a, det_meta_b, det_parquet_rel):
        # choose a base meta and update parquet ref
        det_meta_out = det_meta_b if det_meta_b else det_meta_a
        if isinstance(det_meta_out, dict):
            det_meta_out['trafo_loading_file'] = det_parquet_rel.replace('\\', '/')
            with open(os.path.join(out_dir,'v4_meta_drcc_false.json'),'w',encoding='utf-8') as f:
                json.dump(det_meta_out, f, indent=2)
    # RT-suffixed deterministic metas/parquets
    for tag in ('rt_on','rt_off','rt_unk'):
        a_meta_p = os.path.join(a_dir, f'v4_meta_drcc_false_{tag}.json')
        b_meta_p = os.path.join(b_dir, f'v4_meta_drcc_false_{tag}.json')
        a_meta = {}
        b_meta = {}
        if os.path.exists(a_meta_p):
            try:
                with open(a_meta_p,'r',encoding='utf-8') as f:
                    a_meta = json.load(f)
            except Exception:
                a_meta = {}
        if os.path.exists(b_meta_p):
            try:
                with open(b_meta_p,'r',encoding='utf-8') as f:
                    b_meta = json.load(f)
            except Exception:
                b_meta = {}
        pq_rel = f'v4_loading/trafo_loading_raw_drcc_false_{tag}_combined.parquet'
        if _combine_parquet_from_meta(a_meta, b_meta, pq_rel):
            meta_out = b_meta if b_meta else a_meta
            if isinstance(meta_out, dict):
                meta_out['trafo_loading_file'] = pq_rel.replace('\\', '/')
                with open(os.path.join(out_dir, f'v4_meta_drcc_false_{tag}.json'),'w',encoding='utf-8') as f:
                    json.dump(meta_out, f, indent=2)

    # DRCC epsilons
    for e in EPSILONS:
        tok = epsilon_token(e)
        # unsuffixed (legacy)
        a_sum_p = os.path.join(a_dir, f'v4_summary_drcc_true_epsilon_{tok}.csv')
        b_sum_p = os.path.join(b_dir, f'v4_summary_drcc_true_epsilon_{tok}.csv')
        out_sum_p = os.path.join(out_dir, f'v4_summary_drcc_true_epsilon_{tok}.csv')
        _combine_summary(a_sum_p, b_sum_p, out_sum_p)
        a_meta_p = os.path.join(a_dir, f'v4_meta_drcc_true_epsilon_{tok}.json')
        b_meta_p = os.path.join(b_dir, f'v4_meta_drcc_true_epsilon_{tok}.json')
        a_meta = {}
        b_meta = {}
        if os.path.exists(a_meta_p):
            try:
                with open(a_meta_p,'r',encoding='utf-8') as f:
                    a_meta = json.load(f)
            except Exception:
                a_meta = {}
        if os.path.exists(b_meta_p):
            try:
                with open(b_meta_p,'r',encoding='utf-8') as f:
                    b_meta = json.load(f)
            except Exception:
                b_meta = {}
        pq_rel = f'v4_loading/trafo_loading_raw_epsilon_{tok}_combined.parquet'
        if _combine_parquet_from_meta(a_meta, b_meta, pq_rel):
            meta_out = b_meta if b_meta else a_meta
            if isinstance(meta_out, dict):
                meta_out['trafo_loading_file'] = pq_rel.replace('\\', '/')
                with open(os.path.join(out_dir, f'v4_meta_drcc_true_epsilon_{tok}.json'),'w',encoding='utf-8') as f:
                    json.dump(meta_out, f, indent=2)
        # RT-suffixed variants
        for tag in ('rt_on','rt_off','rt_unk'):
            a_sum_v = os.path.join(a_dir, f'v4_summary_drcc_true_epsilon_{tok}_{tag}.csv')
            b_sum_v = os.path.join(b_dir, f'v4_summary_drcc_true_epsilon_{tok}_{tag}.csv')
            out_sum_v = os.path.join(out_dir, f'v4_summary_drcc_true_epsilon_{tok}_{tag}.csv')
            _combine_summary(a_sum_v, b_sum_v, out_sum_v)
            a_meta_v = os.path.join(a_dir, f'v4_meta_drcc_true_epsilon_{tok}_{tag}.json')
            b_meta_v = os.path.join(b_dir, f'v4_meta_drcc_true_epsilon_{tok}_{tag}.json')
            ma = {}; mb = {}
            if os.path.exists(a_meta_v):
                try:
                    with open(a_meta_v,'r',encoding='utf-8') as f:
                        ma = json.load(f)
                except Exception:
                    ma = {}
            if os.path.exists(b_meta_v):
                try:
                    with open(b_meta_v,'r',encoding='utf-8') as f:
                        mb = json.load(f)
                except Exception:
                    mb = {}
            pq_rel_v = f'v4_loading/trafo_loading_raw_epsilon_{tok}_{tag}_combined.parquet'
            if _combine_parquet_from_meta(ma, mb, pq_rel_v):
                m_out = mb if mb else ma
                if isinstance(m_out, dict):
                    m_out['trafo_loading_file'] = pq_rel_v.replace('\\', '/')
                    with open(os.path.join(out_dir, f'v4_meta_drcc_true_epsilon_{tok}_{tag}.json'),'w',encoding='utf-8') as f:
                        json.dump(m_out, f, indent=2)

    # Copy policy coeffs and SoC envelopes from whichever exists (prefer dist_b then dist_a)
    def _copy_if_exists(rel_name: str):
        src = os.path.join(b_dir, rel_name)
        if not os.path.exists(src):
            src = os.path.join(a_dir, rel_name)
        if os.path.exists(src):
            dst = os.path.join(out_dir, rel_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

    # baseline policy/soc
    _copy_if_exists('policy_coeffs_drcc_false.csv')
    # RT-suffixed policy coeffs (if present)
    for _tag in ('rt_on','rt_off','rt_unk'):
        _copy_if_exists(f'policy_coeffs_drcc_false_{_tag}.csv')
    # baseline SoC envelopes (unsuffixed + RT variants)
    _copy_if_exists('soc_envelope_drcc_false.csv')
    for _tag in ('rt_on','rt_off','rt_unk'):
        _copy_if_exists(f'soc_envelope_drcc_false_{_tag}.csv')
    # baseline SoC raw series (parquet/csv; unsuffixed + RT variants)
    for _ext in ('parquet','csv'):
        _copy_if_exists(f'soc_series_drcc_false.{_ext}')
        for _tag in ('rt_on','rt_off','rt_unk'):
            _copy_if_exists(f'soc_series_drcc_false_{_tag}.{_ext}')
    # per epsilon
    for e in EPSILONS:
        tok = epsilon_token(e)
        _copy_if_exists(f'policy_coeffs_drcc_true_epsilon_{tok}.csv')
        for _tag in ('rt_on','rt_off','rt_unk'):
            _copy_if_exists(f'policy_coeffs_drcc_true_epsilon_{tok}_{_tag}.csv')
        # SoC envelopes (unsuffixed + RT variants)
        _copy_if_exists(f'soc_envelope_drcc_true_epsilon_{tok}.csv')
        for _tag in ('rt_on','rt_off','rt_unk'):
            _copy_if_exists(f'soc_envelope_drcc_true_epsilon_{tok}_{_tag}.csv')
        # SoC raw series (parquet/csv; unsuffixed + RT variants)
        for _ext in ('parquet','csv'):
            _copy_if_exists(f'soc_series_drcc_true_epsilon_{tok}.{_ext}')
            for _tag in ('rt_on','rt_off','rt_unk'):
                _copy_if_exists(f'soc_series_drcc_true_epsilon_{tok}_{_tag}.{_ext}')

    # Now generate full analyses using the combined dataset by pointing RESULTS_DIR to out_dir
    global RESULTS_DIR, AGGREGATE_DISTS
    RESULTS_DIR = out_dir
    AGGREGATE_DISTS = None  # allow tail overview and standard flow
    print(f"[aggregate] Running full analyses on combined dataset at {RESULTS_DIR}")
    try:
        main()
    except Exception as e:
        print(f"[WARN] Combined full analyses failed: {e}")


def main() -> None:
    # --- Legacy aggregate for DRCC epsilons only (kept) ---
    rows: List[Dict[str, float]] = []
    for eps in EPSILONS:
        try:
            df = load_summary_for_epsilon(eps)
        except FileNotFoundError:
            continue
        meta = load_meta_for_epsilon(eps)
        avg_price = compute_avg_price_from_v2(meta)
        agg = aggregate_metrics(df, avg_price)
        agg["epsilon"] = eps
        rows.append(agg)
    legacy_summary = pd.DataFrame(rows)

    # --- Build unified RT-focused summary with deterministic appended ---
    def build_rt_row(df_eps: pd.DataFrame, meta: Dict, eps: float | None, label: str) -> Dict[str, float]:
        horizon = np.nan
        v2_csv = meta.get('v2_results_csv') if isinstance(meta, dict) else None
        if v2_csv and os.path.exists(v2_csv):
            try:
                v2_df = pd.read_csv(v2_csv)
                if 'electricity_price_eur_mwh' in v2_df.columns:
                    horizon = int(pd.to_numeric(v2_df['electricity_price_eur_mwh'], errors='coerce').dropna().shape[0])
                else:
                    horizon = int(v2_df.shape[0])
            except Exception:
                pass
        da_import_cost_mean = float(df_eps.get('da_energy_cost_eur', pd.Series([0.0])).mean())
        da_total_cost_eur = read_v2_da_total_cost(meta)
        rt_imb_cost_mean = float(df_eps.get('rt_imbalance_cost_eur', pd.Series([0.0])).mean())
        rt_pv_cost_mean = float(df_eps.get('rt_pv_curtail_cost_eur', pd.Series([0.0])).mean())
        rt_bess_cost_mean = float(df_eps.get('rt_bess_cycle_cost_eur', pd.Series([0.0])).mean())
        if 'steps_trafo_over_80pct' in df_eps.columns:
            trafo_steps = int(df_eps['steps_trafo_over_80pct'].sum())
        elif 'steps_trafo_over_100pct' in df_eps.columns:
            trafo_steps = int(df_eps['steps_trafo_over_100pct'].sum())
        else:
            trafo_steps = 0
        if 'steps_line_over_80pct' in df_eps.columns:
            line_steps = int(df_eps['steps_line_over_80pct'].sum())
        elif 'steps_line_over_100pct' in df_eps.columns:
            line_steps = int(df_eps['steps_line_over_100pct'].sum())
        else:
            line_steps = 0
        n_traj = len(df_eps)
        if isinstance(horizon, (int, np.integer)) and horizon > 0 and n_traj > 0:
            trafo_violation_probability_pct = (trafo_steps / (n_traj * horizon)) * 100.0
        else:
            trafo_violation_probability_pct = np.nan
        return {
            'epsilon': eps if eps is not None else np.nan,
            'label': label,
            'da_import_cost_mean': da_import_cost_mean,
            'da_total_cost_eur': da_total_cost_eur,
            'rt_imbalance_cost_mean': rt_imb_cost_mean,
            'rt_pv_cost_mean': rt_pv_cost_mean,
            'rt_bess_cost_mean': rt_bess_cost_mean,
            'trafo_steps': trafo_steps,
            'line_steps': line_steps,
            'trafo_violation_probability_pct': trafo_violation_probability_pct,
            'horizon_timesteps': horizon,
            'total_rt_cost_mean': rt_imb_cost_mean + rt_pv_cost_mean + rt_bess_cost_mean,
            'is_deterministic': int(eps is None),
            'n_trajectories': n_traj
        }

    rt_rows: List[Dict[str, float]] = []
    # Pretty label helpers
    def _display_label_for_case(lab: str) -> str:
        """Return a display label: 'deterministic' for baseline; 'DRCC, ε = <lab>' for epsilon cases.

        This only affects tick/legend text; internal keys remain unchanged.
        """
        try:
            # If label parses as a float, it's an epsilon string like '0.10'
            float(lab)
            return f"DRCC, ε = {float(lab):.2f}"
        except Exception:
            # 'deterministic' or other non-epsilon labels pass through
            return lab
    # Helper: compute CVaR at alpha for a 1-D numpy array of loadings (%). Returns np.nan if insufficient data.
    def cvar(loadings: np.ndarray, alpha: float) -> float:
        if loadings is None or loadings.size == 0:
            return float('nan')
        # Tail definition: exceedances over VaR alpha (upper tail severity)
        var_thresh = np.nanpercentile(loadings, alpha * 100.0)
        tail = loadings[loadings >= var_thresh]
        if tail.size == 0:
            return float('nan')
        return float(np.nanmean(tail))

    def load_trafo_loading(meta: Dict) -> Dict[str, float]:
        """Load transformer loading parquet (if present) and compute CVaR90/95 of max loading across trafos/time.

        Assumptions:
          - meta['trafo_loading_file'] is a relative path (e.g., 'v4_loading\\trafo_loading_raw_epsilon_0_05.parquet')
          - Parquet contains columns either like 'loading_pct' per record or per-trafo columns.
        Strategy:
          1. Read parquet to DataFrame (if engine available).
          2. Collect all numeric columns whose name contains 'loading' and '%'.
          3. Flatten into single 1-D array of loading percentages.
          4. Compute CVaR90 & CVaR95 over that array.
        """
        rel_path = meta.get('trafo_loading_file')
        base_dir = RESULTS_DIR  # parquets appear inside RESULTS_DIR/v4_loading
        if not rel_path:
            return {"cvar90": float('nan'), "cvar95": float('nan'), "sev_cvar90": float('nan'), "sev_cvar95": float('nan')}
        abs_path = os.path.join(base_dir, rel_path.replace('/', os.sep))
        if not os.path.exists(abs_path):
            # Try alternative: if meta stored absolute already
            if os.path.exists(rel_path):
                abs_path = rel_path
            else:
                return {"cvar90": float('nan'), "cvar95": float('nan'), "sev_cvar90": float('nan'), "sev_cvar95": float('nan')}
        try:
            df_load = _read_parquet_or_csv(abs_path)
        except Exception:
            df_load = None
        if df_load is None:
            return {"cvar90": float('nan'), "cvar95": float('nan'), "sev_cvar90": float('nan'), "sev_cvar95": float('nan')}
        # Identify loading columns
        cand_cols = [c for c in df_load.columns if 'load' in c.lower() and ('pct' in c.lower() or 'percent' in c.lower())]
        if not cand_cols:
            # Fallback: any column with 'loading'
            cand_cols = [c for c in df_load.columns if 'loading' in c.lower()]
        if not cand_cols:
            return {"cvar90": float('nan'), "cvar95": float('nan'), "sev_cvar90": float('nan'), "sev_cvar95": float('nan')}
        arr = pd.concat([pd.to_numeric(df_load[c], errors='coerce') for c in cand_cols], axis=0).to_numpy()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"cvar90": float('nan'), "cvar95": float('nan'), "sev_cvar90": float('nan'), "sev_cvar95": float('nan')}
        raw_cvar90 = cvar(arr, 0.90)
        raw_cvar95 = cvar(arr, 0.95)
        # Violation severity: only excess over 100% (clip below at 0)
        excess = np.clip(arr - 100.0, a_min=0.0, a_max=None)
        # If all zero (no violations) keep NaN for severity CVaR to distinguish from zero severity distribution
        sev_cvar90 = cvar(excess[excess > 0], 0.90) if np.any(excess > 0) else float('nan')
        sev_cvar95 = cvar(excess[excess > 0], 0.95) if np.any(excess > 0) else float('nan')
        return {"cvar90": raw_cvar90, "cvar95": raw_cvar95, "sev_cvar90": sev_cvar90, "sev_cvar95": sev_cvar95}
    meta_map: Dict[str, Dict] = {}
    for eps in EPSILONS:
        variant_paths = find_summary_variant_paths(eps, RESULTS_DIR) if INCLUDE_RT_SPLIT else []
        if not variant_paths:
            # Strict mode: skip epsilon if no RT-suffixed summaries present.
            continue
        for rt_tag, summary_path, meta_path in variant_paths:
            try:
                df_eps = pd.read_csv(summary_path)
            except Exception:
                continue
            meta = {}
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path,'r',encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            severity = load_trafo_loading(meta)
            label_base = f"{eps:.2f}"
            label_display = f"{label_base} ({_rt_display(rt_tag)})"
            row = build_rt_row(df_eps, meta, eps, label_display)
            row.update({
                'trafo_cvar90_loading_pct': severity['cvar90'],
                'trafo_cvar95_loading_pct': severity['cvar95'],
                'trafo_violation_excess_cvar90_pct': severity['sev_cvar90'],
                'trafo_violation_excess_cvar95_pct': severity['sev_cvar95'],
            })
            row['rt_tag'] = rt_tag
            rt_rows.append(row)
            meta_map[label_display] = meta

    # Deterministic (baseline k=1) appended (ordering handled later so it appears first)
    det_variants = find_deterministic_variant_paths(RESULTS_DIR) if INCLUDE_RT_SPLIT else []
    if INCLUDE_DETERMINISTIC and det_variants:
        for rt_tag, summary_path, meta_path in det_variants:
            try:
                det_df = pd.read_csv(summary_path)
            except Exception:
                continue
            det_meta = {}
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path,'r',encoding='utf-8') as f:
                        det_meta = json.load(f)
                except Exception:
                    det_meta = {}
            sev_det = load_trafo_loading(det_meta)
            label_display = f"{DETERMINISTIC_LABEL} ({_rt_display(rt_tag)})"
            det_row = build_rt_row(det_df, det_meta, None, label_display)
            det_row.update({
                'trafo_cvar90_loading_pct': sev_det['cvar90'],
                'trafo_cvar95_loading_pct': sev_det['cvar95'],
                'trafo_violation_excess_cvar90_pct': sev_det['sev_cvar90'],
                'trafo_violation_excess_cvar95_pct': sev_det['sev_cvar95'],
            })
            det_row['rt_tag'] = rt_tag
            rt_rows.append(det_row)
            meta_map[label_display] = det_meta

    rt_summary = pd.DataFrame(rt_rows)
    # Build label ordering ignoring RT suffix. Deterministic first (if present) then epsilon bases.
    def _label_base(l: str) -> str:
        if l.startswith(DETERMINISTIC_LABEL):
            return DETERMINISTIC_LABEL
        try:
            part = l.split()[0]
            float(part)
            return part
        except Exception:
            return l
    bases_present = [_label_base(l) for l in rt_summary['label']]
    label_order: List[str] = []
    if any(b == DETERMINISTIC_LABEL for b in bases_present):
        label_order.append(DETERMINISTIC_LABEL)
    for e in EPSILONS:
        tok = f"{e:.2f}"
        if tok in bases_present:
            label_order.append(tok)
    rt_summary['plot_order'] = rt_summary['label'].apply(lambda x: label_order.index(_label_base(x)) if _label_base(x) in label_order else 999)
    rt_summary = rt_summary.sort_values('plot_order')

    # Merge legacy (epsilon keyed) for DRCC rows only
    if not legacy_summary.empty:
        legacy_summary = legacy_summary.rename(columns={'epsilon': 'epsilon'}).copy()
        summary = pd.merge(rt_summary, legacy_summary, on='epsilon', how='left', suffixes=('', '_legacy'))
    else:
        summary = rt_summary.copy()

    # Add all-in total cost (DA total + mean RT costs) for convenience in downstream analyses
    try:
        summary['all_in_total_cost_mean'] = summary['da_total_cost_eur'] + summary['total_rt_cost_mean']
    except Exception:
        summary['all_in_total_cost_mean'] = np.nan

    summary.to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)

    # === Radial-only adaptation ===
    # New v4 (post-refactor) provides only radial (Option A) network loading; voltages are NaN/omitted.
    # Detect this to adjust plot labels & console messaging.
    radial_only_mode = True  # currently always true after removal of admittance logic
    if radial_only_mode:
        print("[INFO] Detected radial-only flow evaluation mode (Option A); voltage metrics suppressed / NaN.")

    # Derive transformer violation threshold (default 80%) from any available summary column
    threshold_candidates = []
    if 'loading_violation_threshold_pct' in summary.columns:
        threshold_candidates.extend(list(pd.to_numeric(summary['loading_violation_threshold_pct'], errors='coerce').dropna().unique()))
    # Fallback: look directly into a representative v4_summary file if not populated (older runs)
    if not threshold_candidates:
        for eps in EPSILONS:
            token = epsilon_token(eps)
            fpath = os.path.join(RESULTS_DIR, f"v4_summary_drcc_true_epsilon_{token}.csv")
            if os.path.exists(fpath):
                try:
                    tmp_df = pd.read_csv(fpath, nrows=1)
                    if 'loading_violation_threshold_pct' in tmp_df.columns:
                        val = pd.to_numeric(tmp_df['loading_violation_threshold_pct'], errors='coerce').iloc[0]
                        if np.isfinite(val):
                            threshold_candidates.append(val)
                            break
                except Exception:
                    pass
    # Final fallback constant from summary; use it as-is to match generation (no bump)
    violation_threshold_pct = float(threshold_candidates[0]) if threshold_candidates else 80.0
    # Effective threshold equals the run's own threshold to keep plots consistent with summaries
    viol_threshold_eff = float(violation_threshold_pct)
    print(f"[INFO] Using transformer violation threshold = {viol_threshold_eff:.2f}% (from summaries)")

    # New condensed overview: 9 panels
    # 0: RT imbalance cost (already simplified earlier)
    # 1: DA import cost
    # 2: Transformer violation steps
    # 3: Transformer violation probability
    # 4: Boxplot of transformer loading distribution (per-timestep max loading across trafos per trajectory)
    # 5: Violin plot of the same distribution (requested)
    # 6: Violation severity (mean exceedance over threshold among violated timesteps)
    # 7: Loading CVaR80 (per-timestep maxima)
    # 8: Histogram of top 20% loading by case (overlay)
    fig, axes = plt.subplots(1, 9, figsize=(58, 4.2), constrained_layout=True)
    x = np.arange(len(rt_summary))
    width = 0.6

    # RT imbalance-only cost plot (simplified per user request: removed PV curtail & BESS cycle components + legend)
    c_imb = rt_summary['rt_imbalance_cost_mean'].to_numpy()
    axes[0].bar(x, c_imb, width=width, color='#dd8452')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([_display_label_for_case(l) for l in rt_summary['label']])
    axes[0].set_xlabel('epsilon / mode')
    axes[0].set_ylabel('EUR (mean across samples)')
    axes[0].set_title('RT imbalance cost')
    axes[0].grid(axis='y', alpha=0.3)

    # All-in total cost bar: v2 base cost (no RT proxies) + v4 RT recourse costs
    # Build meta cache for labels
    meta_by_label: Dict[str, Dict] = {}
    det_meta_path = os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json')
    if INCLUDE_DETERMINISTIC and os.path.exists(det_meta_path):
        try:
            with open(det_meta_path, 'r', encoding='utf-8') as f:
                meta_by_label[DETERMINISTIC_LABEL] = json.load(f)
        except Exception:
            meta_by_label[DETERMINISTIC_LABEL] = {}
    for e in EPSILONS:
        lab = f"{e:.2f}"
        meta_by_label[lab] = load_meta_for_epsilon(e)
    # Compute base costs per label
    base_costs: List[float] = []
    for lab in rt_summary['label']:
        meta = meta_by_label.get(lab, {})
        # Prefer direct exported DA total cost; fallback to import-only reconstruction
        da_total = read_v2_da_total_cost(meta)
        if not np.isfinite(da_total):
            da_total = compute_v2_base_cost(meta)
        base_costs.append(da_total if np.isfinite(da_total) else np.nan)
    base_series = pd.Series(base_costs, index=rt_summary.index)
    # RT recourse mean costs from v4
    c_rt = pd.to_numeric(rt_summary['total_rt_cost_mean'], errors='coerce')
    # If base is missing, fallback to DA import mean just to keep the plot complete
    c_da_fallback = pd.to_numeric(rt_summary.get('da_import_cost_mean', pd.Series([np.nan]*len(rt_summary))), errors='coerce')
    base_used = base_series.where(base_series.notna(), c_da_fallback)
    c_allin = base_used.fillna(0.0).to_numpy() + c_rt.fillna(0.0).to_numpy()
    bars_allin = axes[1].bar(x, c_allin, width=width, color='#4c72b0')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_display_label_for_case(l) for l in rt_summary['label']])
    axes[1].set_xlabel('epsilon / mode')
    axes[1].set_ylabel('EUR (mean across samples)')
    #axes[1].set_title('All-in cost: DA import + DA (BESS, flex) + RT')
    axes[1].grid(axis='y', alpha=0.3)
    # Annotate % difference vs deterministic on DRCC bars
    try:
        # Find deterministic baseline index and cost
        det_mask = (rt_summary['label'] == DETERMINISTIC_LABEL)
        if det_mask.any():
            det_idx = int(np.argmax(det_mask.to_numpy()))
            det_val = float(c_allin[det_idx])
            if np.isfinite(det_val) and det_val != 0.0:
                for i, (rect, val, lab) in enumerate(zip(bars_allin, c_allin, rt_summary['label'])):
                    if lab == DETERMINISTIC_LABEL or not np.isfinite(val):
                        continue
                    pct = (val - det_val) / det_val * 100.0
                    sgn = '+' if pct >= 0 else ''
                    txt = f"{sgn}{pct:.1f}%"
                    y = rect.get_height()
                    ann = axes[1].text(rect.get_x() + rect.get_width()/2.0,
                                        y + max(0.01*y, 0.5),
                                        txt,
                                        ha='center', va='bottom', fontsize=8, color='black')
                    try:
                        ann.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground='white')])
                    except Exception:
                        pass
    except Exception:
        pass
    # No stack/legend per request

    # Ensure numeric (guard against dtype/object issues causing a missing bar for stochastic baseline)
    t_steps_series = pd.to_numeric(rt_summary['trafo_steps'], errors='coerce')
    # Sanity log for stochastic row
    try:
        stoch_val = float(t_steps_series.loc[rt_summary['label'] == DETERMINISTIC_LABEL].iloc[0])
        print(f"[DEBUG] Stochastic trafo_steps = {stoch_val}")
    except Exception:
        print("[DEBUG] Stochastic trafo_steps not found in rt_summary")
    t_steps = t_steps_series.to_numpy()
    bars = axes[2].bar(x, t_steps, width=width, color='#dd8452', label='Transformer steps > threshold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([_display_label_for_case(l) for l in rt_summary['label']])
    axes[2].set_xlabel('epsilon / mode')
    axes[2].set_ylabel('Steps (sum across trajectories)')
    axes[2].set_title(f'Transformer loading violations (> {viol_threshold_eff:.2f}%)')
    axes[2].grid(axis='y', alpha=0.3)
    # Annotate bars (always show value, even if very small or large)
    for rect, val in zip(bars, t_steps):
        if not np.isfinite(val):
            continue
        h = rect.get_height()
        axes[2].text(rect.get_x() + rect.get_width()/2, h + max(0.01*h, 0.5), f"{int(val)}", ha='center', va='bottom', fontsize=7, rotation=0)
    axes[2].legend()

    t_prob = rt_summary['trafo_violation_probability_pct'].to_numpy()
    axes[3].bar(x, t_prob, width=width, color='#55a868', label='Trafo violation probability')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([_display_label_for_case(l) for l in rt_summary['label']])
    axes[3].set_xlabel('epsilon / mode')
    axes[3].set_ylabel('% of total timesteps')
    axes[3].set_title(f'Transformer violation probability (> {viol_threshold_eff:.2f}%)')
    axes[3].grid(axis='y', alpha=0.3)
    axes[3].legend()

    # 4. Transformer loading distribution boxplot (per-timestep max loading across trafos per trajectory)
    # Build distributions
    distributions: List[np.ndarray] = []
    labels_box: List[str] = []

    def _load_loading_distribution(meta: Dict) -> np.ndarray:
        """Return flattened array of per-(sample_id,t) max transformer loading percentages.

        Strategy: read parquet referenced in meta['trafo_loading_file'] (if present), group by (sample_id,t)
        to take max loading across trafos, and return all those maxima as a 1-D numpy array.
        """
        if not meta or 'trafo_loading_file' not in meta:
            return np.array([])
        rel_path = meta.get('trafo_loading_file')
        if not rel_path:
            return np.array([])
        abs_path = os.path.join(RESULTS_DIR, rel_path.replace('/', os.sep))
        if not os.path.exists(abs_path):
            # try direct path
            if os.path.exists(rel_path):
                abs_path = rel_path
            else:
                return np.array([])
        try:
            pdf = _read_parquet_or_csv(abs_path)
        except Exception:
            pdf = None
        if pdf is None:
            return np.array([])
        must_cols = {'sample_id','t','trafo_index','loading_pct'}
        if not must_cols <= set(pdf.columns):
            return np.array([])
        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
        arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
        return arr[np.isfinite(arr)]

    # Build distributions strictly per full label using meta_map (constructed earlier)
    for lab in rt_summary['label']:
        meta = meta_map.get(lab, {})
        dist = _load_loading_distribution(meta if isinstance(meta, dict) else {})
        distributions.append(dist if dist.size else np.array([np.nan]))
        labels_box.append(lab)

    # Create boxplot
    ax_box = axes[4]
    # Matplotlib expects a list of arrays; we already have it
    # Display labels for boxplot
    _labels_box_display = [_display_label_for_case(l) for l in labels_box]
    box = ax_box.boxplot(distributions, labels=_labels_box_display, showfliers=False, patch_artist=True,
                         boxprops=dict(facecolor='#cccccc', alpha=0.7), medianprops=dict(color='black'))
    ax_box.set_ylabel('Transformer loading %')
    ax_box.set_xlabel('epsilon / mode')
    ax_box.set_title('Transformer loading distribution (per-timestep maxima)')
    ax_box.grid(axis='y', alpha=0.3)
    # Optional: annotate median values above boxes
    try:
        for i, dist in enumerate(distributions, start=1):
            if dist.size and np.isfinite(np.nanmedian(dist)):
                ax_box.text(i, np.nanmedian(dist)+1.0, f"{np.nanmedian(dist):.1f}", ha='center', va='bottom', fontsize=7)
    except Exception:
        pass

    # 5. Violin plot (same underlying distributions)
    ax_vio = axes[5]
    # Filter out non-finite values per group
    violin_data: List[np.ndarray] = []
    violin_positions: List[int] = []
    violin_labels: List[str] = []
    for i, (lab, dist) in enumerate(zip(labels_box, distributions), start=1):
        vals = dist[np.isfinite(dist)] if isinstance(dist, np.ndarray) else np.array([])
        if vals.size > 0:
            violin_data.append(vals)
            violin_positions.append(i)
            violin_labels.append(lab)
    if violin_data:
        parts = ax_vio.violinplot(violin_data, positions=violin_positions, showmeans=False, showmedians=True, showextrema=False)
        # Style violins
        for pc in parts['bodies']:
            pc.set_facecolor('#b2df8a')
            pc.set_edgecolor('#1b7837')
            pc.set_alpha(0.6)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('#1b7837')
        ax_vio.set_xticks(violin_positions)
        ax_vio.set_xticklabels([_display_label_for_case(l) for l in violin_labels])
    else:
        # No data available: show placeholder
        ax_vio.text(0.5, 0.5, 'No transformer loading data', ha='center', va='center', transform=ax_vio.transAxes, fontsize=9, color='gray')
        ax_vio.set_xticks(range(1, len(labels_box)+1))
        ax_vio.set_xticklabels(labels_box)
    ax_vio.set_ylabel('Transformer loading %')
    ax_vio.set_xlabel('epsilon / mode')
    ax_vio.set_title('Transformer loading (violin)')
    ax_vio.grid(axis='y', alpha=0.3)

    # Standalone multi-case violin figure (optional)
    if PLOT_VIOLIN_ALL_CASES:
        try:
            # Reuse distributions & labels_box already built (labels may include RT modes)
            finite_groups = []
            finite_labels = []
            for lab, dist in zip(labels_box, distributions):
                vals = dist[np.isfinite(dist)] if isinstance(dist, np.ndarray) else np.array([])
                if vals.size:
                    finite_groups.append(vals)
                    finite_labels.append(lab)
            if finite_groups:
                fig_va, ax_va = plt.subplots(figsize=(1.4*len(finite_groups)+2, 4.0))
                vp = ax_va.violinplot(finite_groups, positions=list(range(1, len(finite_groups)+1)), showmeans=False, showmedians=True, showextrema=False)
                for i, body in enumerate(vp['bodies']):
                    base = finite_labels[i]
                    # Color code RT tag if present
                    if 'rt_off' in base:
                        body.set_facecolor('#9ecae1'); body.set_edgecolor('#08519c')
                    elif 'rt_on' in base:
                        body.set_facecolor('#fb6a4a'); body.set_edgecolor('#cb181d')
                    else:
                        body.set_facecolor('#b2df8a'); body.set_edgecolor('#1b7837')
                    body.set_alpha(0.65)
                if 'cmedians' in vp:
                    vp['cmedians'].set_color('black')
                ax_va.set_xticks(range(1, len(finite_groups)+1))
                # Display labels using epsilon display helper; preserve RT mode suffix in parenthesis
                disp = []
                for lab in finite_labels:
                    base_disp = _display_label_for_case(_label_base(lab) if '_label_base' in globals() else lab)
                    if 'rt_on' in lab:
                        base_disp += ' (RT ON)'
                    elif 'rt_off' in lab:
                        base_disp += ' (RT OFF)'
                    ax_va.set_xlabel('epsilon / mode')
                    disp.append(base_disp)
                ax_va.set_xticklabels(disp, rotation=25, ha='right')
                ax_va.set_ylabel('Transformer loading %')
                ax_va.set_title('Transformer loading distributions (all cases)')
                ax_va.grid(axis='y', alpha=0.3)
                out_va = os.path.join(RESULTS_DIR, VIOLIN_ALL_CASES_FIG)
                fig_va.tight_layout()
                fig_va.savefig(out_va, dpi=150)
                print(f"✓ Multi-case violin figure: {out_va}")
            else:
                print('[INFO] Multi-case violin figure skipped (no finite groups).')
        except Exception as e:
            print(f"[WARN] Failed multi-case violin figure: {e}")

    # 6. Violation severity bar plot (mean exceedance above threshold per case)
    ax_sev = axes[6]
    means_sev: List[float] = []
    pos_sev: List[int] = []
    for i, dist in enumerate(distributions, start=1):
        vals = dist[np.isfinite(dist)] if isinstance(dist, np.ndarray) else np.array([])
        if vals.size == 0:
            means_sev.append(0.0)
            pos_sev.append(i)
            continue
        excess = vals - viol_threshold_eff
        excess = excess[excess > 0]
        means_sev.append(float(np.mean(excess)) if excess.size > 0 else 0.0)
        pos_sev.append(i)
    bars_sev = ax_sev.bar(pos_sev, means_sev, width=0.6, color='#c44e52')
    ax_sev.set_xticks(pos_sev)
    ax_sev.set_xticklabels([_display_label_for_case(l) for l in labels_box])
    ax_sev.set_xlabel('epsilon / mode')
    ax_sev.set_ylabel(f'Exceedance over {viol_threshold_eff:.2f}% (pp)')
    ax_sev.set_title('Violation severity (mean among violations)')
    ax_sev.grid(axis='y', alpha=0.3)
    # annotate bars
    for rect, val in zip(bars_sev, means_sev):
        if np.isfinite(val):
            h = rect.get_height()
            ax_sev.text(rect.get_x() + rect.get_width()/2, h + max(0.01*h, 0.5), f"{val:.1f}", ha='center', va='bottom', fontsize=7)

    # 7. CVaR80 of loading (Option A basis, per-timestep maxima)
    ax_cvar = axes[7]
    cvar90_vals: List[float] = []
    pos_cvar: List[int] = []
    for i, dist in enumerate(distributions, start=1):
        vals = dist[np.isfinite(dist)] if isinstance(dist, np.ndarray) else np.array([])
        if vals.size == 0:
            cvar90_vals.append(0.0)
            pos_cvar.append(i)
            continue
        # CVaR of loading itself (not exceedance): take 80th percentile of vals, then tail mean
        var90 = np.percentile(vals, 80)
        tail = vals[vals >= var90]
        cvar90 = float(np.mean(tail)) if tail.size > 0 else 0.0
        cvar90_vals.append(cvar90)
        pos_cvar.append(i)
    bars_cvar = ax_cvar.bar(pos_cvar, cvar90_vals, width=0.6, color='#6a3d9a')
    ax_cvar.set_xticks(pos_cvar)
    ax_cvar.set_xticklabels([_display_label_for_case(l) for l in labels_box])
    ax_cvar.set_xlabel('epsilon / mode')
    ax_cvar.set_ylabel('Transformer loading %')
    ax_cvar.set_title('Loading CVaR80 (per-timestep maxima)')
    ax_cvar.grid(axis='y', alpha=0.3)
    for rect, val in zip(bars_cvar, cvar90_vals):
        if np.isfinite(val):
            h = rect.get_height()
            ax_cvar.text(rect.get_x() + rect.get_width()/2, h + max(0.01*h, 0.5), f"{val:.1f}", ha='center', va='bottom', fontsize=7)

    # 8. Histogram of top 20% loadings per case (overlay)
    ax_hist = axes[8]
    # Prepare color mapping: baseline black; epsilons via colormap
    def parse_eps_label(lab: str):
        try:
            return float(lab)
        except Exception:
            return None
    eps_values = [parse_eps_label(l) for l in labels_box]
    eps_nums = [e for e in eps_values if e is not None]
    cmap = plt.cm.viridis
    vmin, vmax = (min(eps_nums), max(eps_nums)) if eps_nums else (0.0, 1.0)
    # Determine common bins range from all top-20 tails
    tails_by_label: List[Tuple[str, np.ndarray]] = []
    global_max = 0.0
    for lab, dist in zip(labels_box, distributions):
        vals = dist[np.isfinite(dist)] if isinstance(dist, np.ndarray) else np.array([])
        if vals.size == 0:
            continue
        thr = np.percentile(vals, 80)
        tail = vals[vals >= thr]
        if tail.size == 0:
            continue
        tails_by_label.append((lab, tail))
        global_max = max(global_max, float(np.nanmax(tail)))
    if tails_by_label:
        bins = np.linspace(0, max(100.0, global_max), 20)
        for lab, tail in tails_by_label:
            e = parse_eps_label(lab)
            if e is None:
                color = 'black'
                alpha = 0.25
            else:
                normed = 0.0 if vmax == vmin else (e - vmin) / (vmax - vmin)
                color = cmap(normed)
                alpha = 0.35
            ax_hist.hist(tail, bins=bins, histtype='stepfilled', alpha=alpha, color=color,
                         label=('deterministic' if e is None else f"DRCC, ε={lab}"), density=True)
        ax_hist.set_xlabel('Transformer loading % (top 20%)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Top-20% loading histogram (overlay)')
        ax_hist.grid(axis='y', alpha=0.3)
        # Compact legend
        ax_hist.legend(fontsize=7, ncol=1, frameon=False)
    else:
        ax_hist.text(0.5, 0.5, 'No top-20% data', ha='center', va='center', transform=ax_hist.transAxes, fontsize=9, color='gray')

    out_path = os.path.join(RESULTS_DIR, OUT_FIG)
    fig.savefig(out_path, dpi=150)
    print(f"✓ Overview saved: {out_path}")
    print(f"✓ Summary CSV: {os.path.join(RESULTS_DIR, OUT_CSV)}")

    # --- New: Split-violin comparison figure (deterministic vs epsilon=0.05) ---
    if PLOT_VIOLIN_COMPARE:
        try:
            # Variant-aware grouped violin: if rt_tag present and both rt_on/off appear for any epsilon
            if 'rt_tag' in rt_summary.columns and any(rt_summary['rt_tag'].notna()):
                # Build mapping epsilon -> {rt_tag: distribution array}
                # Use distributions list aligned with labels_box
                dist_map = {lab: arr for lab, arr in zip(labels_box, distributions)}
                eps_groups: Dict[str, Dict[str, np.ndarray]] = {}
                for lab in labels_box:
                    base = lab  # labels are raw epsilon strings or 'deterministic'
                    if base == DETERMINISTIC_LABEL:
                        continue
                    # Find rt tags for this epsilon from rt_summary by matching label base
                    matching_rows = rt_summary[rt_summary['label'] == base]
                    for _, row in matching_rows.iterrows():
                        tag = row.get('rt_tag')
                        arr = dist_map.get(base, np.array([]))
                        if tag and isinstance(arr, np.ndarray) and arr.size:
                            eps_groups.setdefault(base, {})[tag] = arr[np.isfinite(arr)]
                # Filter only groups with at least one variant
                eps_groups = {k: v for k, v in eps_groups.items() if v}
                if eps_groups:
                    # Build side-by-side violins: order eps ascending; each epsilon yields consecutive positions for rt_off then rt_on
                    try:
                        ordered_eps = sorted(eps_groups.keys(), key=lambda x: float(x))
                    except Exception:
                        ordered_eps = sorted(eps_groups.keys())
                    data_list: List[np.ndarray] = []
                    pos_list: List[int] = []
                    tick_labels: List[str] = []
                    pos = 1
                    for eps_lab in ordered_eps:
                        variants = eps_groups[eps_lab]
                        # ensure consistent order
                        for tag in ('rt_off','rt_on','rt_unk'):
                            arr = variants.get(tag)
                            if arr is None or arr.size == 0:
                                continue
                            data_list.append(arr)
                            pos_list.append(pos)
                            tick_labels.append(f"ε={eps_lab} {'RT OFF' if tag=='rt_off' else 'RT ON' if tag=='rt_on' else 'RT ?'}")
                            pos += 1
                    if data_list:
                        fig_grp, ax_grp = plt.subplots(figsize=(1.6*len(pos_list)+2, 3.6))
                        vp = ax_grp.violinplot(data_list, positions=pos_list, showmeans=False, showmedians=True, showextrema=False)
                        for i, body in enumerate(vp['bodies']):
                            lab = tick_labels[i]
                            if 'RT OFF' in lab:
                                body.set_facecolor('#9ecae1')
                                body.set_edgecolor('#08519c')
                            elif 'RT ON' in lab:
                                body.set_facecolor('#fb6a4a')
                                body.set_edgecolor('#cb181d')
                            else:
                                body.set_facecolor('#cccccc')
                                body.set_edgecolor('#666666')
                            body.set_alpha(0.65)
                        if 'cmedians' in vp:
                            vp['cmedians'].set_color('black')
                        ax_grp.set_xticks(pos_list)
                        ax_grp.set_xticklabels(tick_labels, rotation=25, ha='right')
                        ax_grp.set_ylabel('Transformer loading %')
                        ax_grp.set_title('Loading distributions by ε and RT mode')
                        ax_grp.grid(axis='y', alpha=0.3)
                        out_grp = os.path.join(RESULTS_DIR, VIOLIN_COMPARE_FIG)
                        fig_grp.tight_layout()
                        fig_grp.savefig(out_grp, dpi=150)
                        print(f"✓ Violin comparison figure (RT variants): {out_grp}")
                        # Do not return here; continue to subsequent plots (SoC envelopes, etc.)
            # Fallback original deterministic vs epsilon=0.05 split violin
            dist_map = {lab: arr for lab, arr in zip(labels_box, distributions)}
            left_label = DETERMINISTIC_LABEL
            right_label = f"{0.05:.2f}"
            left_vals = dist_map.get(left_label, np.array([]))
            right_vals = dist_map.get(right_label, np.array([]))
            left_vals = left_vals[np.isfinite(left_vals)] if isinstance(left_vals, np.ndarray) and left_vals.size else np.array([])
            right_vals = right_vals[np.isfinite(right_vals)] if isinstance(right_vals, np.ndarray) and right_vals.size else np.array([])
            fig_cmp, ax_cmp = plt.subplots(figsize=(4.0, 8.0))
            x0 = 1.0
            if left_vals.size == 0 and right_vals.size == 0:
                ax_cmp.text(0.5, 0.5, 'No transformer loading data', ha='center', va='center', transform=ax_cmp.transAxes, fontsize=9, color='gray')
                ax_cmp.set_xticks([x0])
                ax_cmp.set_xticklabels([f"{_display_label_for_case(left_label)} vs DRCC, ε={right_label}"])
            else:
                vals_list = []
                side_tags = []
                if left_vals.size:
                    vals_list.append(left_vals); side_tags.append('left')
                if right_vals.size:
                    vals_list.append(right_vals); side_tags.append('right')
                stacked = np.concatenate(vals_list) if vals_list else np.array([])
                y_min = float(np.nanmin(stacked)) if stacked.size else 0.0
                y_max = float(np.nanmax(stacked)) if stacked.size else 1.0
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                    y_min, y_max = 0.0, 1.0
                pad = 0.02 * (y_max - y_min + 1e-9)
                y_min -= pad; y_max += pad
                ax_cmp.set_ylim(y_min, y_max)
                ax_cmp.set_xlim(x0 - 0.6, x0 + 0.6)
                vp = ax_cmp.violinplot(vals_list, positions=[x0 for _ in vals_list], showmeans=False, showmedians=False, showextrema=False)
                from matplotlib.patches import Rectangle, Patch
                xmin, xmax = ax_cmp.get_xlim(); ymin, ymax = ax_cmp.get_ylim()
                left_clip = Rectangle((xmin, ymin), width=(x0 - xmin), height=(ymax - ymin), transform=ax_cmp.transData)
                right_clip = Rectangle((x0, ymin), width=(xmax - x0), height=(ymax - ymin), transform=ax_cmp.transData)
                for body, tag in zip(vp['bodies'], side_tags):
                    if tag == 'left':
                        body.set_facecolor('#b2df8a'); body.set_edgecolor('#1b7837'); body.set_alpha(0.6); body.set_clip_path(left_clip)
                    else:
                        body.set_facecolor('#a6cee3'); body.set_edgecolor('#1f78b4'); body.set_alpha(0.6); body.set_clip_path(right_clip)
                if isinstance(vp, dict) and 'cmedians' in vp and vp['cmedians'] is not None:
                    try: vp['cmedians'].set_visible(False)
                    except Exception: pass
                if left_vals.size:
                    m_left = float(np.nanmedian(left_vals)); ax_cmp.plot([x0 - 0.21, x0], [m_left, m_left], color='#1b7837', linewidth=2)
                if right_vals.size:
                    m_right = float(np.nanmedian(right_vals)); ax_cmp.plot([x0, x0 + 0.21], [m_right, m_right], color='#1f78b4', linewidth=2)
                ax_cmp.set_xticks([x0]); ax_cmp.set_xticklabels([f"{left_label} vs {right_label}"])
                ax_cmp.set_ylabel('Transformer loading %'); ax_cmp.grid(axis='y', alpha=0.3)
                ax_cmp.axvline(x0, color='black', linewidth=0.9, alpha=0.8, zorder=3)
                legend_handles = []
                if left_vals.size:
                    legend_handles.append(Patch(facecolor='#b2df8a', edgecolor='#1b7837', label=_display_label_for_case(left_label), alpha=0.6))
                if right_vals.size:
                    legend_handles.append(Patch(facecolor='#a6cee3', edgecolor='#1f78b4', label=f"DRCC, ε={right_label}", alpha=0.6))
                if legend_handles:
                    ax_cmp.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=8)
            cmp_path = os.path.join(RESULTS_DIR, VIOLIN_COMPARE_FIG)
            fig_cmp.savefig(cmp_path, dpi=150)
            print(f"✓ Violin comparison figure: {cmp_path}")
        except Exception as e:
            print(f"[WARN] Failed to build violin comparison figure: {e}")

    # --- New: Focused ε=0.10 RT ON vs RT OFF split violin ---
    if PLOT_VIOLIN_EPS_010_RT_COMPARE:
        try:
            target_eps = f"{0.10:.2f}"  # "0.10"
            dist_map = {lab: arr for lab, arr in zip(labels_box, distributions)}
            # Expected labels like "0.10 (RT ON)" and "0.10 (RT OFF)"
            label_on = None; label_off = None
            for lab in labels_box:
                if lab.startswith(target_eps):
                    if '(RT ON' in lab:
                        label_on = lab
                    elif '(RT OFF' in lab:
                        label_off = lab
            if not label_on and not label_off:
                print(f"[INFO] ε={target_eps} RT ON/OFF distributions not both present; skipping focused violin.")
            else:
                vals_on = dist_map.get(label_on, np.array([])) if label_on else np.array([])
                vals_off = dist_map.get(label_off, np.array([])) if label_off else np.array([])
                vals_on = vals_on[np.isfinite(vals_on)] if isinstance(vals_on, np.ndarray) and vals_on.size else np.array([])
                vals_off = vals_off[np.isfinite(vals_off)] if isinstance(vals_off, np.ndarray) and vals_off.size else np.array([])
                fig_rt, ax_rt = plt.subplots(figsize=(4.2, 8.0))
                x0 = 1.0
                if vals_on.size == 0 and vals_off.size == 0:
                    ax_rt.text(0.5, 0.5, 'No ε=0.10 RT data', ha='center', va='center', transform=ax_rt.transAxes,
                               fontsize=9, color='gray')
                    ax_rt.set_xticks([x0]); ax_rt.set_xticklabels([f"ε={target_eps} RT ON vs RT OFF"])
                else:
                    vals_list = []
                    side_tags = []
                    if vals_off.size: vals_list.append(vals_off); side_tags.append('off')
                    if vals_on.size: vals_list.append(vals_on); side_tags.append('on')
                    stacked = np.concatenate(vals_list) if vals_list else np.array([])
                    y_min = float(np.nanmin(stacked)) if stacked.size else 0.0
                    y_max = float(np.nanmax(stacked)) if stacked.size else 1.0
                    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                        y_min, y_max = 0.0, 1.0
                    pad = 0.02 * (y_max - y_min + 1e-9)
                    y_min -= pad; y_max += pad
                    ax_rt.set_ylim(y_min, y_max)
                    ax_rt.set_xlim(x0 - 0.6, x0 + 0.6)
                    vp = ax_rt.violinplot(vals_list, positions=[x0 for _ in vals_list], showmeans=False, showmedians=False, showextrema=False)
                    from matplotlib.patches import Rectangle, Patch
                    xmin, xmax = ax_rt.get_xlim(); ymin, ymax = ax_rt.get_ylim()
                    left_clip = Rectangle((xmin, ymin), width=(x0 - xmin), height=(ymax - ymin), transform=ax_rt.transData)
                    right_clip = Rectangle((x0, ymin), width=(xmax - x0), height=(ymax - ymin), transform=ax_rt.transData)
                    for body, tag in zip(vp['bodies'], side_tags):
                        if tag == 'off':
                            body.set_facecolor('#9ecae1'); body.set_edgecolor('#08519c'); body.set_alpha(0.65); body.set_clip_path(left_clip)
                        else:
                            body.set_facecolor('#fb6a4a'); body.set_edgecolor('#cb181d'); body.set_alpha(0.65); body.set_clip_path(right_clip)
                    # Medians
                    if vals_off.size:
                        m_off = float(np.nanmedian(vals_off)); ax_rt.plot([x0 - 0.21, x0], [m_off, m_off], color='#08519c', linewidth=2)
                    if vals_on.size:
                        m_on = float(np.nanmedian(vals_on)); ax_rt.plot([x0, x0 + 0.21], [m_on, m_on], color='#cb181d', linewidth=2)
                    ax_rt.axvline(x0, color='black', linewidth=0.9, alpha=0.8, zorder=3)
                    ax_rt.set_xticks([x0])
                    ax_rt.set_xticklabels([f"ε={target_eps}"])
                    ax_rt.set_ylabel('Transformer loading %')
                    ax_rt.grid(axis='y', alpha=0.3)
                    legend_handles = []
                    if vals_off.size:
                        legend_handles.append(Patch(facecolor='#9ecae1', edgecolor='#08519c', label=f"ε={target_eps} RT OFF", alpha=0.65))
                    if vals_on.size:
                        legend_handles.append(Patch(facecolor='#fb6a4a', edgecolor='#cb181d', label=f"ε={target_eps} RT ON", alpha=0.65))
                    if legend_handles:
                        ax_rt.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=8)
                out_rt = os.path.join(RESULTS_DIR, VIOLIN_COMPARE_010_RT_FIG)
                fig_rt.tight_layout()
                fig_rt.savefig(out_rt, dpi=150)
                print(f"✓ Focused ε=0.10 RT ON vs OFF violin: {out_rt}")
        except Exception as e:
            print(f"[WARN] Failed focused ε=0.10 RT violin: {e}")

    # --- New: Overload energy comparison (deterministic vs ε=0.05) ---
    if PLOT_OVERLOAD_ENERGY_COMPARE:
        try:
            # Helper to compute total overload energy (excess above threshold) in MVAh from parquet
            threshold_pct_local = viol_threshold_eff  # align with summaries/plots threshold
            def _compute_overload_energy_from_parquet(parquet_path: str) -> float:
                try:
                    pdf = _read_parquet_or_csv(parquet_path)
                except Exception:
                    pdf = None
                if pdf is None:
                    return float('nan')
                must = {'sample_id','t','trafo_index','loading_pct'}
                if not must <= set(pdf.columns):
                    return float('nan')
                lp = pd.to_numeric(pdf['loading_pct'], errors='coerce').to_numpy()
                mask = np.isfinite(lp) & (lp > threshold_pct_local)
                if not np.any(mask):
                    # No exceedances => zero overload energy
                    # Still return 0.0 kWh per sample
                    return 0.0
                excess_pct = lp[mask] - threshold_pct_local
                excess_mva = (excess_pct / 100.0) * RATED_TRAFO_MVA
                total_mvah = float(np.sum(excess_mva) * STEP_HOURS)
                # Convert to kWh and average per sample
                try:
                    n_samples = int(pd.to_numeric(pdf['sample_id'], errors='coerce').dropna().nunique())
                except Exception:
                    n_samples = OVERLOAD_SAMPLE_COUNT_DEFAULT
                n_samples = n_samples if n_samples > 0 else OVERLOAD_SAMPLE_COUNT_DEFAULT
                total_kwh_per_sample = (total_mvah * 1000.0) / float(n_samples)
                return total_kwh_per_sample

            # Load baseline meta for path
            det_meta_path = os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json')
            det_over_mvah = float('nan')
            if os.path.exists(det_meta_path):
                try:
                    with open(det_meta_path,'r',encoding='utf-8') as f:
                        det_meta = json.load(f)
                    rel = det_meta.get('trafo_loading_file')
                    if rel:
                        det_parquet = os.path.join(RESULTS_DIR, rel)
                        det_over_mvah = _compute_overload_energy_from_parquet(det_parquet)
                except Exception:
                    pass
            # Load epsilon=0.05 meta for path
            eps_label = f"{0.05:.2f}"
            meta_005 = load_meta_for_epsilon(0.05)
            eps_over_mvah = float('nan')
            if meta_005 and meta_005.get('trafo_loading_file'):
                pq_005 = os.path.join(RESULTS_DIR, meta_005['trafo_loading_file'])
                eps_over_mvah = _compute_overload_energy_from_parquet(pq_005)

            # If either is available, plot
            if np.isfinite(det_over_mvah) or np.isfinite(eps_over_mvah):
                labels = ['deterministic', f"DRCC, ε={eps_label}"]
                values = [det_over_mvah if np.isfinite(det_over_mvah) else 0.0,
                          eps_over_mvah if np.isfinite(eps_over_mvah) else 0.0]
                fig_ovl, ax_ovl = plt.subplots(figsize=(4.0, 8.0))
                x = np.arange(len(labels))
                bars = ax_ovl.bar(x, values, color='#1f78b4', alpha=0.85)
                ax_ovl.set_xticks(x)
                ax_ovl.set_xticklabels(labels)
                ax_ovl.set_ylabel('Total overload energy per sample (kWh)')
                #ax_ovl.set_title(f'Total transformer overload energy (> {int(OVERLOAD_THRESHOLD_PCT)}%)')
                ax_ovl.grid(axis='y', alpha=0.3)
                for rect, val in zip(bars, values):
                    ax_ovl.text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(0.01*rect.get_height(), 0.01),
                                f"{val:.2f}", ha='center', va='bottom', fontsize=9)
                out_cmp = os.path.join(RESULTS_DIR, OVERLOAD_ENERGY_COMPARE_FIG)
                fig_ovl.tight_layout()
                fig_ovl.savefig(out_cmp, dpi=150)
                print(f"✓ Overload energy comparison figure: {out_cmp}")
                # CSV dump
                csv_df = pd.DataFrame({'label': labels, 'overload_energy_kwh_per_sample': values})
                out_csv = os.path.join(RESULTS_DIR, OVERLOAD_ENERGY_COMPARE_CSV)
                csv_df.to_csv(out_csv, index=False)
                print(f"✓ Overload energy CSV: {out_csv}")
            else:
                print('[INFO] Skipped overload energy comparison (no transformer loading parquet data found).')
        except Exception as e:
            print(f"[WARN] Failed to build overload energy comparison: {e}")

    # --- New: Overload energy comparison (ε=0.10 RT ON vs RT OFF) ---
    if PLOT_OVERLOAD_ENERGY_010_RT_COMPARE:
        try:
            threshold_pct_local = viol_threshold_eff
            def _compute_overload_energy_from_parquet_rt(parquet_path: str) -> float:
                try:
                    pdf = _read_parquet_or_csv(parquet_path)
                except Exception:
                    pdf = None
                if pdf is None:
                    return float('nan')
                must = {'sample_id','t','trafo_index','loading_pct'}
                if not must <= set(pdf.columns):
                    return float('nan')
                lp = pd.to_numeric(pdf['loading_pct'], errors='coerce').to_numpy()
                mask = np.isfinite(lp) & (lp > threshold_pct_local)
                if not np.any(mask):
                    return 0.0
                excess_pct = lp[mask] - threshold_pct_local
                excess_mva = (excess_pct / 100.0) * RATED_TRAFO_MVA
                total_mvah = float(np.sum(excess_mva) * STEP_HOURS)
                try:
                    n_samples = int(pd.to_numeric(pdf['sample_id'], errors='coerce').dropna().nunique())
                except Exception:
                    n_samples = OVERLOAD_SAMPLE_COUNT_DEFAULT
                n_samples = n_samples if n_samples > 0 else OVERLOAD_SAMPLE_COUNT_DEFAULT
                total_kwh_per_sample = (total_mvah * 1000.0) / float(n_samples)
                return total_kwh_per_sample

            tok = epsilon_token(0.10)
            meta_on_path = os.path.join(RESULTS_DIR, f'v4_meta_drcc_true_epsilon_{tok}_rt_on.json')
            meta_off_path = os.path.join(RESULTS_DIR, f'v4_meta_drcc_true_epsilon_{tok}_rt_off.json')
            val_on = float('nan')
            val_off = float('nan')
            # RT ON
            if os.path.exists(meta_on_path):
                try:
                    with open(meta_on_path,'r',encoding='utf-8') as f:
                        m_on = json.load(f)
                    rel = m_on.get('trafo_loading_file') if isinstance(m_on, dict) else None
                    if rel:
                        pq = os.path.join(RESULTS_DIR, rel.replace('/', os.sep))
                        if not os.path.exists(pq) and os.path.exists(rel):
                            pq = rel
                        if os.path.exists(pq):
                            val_on = _compute_overload_energy_from_parquet_rt(pq)
                except Exception:
                    pass
            # RT OFF
            if os.path.exists(meta_off_path):
                try:
                    with open(meta_off_path,'r',encoding='utf-8') as f:
                        m_off = json.load(f)
                    rel = m_off.get('trafo_loading_file') if isinstance(m_off, dict) else None
                    if rel:
                        pq = os.path.join(RESULTS_DIR, rel.replace('/', os.sep))
                        if not os.path.exists(pq) and os.path.exists(rel):
                            pq = rel
                        if os.path.exists(pq):
                            val_off = _compute_overload_energy_from_parquet_rt(pq)
                except Exception:
                    pass

            if np.isfinite(val_on) or np.isfinite(val_off):
                labels = []
                values = []
                colors = []
                # Order: RT OFF (left, blue) then RT ON (right, red) if present
                if np.isfinite(val_off):
                    labels.append('DRCC without RT Recourse, ε = 0.10')
                    values.append(val_off)
                    colors.append('#9ecae1')
                if np.isfinite(val_on):
                    labels.append('DRCC with RT Recourse, ε = 0.10')
                    values.append(val_on)
                    colors.append('#fb6a4a')
                fig_rtovl, ax_rtovl = plt.subplots(figsize=(4.6, 6.0))
                x = np.arange(len(labels))
                bars = ax_rtovl.bar(x, values, color=colors, edgecolor=['#08519c' if 'without' in lab else '#cb181d' for lab in labels], alpha=0.85)
                ax_rtovl.set_xticks(x)
                ax_rtovl.set_xticklabels(labels, rotation=0)
                ax_rtovl.set_ylabel('Total overload energy per sample (kWh)')
                ax_rtovl.grid(axis='y', alpha=0.3)
                for rect, val in zip(bars, values):
                    h = rect.get_height()
                    ax_rtovl.text(rect.get_x() + rect.get_width()/2, h + max(0.01*h, 0.01), f"{val:.2f}", ha='center', va='bottom', fontsize=9)
                out_fig = os.path.join(RESULTS_DIR, OVERLOAD_ENERGY_010_RT_FIG)
                fig_rtovl.tight_layout()
                fig_rtovl.savefig(out_fig, dpi=150)
                print(f"✓ Overload energy ε=0.10 RT ON vs OFF: {out_fig}")
                # CSV
                df_out = pd.DataFrame({'label': labels, 'overload_energy_kwh_per_sample': values})
                out_csv = os.path.join(RESULTS_DIR, OVERLOAD_ENERGY_010_RT_CSV)
                df_out.to_csv(out_csv, index=False)
                print(f"✓ Overload energy ε=0.10 RT CSV: {out_csv}")
            else:
                print('[INFO] Skipped ε=0.10 RT ON/OFF overload energy plot (no parquet data).')
        except Exception as e:
            print(f"[WARN] Failed to build ε=0.10 RT overload energy comparison: {e}")

    # --- New: CVaR90 transformer loading comparison (deterministic vs ε=0.05) ---
    if PLOT_CVAR90_COMPARE:
        try:
            # Deterministic meta and CVaR90
            det_meta_path = os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json')
            det_cvar = float('nan')
            if os.path.exists(det_meta_path):
                try:
                    with open(det_meta_path, 'r', encoding='utf-8') as f:
                        det_meta = json.load(f)
                    sev_det = load_trafo_loading(det_meta)
                    det_cvar = float(sev_det.get('cvar90', float('nan')))
                except Exception:
                    pass
            # Epsilon 0.05 meta and CVaR90
            meta_005 = load_meta_for_epsilon(0.05)
            eps_cvar = float('nan')
            if meta_005:
                try:
                    sev_005 = load_trafo_loading(meta_005)
                    eps_cvar = float(sev_005.get('cvar90', float('nan')))
                except Exception:
                    pass
            if np.isfinite(det_cvar) or np.isfinite(eps_cvar):
                labels = ['deterministic', 'DRCC, ε=0.05']
                values = [det_cvar if np.isfinite(det_cvar) else 0.0,
                          eps_cvar if np.isfinite(eps_cvar) else 0.0]
                fig_cv, ax_cv = plt.subplots(figsize=(4.0, 8.0))
                x = np.arange(len(labels))
                bars = ax_cv.bar(x, values, color='#1f78b4', alpha=0.9)
                ax_cv.set_xticks(x)
                ax_cv.set_xticklabels(labels)
                ax_cv.set_ylabel('CVaR90 transformer loading (%)')
                ax_cv.set_title('CVaR90 Transformer Loading')
                ax_cv.grid(axis='y', alpha=0.3)
                # Start y-axis at 80%
                ax_cv.set_ylim(bottom=80)
                for rect, val in zip(bars, values):
                    ax_cv.text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(0.01*rect.get_height(), 0.05),
                               f"{val:.1f}", ha='center', va='bottom', fontsize=9)
                out_cv = os.path.join(RESULTS_DIR, CVAR90_COMPARE_FIG)
                fig_cv.tight_layout()
                fig_cv.savefig(out_cv, dpi=150)
                print(f"✓ CVaR90 comparison figure: {out_cv}")
                # CSV output
                df_cv = pd.DataFrame({'label': labels, 'cvar90_loading_pct': values})
                out_csv = os.path.join(RESULTS_DIR, CVAR90_COMPARE_CSV)
                df_cv.to_csv(out_csv, index=False)
                print(f"✓ CVaR90 comparison CSV: {out_csv}")
            else:
                print('[INFO] Skipped CVaR90 comparison (no transformer loading data found).')
        except Exception as e:
            print(f"[WARN] Failed to build CVaR90 comparison: {e}")

    # --- Build cost-risk frontier (VaR95 + mean) using per-trajectory v4_summary_* CSVs ---
    def build_frontier(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        pattern = os.path.join(results_dir, 'v4_summary_*.csv')
        for path in glob.glob(pattern):
            try:
                df_sum = pd.read_csv(path)
            except Exception:
                continue
            fname = os.path.basename(path)
            # Skip legacy simple names if a preferred drcc_true exists for same epsilon
            legacy_match = re.match(r'v4_summary_epsilon_([0-9]+_[0-9]+)\.csv', fname)
            if legacy_match:
                tok = legacy_match.group(1)
                preferred = os.path.join(results_dir, f"v4_summary_drcc_true_epsilon_{tok}.csv")
                if os.path.exists(preferred):
                    continue  # ignore legacy because updated file present
            # Skip misnamed drcc_false_epsilon_ variants (deterministic should not carry epsilon)
            if 'drcc_false_epsilon_' in fname:
                continue
            # Mode & epsilon inference
            if 'drcc_false' in fname:
                mode = 'stochastic'
                eps_val = None
                meta_path = os.path.join(results_dir, 'v4_meta_drcc_false.json')
                tok_str = None
            else:
                mode_match = re.search(r'v4_summary_(drcc_[a-zA-Z]+)_epsilon_', fname)
                mode = mode_match.group(1) if mode_match else 'drcc_true'
                tok_match = re.search(r'_epsilon_([0-9]+_[0-9]+)', fname)
                eps_val = None
                meta_path = None
                tok_str = None
                if tok_match:
                    try:
                        tok_str = tok_match.group(1)
                        eps_val = float(tok_str.replace('_', '.'))
                        meta_path = os.path.join(results_dir, f"{ 'v4_meta_drcc_true_epsilon_' + tok_str }.json")
                    except Exception:
                        eps_val = None
                        meta_path = None
            if df_sum.empty:
                continue
            # Mean & VaR95 (quantile) of total cost
            if 'total_cost_eur' not in df_sum.columns:
                continue
            mean_cost = float(pd.to_numeric(df_sum['total_cost_eur'], errors='coerce').dropna().mean())
            series_cost = pd.to_numeric(df_sum['total_cost_eur'], errors='coerce').dropna()
            var95_cost = float(series_cost.quantile(0.95)) if len(series_cost) else float('nan')
            # CVaR95 (tail mean)
            if len(series_cost):
                thr = series_cost.quantile(0.95)
                tail = series_cost[series_cost >= thr]
                cvar95_cost = float(tail.mean()) if len(tail) else float('nan')
            else:
                cvar95_cost = float('nan')
            # Transformer violation rate (average of per-traj ratios)
            if 'steps_trafo_over_80pct' in df_sum.columns and 'n_steps' in df_sum.columns:
                vrates = []
                for _, r in df_sum.iterrows():
                    try:
                        ns = float(r.get('n_steps', float('nan')))
                        st = float(r.get('steps_trafo_over_80pct', float('nan')))
                        if ns > 0 and np.isfinite(st):
                            vrates.append(st / ns)
                    except Exception:
                        continue
                trafo_vrate = float(np.mean(vrates)) if vrates else float('nan')
            else:
                trafo_vrate = float('nan')

            # New: per-timestep violation probability (max over time) computed from parquet, if available
            trafo_vrate_max = float('nan')
            try:
                # Determine meta path if not set above (baseline case)
                if meta_path is None and mode == 'stochastic':
                    meta_path = os.path.join(results_dir, 'v4_meta_drcc_false.json')
                # Attempt computation only if meta exists
                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_obj = json.load(f)
                    rel = meta_obj.get('trafo_loading_file') if isinstance(meta_obj, dict) else None
                    if rel:
                        pq_path = os.path.join(results_dir, rel.replace('/', os.sep))
                        if not os.path.exists(pq_path) and os.path.exists(rel):
                            pq_path = rel
                        if os.path.exists(pq_path):
                            try:
                                pdf = _read_parquet_or_csv(pq_path)
                                if pdf is not None:
                                    must = {'sample_id','t','trafo_index','loading_pct'}
                                    if must <= set(pdf.columns):
                                        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
                                        counts = grp.groupby('t')['sample_id'].nunique()
                                        viol = grp[grp['loading_pct'] > viol_threshold_eff].groupby('t')['sample_id'].nunique()
                                        rate_series = (viol / counts).reindex(counts.index).fillna(0.0)
                                        trafo_vrate_max = float(np.nanmax(rate_series.to_numpy())) if len(rate_series) else float('nan')
                            except Exception:
                                pass
            except Exception:
                trafo_vrate_max = float('nan')
            n_traj = int(len(df_sum))
            n_steps = int(df_sum.get('n_steps', pd.Series([np.nan])).iloc[0]) if 'n_steps' in df_sum.columns else np.nan
            rows.append({
                'file': fname,
                'mode': mode,
                'epsilon': eps_val,
                'mean_cost_eur': mean_cost,
                'var95_cost_eur': var95_cost,
                'cvar95_cost_eur': cvar95_cost,
                'trafo_violation_rate_mean': trafo_vrate,
                'trafo_violation_rate_max': trafo_vrate_max,
                'n_trajectories': n_traj,
                'n_steps': n_steps
            })
        frontier_df = pd.DataFrame(rows)
        if not frontier_df.empty:
            # Order: stochastic first then ascending epsilon
            def sort_key(row):
                if row['mode'] == 'stochastic':
                    return (-1, -1.0)
                return (0, row['epsilon'] if row['epsilon'] is not None else 999)
            frontier_df = frontier_df.sort_values(by=['mode','epsilon'], key=None)
            # Sorting custom since key param for multiple columns not directly combining; so re-sort manually:
            frontier_df = frontier_df.reindex(sorted(frontier_df.index, key=lambda i: sort_key(frontier_df.loc[i])))
            out_fp = os.path.join(results_dir, FRONTIER_CSV)
            frontier_df.to_csv(out_fp, index=False)
            print(f"✓ Frontier CSV: {out_fp}")
        return frontier_df

    frontier_df = build_frontier()

    # --- Frontier scatter plot (mean cost vs trafo violation rate) ---
    if PLOT_FRONTIER_SCATTER and frontier_df is not None and not frontier_df.empty:
        # Deduplicate by picking row with max trajectories per (mode, epsilon)
        subset_rows = []
        for (mode, eps), grp in frontier_df.groupby(['mode','epsilon'], dropna=False):
            # Prefer rows with non-null MAX violation rate; among those pick highest trajectory count
            grp_valid = grp[grp['trafo_violation_rate_max'].notna()]
            if not grp_valid.empty:
                pick = grp_valid.sort_values('n_trajectories', ascending=False).iloc[0]
            else:
                pick = grp.sort_values('n_trajectories', ascending=False).iloc[0]
            subset_rows.append(pick)
        plot_df = pd.DataFrame(subset_rows)
        fig_f, ax_f = plt.subplots(figsize=(6,5))
        # Separate baseline
        base_df = plot_df[plot_df['mode'] == 'stochastic']
        drcc_df = plot_df[plot_df['mode'] != 'stochastic']
        # Plot baseline first (bottom layer)
        if not base_df.empty:
            ax_f.scatter(
                base_df['trafo_violation_rate_max'],
                base_df['mean_cost_eur'],
                marker='o', s=80, c='black', edgecolors='none', zorder=2
            )
            # Annotate deterministic mean point
            for _, r in base_df.iterrows():
                if np.isfinite(r['trafo_violation_rate_max']) and np.isfinite(r['mean_cost_eur']):
                    _ann = ax_f.annotate('deterministic', (r['trafo_violation_rate_max'], r['mean_cost_eur']),
                                         textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                    _ann.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
        # DRCC points layered (preferred ordering): 0.10, 0.15, 0.05 (others, if any, first)
        if not drcc_df.empty:
            eps_vals = drcc_df['epsilon'].to_numpy(dtype=float)
            vmin, vmax = float(np.nanmin(eps_vals)), float(np.nanmax(eps_vals))
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
            preferred = [0.10, 0.15, 0.05]
            present = [float(e) for e in sorted(pd.unique(drcc_df['epsilon'].dropna()))]
            extras = [e for e in present if e not in preferred]
            order_eps = extras + [e for e in preferred if e in present]
            for e in order_eps:
                sub = drcc_df[np.isclose(drcc_df['epsilon'].astype(float), e)]
                if sub.empty:
                    continue
                ax_f.scatter(
                    sub['trafo_violation_rate_max'],
                    sub['mean_cost_eur'],
                    color=cmap(norm(e)), s=70, edgecolors='k', linewidths=0.4, zorder=3
                )
            # Colorbar using ScalarMappable
            # Removed colorbar per user request (keep colormap encoding but no legend bar)
            # sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            # sm.set_array([])
            # fig_f.colorbar(sm, ax=ax_f, label='risk level (ε)')
        # Annotate epsilon values
        for _, r in drcc_df.iterrows():
            if r['epsilon'] is not None and np.isfinite(r['epsilon']):
                _ann2 = ax_f.annotate(f"ε = {r['epsilon']:.2f}", (r['trafo_violation_rate_max'], r['mean_cost_eur']),
                                      textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                _ann2.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
        ax_f.set_xlabel('Transformer violation rate (max over time)')
        ax_f.set_ylabel('Mean total cost (EUR)')
        ax_f.set_title('Cost–Risk Frontier (Mean vs Violation Rate)')
        ax_f.grid(alpha=0.35)
        # Percent formatting on x-axis
        ax_f.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        # Legend removed per request
        frontier_fig_path = os.path.join(RESULTS_DIR, FRONTIER_SCATTER_FIG)
        fig_f.savefig(frontier_fig_path, dpi=150)
        print(f"✓ Frontier scatter: {frontier_fig_path}")

    # --- New: BESS clipping vs transformer violations correlation (ε selection) ---
    if PLOT_BESS_CLIPPING_CORRELATION:
        try:
            # Prefer epsilon=0.10; else pick the smallest available that has a summary
            chosen_eps = None
            chosen_df = None
            for e in ([0.10] + sorted(EPSILONS)):
                try:
                    df_try = load_summary_for_epsilon(e)
                except Exception:
                    df_try = None
                if df_try is not None and not df_try.empty:
                    chosen_eps = e
                    chosen_df = df_try
                    break
            if chosen_df is None:
                print('[INFO] Skipped BESS clipping correlation (no v4_summary for any epsilon).')
            else:
                df = chosen_df.copy()
                must = {
                    'n_steps','steps_trafo_over_80pct',
                    'bess_clip_power_steps','bess_clip_energy_steps','bess_clip_both_steps',
                    'bess_clip_total_energy_mwh','bess_clip_avg_abs_mw'
                }
                missing = [c for c in must if c not in df.columns]
                if missing:
                    print(f"[INFO] Skipped BESS clipping correlation (missing columns: {missing})")
                else:
                    n_steps = pd.to_numeric(df['n_steps'], errors='coerce')
                    with np.errstate(divide='ignore', invalid='ignore'):
                        df['violation_rate'] = pd.to_numeric(df['steps_trafo_over_80pct'], errors='coerce') / n_steps
                        df['clip_power_rate'] = pd.to_numeric(df['bess_clip_power_steps'], errors='coerce') / n_steps
                        df['clip_energy_rate'] = pd.to_numeric(df['bess_clip_energy_steps'], errors='coerce') / n_steps
                        df['clip_both_rate'] = pd.to_numeric(df['bess_clip_both_steps'], errors='coerce') / n_steps
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.dropna(subset=['violation_rate','clip_power_rate','clip_energy_rate','clip_both_rate','bess_clip_avg_abs_mw'])

                    # Persist per-trajectory CSV
                    tok = epsilon_token(chosen_eps)
                    out_csv = os.path.join(RESULTS_DIR, f"{BESS_CLIP_SUMMARY_PREFIX}{tok}.csv")
                    export_cols = []
                    if 'sample_id' in df.columns:
                        export_cols.append('sample_id')
                    export_cols += [
                        'n_steps','steps_trafo_over_80pct','violation_rate',
                        'bess_clip_power_steps','bess_clip_energy_steps','bess_clip_both_steps',
                        'clip_power_rate','clip_energy_rate','clip_both_rate',
                        'bess_clip_total_energy_mwh','bess_clip_avg_abs_mw'
                    ]
                    for c in ['bess_headroom_charge_avg_mw','bess_headroom_discharge_avg_mw']:
                        if c in df.columns:
                            export_cols.append(c)
                    df.loc[:, export_cols].to_csv(out_csv, index=False)
                    print(f"✓ BESS clipping summary CSV (ε={chosen_eps:.2f}): {out_csv}")

                    # Build scatter figure
                    fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
                    pairs = [
                        ('clip_power_rate', 'Power-clip rate'),
                        ('clip_energy_rate', 'Energy-clip rate'),
                        ('bess_clip_avg_abs_mw', 'Avg |clip| (MW)')
                    ]
                    x = df['violation_rate'].to_numpy(dtype=float)
                    for ax, (col, title) in zip(axes_c, pairs):
                        y = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
                        ax.scatter(x, y, s=28, alpha=0.8, color='#1f77b4', edgecolors='none')
                        # Pearson correlation
                        r = np.nan
                        try:
                            mask = np.isfinite(x) & np.isfinite(y)
                            if np.count_nonzero(mask) > 1:
                                r = float(np.corrcoef(x[mask], y[mask])[0,1])
                        except Exception:
                            r = np.nan
                        ax.set_title(f"{title}\n r={r:.2f}" if np.isfinite(r) else title)
                        ax.set_xlabel('Transformer violation rate')
                        ax.grid(alpha=0.3)
                        if col.endswith('_rate'):
                            ax.set_ylabel('rate')
                            ax.set_ylim(0, 1.0)
                            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                        else:
                            ax.set_ylabel('MW')
                            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                    fig_c.suptitle(f"BESS clipping vs transformer violations (ε={chosen_eps:.2f})")
                    out_fig = os.path.join(RESULTS_DIR, BESS_CLIP_CORR_FIG.replace('.png', f'_epsilon_{tok}.png'))
                    fig_c.savefig(out_fig, dpi=150)
                    print(f"✓ BESS clipping correlation figure: {out_fig}")
        except Exception as e:
            print(f"[WARN] Failed to build BESS clipping correlation: {e}")

    # --- Per-trajectory frontier scatter (many dots) ---
    if PLOT_FRONTIER_TRAJECTORY_SCATTER:
        traj_points = []  # list of dicts: {'epsilon':..., 'mode':..., 'vrate':..., 'cost':...}
        # Baseline first
        base_summary = os.path.join(RESULTS_DIR, 'v4_summary_drcc_false.csv')
        if os.path.exists(base_summary):
            try:
                dfb = pd.read_csv(base_summary)
                if 'steps_trafo_over_80pct' in dfb.columns:
                    n_steps_b = dfb.get('n_steps')
                    if n_steps_b is not None and not n_steps_b.isna().all():
                        for _, r in dfb.iterrows():
                            try:
                                ns = float(r.get('n_steps', np.nan))
                                st = float(r.get('steps_trafo_over_80pct', np.nan))
                                if ns > 0 and np.isfinite(st):
                                    traj_points.append({
                                        'epsilon': None,
                                        'mode': 'stochastic',
                                        'vrate': st / ns,
                                        'cost': float(r.get('total_cost_eur', np.nan))
                                    })
                            except Exception:
                                pass
            except Exception:
                pass
        # DRCC runs (include RT variants if present)
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            variant_paths = find_summary_variant_paths(eps, RESULTS_DIR) if INCLUDE_RT_SPLIT else []
            if variant_paths:
                for rt_tag, summary_path, _meta in variant_paths:
                    if not os.path.exists(summary_path):
                        continue
                    try:
                        df_eps = pd.read_csv(summary_path)
                    except Exception:
                        continue
                    if 'steps_trafo_over_80pct' not in df_eps.columns or 'n_steps' not in df_eps.columns:
                        continue
                    for _, r in df_eps.iterrows():
                        try:
                            ns = float(r.get('n_steps', np.nan))
                            st = float(r.get('steps_trafo_over_80pct', np.nan))
                            if ns > 0 and np.isfinite(st):
                                traj_points.append({
                                    'epsilon': eps,
                                    'mode': 'drcc_true',
                                    'rt_tag': rt_tag,
                                    'vrate': st / ns,
                                    'cost': float(r.get('total_cost_eur', np.nan))
                                })
                        except Exception:
                            pass
            else:
                fpath = os.path.join(RESULTS_DIR, f"v4_summary_drcc_true_epsilon_{tok}.csv")
                if not os.path.exists(fpath):
                    continue
                try:
                    df_eps = pd.read_csv(fpath)
                except Exception:
                    continue
                if 'steps_trafo_over_80pct' not in df_eps.columns or 'n_steps' not in df_eps.columns:
                    continue
                for _, r in df_eps.iterrows():
                    try:
                        ns = float(r.get('n_steps', np.nan))
                        st = float(r.get('steps_trafo_over_80pct', np.nan))
                        if ns > 0 and np.isfinite(st):
                            traj_points.append({
                                'epsilon': eps,
                                'mode': 'drcc_true',
                                'rt_tag': None,
                                'vrate': st / ns,
                                'cost': float(r.get('total_cost_eur', np.nan))
                            })
                    except Exception:
                        pass
        if traj_points:
            traj_df = pd.DataFrame(traj_points)
            fig_t, ax_t = plt.subplots(figsize=(6.5,5))
            # Scatter DRCC points colored by epsilon
            drcc_pts = traj_df[traj_df['mode'] == 'drcc_true']
            base_pts = traj_df[traj_df['mode'] == 'stochastic']
            # Draw baseline first (under)
            if not base_pts.empty:
                ax_t.scatter(
                    base_pts['vrate'], base_pts['cost'], marker='o', s=55, c='black', edgecolors='none', alpha=0.85, zorder=3
                )
            # Then DRCC cloud on top
            if not drcc_pts.empty:
                eps_vals = drcc_pts['epsilon'].to_numpy()
                norm = plt.Normalize(vmin=np.nanmin(eps_vals), vmax=np.nanmax(eps_vals))
                cmap = plt.cm.plasma
                sc2 = ax_t.scatter(
                    drcc_pts['vrate'], drcc_pts['cost'], c=eps_vals, cmap=cmap,
                    s=25, alpha=0.65, edgecolors='none', zorder=2
                )
                fig_t.colorbar(sc2, ax=ax_t, label='risk level (ε)')
            ax_t.set_xlabel('Transformer violation rate (trajectory)')
            ax_t.set_ylabel('Trajectory total cost (EUR)')
            ax_t.set_title('Per-Trajectory Cost–Risk Cloud')
            ax_t.grid(alpha=0.35)
            # Percent formatting on x-axis
            ax_t.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            from matplotlib.lines import Line2D
            legend_handles_traj = [
                Line2D([], [], marker='o', linestyle='None', color='black', markersize=6, label='DRCC trajectories'),
                Line2D([], [], marker='o', linestyle='None', color='black', markersize=7, label='deterministic trajectories'),
            ]
            # Legend removed per request
            # ax_t.legend(handles=legend_handles_traj, fontsize=8, frameon=True)
            traj_fig_path = os.path.join(RESULTS_DIR, FRONTIER_TRAJECTORY_SCATTER_FIG)
            fig_t.savefig(traj_fig_path, dpi=150)
            print(f"✓ Frontier trajectory scatter: {traj_fig_path}")
        else:
            print('[INFO] No trajectory-level points available for trajectory scatter plot.')

    # --- Hybrid frontier scatter (faded cloud + mean overlay) ---
    if (PLOT_FRONTIER_SCATTER and PLOT_FRONTIER_TRAJECTORY_SCATTER and
            frontier_df is not None and not frontier_df.empty):
        # Rebuild mean subset (same logic as mean frontier) for consistency
        subset_rows = []
        for (mode, eps), grp in frontier_df.groupby(['mode','epsilon'], dropna=False):
            grp_valid = grp[grp['trafo_violation_rate_max'].notna()]
            if not grp_valid.empty:
                pick = grp_valid.sort_values('n_trajectories', ascending=False).iloc[0]
            else:
                pick = grp.sort_values('n_trajectories', ascending=False).iloc[0]
            subset_rows.append(pick)
        mean_df = pd.DataFrame(subset_rows)
        # Gather trajectory cloud points (baseline + DRCC) using per-trajectory max fraction of trafos violating at any timestep
        cloud_points: List[Dict] = []
        # Helpers
        def _sample_max_violation_fraction_from_parquet(parquet_path: str, threshold_pct: float) -> Dict[int, float]:
            try:
                pdf = _read_parquet_or_csv(parquet_path)
            except Exception:
                pdf = None
            if pdf is None:
                return {}
            must = {'sample_id','t','trafo_index','loading_pct'}
            if not must <= set(pdf.columns):
                return {}
            grp = pdf.groupby(['sample_id','t'])['loading_pct'].apply(lambda s: np.mean(pd.to_numeric(s, errors='coerce') > threshold_pct)).reset_index(name='viol_frac')
            smax = grp.groupby('sample_id')['viol_frac'].max()
            out: Dict[int, float] = {}
            for k, v in smax.items():
                try:
                    out[int(k)] = float(v)
                except Exception:
                    continue
            return out
        def _extract_sample_id(row: pd.Series, row_index: int) -> int:
            for c in ('sample_id','case_id','case_index','trajectory_id','traj_id','i','sample'):
                if c in row and pd.notna(row[c]):
                    try:
                        return int(row[c])
                    except Exception:
                        pass
            return int(row_index)

        base_summary = os.path.join(RESULTS_DIR, 'v4_summary_drcc_false.csv')
        if os.path.exists(base_summary):
            try:
                dfb = pd.read_csv(base_summary)
                # Build baseline sample->max map from parquet
                sample_max_map: Dict[int, float] = {}
                base_meta_path = os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json')
                if os.path.exists(base_meta_path):
                    try:
                        with open(base_meta_path,'r',encoding='utf-8') as f:
                            base_meta = json.load(f)
                        rel = base_meta.get('trafo_loading_file') if isinstance(base_meta, dict) else None
                        if rel:
                            pq_path = os.path.join(RESULTS_DIR, rel.replace('/', os.sep))
                            if not os.path.exists(pq_path) and os.path.exists(rel):
                                pq_path = rel
                            if os.path.exists(pq_path):
                                sample_max_map = _sample_max_violation_fraction_from_parquet(pq_path, viol_threshold_eff)
                    except Exception:
                        sample_max_map = {}
                if 'total_cost_eur' in dfb.columns:
                    for idx, r in dfb.iterrows():
                        sid = _extract_sample_id(r, idx)
                        xval = sample_max_map.get(sid, np.nan)
                        cloud_points.append({'epsilon': None,'mode':'stochastic','vrate': xval,'cost': float(r.get('total_cost_eur', np.nan))})
            except Exception:
                pass
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            fpath = os.path.join(RESULTS_DIR, f'v4_summary_drcc_true_epsilon_{tok}.csv')
            if not os.path.exists(fpath):
                continue
            try:
                df_eps = pd.read_csv(fpath)
            except Exception:
                continue
            if 'total_cost_eur' not in df_eps.columns:
                continue
            # sample->max map for this epsilon
            sample_max_map: Dict[int, float] = {}
            meta_path = os.path.join(RESULTS_DIR, f'v4_meta_drcc_true_epsilon_{tok}.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path,'r',encoding='utf-8') as f:
                        meta = json.load(f)
                    rel = meta.get('trafo_loading_file') if isinstance(meta, dict) else None
                    if rel:
                        pq_path = os.path.join(RESULTS_DIR, rel.replace('/', os.sep))
                        if not os.path.exists(pq_path) and os.path.exists(rel):
                            pq_path = rel
                        if os.path.exists(pq_path):
                            sample_max_map = _sample_max_violation_fraction_from_parquet(pq_path, viol_threshold_eff)
                except Exception:
                    sample_max_map = {}
            for idx, r in df_eps.iterrows():
                sid = _extract_sample_id(r, idx)
                xval = sample_max_map.get(sid, np.nan)
                cloud_points.append({'epsilon': eps,'mode':'drcc_true','vrate': xval,'cost': float(r.get('total_cost_eur', np.nan))})
        if cloud_points and not mean_df.empty:
            cloud_df = pd.DataFrame(cloud_points)
            fig_h, ax_h = plt.subplots(figsize=(6,5))
            # Faded cloud first
            drcc_cloud = cloud_df[cloud_df['mode'] == 'drcc_true']
            base_cloud = cloud_df[cloud_df['mode'] == 'stochastic']
            if not drcc_cloud.empty:
                eps_vals = drcc_cloud['epsilon'].to_numpy(dtype=float)
                norm_c = plt.Normalize(vmin=np.nanmin(eps_vals), vmax=np.nanmax(eps_vals))
                cmap_c = plt.cm.viridis
                ax_h.scatter(drcc_cloud['vrate'], drcc_cloud['cost'], c=eps_vals, cmap=cmap_c,
                             s=20, alpha=0.12, edgecolors='none', zorder=2)
            if not base_cloud.empty:
                ax_h.scatter(base_cloud['vrate'], base_cloud['cost'], marker='o', s=30,
                             c='black', alpha=0.15, edgecolors='none', zorder=1)
            # Mean overlay (reuse style from mean frontier)
            drcc_mean = mean_df[mean_df['mode'] != 'stochastic']
            base_mean = mean_df[mean_df['mode'] == 'stochastic']
            if not drcc_mean.empty:
                eps_mean = drcc_mean['epsilon'].to_numpy(dtype=float)
                vmin_m, vmax_m = float(np.nanmin(eps_mean)), float(np.nanmax(eps_mean))
                norm_m = plt.Normalize(vmin=vmin_m, vmax=vmax_m)
                cmap_m = plt.cm.viridis
                preferred = [0.10, 0.15, 0.05]
                present_m = [float(e) for e in sorted(pd.unique(drcc_mean['epsilon'].dropna()))]
                extras_m = [e for e in present_m if e not in preferred]
                order_eps_m = extras_m + [e for e in preferred if e in present_m]
                for e in order_eps_m:
                    subm = drcc_mean[np.isclose(drcc_mean['epsilon'].astype(float), e)]
                    if subm.empty:
                        continue
                    ax_h.scatter(subm['trafo_violation_rate_max'], subm['mean_cost_eur'],
                                 color=cmap_m(norm_m(e)), s=70, edgecolors='k', linewidths=0.4, zorder=4)
                # Removed colorbar on hybrid mean overlay per user request
                # sm_m = mpl.cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
                # sm_m.set_array([])
                # fig_h.colorbar(sm_m, ax=ax_h, label='risk level (ε)')
            if not base_mean.empty:
                ax_h.scatter(base_mean['trafo_violation_rate_max'], base_mean['mean_cost_eur'],
                             marker='o', s=85, c='black', edgecolors='white', linewidths=0.4, zorder=3)
                # Annotate deterministic mean point(s)
                for _, r in base_mean.iterrows():
                    if np.isfinite(r['trafo_violation_rate_max']) and np.isfinite(r['mean_cost_eur']):
                        _ann3 = ax_h.annotate('deterministic', (r['trafo_violation_rate_max'], r['mean_cost_eur']),
                                               textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                        _ann3.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
            for _, r in drcc_mean.iterrows():
                if r['epsilon'] is not None and np.isfinite(r['epsilon']):
                    _ann4 = ax_h.annotate(f"ε = {r['epsilon']:.2f}", (r['trafo_violation_rate_max'], r['mean_cost_eur']),
                                          textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                    _ann4.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
            ax_h.set_xlabel('Transformer violation rate (trajectory / max over time)')
            ax_h.set_ylabel('Total cost (EUR)')
            ax_h.set_title('Hybrid Cost–Risk Frontier (Cloud + Mean)')
            ax_h.grid(alpha=0.35)
            # Percent formatting on x-axis
            ax_h.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            hybrid_path = os.path.join(RESULTS_DIR, FRONTIER_HYBRID_SCATTER_FIG)
            fig_h.savefig(hybrid_path, dpi=150)
            print(f"✓ Frontier hybrid scatter: {hybrid_path}")
        else:
            print('[INFO] Hybrid frontier scatter skipped (insufficient data).')

    # --- Transformer violation probability per timestep (now RT variants distinguished) ---
    if PLOT_TRAFO_VIOLATION_TIME_PROFILE:
        threshold_pct = viol_threshold_eff  # strict '>' threshold
        profiles: List[Tuple[str, np.ndarray]] = []  # (label, rate_series)
        profiles_detail: Dict[str, Dict[str, np.ndarray]] = {}
        t_axis: np.ndarray | None = None

        def compute_profile(parquet_path: str):
            """Return (t_index_array, n_counts, n_violations, rate_series)."""
            try:
                pdf = _read_parquet_or_csv(parquet_path)
            except Exception:
                pdf = None
            if pdf is None:
                return None
            must = {'sample_id','t','trafo_index','loading_pct'}
            if not must <= set(pdf.columns):
                return None
            grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
            counts = grp.groupby('t')['sample_id'].nunique()
            viol = grp[grp['loading_pct'] > threshold_pct].groupby('t')['sample_id'].nunique()
            viol = viol.reindex(counts.index).fillna(0.0)
            rate_series = (viol / counts).astype(float)
            return counts.index.to_numpy(), counts.to_numpy(), viol.to_numpy(), rate_series.to_numpy()

        # Helper to add a meta path (with label) for baseline or epsilon variants
        def add_meta_variant(label: str, meta_path: str):
            nonlocal t_axis
            if not os.path.exists(meta_path):
                return
            try:
                with open(meta_path,'r',encoding='utf-8') as f:
                    meta = json.load(f)
                rel = meta.get('trafo_loading_file') if isinstance(meta, dict) else None
                if not rel:
                    return
                pq_path = os.path.join(RESULTS_DIR, rel.replace('/', os.sep))
                if not os.path.exists(pq_path):
                    # allow absolute path fallback
                    if os.path.exists(rel):
                        pq_path = rel
                if not os.path.exists(pq_path):
                    return
                res = compute_profile(pq_path)
                if not res:
                    return
                t_local, cnts, viols, rate = res
                if t_axis is None:
                    t_axis = t_local
                else:
                    if len(t_local) != len(t_axis):
                        min_len = min(len(t_local), len(t_axis))
                        t_axis = t_axis[:min_len]
                        rate = rate[:min_len]; cnts = cnts[:min_len]; viols = viols[:min_len]
                profiles.append((label, rate))
                profiles_detail[label] = {'t': t_axis.copy(), 'n': cnts.copy(), 'k': viols.copy(), 'p': rate.copy()}
            except Exception as e:
                print(f"[WARN] Failed adding trafo profile for {label}: {e}")

        # Baseline RT variants (deterministic case) strict enumeration
        for tag in ('rt_on','rt_off','rt_unk'):
            add_meta_variant(f"{DETERMINISTIC_LABEL} ({_rt_display(tag)})", os.path.join(RESULTS_DIR, f'v4_meta_drcc_false_{tag}.json'))
        # Fallback unsuffixed deterministic if none added
        if not any(lab.startswith(DETERMINISTIC_LABEL) for lab, _ in profiles):
            add_meta_variant(DETERMINISTIC_LABEL, os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json'))

        # DRCC epsilon RT variants
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            per_eps_added = False
            for tag in ('rt_on','rt_off','rt_unk'):
                label = f"{eps:.2f} ({_rt_display(tag)})"
                path_meta = os.path.join(RESULTS_DIR, f'v4_meta_drcc_true_epsilon_{tok}_{tag}.json')
                before_ct = len(profiles)
                add_meta_variant(label, path_meta)
                if len(profiles) > before_ct:
                    per_eps_added = True
            if not per_eps_added:  # fallback unsuffixed
                add_meta_variant(f"{eps:.2f}", os.path.join(RESULTS_DIR, f'v4_meta_drcc_true_epsilon_{tok}.json'))

        if profiles and t_axis is not None:
            # Optional sanity check: investigate timesteps that appear as ~100% violations while viol threshold is strict '>'
            if str(os.getenv('V4_DEBUG_VIOL_CHECK', '0')).strip() in {'1','true','True'}:
                try:
                    def _debug_scan_case(label: str, meta: dict, rate: np.ndarray, threshold: float):
                        # Re-load underlying table and compute per-(sample,t) maxima
                        if not isinstance(meta, dict) or 'trafo_loading_file' not in meta:
                            return
                        pq = os.path.join(RESULTS_DIR, meta['trafo_loading_file'].replace('/', os.sep))
                        pdf = _read_parquet_or_csv(pq)
                        if pdf is None or {'sample_id','t','trafo_index','loading_pct'} - set(pdf.columns):
                            return
                        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
                        # Find t where plotted rate >= 0.999
                        idx_full = np.where(rate >= 0.999)[0]
                        if idx_full.size == 0:
                            return
                        print(f"[DEBUG] Case {label}: {len(idx_full)} timesteps with ~100% violation prob (strict '>' {threshold:.2f}%)")
                        # Map from local 0..T-1 index to actual t value used in parquet
                        # Our profile rate arrays are ordered by increasing 't'; align via unique sorted t values
                        unique_t = np.sort(grp['t'].unique())
                        for k in idx_full[:10]:  # print first few
                            if k >= len(unique_t):
                                continue
                            tval = unique_t[k]
                            rows = grp[grp['t'] == tval]['loading_pct']
                            vals = pd.to_numeric(rows, errors='coerce').to_numpy()
                            vals = vals[np.isfinite(vals)]
                            if vals.size == 0:
                                continue
                            eq_cnt = int(np.sum(vals == threshold))
                            gt_cnt = int(np.sum(vals > threshold))
                            n = int(len(vals))
                            print(f"  - t={tval}: n={n}, min={np.min(vals):.6f}, max={np.max(vals):.6f}, ==thr={eq_cnt}, >thr={gt_cnt}")

                    # Baseline
                    base_meta_path = os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json')
                    if os.path.exists(base_meta_path):
                        with open(base_meta_path,'r',encoding='utf-8') as f:
                            m0 = json.load(f)
                        # Find its rate series from profiles
                        for lab, arr in profiles:
                            if lab == 'stochastic':
                                _debug_scan_case('deterministic', m0, arr, viol_threshold_eff)
                                break
                    # DRCC eps
                    for eps in EPSILONS:
                        lab = f"{eps:.2f}"
                        m = load_meta_for_epsilon(eps)
                        for lab2, arr in profiles:
                            if lab2 == lab:
                                _debug_scan_case(lab, m, arr, viol_threshold_eff)
                                break
                except Exception as _dbg:
                    print(f"[WARN] Debug violation check failed: {_dbg}")
            # Align all series by the union of available timestep keys instead of truncation
            try:
                # Build union of t across all cases we collected in profiles_detail
                union_t_vals = None
                for lab in [lb for lb, _ in profiles]:
                    det = profiles_detail.get(lab)
                    if isinstance(det, dict) and 't' in det and det['t'] is not None:
                        t_vals = np.asarray(det['t'])
                        union_t_vals = t_vals if union_t_vals is None else np.union1d(union_t_vals, t_vals)
                if union_t_vals is None:
                    # Fallback to previous behavior if no t arrays found
                    min_len = min(len(r) for _, r in profiles)
                    profiles = [(lab, r[:min_len]) for lab, r in profiles]
                    t_axis = t_axis[:min_len]
                else:
                    # Reindex each rate series onto the union t-axis
                    aligned_profiles = []
                    for lab, _rate in profiles:
                        det = profiles_detail.get(lab, {})
                        t_vals = np.asarray(det.get('t', []))
                        p_vals = np.asarray(det.get('p', []), dtype=float)
                        # Map existing t positions to union indices
                        aligned = np.full(shape=(len(union_t_vals),), fill_value=np.nan, dtype=float)
                        if t_vals.size and p_vals.size:
                            # Build index via match
                            # Use dict for speed on generic small horizons
                            idx_map = {int(tv): i for i, tv in enumerate(union_t_vals)}
                            for j, tv in enumerate(t_vals):
                                i_u = idx_map.get(int(tv))
                                if i_u is not None and j < len(p_vals):
                                    aligned[i_u] = float(p_vals[j])
                        aligned_profiles.append((lab, aligned))
                    profiles = aligned_profiles
                    t_axis = union_t_vals
            except Exception:
                # On any alignment failure, revert to safe truncation
                min_len = min(len(r) for _, r in profiles)
                profiles = [(lab, r[:min_len]) for lab, r in profiles]
                t_axis = t_axis[:min_len]
            fig_tp, ax_tp = plt.subplots(figsize=(10,4.8))
            # Plot with ordering: deterministic variants first (rt_on, rt_off, rt_unk), then epsilons ascending with their RT tags
            def _order_key(lab: str):
                if lab.startswith(DETERMINISTIC_LABEL):
                    # deterministic (rt_on first, then rt_off, rt_unk, unsuffixed)
                    if '(RT ON)' in lab: return (0,0)
                    if '(RT OFF)' in lab: return (0,1)
                    if '(RT ?)' in lab or '(RT UNK)' in lab: return (0,2)
                    return (0,3)
                # epsilon labels start with number
                try:
                    base = lab.split()[0]
                    val = float(base)
                except Exception:
                    val = 999.0
                # keep rt_on before rt_off before rt_unk before unsuffixed for same epsilon
                if '(RT ON)' in lab: tag_rank = 0
                elif '(RT OFF)' in lab: tag_rank = 1
                elif '(RT ?)' in lab or '(RT UNK)' in lab: tag_rank = 2
                else: tag_rank = 3
                return (1, val, tag_rank)
            for lab, arr in sorted(profiles, key=lambda x: _order_key(x[0])):
                # Choose style: deterministic dashed, others solid
                style = '--' if lab.startswith(DETERMINISTIC_LABEL) else '-'
                ax_tp.plot(t_axis, arr, linewidth=1.2 if style=='-' else 1.6, linestyle=style, alpha=0.95, label=lab)
            ax_tp.set_xlabel('Timestep index')
            ax_tp.set_ylabel(f"P(any trafo > {threshold_pct:.2f}%)")
            ax_tp.set_title('Per-Timestep Transformer Violation Probability')
            ax_tp.grid(alpha=0.3, linewidth=0.5)
            ax_tp.set_ylim(0, 1.0)
            ax_tp.legend(fontsize=8, ncol=min(4, len(profiles)), frameon=False)
            out_tp = os.path.join(RESULTS_DIR, TRAFO_VIOLATION_TIME_PROFILE_FIG)
            fig_tp.tight_layout()
            fig_tp.savefig(out_tp, dpi=160)
            print(f"✓ Transformer violation time profile: {out_tp}")
            # Also export a CSV with per-timestep counts, violations, and probabilities per case
            try:
                rows_tp: List[Dict[str, float]] = []
                for lab, det in profiles_detail.items():
                    t_vals = det['t']
                    n_vals = det['n']
                    k_vals = det['k']
                    p_vals = det['p']
                    L = min(len(t_vals), len(n_vals), len(k_vals), len(p_vals))
                    for i in range(L):
                        rows_tp.append({
                            'case': lab,
                            't': int(t_vals[i]),
                            'n_samples': int(n_vals[i]),
                            'n_violations': int(k_vals[i]),
                            'violation_probability': float(p_vals[i])
                        })
                out_tp_csv = os.path.join(RESULTS_DIR, 'trafo_violation_time_profile.csv')
                pd.DataFrame(rows_tp).to_csv(out_tp_csv, index=False)
                print(f"✓ Transformer violation time profile CSV: {out_tp_csv}")
            except Exception as _e:
                print(f"[WARN] Could not write time profile CSV: {_e}")
        else:
            print('[INFO] Skipped transformer violation time profile (no loading parquet data).')

    # --- Transformer violation probability heatmap (cases x timesteps, RT variants) ---
    if 'profiles' in locals() and profiles and 't_axis' in locals() and t_axis is not None and PLOT_TRAFO_VIOLATION_HEATMAP:
        try:
            # Re-order using same key function as time profile
            def _hm_order_key(lab: str):
                if lab.startswith(DETERMINISTIC_LABEL):
                    if '(RT ON)' in lab: return (0,0)
                    if '(RT OFF)' in lab: return (0,1)
                    if '(RT ?)' in lab or '(RT UNK)' in lab: return (0,2)
                    return (0,3)
                try:
                    base = lab.split()[0]
                    val = float(base)
                except Exception:
                    val = 999.0
                if '(RT ON)' in lab: tag_rank = 0
                elif '(RT OFF)' in lab: tag_rank = 1
                elif '(RT ?)' in lab or '(RT UNK)' in lab: tag_rank = 2
                else: tag_rank = 3
                return (1, val, tag_rank)
            profiles_sorted = sorted(profiles, key=lambda it: _hm_order_key(it[0]))
            # Build a shared t-axis using profiles_detail if available; otherwise, use current t_axis
            try:
                union_t_vals = None
                if 'profiles_detail' in locals() and isinstance(profiles_detail, dict):
                    for lab, _ in profiles_sorted:
                        det = profiles_detail.get(lab)
                        if isinstance(det, dict) and 't' in det and det['t'] is not None:
                            t_vals = np.asarray(det['t'])
                            union_t_vals = t_vals if union_t_vals is None else np.union1d(union_t_vals, t_vals)
                if union_t_vals is None:
                    # Fall back to existing axis
                    union_t_vals = np.asarray(t_axis)
                # For each profile, align to union_t_vals
                aligned_rows = []
                for lab, r in profiles_sorted:
                    if 'profiles_detail' in locals() and isinstance(profiles_detail, dict) and lab in profiles_detail:
                        det = profiles_detail[lab]
                        t_vals = np.asarray(det.get('t', []))
                        p_vals = np.asarray(det.get('p', []), dtype=float)
                        row = np.full(shape=(len(union_t_vals),), fill_value=np.nan, dtype=float)
                        if t_vals.size and p_vals.size:
                            idx_map = {int(tv): i for i, tv in enumerate(union_t_vals)}
                            for j, tv in enumerate(t_vals):
                                i_u = idx_map.get(int(tv))
                                if i_u is not None and j < len(p_vals):
                                    row[i_u] = float(p_vals[j])
                        aligned_rows.append(row)
                    else:
                        # If we don't have detailed t, assume r already matches current t_axis
                        # Reindex to union_t by simple truncation/padding to the left
                        row = np.full(shape=(len(union_t_vals),), fill_value=np.nan, dtype=float)
                        L = min(len(r), len(union_t_vals))
                        row[:L] = r[:L]
                        aligned_rows.append(row)
                mat = np.vstack(aligned_rows)
                T = mat.shape[1]
            except Exception:
                # Fallback: previous behavior (min-length truncation)
                min_len = min(len(r) for _, r in profiles_sorted)
                mat = np.vstack([r[:min_len] for _, r in profiles_sorted])
                T = mat.shape[1]
            # Use full probability range [0,1]
            mat = np.clip(mat, 0.0, 1.0)
            case_labels = [lab for lab, _ in profiles_sorted]
            # Build heatmap
            fig_hm, ax_hm = plt.subplots(figsize=(T/10 + 2.5, 0.6*len(case_labels) + 1.8))
            # Mask NaNs so truly-missing timesteps are not misinterpreted
            mat_masked = np.ma.masked_invalid(mat)
            # Improve contrast by saturating at 0.4 probability
            vmax_limit = 0.4
            im = ax_hm.imshow(mat_masked, aspect='auto', cmap=WHITE_BLUE_CMAP, vmin=0.0, vmax=vmax_limit, interpolation='nearest')
            ax_hm.set_yticks(np.arange(len(case_labels)))
            ax_hm.set_yticklabels(case_labels)
            ax_hm.set_xlabel('timestep index')
            ax_hm.set_ylabel('case (ε + RT)')
            ax_hm.set_title('Transformer Overload Probability per Timestep (All RT Variants)')
            # Colorbar on the right
            cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
            cbar.set_label('violation probability (saturated at 0.4)')
            # Optionally thin x ticks for readability on long horizons
            if T > 48:
                step = 24 if T > 200 else 12
                ax_hm.set_xticks(np.arange(0, T, step))
            # Save figure
            out_hm = os.path.join(RESULTS_DIR, TRAFO_VIOLATION_HEATMAP_FIG)
            fig_hm.tight_layout()
            fig_hm.savefig(out_hm, dpi=160)
            print(f"✓ Transformer violation heatmap: {out_hm}")
        except Exception as e:
            print(f"[WARN] Failed to build transformer violation heatmap: {e}")

    # --- Policy heatmaps ---
    if PLOT_POLICY_HEATMAPS:
        # Collect coefficient files: baseline + each epsilon
        heat_cases: List[Tuple[str, str]] = []
        base_pol = os.path.join(RESULTS_DIR, 'policy_coeffs_drcc_false.csv')
        if os.path.exists(base_pol):
            heat_cases.append((DETERMINISTIC_LABEL, base_pol))
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            # Support RT-suffixed variants; prefer rt_on then rt_off then unsuffixed
            pol_candidates = [
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_on.csv'),
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_off.csv'),
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_unk.csv'),
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}.csv'),
            ]
            chosen_pol = None
            for _p in pol_candidates:
                if os.path.exists(_p):
                    chosen_pol = _p
                    break
            if chosen_pol:
                heat_cases.append((f"{eps:.2f}", chosen_pol))
        if heat_cases:
            # Load first to get coeff order
            coeff_order: List[str] = []
            data_mats: List[Tuple[str, np.ndarray, List[str]]] = []
            for label, path_pol in heat_cases:
                try:
                    pdf = pd.read_csv(path_pol)
                except Exception:
                    continue
                if 'timestamp' in pdf.columns:
                    pdf = pdf.drop(columns=['timestamp'])
                # Identify coefficients: prefer K-gain columns if present; otherwise use all numeric
                k_pref = ['K_pv_bess','K_hp_bess','K_pv_pvcurt','K_hp_pvcurt']
                available_k = [c for c in k_pref if c in pdf.columns]
                if available_k:
                    cols = available_k
                else:
                    cols = [c for c in pdf.columns if np.issubdtype(pdf[c].dtype, np.number)]
                if not cols:
                    continue
                if not coeff_order:
                    coeff_order = cols
                # Reindex columns to coeff_order subset intersection
                use_cols = [c for c in coeff_order if c in cols]
                mat = pdf[use_cols].to_numpy(dtype=float).T  # shape (coeffs, T)
                # Min-max normalize per coefficient for visualization only
                mat_norm = mat.copy()
                for i_c in range(mat_norm.shape[0]):
                    row = mat_norm[i_c]
                    r_min = np.nanmin(row)
                    r_max = np.nanmax(row)
                    if np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
                        mat_norm[i_c] = (row - r_min) / (r_max - r_min)
                    else:
                        mat_norm[i_c] = 0.0
                data_mats.append((label, mat_norm, use_cols))
            if data_mats:
                n_cases = len(data_mats)
                fig_h, axes_h = plt.subplots(1, n_cases, figsize=(3.5*n_cases, 0.5*len(coeff_order)+2), constrained_layout=True)
                if n_cases == 1:
                    axes_h = [axes_h]
                # Pretty math labels for coefficients (λ, χ, γ, ρ)
                import re as _re
                def _pretty_coeff_label(name: str) -> str:
                    s = str(name)
                    # strip common unit suffixes
                    for suf in ('_mw','_mwh','_pu','_eur'):
                        if s.endswith(suf):
                            s = s[: -len(suf)]
                    low = s.lower()
                    # K gains
                    if low == 'k_pv_bess':
                        return r'$K_{pv\to bess}$'
                    if low == 'k_hp_bess':
                        return r'$K_{hp\to bess}$'
                    if low == 'k_pv_pvcurt':
                        return r'$K_{pv\to curt}$'
                    if low == 'k_hp_pvcurt':
                        return r'$K_{hp\to curt}$'
                    # lambda family
                    if low.startswith('lambda0') or low == 'lambda0':
                        return r'$\lambda^{(0)}$'
                    if low.startswith('lambda_plus'):
                        return r'$\lambda^{(+)}$'
                    if low.startswith('lambda_minus'):
                        return r'$\lambda^{(-)}$'
                    # chi family
                    if low.startswith('chi0') or low == 'chi0':
                        return r'$\chi^{(0)}$'
                    if low.startswith('chi_plus'):
                        return r'$\chi^{(+)}$'
                    if low.startswith('chi_minus'):
                        return r'$\chi^{(-)}$'
                    # gamma family
                    if low.startswith('gamma0') or low == 'gamma0':
                        return r'$\gamma^{(0)}$'
                    if low.startswith('gamma_plus'):
                        return r'$\gamma^{(+)}$'
                    if low.startswith('gamma_minus'):
                        return r'$\gamma^{(-)}$'
                    # rho family with sign and index
                    m = _re.match(r'rho_(plus|minus)_([0-9]+)', low)
                    if m:
                        sign = '+' if m.group(1) == 'plus' else '-'
                        idx = m.group(2)
                        return rf'$\\rho_{{{sign}}}^{{({idx})}}$'
                    # fallback to raw name
                    return s

                for ax_h, (lab, mat_norm, cols) in zip(axes_h, data_mats):
                    im = ax_h.imshow(mat_norm, aspect='auto', cmap=WHITE_BLUE_CMAP, interpolation='nearest', vmin=0.0, vmax=1.0)
                    # Use standardized labeling for cases
                    ax_h.set_title(_display_label_for_case(lab))
                    ax_h.set_yticks(range(len(cols)))
                    ax_h.set_yticklabels([_pretty_coeff_label(c) for c in cols], fontsize=8)
                    ax_h.set_xlabel('time step')
                    if ax_h == axes_h[0]:
                        ax_h.set_ylabel('coefficients')
                cbar = fig_h.colorbar(im, ax=axes_h, shrink=0.6, location='right', pad=0.02)
                cbar.set_label('normalized value (per coeff)')
                heat_path = os.path.join(RESULTS_DIR, POLICY_HEATMAP_FIG)
                fig_h.savefig(heat_path, dpi=150)
                print(f"✓ Policy heatmaps: {heat_path}")
        else:
            print('[INFO] No policy coefficient files found for heatmap plotting.')

    # --- Policy lambda time series (lambda0_mw & lambda_plus) ---
    if PLOT_POLICY_LAMBDA_TIME_SERIES:
        lambda_cases: List[Tuple[str, str, float | None]] = []  # (label, path, epsilon)
        base_pol = os.path.join(RESULTS_DIR, 'policy_coeffs_drcc_false.csv')
        if os.path.exists(base_pol):
            lambda_cases.append((DETERMINISTIC_LABEL, base_pol, None))
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            pol_candidates = [
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_on.csv'),
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_off.csv'),
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_unk.csv'),
                os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}.csv'),
            ]
            chosen_pol = None
            for _p in pol_candidates:
                if os.path.exists(_p):
                    chosen_pol = _p
                    break
            if chosen_pol:
                lambda_cases.append((f"{eps:.2f}", chosen_pol, eps))
        # Need both lambda0_mw and lambda_plus to proceed
        if lambda_cases:
            try:
                import matplotlib.dates as mdates
                # Load all and align timestamps (assume same horizon)
                series_lambda0 = []  # (label, values)
                series_lambdap = []
                timestamps = None
                for lab, path_pol, eps_val in lambda_cases:
                    try:
                        pdf = pd.read_csv(path_pol)
                    except Exception:
                        continue
                    if 'lambda0_mw' not in pdf.columns or 'lambda_plus' not in pdf.columns:
                        continue
                    if timestamps is None and 'timestamp' in pdf.columns:
                        timestamps = pd.to_datetime(pdf['timestamp'])
                    series_lambda0.append((lab, pdf['lambda0_mw'].to_numpy(dtype=float)))
                    series_lambdap.append((lab, pdf['lambda_plus'].to_numpy(dtype=float)))
                if series_lambda0 and series_lambdap:
                    fig_l, axes_l = plt.subplots(2, 1, figsize=(10,5.8), sharex=True)
                    # Color mapping for epsilons; baseline black
                    # Collect eps labels (exclude baseline) for consistent color scale ordering
                    eps_labels = sorted([lab for lab, _, e in lambda_cases if e is not None], key=lambda x: float(x))
                    cmap = plt.cm.plasma
                    def color_for_label(lab: str):
                        if lab == DETERMINISTIC_LABEL:
                            return 'black'
                        # map label string (like '0.05') to index
                        if lab in eps_labels:
                            idx = eps_labels.index(lab)
                            return cmap((idx+1)/(len(eps_labels)+1))
                        return 'gray'
                    for lab, vals in series_lambda0:
                        axes_l[0].plot(vals, label=lab, color=color_for_label(lab), linewidth=1.2,
                                       linestyle='--' if lab==DETERMINISTIC_LABEL else '-')
                    axes_l[0].set_ylabel('lambda0_mw')
                    axes_l[0].grid(alpha=0.25)
                    for lab, vals in series_lambdap:
                        axes_l[1].plot(vals, label=lab, color=color_for_label(lab), linewidth=1.2,
                                       linestyle='--' if lab==DETERMINISTIC_LABEL else '-')
                    axes_l[1].set_ylabel('lambda_plus')
                    axes_l[1].grid(alpha=0.25)
                    axes_l[1].set_xlabel('Timestep index')
                    # Legend outside to avoid clutter
                    handles, labels = axes_l[0].get_legend_handles_labels()
                    # Convert epsilon labels to display form for legend
                    disp_labels = [_display_label_for_case(l) for l in labels]
                    fig_l.legend(handles, disp_labels, loc='upper center', ncol=min(6, len(disp_labels)), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
                    fig_l.tight_layout(rect=[0,0,1,0.97])
                    out_lam = os.path.join(RESULTS_DIR, POLICY_LAMBDA_TIME_SERIES_FIG)
                    fig_l.savefig(out_lam, dpi=150)
                    print(f"✓ Policy lambda time series: {out_lam}")
            except Exception as e:
                print(f"[WARN] Failed to build lambda time series plot: {e}")

        # --- Policy K-gain time series (K_pv_bess, K_hp_bess, K_pv_pvcurt, K_hp_pvcurt) ---
        if 'PLOT_POLICY_K_TIME_SERIES' in globals() and PLOT_POLICY_K_TIME_SERIES:
            k_cases: List[Tuple[str, str, float | None]] = []  # (label, path, epsilon)
            base_pol = os.path.join(RESULTS_DIR, 'policy_coeffs_drcc_false.csv')
            if os.path.exists(base_pol):
                k_cases.append((DETERMINISTIC_LABEL, base_pol, None))
            for eps in EPSILONS:
                tok = epsilon_token(eps)
                pol_candidates = [
                    os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_on.csv'),
                    os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_off.csv'),
                    os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}_rt_unk.csv'),
                    os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}.csv'),
                ]
                chosen_pol = None
                for _p in pol_candidates:
                    if os.path.exists(_p):
                        chosen_pol = _p
                        break
                if chosen_pol:
                    k_cases.append((f"{eps:.2f}", chosen_pol, eps))
            if k_cases:
                try:
                    series_map: Dict[str, List[Tuple[str, np.ndarray]]] = {  # coeff -> list of (label, values)
                        'K_pv_bess': [], 'K_hp_bess': [], 'K_pv_pvcurt': [], 'K_hp_pvcurt': []
                    }
                    for lab, path_pol, eps_val in k_cases:
                        try:
                            pdf = pd.read_csv(path_pol)
                        except Exception:
                            continue
                        # Only proceed if at least one K column exists
                        present = [k for k in series_map.keys() if k in pdf.columns]
                        if not present:
                            continue
                        for k in present:
                            series_map[k].append((lab, pd.to_numeric(pdf[k], errors='coerce').to_numpy(dtype=float)))
                    # Filter out coefficients with no data
                    series_map = {k: v for k, v in series_map.items() if v}
                    if series_map:
                        # Determine subplot rows: group BESS-target K on row 1, PV-curt on row 2
                        def _color_for(lab: str, ordered_labels: List[str]) -> str:
                            if lab == DETERMINISTIC_LABEL:
                                return 'black'
                            cmap = plt.cm.plasma
                            if lab in ordered_labels:
                                idx = ordered_labels.index(lab)
                                return cmap((idx+1)/(len(ordered_labels)+1))
                            return 'gray'
                        # Order epsilon labels for consistent colors
                        eps_labels = sorted([lab for lab, _, e in k_cases if e is not None], key=lambda x: float(x))
                        figk, axk = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True, constrained_layout=True)
                        # Row 1: K to BESS
                        for coeff in ['K_pv_bess','K_hp_bess']:
                            if coeff in series_map:
                                for lab, vals in series_map[coeff]:
                                    axk[0].plot(vals, label=lab, linewidth=1.2,
                                                color=_color_for(lab, eps_labels), linestyle='--' if lab==DETERMINISTIC_LABEL else '-')
                        axk[0].set_ylabel('K to BESS')
                        axk[0].grid(alpha=0.25)
                        # Row 2: K to PV curtailment
                        for coeff in ['K_pv_pvcurt','K_hp_pvcurt']:
                            if coeff in series_map:
                                for lab, vals in series_map[coeff]:
                                    axk[1].plot(vals, label=lab, linewidth=1.2,
                                                color=_color_for(lab, eps_labels), linestyle='--' if lab==DETERMINISTIC_LABEL else '-')
                        axk[1].set_ylabel('K to Curtailment')
                        axk[1].set_xlabel('Timestep index')
                        axk[1].grid(alpha=0.25)
                        # Legend once at top
                        handles, labels = axk[0].get_legend_handles_labels()
                        disp_labels = [_display_label_for_case(l) for l in labels]
                        if handles:
                            figk.legend(handles, disp_labels, loc='upper center', ncol=min(6, len(disp_labels)), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
                        out_k = os.path.join(RESULTS_DIR, POLICY_K_TIME_SERIES_FIG)
                        figk.savefig(out_k, dpi=150)
                        print(f"✓ Policy K-gain time series: {out_k}")
                except Exception as e:
                    print(f"[WARN] Failed to build K time series plot: {e}")

    # --- SoC envelope plotting (raw trajectory based if available) ---
    if PLOT_SOC_ENVELOPES:
        def _load_raw_soc(meta: Dict) -> pd.DataFrame | None:
            """Load raw SoC time series from meta if available.

            Returns DataFrame with columns ['timestamp','sample_id','soc_frac'] or None.
            Supports parquet or CSV; expects file referenced by 'soc_series_file'.
            """
            if not isinstance(meta, dict):
                return None
            fname = meta.get('soc_series_file') or meta.get('soc_envelope_file')  # fallback legacy
            if not fname:
                return None
            abs_path = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(abs_path):
                # try direct if meta stored absolute
                if os.path.exists(fname):
                    abs_path = fname
                else:
                    return None
            try:
                if abs_path.lower().endswith('.parquet'):
                    df_raw = _read_parquet_or_csv(abs_path)
                else:
                    df_raw = pd.read_csv(abs_path)
            except Exception:
                return None
            if df_raw is None or df_raw.empty:
                return None
            # Detect envelope file shape and skip (want raw series). Raw has soc_frac column.
            if {'soc_p05','soc_p50','soc_p95'} <= set(df_raw.columns):
                return None  # it's an envelope, not raw series
            needed = {'timestamp','sample_id','soc_frac'}
            if not needed <= set(df_raw.columns):
                return None
            # Ensure timestamp dtype
            try:
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            except Exception:
                pass
            return df_raw

        # Build case list from meta_map (constructed earlier): label -> meta
        soc_cases: List[Tuple[str, pd.DataFrame]] = []  # (display_label, raw_df or envelope-derived df)
        for lab, meta in meta_map.items():
            raw_df = _load_raw_soc(meta)
            if raw_df is not None:
                soc_cases.append((lab, raw_df))
        # Fallback to directory scan for raw series if meta-based load failed
        if not soc_cases:
            try:
                cand_paths: List[str] = []
                cand_paths += glob.glob(os.path.join(RESULTS_DIR, 'soc_series_drcc_true_epsilon_*.parquet'))
                cand_paths += glob.glob(os.path.join(RESULTS_DIR, 'soc_series_drcc_true_epsilon_*.csv'))
                # Include deterministic baseline variants if present
                cand_paths += glob.glob(os.path.join(RESULTS_DIR, 'soc_series_drcc_false*.parquet'))
                cand_paths += glob.glob(os.path.join(RESULTS_DIR, 'soc_series_drcc_false*.csv'))
                for p in sorted(cand_paths):
                    name = os.path.basename(p)
                    # Determine label
                    lab: str
                    if name.startswith('soc_series_drcc_true_epsilon_'):
                        # extract epsilon token between prefix and possible RT tag
                        core = name[len('soc_series_drcc_true_epsilon_'):]  # e.g., 0_10_rt_on.parquet
                        token = core.split('_rt_')[0].split('.')[0]
                        try:
                            eps_val = float(token.replace('_', '.'))
                        except Exception:
                            continue
                        rt_tag = _extract_rt_tag_from_name(name) or ''
                        disp = f"{eps_val:.2f} ({_rt_display(rt_tag)})" if rt_tag else f"{eps_val:.2f}"
                        lab = disp
                    else:
                        # deterministic baseline
                        rt_tag = _extract_rt_tag_from_name(name) or ''
                        lab = f"{DETERMINISTIC_LABEL} ({_rt_display(rt_tag)})" if rt_tag else DETERMINISTIC_LABEL
                    df_raw = _read_parquet_or_csv(p)
                    if df_raw is None or df_raw.empty:
                        continue
                    cols = set(df_raw.columns)
                    if {'soc_p05','soc_p50','soc_p95'} <= cols:
                        # envelope, skip
                        continue
                    if not {'timestamp','sample_id','soc_frac'} <= cols:
                        continue
                    try:
                        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                    except Exception:
                        pass
                    soc_cases.append((lab, df_raw))
            except Exception:
                pass
        # Fallback to envelope CSVs only if still no raw cases found
        if not soc_cases:
            # Reuse previous envelope search logic minimally (legacy fallback)
            legacy_env_paths: List[Tuple[str,str]] = []
            # Deterministic variants
            for tag in ('rt_on','rt_off','rt_unk'):
                p_env = os.path.join(RESULTS_DIR, f'soc_envelope_drcc_false_{tag}.csv')
                if os.path.exists(p_env):
                    legacy_env_paths.append((f"{DETERMINISTIC_LABEL} ({_rt_display(tag)})", p_env))
            if not legacy_env_paths:
                p_env_det = os.path.join(RESULTS_DIR, 'soc_envelope_drcc_false.csv')
                if os.path.exists(p_env_det):
                    legacy_env_paths.append((DETERMINISTIC_LABEL, p_env_det))
            for eps in EPSILONS:
                tok = epsilon_token(eps)
                added = False
                for tag in ('rt_on','rt_off','rt_unk'):
                    p_env = os.path.join(RESULTS_DIR, f'soc_envelope_drcc_true_epsilon_{tok}_{tag}.csv')
                    if os.path.exists(p_env):
                        legacy_env_paths.append((f"{eps:.2f} ({_rt_display(tag)})", p_env))
                        added = True
                if not added:
                    p_env_uns = os.path.join(RESULTS_DIR, f'soc_envelope_drcc_true_epsilon_{tok}.csv')
                    if os.path.exists(p_env_uns):
                        legacy_env_paths.append((f"{eps:.2f}", p_env_uns))
            if legacy_env_paths:
                cols = len(legacy_env_paths)
                fig_soc, ax_soc = plt.subplots(1, cols, figsize=(4*cols, 3), constrained_layout=True)
                if cols == 1:
                    ax_soc = [ax_soc]
                final_rows_env = []
                for ax, (lab, pth) in zip(ax_soc, legacy_env_paths):
                    try:
                        env_df = pd.read_csv(pth)
                    except Exception:
                        continue
                    if not {'soc_p05','soc_p50','soc_p95'} <= set(env_df.columns):
                        continue
                    p05 = pd.to_numeric(env_df['soc_p05'], errors='coerce')
                    p50 = pd.to_numeric(env_df['soc_p50'], errors='coerce')
                    p95 = pd.to_numeric(env_df['soc_p95'], errors='coerce')
                    t = np.arange(len(env_df))
                    bw = float(np.nanmax(p95 - p05)) if len(env_df) else 0.0
                    if not np.isfinite(bw):
                        bw = 0.0
                    if bw <= 1e-6:
                        ax.plot(t, p50, color='#08519c', linewidth=1.5, label='Median (degenerate)')
                    else:
                        ax.fill_between(t, p05, p95, color='#c6dbef', alpha=0.6, label='5–95% band')
                        ax.plot(t, p50, color='#08519c', linewidth=1.5, label='Median')
                    ax.set_ylim(0, 1.02)
                    ax.set_title(f"SoC envelope ({_display_label_for_case(lab)})")
                    ax.set_xlabel('t step')
                    if ax == ax_soc[0]:
                        ax.set_ylabel('SoC fraction')
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8)
                    last = env_df.iloc[-1]
                    final_rows_env.append((_display_label_for_case(lab), float(last['soc_p05']), float(last['soc_p50']), float(last['soc_p95'])))
                soc_fig_path = os.path.join(RESULTS_DIR, SOC_ENV_FIG)
                fig_soc.savefig(soc_fig_path, dpi=150)
                print(f"✓ SoC envelopes figure (legacy envelope fallback): {soc_fig_path}")
                if final_rows_env:
                    labels = [r[0] for r in final_rows_env]
                    medians = np.array([r[2] for r in final_rows_env])
                    low_err = medians - np.array([r[1] for r in final_rows_env])
                    high_err = np.array([r[3] for r in final_rows_env]) - medians
                    low_err = np.clip(low_err, 0, None); high_err = np.clip(high_err, 0, None)
                    fig_final, ax_final = plt.subplots(figsize=(max(6, 0.6*len(final_rows_env)+2), 4))
                    x = np.arange(len(final_rows_env))
                    ax_final.bar(x, medians, color='#3182bd', alpha=0.85, label='Median final SoC')
                    ax_final.errorbar(x, medians, yerr=[low_err, high_err], fmt='none', ecolor='#08306b', elinewidth=1.2, capsize=4, label='5–95% range')
                    ax_final.set_xticks(x)
                    ax_final.set_xticklabels(labels, rotation=45, ha='right')
                    ax_final.set_ylabel('Final timestep SoC fraction')
                    ax_final.set_ylim(0, 1.05)
                    ax_final.grid(axis='y', alpha=0.3)
                    ax_final.legend(fontsize=8)
                    final_fig_path = os.path.join(RESULTS_DIR, SOC_FINAL_FIG)
                    fig_final.tight_layout(); fig_final.savefig(final_fig_path, dpi=150)
                    print(f"✓ Final timestep SoC summary figure (legacy): {final_fig_path}")
                    print("(Raw series missing; consider re-running v4 with EXPORT_SOC_SERIES=True)")
            else:
                print('[INFO] No SoC envelope or raw series files found.')
        else:
            # Raw series path: compute envelopes from actual trajectories
            # Group cases by display label retaining RT tags & epsilon identification
            case_envelopes: List[Tuple[str, pd.DataFrame]] = []  # (label, env_df with p05/p50/p95)
            for lab, raw_df in soc_cases:
                # Ensure ordering by timestamp
                raw_df = raw_df.sort_values(['timestamp','sample_id'])
                # Group by timestamp and compute quantiles from raw soc_frac
                g = raw_df.groupby('timestamp')['soc_frac']
                q05 = g.quantile(0.05)
                q50 = g.quantile(0.50)
                q95 = g.quantile(0.95)
                env_df = pd.DataFrame({
                    'timestamp': q05.index,
                    'soc_p05': q05.to_numpy(),
                    'soc_p50': q50.to_numpy(),
                    'soc_p95': q95.to_numpy(),
                })
                case_envelopes.append((lab, env_df))
            # Plot envelopes
            cols = len(case_envelopes)
            fig_soc, ax_soc = plt.subplots(1, cols, figsize=(4*cols, 3), constrained_layout=True)
            if cols == 1:
                ax_soc = [ax_soc]
            final_rows = []
            for ax, (lab, env_df) in zip(ax_soc, case_envelopes):
                p05 = pd.to_numeric(env_df['soc_p05'], errors='coerce')
                p50 = pd.to_numeric(env_df['soc_p50'], errors='coerce')
                p95 = pd.to_numeric(env_df['soc_p95'], errors='coerce')
                t = np.arange(len(env_df))
                bw = float(np.nanmax(p95 - p05)) if len(env_df) else 0.0
                if not np.isfinite(bw):
                    bw = 0.0
                if bw <= 1e-6:
                    ax.plot(t, p50, color='#08519c', linewidth=1.5, label='Median (degenerate)')
                else:
                    ax.fill_between(t, p05, p95, color='#c6dbef', alpha=0.6, label='5–95% band')
                    ax.plot(t, p50, color='#08519c', linewidth=1.5, label='Median')
                ax.set_ylim(0, 1.02)
                ax.set_title(f"SoC envelope ({_display_label_for_case(lab)})")
                ax.set_xlabel('t step')
                if ax == ax_soc[0]:
                    ax.set_ylabel('SoC fraction')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                last = env_df.iloc[-1]
                final_rows.append((_display_label_for_case(lab), float(last['soc_p05']), float(last['soc_p50']), float(last['soc_p95'])))
            soc_fig_path = os.path.join(RESULTS_DIR, SOC_ENV_FIG)
            fig_soc.savefig(soc_fig_path, dpi=150)
            print(f"✓ SoC envelopes figure (raw series): {soc_fig_path}")
            if final_rows:
                labels = [r[0] for r in final_rows]
                medians = np.array([r[2] for r in final_rows])
                low_err = medians - np.array([r[1] for r in final_rows])
                high_err = np.array([r[3] for r in final_rows]) - medians
                low_err = np.clip(low_err, 0, None); high_err = np.clip(high_err, 0, None)
                fig_final, ax_final = plt.subplots(figsize=(max(6, 0.6*len(final_rows)+2), 4))
                x = np.arange(len(final_rows))
                ax_final.bar(x, medians, color='#3182bd', alpha=0.85, label='Median final SoC')
                ax_final.errorbar(x, medians, yerr=[low_err, high_err], fmt='none', ecolor='#08306b', elinewidth=1.2, capsize=4, label='5–95% range')
                ax_final.set_xticks(x)
                ax_final.set_xticklabels(labels, rotation=45, ha='right')
                ax_final.set_ylabel('Final timestep SoC fraction')
                ax_final.set_ylim(0, 1.05)
                ax_final.grid(axis='y', alpha=0.3)
                ax_final.legend(fontsize=8)
                final_fig_path = os.path.join(RESULTS_DIR, SOC_FINAL_FIG)
                fig_final.tight_layout(); fig_final.savefig(final_fig_path, dpi=150)
                print(f"✓ Final timestep SoC summary figure (raw series): {final_fig_path}")

            # Also produce a boxplot of final-step SoC across cases using raw trajectories
            try:
                final_values: List[np.ndarray] = []
                box_labels: List[str] = []
                for lab, raw_df in soc_cases:
                    if raw_df is None or raw_df.empty:
                        continue
                    # final timestamp present in this case
                    try:
                        t_final = raw_df['timestamp'].max()
                        vals = pd.to_numeric(raw_df.loc[raw_df['timestamp'] == t_final, 'soc_frac'], errors='coerce').dropna().to_numpy()
                    except Exception:
                        vals = np.array([])
                    if vals.size == 0:
                        continue
                    final_values.append(vals)
                    box_labels.append(_display_label_for_case(lab))
                if final_values:
                    fig_box, ax_box = plt.subplots(figsize=(max(6, 0.8*len(final_values)+2), 4))
                    ax_box.boxplot(final_values, labels=box_labels, showfliers=False, patch_artist=True,
                                   boxprops=dict(facecolor='#c6dbef', color='#08519c'),
                                   medianprops=dict(color='#08306b', linewidth=1.5),
                                   whiskerprops=dict(color='#08519c'), capprops=dict(color='#08519c'))
                    ax_box.set_ylabel('Final timestep SoC fraction')
                    ax_box.set_ylim(0, 1.05)
                    ax_box.grid(axis='y', alpha=0.3)
                    plt.setp(ax_box.get_xticklabels(), rotation=45, ha='right')
                    box_fig_path = os.path.join(RESULTS_DIR, SOC_FINAL_BOXPLOT_FIG)
                    fig_box.tight_layout(); fig_box.savefig(box_fig_path, dpi=150)
                    print(f"✓ Final timestep SoC boxplot (raw series): {box_fig_path}")
            except Exception as e:
                print(f"[WARN] Could not generate final-step SoC boxplot: {e}")

            # Produce a boxplot of SoC over the whole day (all timesteps) across cases
            try:
                daily_values: List[np.ndarray] = []
                daily_labels: List[str] = []
                for lab, raw_df in soc_cases:
                    if raw_df is None or raw_df.empty:
                        continue
                    vals = pd.to_numeric(raw_df.get('soc_frac', pd.Series(dtype=float)), errors='coerce').dropna().to_numpy()
                    if vals.size == 0:
                        continue
                    daily_values.append(vals)
                    daily_labels.append(_display_label_for_case(lab))
                if daily_values:
                    fig_day, ax_day = plt.subplots(figsize=(max(6, 0.8*len(daily_values)+2), 4))
                    ax_day.boxplot(daily_values, labels=daily_labels, showfliers=False, patch_artist=True,
                                   boxprops=dict(facecolor='#c6dbef', color='#08519c'),
                                   medianprops=dict(color='#08306b', linewidth=1.5),
                                   whiskerprops=dict(color='#08519c'), capprops=dict(color='#08519c'))
                    ax_day.set_ylabel('SoC fraction (all timesteps)')
                    ax_day.set_ylim(0, 1.05)
                    ax_day.grid(axis='y', alpha=0.3)
                    plt.setp(ax_day.get_xticklabels(), rotation=45, ha='right')
                    daily_fig_path = os.path.join(RESULTS_DIR, SOC_DAILY_BOXPLOT_FIG)
                    fig_day.tight_layout(); fig_day.savefig(daily_fig_path, dpi=150)
                    print(f"✓ Whole-day SoC boxplot (raw series): {daily_fig_path}")
            except Exception as e:
                print(f"[WARN] Could not generate whole-day SoC boxplot: {e}")

    if SHOW:
        plt.show()

    # --- Evening transformer sigma decomposition (from v2 CSV) ---
    if PLOT_EVENING_TRAFO_SIGMA_DECOMP:
        try:
            def _resolve_v2_path(v2_csv: str | None) -> str | None:
                if not v2_csv:
                    return None
                if os.path.exists(v2_csv):
                    return v2_csv
                # try resolve relative to RESULTS_DIR
                cand = os.path.join(RESULTS_DIR, v2_csv)
                if os.path.exists(cand):
                    return cand
                # try workspace root
                if os.path.exists(os.path.basename(v2_csv)):
                    return os.path.basename(v2_csv)
                return None

            # Prefer epsilon=0.10 case; else pick the smallest epsilon available
            chosen_eps = None
            chosen_meta = None
            for e in ([0.10] + sorted(EPSILONS)):
                try:
                    m = load_meta_for_epsilon(e)
                except Exception:
                    m = {}
                v2p = _resolve_v2_path(m.get('v2_results_csv') if isinstance(m, dict) else None)
                if v2p:
                    chosen_eps = e
                    chosen_meta = m
                    break
            if chosen_meta is None:
                print('[INFO] Skipped evening sigma decomposition (no v2_results_csv found in meta).')
            else:
                v2_path = _resolve_v2_path(chosen_meta.get('v2_results_csv'))
                if not v2_path or not os.path.exists(v2_path):
                    print(f"[INFO] Skipped evening sigma decomposition (v2 CSV not found: {v2_path}).")
                else:
                    df2 = pd.read_csv(v2_path)
                    # Required columns (soft-check: proceed with what we have)
                    cols_sigma = [
                        'sigma_tr0_comp_pvP_mva',
                        'sigma_tr0_comp_hpP_mva',
                        'sigma_tr0_comp_hpQ_mva',
                        'sigma_tr0_calc_mva',
                    ]
                    cols_k = ['K_pv_bess','K_hp_bess','K_pv_pvcurt','K_hp_pvcurt']
                    col_slack_min = 'bess_power_robust_slack_min_mw'
                    # Evening mask: pv availability near zero
                    if 'pv_avail_sum_mw' in df2.columns:
                        pv_sum = pd.to_numeric(df2['pv_avail_sum_mw'], errors='coerce').fillna(0.0)
                        eve_mask = pv_sum <= max(1e-3, 0.01)  # <= 0.01 MW considered zero
                    else:
                        # fallback: evening hours 17–21 if timestamp available, else middle third of horizon
                        eve_mask = None
                    if eve_mask is None and 'timestamp' in df2.columns:
                        try:
                            ts = pd.to_datetime(df2['timestamp'])
                            hrs = ts.dt.hour.to_numpy()
                            eve_mask = (hrs >= 17) & (hrs <= 21)
                        except Exception:
                            eve_mask = None
                    if eve_mask is None:
                        n = len(df2)
                        s = n//3
                        eve_mask = np.zeros(n, dtype=bool)
                        eve_mask[s:2*s] = True
                    # Slice
                    df_eve = df2.loc[eve_mask].reset_index(drop=True)
                    if df_eve.empty:
                        print('[INFO] Evening mask matched no rows; plotting full horizon instead.')
                        df_eve = df2.reset_index(drop=True)
                    # X axis
                    if 'timestamp' in df_eve.columns:
                        try:
                            x = pd.to_datetime(df_eve['timestamp'])
                            x_labels = x.dt.strftime('%H:%M')
                        except Exception:
                            x_labels = pd.Series(range(len(df_eve)))
                    else:
                        x_labels = pd.Series(range(len(df_eve)))
                    # Build figure
                    fig_ev, axes_ev = plt.subplots(2, 1, figsize=(12, 7.0), constrained_layout=True, sharex=True)
                    # Panel 1: sigma components (MVA)
                    present_sigma = [c for c in cols_sigma if c in df_eve.columns]
                    if present_sigma:
                        for c in present_sigma:
                            axes_ev[0].plot(df_eve.index, pd.to_numeric(df_eve[c], errors='coerce'), label=c.replace('sigma_tr0_comp_','').replace('_mva','').replace('sigma_tr0_calc_','calc_'))
                        axes_ev[0].set_ylabel('Transformer σ (MVA)')
                        axes_ev[0].set_title(f"Trafo 0 sigma decomposition (ε={chosen_eps:.2f})")
                        axes_ev[0].grid(alpha=0.3)
                        axes_ev[0].legend(fontsize=8, ncol=2, frameon=False)
                    else:
                        axes_ev[0].text(0.5, 0.5, 'No sigma decomposition columns', ha='center', va='center', transform=axes_ev[0].transAxes, color='gray')
                        axes_ev[0].set_title(f"Trafo 0 sigma decomposition (ε={chosen_eps:.2f})")
                        axes_ev[0].grid(alpha=0.3)
                    # Panel 2: K gains (dimensionless) and BESS robust slack (MW)
                    present_k = [c for c in cols_k if c in df_eve.columns]
                    axk = axes_ev[1]
                    axk2 = axk.twinx()
                    if present_k:
                        for c in present_k:
                            axk.plot(df_eve.index, pd.to_numeric(df_eve[c], errors='coerce'), linewidth=1.3, label=c)
                    if col_slack_min in df_eve.columns:
                        axk2.plot(df_eve.index, pd.to_numeric(df_eve[col_slack_min], errors='coerce'), color='#d62728', linewidth=1.6, alpha=0.85, label='bess_power_robust_slack_min_mw')
                    axk.set_ylabel('K gains (dimensionless)')
                    axk2.set_ylabel('BESS robust slack (MW)')
                    axk.grid(alpha=0.3)
                    # Legends
                    h1, l1 = axk.get_legend_handles_labels()
                    h2, l2 = axk2.get_legend_handles_labels()
                    if h1 or h2:
                        axes_ev[1].legend(h1+h2, l1+l2, loc='upper left', fontsize=8, frameon=False)
                    # X labels
                    axes_ev[1].set_xticks(range(len(df_eve)))
                    try:
                        axes_ev[1].set_xticklabels(list(x_labels))
                    except Exception:
                        pass
                    axes_ev[1].set_xlabel('time (evening window)')
                    # Save
                    out_ev = os.path.join(RESULTS_DIR, EVENING_TRAFO_SIGMA_DECOMP_FIG)
                    fig_ev.savefig(out_ev, dpi=150)
                    print(f"✓ Evening sigma decomposition figure: {out_ev}")
                    # Also export CSV with the evening slice
                    try:
                        export_cols = []
                        # Keep timestamp if present
                        if 'timestamp' in df_eve.columns:
                            export_cols.append('timestamp')
                        # Context columns if present
                        for c in ['pv_avail_sum_mw']:
                            if c in df_eve.columns:
                                export_cols.append(c)
                        # Sigma components and calc (subset of available)
                        for c in cols_sigma:
                            if c in df_eve.columns:
                                export_cols.append(c)
                        # K gains and BESS slack
                        for c in cols_k:
                            if c in df_eve.columns:
                                export_cols.append(c)
                        if col_slack_min in df_eve.columns:
                            export_cols.append(col_slack_min)
                        # Build export DF
                        exp = df_eve.loc[:, [c for c in export_cols if c in df_eve.columns]].copy()
                        exp.insert(0, 't_idx_evening', list(range(len(exp))))
                        exp.insert(1, 'epsilon', chosen_eps)
                        out_csv = os.path.join(RESULTS_DIR, EVENING_TRAFO_SIGMA_DECOMP_CSV)
                        exp.to_csv(out_csv, index=False)
                        print(f"✓ Evening sigma decomposition CSV: {out_csv}")
                    except Exception as _e:
                        print(f"[WARN] Failed to write evening sigma CSV: {_e}")
        except Exception as e:
            print(f"[WARN] Failed to build evening sigma decomposition plot: {e}")


if __name__ == "__main__":
    # If dual-aggregation requested, run it and exit; else run standard single-distribution analysis.
    if AGGREGATE_DISTS is not None:
        _d1, _d2 = AGGREGATE_DISTS
        _w = AGGREGATE_WEIGHTS if AGGREGATE_WEIGHTS is not None else (0.5, 0.5)
        _run_dual_aggregation(_d1, _d2, _w)
    else:
        main()
    # --- Tail-focused overview (additional figure) ---
    if (AGGREGATE_DISTS is None) and PLOT_TAIL_OVERVIEW:
        try:
            # Helper to load per-(sample,t) max loading arrays for a case via meta parquet
            def _load_case_profile(label: str, eps: float | None) -> tuple[np.ndarray, np.ndarray] | None:
                # Return (t_axis, p_viol_t) where p_viol_t is violation probability per timestep
                if label == DETERMINISTIC_LABEL:
                    meta_path = os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json')
                    if not os.path.exists(meta_path):
                        return None
                    with open(meta_path,'r',encoding='utf-8') as f:
                        meta_b = json.load(f)
                    rel = meta_b.get('trafo_loading_file')
                    if not rel:
                        return None
                    pq = os.path.join(RESULTS_DIR, rel)
                else:
                    meta_e = load_meta_for_epsilon(float(label)) if eps is None else load_meta_for_epsilon(eps)
                    if not meta_e or 'trafo_loading_file' not in meta_e:
                        return None
                    pq = os.path.join(RESULTS_DIR, meta_e['trafo_loading_file'])
                # Load transformer loading table with CSV fallback
                pdf = _read_parquet_or_csv(pq)
                if pdf is None:
                    return None
                must = {'sample_id','t','trafo_index','loading_pct'}
                if not must <= set(pdf.columns):
                    return None
                grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
                counts = grp.groupby('t')['sample_id'].nunique()
                viol = grp[grp['loading_pct'] > OVERLOAD_THRESHOLD_PCT].groupby('t')['sample_id'].nunique()
                rate_series = (viol / counts).reindex(counts.index).fillna(0.0)
                return counts.index.to_numpy(), rate_series.to_numpy()

            # Load deterministic profile
            det = _load_case_profile(DETERMINISTIC_LABEL, None)
            # Load DRCC profiles for EPSILONS
            drcc_profiles = []  # list of (label, (t_axis, rate))
            for e in EPSILONS:
                lab = f"{e:.2f}"
                prof = _load_case_profile(lab, e)
                if prof:
                    drcc_profiles.append((lab, prof))

            # Need at least deterministic and one DRCC
            if det and drcc_profiles:
                t_det, r_det = det
                # Align length by truncation for simplicity
                min_len_all = min([len(t_det)] + [len(p[1][0]) for p in drcc_profiles])
                t_axis = t_det[:min_len_all]
                r_det = r_det[:min_len_all]
                drcc_profiles = [(lab, (p[0][:min_len_all], p[1][:min_len_all])) for lab, (p) in drcc_profiles]

                # 1) Evening window (17:00–21:00) zoom
                # If we don't have real timestamps, use index ranges: take middle third as proxy if needed.
                idx_evening = slice(None)
                try:
                    # If original CSV exists, we can infer timestamps from any meta v2 file
                    any_meta = load_meta_for_epsilon(EPSILONS[0]) if EPSILONS else {}
                    v2_csv = any_meta.get('v2_results_csv') if isinstance(any_meta, dict) else None
                    if v2_csv and os.path.exists(v2_csv):
                        v2_df = pd.read_csv(v2_csv)
                        if 'timestamp' in v2_df.columns and len(v2_df['timestamp']) >= len(t_axis):
                            ts = pd.to_datetime(v2_df['timestamp']).to_series().reset_index(drop=True)
                            hours = ts.dt.hour.to_numpy()
                            mask = (hours >= 17) & (hours <= 21)
                            if mask.sum() > 0:
                                idx_evening = np.where(mask[:len(t_axis)])[0]
                except Exception:
                    pass

                # 2) Top-k risky timesteps by deterministic
                k = min(12, len(t_axis))
                order = np.argsort(-r_det)  # descending
                topk_idx = order[:k]

                # 3) Distributions: CVaR tails and severity
                def _load_flat_loadings(meta_dict: Dict) -> np.ndarray:
                    rel = meta_dict.get('trafo_loading_file') if isinstance(meta_dict, dict) else None
                    if not rel:
                        return np.array([])
                    pq = os.path.join(RESULTS_DIR, rel)
                    # Load parquet if available, else CSV
                    if not os.path.exists(pq):
                        pq_csv = pq.replace('.parquet', '.csv') if pq.endswith('.parquet') else pq + '.csv'
                        if not os.path.exists(pq_csv):
                            return np.array([])
                        try:
                            pdf = pd.read_csv(pq_csv)
                        except Exception:
                            return np.array([])
                    else:
                        try:
                            pdf = pd.read_parquet(pq)
                        except Exception:
                            pq_csv = pq.replace('.parquet', '.csv') if pq.endswith('.parquet') else pq + '.csv'
                            if os.path.exists(pq_csv):
                                try:
                                    pdf = pd.read_csv(pq_csv)
                                except Exception:
                                    return np.array([])
                            else:
                                return np.array([])
                    must = {'sample_id','t','trafo_index','loading_pct'}
                    if not must <= set(pdf.columns):
                        return np.array([])
                    grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
                    arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
                    return arr[np.isfinite(arr)]

                det_meta = {}
                try:
                    with open(os.path.join(RESULTS_DIR, 'v4_meta_drcc_false.json'),'r',encoding='utf-8') as f:
                        det_meta = json.load(f)
                except Exception:
                    pass
                drcc_meta_map = {f"{e:.2f}": load_meta_for_epsilon(e) for e in EPSILONS}
                arr_det = _load_flat_loadings(det_meta)
                arr_drcc = {lab: _load_flat_loadings(m) for lab, m in drcc_meta_map.items()}

                # Build figure (3x3 grid)
                fig_tail, axes_tail = plt.subplots(3, 3, figsize=(18, 10), constrained_layout=True)

                # A1: Evening-window violation probability
                ax = axes_tail[0,0]
                if isinstance(idx_evening, np.ndarray) and idx_evening.size > 0:
                    x = idx_evening
                else:
                    # fallback: use full axis
                    x = np.arange(len(t_axis))
                ax.plot(x, (r_det[x] if isinstance(x, np.ndarray) else r_det), color='black', linestyle='--', linewidth=1.6, label='deterministic')
                for lab, (t_l, r_l) in drcc_profiles:
                    ax.plot(x, (r_l[x] if isinstance(x, np.ndarray) else r_l), linewidth=1.2, alpha=0.9, label=f"DRCC, ε={lab}")
                ax.set_title('Evening-window violation probability')
                ax.set_xlabel('timestep index')
                ax.set_ylabel(f"P(any trafo > {OVERLOAD_THRESHOLD_PCT:.2f}%)")
                ax.set_ylim(0, 1.0)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, frameon=False)

                # A2: Top-k risky timesteps (deterministic ranking)
                ax = axes_tail[0,1]
                ax.plot(topk_idx, r_det[topk_idx], marker='o', color='black', linestyle='--', label='deterministic')
                for lab, (_, r_l) in drcc_profiles:
                    ax.plot(topk_idx, r_l[topk_idx], marker='o', linestyle='-', alpha=0.9, label=f"DRCC, ε={lab}")
                ax.set_title(f'Top-{k} risky timesteps (by deterministic)')
                ax.set_xlabel('timestep index (ranked)')
                ax.set_ylabel('violation probability')
                ax.set_ylim(0, 1.0)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, frameon=False)

                # A3: Δ profile (det − DRCC_0.10) over time
                ax = axes_tail[0,2]
                # pick epsilon 0.10 if present, else the smallest epsilon
                chosen = next((r for lab, r in drcc_profiles if lab == f"{0.10:.2f}"), None)
                if chosen is None and drcc_profiles:
                    chosen = drcc_profiles[-1][1]
                if chosen is not None:
                    _, r_e = chosen
                    delta = r_det - r_e
                    ax.plot(np.arange(len(t_axis)), delta, color='#1f77b4', linewidth=1.2)
                    ax.axhline(0.0, color='black', linewidth=0.8)
                    ax.set_title('Δ violation probability (det − DRCC)')
                    ax.set_xlabel('timestep index')
                    ax.set_ylabel('Δ probability')
                    ax.grid(alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No DRCC data', ha='center', va='center', transform=ax.transAxes, color='gray')

                # B1: Loading CVaR95 (per-(sample,t) maxima)
                ax = axes_tail[1,0]
                def _cvar(vals: np.ndarray, alpha: float) -> float:
                    if vals is None or vals.size == 0:
                        return float('nan')
                    var = np.nanpercentile(vals, alpha * 100.0)
                    tail = vals[vals >= var]
                    return float(np.nanmean(tail)) if tail.size else float('nan')
                labels_tail = [DETERMINISTIC_LABEL] + [lab for lab, _ in drcc_profiles]
                c95_vals = []
                def _vals_for_label(lab: str) -> np.ndarray:
                    if lab == DETERMINISTIC_LABEL:
                        return arr_det
                    return arr_drcc.get(lab, np.array([]))
                for lab in labels_tail:
                    v = _vals_for_label(lab)
                    c95_vals.append(_cvar(v, 0.95))
                x = np.arange(len(labels_tail))
                ax.bar(x, c95_vals, color='#6a3d9a', alpha=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels([('deterministic' if lab==DETERMINISTIC_LABEL else f"DRCC, ε={lab}") for lab in labels_tail], rotation=20)
                ax.set_ylabel('CVaR95(loading %)')
                ax.set_title('Loading CVaR95 (per-(sample,t) maxima)')
                ax.grid(axis='y', alpha=0.3)

                # B2: Exceedance severity distribution (top-5% exceedances)
                ax = axes_tail[1,1]
                def _top5_excess(vals: np.ndarray) -> np.ndarray:
                    if vals is None or vals.size == 0:
                        return np.array([])
                    excess = vals - OVERLOAD_THRESHOLD_PCT
                    excess = excess[excess > 0]
                    if excess.size == 0:
                        return np.array([])
                    thr = np.nanpercentile(excess, 95)
                    tail = excess[excess >= thr]
                    return tail
                datasets = [(DETERMINISTIC_LABEL, _top5_excess(arr_det))] + [(lab, _top5_excess(arr_drcc.get(lab, np.array([])))) for lab, _ in drcc_profiles]
                for lab, arr in datasets:
                    if arr.size:
                        ax.hist(arr, bins=15, alpha=0.4, label=('deterministic' if lab==DETERMINISTIC_LABEL else f"DRCC, ε={lab}"), density=True)
                ax.set_xlabel('Exceedance over threshold (pp)')
                ax.set_ylabel('Density (top-5% exceedances)')
                ax.set_title('Severity tail (top-5% exceedances)')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, frameon=False)

                # B3: Per-trajectory violation rate distribution
                ax = axes_tail[1,2]
                def _traj_violation_rates(summary_path: str) -> np.ndarray:
                    if not os.path.exists(summary_path):
                        return np.array([])
                    try:
                        df_s = pd.read_csv(summary_path)
                    except Exception:
                        return np.array([])
                    if {'steps_trafo_over_80pct','n_steps'} - set(df_s.columns):
                        return np.array([])
                    ns = pd.to_numeric(df_s['n_steps'], errors='coerce')
                    st = pd.to_numeric(df_s['steps_trafo_over_80pct'], errors='coerce')
                    mask = (ns > 0) & np.isfinite(st)
                    return (st[mask] / ns[mask]).to_numpy()
                # deterministic summary
                det_sum = os.path.join(RESULTS_DIR, 'v4_summary_drcc_false.csv')
                arr_det_vr = _traj_violation_rates(det_sum)
                # drcc summaries
                arr_drcc_vr = []
                for e in EPSILONS:
                    tok = epsilon_token(e)
                    sp = os.path.join(RESULTS_DIR, f'v4_summary_drcc_true_epsilon_{tok}.csv')
                    arr = _traj_violation_rates(sp)
                    if arr.size:
                        arr_drcc_vr.append((f"{e:.2f}", arr))
                # Build violin/box
                data = [arr_det_vr] + [arr for _, arr in arr_drcc_vr]
                labels = ['deterministic'] + [f"DRCC, ε={lab}" for lab, _ in arr_drcc_vr]
                if any(len(d)>0 for d in data):
                    parts = ax.violinplot([d for d in data if d.size], positions=np.arange(1, 1+len([d for d in data if d.size])), showmeans=False, showmedians=True, showextrema=False)
                    # Tweak x labels
                    ax.set_xticks(np.arange(1, 1+len(labels)))
                    ax.set_xticklabels(labels, rotation=20)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.set_ylabel('Per-trajectory violation rate')
                ax.set_title('Trajectory violation rate distribution')
                ax.grid(alpha=0.3)

                # C1: Horizon reliability P(any exceedance in trajectory)
                ax = axes_tail[2,0]
                def _horizon_reliability_from_pdf(pdf: pd.DataFrame) -> float:
                    try:
                        mx = pdf.groupby('sample_id')['loading_pct'].max()
                        return float(((mx > OVERLOAD_THRESHOLD_PCT).sum()) / len(mx)) if len(mx) else float('nan')
                    except Exception:
                        return float('nan')
                # Load full parquets
                def _load_pdf_from_meta(m: Dict) -> pd.DataFrame | None:
                    if not isinstance(m, dict):
                        return None
                    rel = m.get('trafo_loading_file')
                    if not rel:
                        return None
                    pq = os.path.join(RESULTS_DIR, rel)
                    df = _read_parquet_or_csv(pq)
                    return df
                det_pdf = _load_pdf_from_meta(det_meta)
                vals = []
                labs = []
                if det_pdf is not None:
                    vals.append(_horizon_reliability_from_pdf(det_pdf))
                    labs.append('deterministic')
                for lab, m in drcc_meta_map.items():
                    pdf = _load_pdf_from_meta(m)
                    if pdf is not None:
                        vals.append(_horizon_reliability_from_pdf(pdf))
                        labs.append(f"DRCC, ε={lab}")
                if vals:
                    ax.bar(np.arange(len(vals)), vals, color='#2ca02c', alpha=0.85)
                    ax.set_xticks(np.arange(len(vals)))
                    ax.set_xticklabels(labs, rotation=20)
                    ax.set_ylim(0, 1.0)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.set_ylabel('Probability')
                ax.set_title('Horizon reliability P(any exceedance)')
                ax.grid(alpha=0.3)

                # C2: Per-transformer violation rates (top-3 by deterministic)
                ax = axes_tail[2,1]
                def _per_trafo_rates(pdf: pd.DataFrame) -> pd.Series:
                    # fraction over all (sample,t) points for each trafo
                    grp_all = pdf.groupby('trafo_index')['loading_pct']
                    counts = pdf.groupby('trafo_index').size()
                    viol = pdf[pdf['loading_pct'] > OVERLOAD_THRESHOLD_PCT].groupby('trafo_index').size()
                    rate = (viol / counts).reindex(counts.index).fillna(0.0)
                    return rate
                if det_pdf is not None:
                    det_rates = _per_trafo_rates(det_pdf)
                    top3_idx = det_rates.sort_values(ascending=False).head(3).index.to_list()
                    x = np.arange(len(top3_idx))
                    ax.bar(x - 0.15, det_rates.loc[top3_idx].to_numpy(), width=0.3, label='deterministic', color='black', alpha=0.7)
                    # choose DRCC 0.10 if available, else first available
                    drcc_choice = drcc_meta_map.get(f"{0.10:.2f}") or (drcc_meta_map[list(drcc_meta_map.keys())[0]] if drcc_meta_map else None)
                    if drcc_choice is not None:
                        pdf = _load_pdf_from_meta(drcc_choice)
                        if pdf is not None:
                            r = _per_trafo_rates(pdf)
                            vals = r.reindex(top3_idx).fillna(0.0).to_numpy()
                            ax.bar(x + 0.15, vals, width=0.3, label='DRCC', color='#1f77b4', alpha=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"Trafo {i}" for i in top3_idx])
                    ax.set_ylim(0, 1.0)
                    ax.set_ylabel('Violation rate')
                    ax.set_title('Per-transformer violation rates (top-3)')
                    ax.legend(fontsize=8, frameon=False)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.grid(alpha=0.3)

                # C3: Overload episode energy distribution per trajectory
                ax = axes_tail[2,2]
                def _overload_energy(pdf: pd.DataFrame) -> np.ndarray:
                    # For each (sample,t) take max exceedance across trafos, then sum over t per sample
                    df = pdf.copy()
                    df['excess'] = (df['loading_pct'] - OVERLOAD_THRESHOLD_PCT).clip(lower=0)
                    per_st = df.groupby(['sample_id','t'])['excess'].max().reset_index()
                    per_s = per_st.groupby('sample_id')['excess'].sum()
                    arr = pd.to_numeric(per_s, errors='coerce').to_numpy()
                    return arr[np.isfinite(arr)]
                data = []
                labels = []
                if det_pdf is not None:
                    data.append(_overload_energy(det_pdf))
                    labels.append('deterministic')
                if drcc_choice is not None:
                    pdf = _load_pdf_from_meta(drcc_choice)
                    if pdf is not None:
                        data.append(_overload_energy(pdf))
                        labels.append('DRCC')
                if data:
                    ax.boxplot(data, labels=labels, showfliers=False)
                    ax.set_ylabel('Sum of exceedance (pp) over horizon')
                    ax.set_title('Overload episode energy by trajectory')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
                ax.grid(alpha=0.3)

                # Save tail overview
                out_tail = os.path.join(RESULTS_DIR, TAIL_OVERVIEW_FIG)
                fig_tail.savefig(out_tail, dpi=150)
                print(f"✓ Tail overview saved: {out_tail}")
        except Exception as e:
            print(f"[WARN] Failed to build tail overview figure: {e}")

