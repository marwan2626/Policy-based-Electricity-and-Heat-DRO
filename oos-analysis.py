# Out-of-sample analysis for v3 results
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
# Distribution toggle: 'gaussian' (default) -> v3_oos; 'uniform' -> v3_oos_uniform; 'contaminated' -> v3_oos_contaminated; 'studentt' -> v3_oos_studentt
DISTRIBUTION: str = os.getenv('V3_SAMPLE_DISTRIBUTION', 'studentt').strip().lower()
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
    "v3_oos_uniform" if DISTRIBUTION == 'uniform' else (
        "v3_oos_contaminated" if DISTRIBUTION == 'contaminated' else (
            "v3_oos_studentt" if DISTRIBUTION == 'studentt' else "v3_oos"
        )
    )
)
print(f"[config] DISTRIBUTION = {DISTRIBUTION} | RESULTS_DIR = {RESULTS_DIR}")
if AGGREGATE_DISTS is None and not os.path.isdir(RESULTS_DIR):
    raise FileNotFoundError(
        "Results directory '" + RESULTS_DIR + "' not found. "
        "If you intended to analyze uniform runs, set --dist uniform (or V3_SAMPLE_DISTRIBUTION=uniform) and run v3 in uniform mode first; "
        "for contaminated, use --dist contaminated and run v3 in contaminated mode; for studentt, use --dist studentt and run v3 in studentt mode."
    )
# Global Matplotlib style: Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
# Reduced epsilon set (removed 0.25 and 0.15 per user request)
EPSILONS: List[float] = [0.30, 0.20, 0.10, 0.05]
# Include baseline (k=1, no network tightening) summary as an extra category.
# We now call this 'stochastic' (it still has RT budgets sized by forecast std but no quantile amplification).
INCLUDE_DETERMINISTIC: bool = True
DETERMINISTIC_LABEL: str = "deterministic"  # displayed label for drcc_false baseline run
OUT_FIG = "oos_overview.png"
OUT_CSV = "oos_overview_summary.csv"
SHOW: bool = False  # set True to display interactively
PLOT_SOC_ENVELOPES: bool = True
SOC_ENV_FIG = "soc_envelopes.png"
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
# New: time series of lambda_plus and lambda0 across epsilon cases
PLOT_POLICY_LAMBDA_TIME_SERIES: bool = True
POLICY_LAMBDA_TIME_SERIES_FIG = "policy_lambda_time_series.png"
# New: Tail/zoomed overview figure (additional diagnostics; does not replace existing plots)
PLOT_TAIL_OVERVIEW: bool = True
TAIL_OVERVIEW_FIG = "oos_overview_tail.png"

# New: dedicated two-violin comparison (deterministic vs epsilon=0.10)
PLOT_VIOLIN_COMPARE: bool = True
VIOLIN_COMPARE_FIG = "violin_compare_det_vs_010.png"
VIOLIN_SPLIT_USE_KDE: bool = True
VIOLIN_SPLIT_KDE_BW_ADJ: float = 2.0  # 1.0 ~ Matplotlib default smoothness; <1 sharper, >1 smoother

# New: deterministic vs ε=0.10 comparison of total transformer overload energy (MVAh)
PLOT_OVERLOAD_ENERGY_COMPARE: bool = True
OVERLOAD_ENERGY_COMPARE_FIG = "overload_energy_compare_det_vs_010.png"
OVERLOAD_ENERGY_COMPARE_CSV = "overload_energy_compare_det_vs_010.csv"
# Parameters per user instruction
OVERLOAD_THRESHOLD_PCT: float = 80.0
RATED_TRAFO_MVA: float = 0.5
STEP_HOURS: float = 0.25  # 15-minute steps
OVERLOAD_SAMPLE_COUNT_DEFAULT: int = 1000  # divide by number of samples to get per-sample energy

# New: deterministic vs ε=0.10 comparison of CVaR90 transformer loading (%)
PLOT_CVAR90_COMPARE: bool = True
CVAR90_COMPARE_FIG = "cvar90_loading_compare_det_vs_010.png"
CVAR90_COMPARE_CSV = "cvar90_loading_compare_det_vs_010.csv"

# Cost model parameters for OOS components
PV_CURT_PRICE_FACTOR = 1.0  # EUR per MWh of curtailed PV is factor * price
BESS_THROUGHPUT_COST_EUR_PER_MWH = 0.0  # cost per MWh of RT BESS throughput (set >0 if you price cycling)

# Shared colormap for heatmaps: white at 0, blue at max
WHITE_BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list('white_blue', ['#ffffff', '#1f77b4'])


def epsilon_token(eps: float) -> str:
    return f"{eps:.2f}".replace(".", "_")


def load_summary_for_epsilon(eps: float) -> pd.DataFrame:
    token = epsilon_token(eps)
    # Preferred new naming (strict): v3_summary_drcc_true_epsilon_<token>.csv
    preferred = os.path.join(RESULTS_DIR, f"v3_summary_drcc_true_epsilon_{token}.csv")
    if os.path.exists(preferred):
        return pd.read_csv(preferred)
    # Fallback 1: legacy (pre-refactor) name (only use if no new drcc_true file found)
    legacy = os.path.join(RESULTS_DIR, f"v3_summary_epsilon_{token}.csv")
    if os.path.exists(legacy):
        print(f"[WARN] Using legacy summary file for epsilon={eps:.2f}: {os.path.basename(legacy)} (consider re-running to produce drcc_true file)")
        return pd.read_csv(legacy)
    # Fallback 2: stray misnamed files (e.g., v3_summary_drcc_false_epsilon_<token>.csv) – ignore unless nothing else
    stray = os.path.join(RESULTS_DIR, f"v3_summary_drcc_false_epsilon_{token}.csv")
    if os.path.exists(stray):
        print(f"[WARN] Falling back to stray drcc_false_epsilon file for epsilon={eps:.2f} (treating as placeholder): {os.path.basename(stray)}")
        return pd.read_csv(stray)
    raise FileNotFoundError(f"Missing summary for epsilon={eps:.2f}: expected {os.path.basename(preferred)} (or legacy {os.path.basename(legacy)})")

# Dir-parameterized variants for aggregation mode
def _load_summary_for_epsilon_in_dir(results_dir: str, eps: float) -> pd.DataFrame:
    token = epsilon_token(eps)
    preferred = os.path.join(results_dir, f"v3_summary_drcc_true_epsilon_{token}.csv")
    if os.path.exists(preferred):
        return pd.read_csv(preferred)
    legacy = os.path.join(results_dir, f"v3_summary_epsilon_{token}.csv")
    if os.path.exists(legacy):
        return pd.read_csv(legacy)
    stray = os.path.join(results_dir, f"v3_summary_drcc_false_epsilon_{token}.csv")
    if os.path.exists(stray):
        return pd.read_csv(stray)
    raise FileNotFoundError(f"[{results_dir}] Missing summary for epsilon={eps:.2f}")


def load_meta_for_epsilon(eps: float) -> Dict:
    token = epsilon_token(eps)
    # Preferred
    preferred = os.path.join(RESULTS_DIR, f"v3_meta_drcc_true_epsilon_{token}.json")
    if os.path.exists(preferred):
        try:
            with open(preferred, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    # Legacy fallback
    legacy = os.path.join(RESULTS_DIR, f"v3_meta_epsilon_{token}.json")
    if os.path.exists(legacy):
        try:
            with open(legacy, 'r', encoding='utf-8') as f:
                print(f"[WARN] Using legacy meta file for epsilon={eps:.2f}: {os.path.basename(legacy)}")
                return json.load(f)
        except Exception:
            pass
    # Stray drcc_false_epsilon (should not exist – fallback warning)
    stray = os.path.join(RESULTS_DIR, f"v3_meta_drcc_false_epsilon_{token}.json")
    if os.path.exists(stray):
        try:
            with open(stray, 'r', encoding='utf-8') as f:
                print(f"[WARN] Falling back to stray drcc_false meta for epsilon={eps:.2f}: {os.path.basename(stray)}")
                return json.load(f)
        except Exception:
            pass
    return {}

def _load_meta_for_epsilon_in_dir(results_dir: str, eps: float) -> Dict:
    token = epsilon_token(eps)
    preferred = os.path.join(results_dir, f"v3_meta_drcc_true_epsilon_{token}.json")
    if os.path.exists(preferred):
        try:
            with open(preferred, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    legacy = os.path.join(results_dir, f"v3_meta_epsilon_{token}.json")
    if os.path.exists(legacy):
        try:
            with open(legacy, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    stray = os.path.join(results_dir, f"v3_meta_drcc_false_epsilon_{token}.json")
    if os.path.exists(stray):
        try:
            with open(stray, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
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
        "v3_oos" if name == 'gaussian' else (
            "v3_oos_uniform" if name == 'uniform' else (
                "v3_oos_contaminated" if name == 'contaminated' else (
                    "v3_oos_studentt" if name == 'studentt' else name
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
            'rt_imbalance_cost_mean': rt_imb_cost_mean,
            'rt_pv_cost_mean': rt_pv_cost_mean,
            'rt_bess_cost_mean': rt_bess_cost_mean,
            'trafo_steps': trafo_steps,
            'trafo_violation_probability_pct': trafo_violation_probability_pct,
            'horizon_timesteps': horizon,
            'total_rt_cost_mean': rt_imb_cost_mean + rt_pv_cost_mean + rt_bess_cost_mean,
        }

    # summaries
    rows: List[Dict[str, float]] = []
    # deterministic
    det_path = os.path.join(results_dir, 'v3_summary_drcc_false.csv')
    if os.path.exists(det_path):
        try:
            det_df = pd.read_csv(det_path)
            det_meta = {}
            det_meta_path = os.path.join(results_dir, 'v3_meta_drcc_false.json')
            if os.path.exists(det_meta_path):
                with open(det_meta_path,'r',encoding='utf-8') as f:
                    det_meta = json.load(f)
            rows.append(_build_rt_row_local(det_df, det_meta, None, DETERMINISTIC_LABEL))
        except Exception:
            pass
    # epsilons
    for e in EPSILONS:
        try:
            df_e = _load_summary_for_epsilon_in_dir(results_dir, e)
        except Exception:
            continue
        meta_e = _load_meta_for_epsilon_in_dir(results_dir, e)
        rows.append(_build_rt_row_local(df_e, meta_e, e, f"{e:.2f}"))
    rt_df = pd.DataFrame(rows)

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
            pdf = pd.read_parquet(pq)
        except Exception:
            return np.array([])
        must = {'sample_id','t','trafo_index','loading_pct'}
        if not must <= set(pdf.columns):
            return np.array([])
        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
        arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
        return arr[np.isfinite(arr)]

    dist_map: Dict[str, np.ndarray] = {}
    # det
    det_meta_p = os.path.join(results_dir, 'v3_meta_drcc_false.json')
    if os.path.exists(det_meta_p):
        try:
            with open(det_meta_p,'r',encoding='utf-8') as f:
                m = json.load(f)
            dist_map[DETERMINISTIC_LABEL] = _load_flat_distribution_from_meta(m)
        except Exception:
            pass
    for e in EPSILONS:
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
    out_dir = f"v3_oos_agg_{dist_a}_{dist_b}"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    print(f"[aggregate] Building dual-distribution aggregation: {dist_a} + {dist_b} -> {out_dir} (weights={w})")

    a_sum, a_dist = _build_bundle_for_dir(a_dir)
    b_sum, b_dist = _build_bundle_for_dir(b_dir)
    # unify labels
    labels = sorted(set(a_sum['label']).union(set(b_sum['label'])), key=lambda s: (0 if s==DETERMINISTIC_LABEL else 1, s))
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
        # epsilon value if available
        eps_val = None
        try:
            eps_val = float(lab) if lab != DETERMINISTIC_LABEL else None
        except Exception:
            eps_val = None
        rows.append({
            'label': lab,
            'epsilon': eps_val,
            'rt_imbalance_cost_mean': _wavg('rt_imbalance_cost_mean'),
            'total_rt_cost_mean': _wavg('total_rt_cost_mean'),
            'trafo_steps': _wavg('trafo_steps'),
            'trafo_violation_probability_pct': _wavg('trafo_violation_probability_pct')
        })
    agg_df = pd.DataFrame(rows)
    agg_csv = os.path.join(out_dir, 'agg_summary.csv')
    agg_df.to_csv(agg_csv, index=False)
    print(f"✓ Aggregated summary CSV: {agg_csv}")

    # Build compact 3-panel plot: (1) RT imbalance cost (2) Trafo steps (3) Violin of loading per case (mixture)
    fig, axes = plt.subplots(1, 3, figsize=(22, 4.5), constrained_layout=True)
    x = np.arange(len(labels))

    # 1) RT imbalance bars (show A, B, and weighted)
    def _values_for(col: str, df: pd.DataFrame):
        vals = []
        for lab in labels:
            row = df[df['label']==lab]
            vals.append(float(row[col].iloc[0]) if not row.empty and np.isfinite(row[col].iloc[0]) else np.nan)
        return np.array(vals)
    # augment a_sum/b_sum to ensure necessary columns
    for df in (a_sum, b_sum):
        for c in ['rt_imbalance_cost_mean','total_rt_cost_mean','trafo_steps','trafo_violation_probability_pct']:
            if c not in df.columns:
                df[c] = np.nan
    a_vals = _values_for('rt_imbalance_cost_mean', a_sum)
    b_vals = _values_for('rt_imbalance_cost_mean', b_sum)
    agg_vals = w[0]*np.nan_to_num(a_vals, nan=0.0) + w[1]*np.nan_to_num(b_vals, nan=0.0)
    width = 0.25
    axes[0].bar(x - width, a_vals, width=width, color='#1f77b4', label=dist_a)
    axes[0].bar(x, agg_vals, width=width, color='#636363', label='aggregated')
    axes[0].bar(x + width, b_vals, width=width, color='#ff7f0e', label=dist_b)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([('deterministic' if lab==DETERMINISTIC_LABEL else f"DRCC, ε={lab}") for lab in labels])
    axes[0].set_ylabel('EUR (mean across samples)')
    axes[0].set_title('RT imbalance cost (mean)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)

    # 2) Trafo steps
    a_steps = _values_for('trafo_steps', a_sum)
    b_steps = _values_for('trafo_steps', b_sum)
    agg_steps = w[0]*np.nan_to_num(a_steps, nan=0.0) + w[1]*np.nan_to_num(b_steps, nan=0.0)
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

        paths = []
        for m in (a_meta, b_meta):
            p = None
            if isinstance(m, dict) and 'trafo_loading_file' in m:
                rel = m['trafo_loading_file']
                cand = os.path.join(a_dir, rel)
                if os.path.exists(cand):
                    p = cand
                else:
                    cand2 = os.path.join(b_dir, rel)
                    if os.path.exists(cand2):
                        p = cand2
                    elif os.path.exists(rel):
                        p = rel
            if p and os.path.exists(p):
                paths.append(p)
        if not paths:
            return False
        try:
            dfs = [pd.read_parquet(p) for p in paths]
            pdf = pd.concat(dfs, ignore_index=True)
            out_abs = os.path.join(out_dir, out_rel.replace('/', os.sep))
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)
            pdf.to_parquet(out_abs, index=False)
            return True
        except Exception as e:
            print(f"[WARN] Failed to write combined parquet for {out_rel}: {e}")
            return False

    # Baseline: summaries and meta/parquet
    det_a_sum = os.path.join(a_dir, 'v3_summary_drcc_false.csv')
    det_b_sum = os.path.join(b_dir, 'v3_summary_drcc_false.csv')
    det_out_sum = os.path.join(out_dir, 'v3_summary_drcc_false.csv')
    _combine_summary(det_a_sum, det_b_sum, det_out_sum)
    # meta
    det_a_meta_p = os.path.join(a_dir, 'v3_meta_drcc_false.json')
    det_b_meta_p = os.path.join(b_dir, 'v3_meta_drcc_false.json')
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
    det_parquet_rel = 'v3_loading/trafo_loading_raw_drcc_false_combined.parquet'
    if _combine_parquet_from_meta(det_meta_a, det_meta_b, det_parquet_rel):
        # choose a base meta and update parquet ref
        det_meta_out = det_meta_b if det_meta_b else det_meta_a
        if isinstance(det_meta_out, dict):
            det_meta_out['trafo_loading_file'] = det_parquet_rel.replace('\\', '/')
            with open(os.path.join(out_dir,'v3_meta_drcc_false.json'),'w',encoding='utf-8') as f:
                json.dump(det_meta_out, f, indent=2)

    # DRCC epsilons
    for e in EPSILONS:
        tok = epsilon_token(e)
        # summaries
        a_sum_p = os.path.join(a_dir, f'v3_summary_drcc_true_epsilon_{tok}.csv')
        b_sum_p = os.path.join(b_dir, f'v3_summary_drcc_true_epsilon_{tok}.csv')
        out_sum_p = os.path.join(out_dir, f'v3_summary_drcc_true_epsilon_{tok}.csv')
        _combine_summary(a_sum_p, b_sum_p, out_sum_p)
        # metas
        a_meta_p = os.path.join(a_dir, f'v3_meta_drcc_true_epsilon_{tok}.json')
        b_meta_p = os.path.join(b_dir, f'v3_meta_drcc_true_epsilon_{tok}.json')
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
        pq_rel = f'v3_loading/trafo_loading_raw_epsilon_{tok}_combined.parquet'
        if _combine_parquet_from_meta(a_meta, b_meta, pq_rel):
            meta_out = b_meta if b_meta else a_meta
            if isinstance(meta_out, dict):
                meta_out['trafo_loading_file'] = pq_rel.replace('\\', '/')
                with open(os.path.join(out_dir, f'v3_meta_drcc_true_epsilon_{tok}.json'),'w',encoding='utf-8') as f:
                    json.dump(meta_out, f, indent=2)

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
    _copy_if_exists('soc_envelope_drcc_false.csv')
    # per epsilon
    for e in EPSILONS:
        tok = epsilon_token(e)
        _copy_if_exists(f'policy_coeffs_drcc_true_epsilon_{tok}.csv')
        _copy_if_exists(f'soc_envelope_drcc_true_epsilon_{tok}.csv')

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
            'rt_imbalance_cost_mean': rt_imb_cost_mean,
            'rt_pv_cost_mean': rt_pv_cost_mean,
            'rt_bess_cost_mean': rt_bess_cost_mean,
            'trafo_steps': trafo_steps,
            'line_steps': line_steps,
            'trafo_violation_probability_pct': trafo_violation_probability_pct,
            'horizon_timesteps': horizon,
            'total_rt_cost_mean': rt_imb_cost_mean + rt_pv_cost_mean + rt_bess_cost_mean,
            'is_deterministic': int(eps is None)
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
            return f"DRCC, ε = {lab}"
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
          - meta['trafo_loading_file'] is a relative path (e.g., 'v3_loading\\trafo_loading_raw_epsilon_0_05.parquet')
          - Parquet contains columns either like 'loading_pct' per record or per-trafo columns.
        Strategy:
          1. Read parquet to DataFrame (if engine available).
          2. Collect all numeric columns whose name contains 'loading' and '%'.
          3. Flatten into single 1-D array of loading percentages.
          4. Compute CVaR90 & CVaR95 over that array.
        """
        rel_path = meta.get('trafo_loading_file')
        base_dir = RESULTS_DIR  # parquets appear inside RESULTS_DIR/v3_loading
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
            df_load = pd.read_parquet(abs_path)
        except Exception:
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
    for eps in EPSILONS:
        try:
            df_eps = load_summary_for_epsilon(eps)
        except FileNotFoundError:
            continue
        meta = load_meta_for_epsilon(eps)
        severity = load_trafo_loading(meta)
        row = build_rt_row(df_eps, meta, eps, f"{eps:.2f}")
        row.update({
            'trafo_cvar90_loading_pct': severity['cvar90'],
            'trafo_cvar95_loading_pct': severity['cvar95'],
            'trafo_violation_excess_cvar90_pct': severity['sev_cvar90'],
            'trafo_violation_excess_cvar95_pct': severity['sev_cvar95'],
        })
        rt_rows.append(row)

    # Deterministic (baseline k=1) appended (ordering handled later so it appears first)
    det_path = os.path.join(RESULTS_DIR, 'v3_summary_drcc_false.csv')
    if INCLUDE_DETERMINISTIC and os.path.exists(det_path):
        det_df = pd.read_csv(det_path)
        # Try load its meta
        det_meta_path = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
        if os.path.exists(det_meta_path):
            try:
                with open(det_meta_path, 'r', encoding='utf-8') as f:
                    det_meta = json.load(f)
            except Exception:
                det_meta = {}
        else:
            det_meta = {}
        sev_det = load_trafo_loading(det_meta)
        det_row = build_rt_row(det_df, det_meta, None, DETERMINISTIC_LABEL)
        det_row.update({
            'trafo_cvar90_loading_pct': sev_det['cvar90'],
            'trafo_cvar95_loading_pct': sev_det['cvar95'],
            'trafo_violation_excess_cvar90_pct': sev_det['sev_cvar90'],
            'trafo_violation_excess_cvar95_pct': sev_det['sev_cvar95'],
        })
        rt_rows.append(det_row)

    rt_summary = pd.DataFrame(rt_rows)
    # Build label ordering: deterministic first (if present) then epsilon cases in given order
    if INCLUDE_DETERMINISTIC and os.path.exists(det_path):
        label_order = [DETERMINISTIC_LABEL] + [f"{e:.2f}" for e in EPSILONS]
    else:
        label_order = [f"{e:.2f}" for e in EPSILONS]
    rt_summary['plot_order'] = rt_summary['label'].apply(lambda x: label_order.index(x) if x in label_order else 999)
    rt_summary = rt_summary.sort_values('plot_order')

    # Merge legacy (epsilon keyed) for DRCC rows only
    if not legacy_summary.empty:
        legacy_summary = legacy_summary.rename(columns={'epsilon': 'epsilon'}).copy()
        summary = pd.merge(rt_summary, legacy_summary, on='epsilon', how='left', suffixes=('', '_legacy'))
    else:
        summary = rt_summary.copy()

    summary.to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)

    # === Radial-only adaptation ===
    # New v3 (post-refactor) provides only radial (Option A) network loading; voltages are NaN/omitted.
    # Detect this to adjust plot labels & console messaging.
    radial_only_mode = True  # currently always true after removal of admittance logic
    if radial_only_mode:
        print("[INFO] Detected radial-only flow evaluation mode (Option A); voltage metrics suppressed / NaN.")

    # Derive transformer violation threshold (default 80%) from any available summary column
    threshold_candidates = []
    if 'loading_violation_threshold_pct' in summary.columns:
        threshold_candidates.extend(list(pd.to_numeric(summary['loading_violation_threshold_pct'], errors='coerce').dropna().unique()))
    # Fallback: look directly into a representative v3_summary file if not populated (older runs)
    if not threshold_candidates:
        for eps in EPSILONS:
            token = epsilon_token(eps)
            fpath = os.path.join(RESULTS_DIR, f"v3_summary_drcc_true_epsilon_{token}.csv")
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
    # Final fallback constant
    violation_threshold_pct = float(threshold_candidates[0]) if threshold_candidates else 80.0
    print(f"[INFO] Using transformer violation threshold = {violation_threshold_pct:.0f}% for plots.")

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

    # All-in total cost bar: v2 base cost (no RT proxies) + v3 RT recourse costs
    # Build meta cache for labels
    meta_by_label: Dict[str, Dict] = {}
    det_meta_path = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
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
        base_val = compute_v2_base_cost(meta)
        base_costs.append(base_val if np.isfinite(base_val) else np.nan)
    base_series = pd.Series(base_costs, index=rt_summary.index)
    # RT recourse mean costs from v3
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
    axes[2].set_title(f'Transformer loading violations (> {violation_threshold_pct:.0f}%)')
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
    axes[3].set_title(f'Transformer violation probability (> {violation_threshold_pct:.0f}%)')
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
            pdf = pd.read_parquet(abs_path)
        except Exception:
            return np.array([])
        must_cols = {'sample_id','t','trafo_index','loading_pct'}
        if not must_cols <= set(pdf.columns):
            return np.array([])
        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
        arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
        return arr[np.isfinite(arr)]

    # Collect meta mapping for epsilons during earlier loop wasn't stored; reload here
    meta_cache: Dict[str, Dict] = {}
    # Deterministic baseline
    if INCLUDE_DETERMINISTIC and os.path.exists(os.path.join(RESULTS_DIR,'v3_meta_drcc_false.json')):
        try:
            with open(os.path.join(RESULTS_DIR,'v3_meta_drcc_false.json'),'r',encoding='utf-8') as f:
                meta_cache[DETERMINISTIC_LABEL] = json.load(f)
        except Exception:
            meta_cache[DETERMINISTIC_LABEL] = {}
    for eps in EPSILONS:
        lab = f"{eps:.2f}"
        meta_cache[lab] = load_meta_for_epsilon(eps)

    for lab in rt_summary['label']:
        meta = meta_cache.get(lab, {})
        dist = _load_loading_distribution(meta)
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
        excess = vals - violation_threshold_pct
        excess = excess[excess > 0]
        means_sev.append(float(np.mean(excess)) if excess.size > 0 else 0.0)
        pos_sev.append(i)
    bars_sev = ax_sev.bar(pos_sev, means_sev, width=0.6, color='#c44e52')
    ax_sev.set_xticks(pos_sev)
    ax_sev.set_xticklabels([_display_label_for_case(l) for l in labels_box])
    ax_sev.set_xlabel('epsilon / mode')
    ax_sev.set_ylabel(f'Exceedance over {int(violation_threshold_pct)}% (pp)')
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

    # --- New: Split-violin comparison figure (deterministic vs epsilon=0.10) ---
    if PLOT_VIOLIN_COMPARE:
        try:
            # Map label -> distribution for quick access
            dist_map = {lab: arr for lab, arr in zip(labels_box, distributions)}
            left_label = DETERMINISTIC_LABEL
            right_label = f"{0.10:.2f}"
            left_vals = dist_map.get(left_label, np.array([]))
            right_vals = dist_map.get(right_label, np.array([]))
            # Filter finite values
            left_vals = left_vals[np.isfinite(left_vals)] if isinstance(left_vals, np.ndarray) and left_vals.size else np.array([])
            right_vals = right_vals[np.isfinite(right_vals)] if isinstance(right_vals, np.ndarray) and right_vals.size else np.array([])
            fig_cmp, ax_cmp = plt.subplots(figsize=(4.0, 8.0))
            x0 = 1.0
            if left_vals.size == 0 and right_vals.size == 0:
                ax_cmp.text(0.5, 0.5, 'No transformer loading data', ha='center', va='center', transform=ax_cmp.transAxes, fontsize=9, color='gray')
                ax_cmp.set_xticks([x0])
                ax_cmp.set_xticklabels([f"{_display_label_for_case(left_label)} vs DRCC, ε={right_label}"])
            else:
                # Determine y-limits from data
                vals_list = []
                side_tags = []  # 'left' or 'right' matching bodies index order
                if left_vals.size:
                    vals_list.append(left_vals)
                    side_tags.append('left')
                if right_vals.size:
                    vals_list.append(right_vals)
                    side_tags.append('right')
                stacked = np.concatenate(vals_list) if vals_list else np.array([])
                y_min = float(np.nanmin(stacked)) if stacked.size else 0.0
                y_max = float(np.nanmax(stacked)) if stacked.size else 1.0
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                    y_min, y_max = 0.0, 1.0
                pad = 0.02 * (y_max - y_min + 1e-9)
                y_min -= pad
                y_max += pad
                ax_cmp.set_ylim(y_min, y_max)
                ax_cmp.set_xlim(x0 - 0.6, x0 + 0.6)

                # Build violins using Matplotlib's KDE and aesthetics
                positions = [x0 for _ in vals_list]
                vp = ax_cmp.violinplot(vals_list, positions=positions, showmeans=False, showmedians=False, showextrema=False)

                # Clip bodies to halves and color them
                from matplotlib.patches import Rectangle, Patch
                xmin, xmax = ax_cmp.get_xlim()
                ymin, ymax = ax_cmp.get_ylim()
                left_clip = Rectangle((xmin, ymin), width=(x0 - xmin), height=(ymax - ymin), transform=ax_cmp.transData)
                right_clip = Rectangle((x0, ymin), width=(xmax - x0), height=(ymax - ymin), transform=ax_cmp.transData)
                for body, tag in zip(vp['bodies'], side_tags):
                    if tag == 'left':
                        body.set_facecolor('#b2df8a')
                        body.set_edgecolor('#1b7837')
                        body.set_alpha(0.6)
                        body.set_clip_path(left_clip)
                    else:
                        body.set_facecolor('#a6cee3')
                        body.set_edgecolor('#1f78b4')
                        body.set_alpha(0.6)
                        body.set_clip_path(right_clip)

                # Ensure default median lines (if any) are hidden
                if isinstance(vp, dict) and 'cmedians' in vp and vp['cmedians'] is not None:
                    try:
                        vp['cmedians'].set_visible(False)
                    except Exception:
                        pass

                # Replace median with short ticks per side for clarity
                if left_vals.size:
                    m_left = float(np.nanmedian(left_vals))
                    # Median tick starting at center and extending left
                    ax_cmp.plot([x0 - 0.21, x0], [m_left, m_left], color='#1b7837', linewidth=2)
                    # Extrema ticks (min/max)
                    try:
                        l_min = float(np.nanmin(left_vals))
                        l_max = float(np.nanmax(left_vals))
                        # Extrema ticks starting at center and extending left
                        ax_cmp.plot([x0 - 0.10, x0], [l_min, l_min], color='#1b7837', linewidth=1.6)
                        ax_cmp.plot([x0 - 0.10, x0], [l_max, l_max], color='#1b7837', linewidth=1.6)
                    except Exception:
                        pass
                if right_vals.size:
                    m_right = float(np.nanmedian(right_vals))
                    # Median tick starting at center and extending right
                    ax_cmp.plot([x0, x0 + 0.21], [m_right, m_right], color='#1f78b4', linewidth=2)
                    # Extrema ticks (min/max)
                    try:
                        r_min = float(np.nanmin(right_vals))
                        r_max = float(np.nanmax(right_vals))
                        # Extrema ticks starting at center and extending right
                        ax_cmp.plot([x0, x0 + 0.10], [r_min, r_min], color='#1f78b4', linewidth=1.6)
                        ax_cmp.plot([x0, x0 + 0.10], [r_max, r_max], color='#1f78b4', linewidth=1.6)
                    except Exception:
                        pass
                                
                ax_cmp.set_xticks([x0])
                ax_cmp.set_xticklabels([f"{left_label} vs {right_label}"])
                ax_cmp.set_ylabel('Transformer loading %')
                #ax_cmp.set_title('Transformer loading comparison')
                ax_cmp.grid(axis='y', alpha=0.3)
                # Center divider line
                ax_cmp.axvline(x0, color='black', linewidth=0.9, alpha=0.8, zorder=3)
                # Legend
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

    # --- New: Overload energy comparison (deterministic vs ε=0.10) ---
    if PLOT_OVERLOAD_ENERGY_COMPARE:
        try:
            # Helper to compute total overload energy (excess above threshold) in MVAh from parquet
            def _compute_overload_energy_from_parquet(parquet_path: str) -> float:
                try:
                    pdf = pd.read_parquet(parquet_path)
                except Exception:
                    return float('nan')
                must = {'sample_id','t','trafo_index','loading_pct'}
                if not must <= set(pdf.columns):
                    return float('nan')
                lp = pd.to_numeric(pdf['loading_pct'], errors='coerce').to_numpy()
                mask = np.isfinite(lp) & (lp > OVERLOAD_THRESHOLD_PCT)
                if not np.any(mask):
                    # No exceedances => zero overload energy
                    # Still return 0.0 kWh per sample
                    return 0.0
                excess_pct = lp[mask] - OVERLOAD_THRESHOLD_PCT
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
            det_meta_path = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
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
            # Load epsilon=0.10 meta for path
            eps_label = f"{0.10:.2f}"
            meta_010 = load_meta_for_epsilon(0.10)
            eps_over_mvah = float('nan')
            if meta_010 and meta_010.get('trafo_loading_file'):
                pq_010 = os.path.join(RESULTS_DIR, meta_010['trafo_loading_file'])
                eps_over_mvah = _compute_overload_energy_from_parquet(pq_010)

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

    # --- New: CVaR90 transformer loading comparison (deterministic vs ε=0.10) ---
    if PLOT_CVAR90_COMPARE:
        try:
            # Deterministic meta and CVaR90
            det_meta_path = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
            det_cvar = float('nan')
            if os.path.exists(det_meta_path):
                try:
                    with open(det_meta_path, 'r', encoding='utf-8') as f:
                        det_meta = json.load(f)
                    sev_det = load_trafo_loading(det_meta)
                    det_cvar = float(sev_det.get('cvar90', float('nan')))
                except Exception:
                    pass
            # Epsilon 0.10 meta and CVaR90
            meta_010 = load_meta_for_epsilon(0.10)
            eps_cvar = float('nan')
            if meta_010:
                try:
                    sev_010 = load_trafo_loading(meta_010)
                    eps_cvar = float(sev_010.get('cvar90', float('nan')))
                except Exception:
                    pass
            if np.isfinite(det_cvar) or np.isfinite(eps_cvar):
                labels = ['deterministic', 'DRCC, ε=0.10']
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

    # --- Build cost-risk frontier (VaR95 + mean) using per-trajectory v3_summary_* CSVs ---
    def build_frontier(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        pattern = os.path.join(results_dir, 'v3_summary_*.csv')
        for path in glob.glob(pattern):
            try:
                df_sum = pd.read_csv(path)
            except Exception:
                continue
            fname = os.path.basename(path)
            # Skip legacy simple names if a preferred drcc_true exists for same epsilon
            legacy_match = re.match(r'v3_summary_epsilon_([0-9]+_[0-9]+)\.csv', fname)
            if legacy_match:
                tok = legacy_match.group(1)
                preferred = os.path.join(results_dir, f"v3_summary_drcc_true_epsilon_{tok}.csv")
                if os.path.exists(preferred):
                    continue  # ignore legacy because updated file present
            # Skip misnamed drcc_false_epsilon_ variants (deterministic should not carry epsilon)
            if 'drcc_false_epsilon_' in fname:
                continue
            # Mode & epsilon inference
            if 'drcc_false' in fname:
                mode = 'stochastic'
                eps_val = None
            else:
                mode_match = re.search(r'v3_summary_(drcc_[a-zA-Z]+)_epsilon_', fname)
                mode = mode_match.group(1) if mode_match else 'drcc_true'
                tok_match = re.search(r'_epsilon_([0-9]+_[0-9]+)', fname)
                eps_val = None
                if tok_match:
                    try:
                        eps_val = float(tok_match.group(1).replace('_', '.'))
                    except Exception:
                        eps_val = None
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
            # Prefer rows with non-null violation rate; among those pick highest trajectory count
            grp_valid = grp[grp['trafo_violation_rate_mean'].notna()]
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
                base_df['trafo_violation_rate_mean'],
                base_df['mean_cost_eur'],
                marker='o', s=80, c='black', edgecolors='none', zorder=2
            )
            # Annotate deterministic mean point
            for _, r in base_df.iterrows():
                if np.isfinite(r['trafo_violation_rate_mean']) and np.isfinite(r['mean_cost_eur']):
                    _ann = ax_f.annotate('deterministic', (r['trafo_violation_rate_mean'], r['mean_cost_eur']),
                                         textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                    _ann.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
        # DRCC points layered: 0.10, 0.20, 0.30 (others, if any, first)
        if not drcc_df.empty:
            eps_vals = drcc_df['epsilon'].to_numpy(dtype=float)
            vmin, vmax = float(np.nanmin(eps_vals)), float(np.nanmax(eps_vals))
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
            preferred = [0.10, 0.20, 0.30]
            present = [float(e) for e in sorted(pd.unique(drcc_df['epsilon'].dropna()))]
            extras = [e for e in present if e not in preferred]
            order_eps = extras + [e for e in preferred if e in present]
            for e in order_eps:
                sub = drcc_df[np.isclose(drcc_df['epsilon'].astype(float), e)]
                if sub.empty:
                    continue
                ax_f.scatter(
                    sub['trafo_violation_rate_mean'],
                    sub['mean_cost_eur'],
                    color=cmap(norm(e)), s=70, edgecolors='k', linewidths=0.4, zorder=3
                )
            # Colorbar using ScalarMappable
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig_f.colorbar(sm, ax=ax_f, label='risk level (ε)')
        # Annotate epsilon values
        for _, r in drcc_df.iterrows():
            if r['epsilon'] is not None and np.isfinite(r['epsilon']):
                _ann2 = ax_f.annotate(f"ε = {r['epsilon']:.2f}", (r['trafo_violation_rate_mean'], r['mean_cost_eur']),
                                      textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                _ann2.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
        ax_f.set_xlabel('Transformer violation rate (mean)')
        ax_f.set_ylabel('Mean total cost (EUR)')
        ax_f.set_title('Cost–Risk Frontier (Mean vs Violation Rate)')
        ax_f.grid(alpha=0.35)
        # Percent formatting on x-axis
        ax_f.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        # Legend removed per request
        frontier_fig_path = os.path.join(RESULTS_DIR, FRONTIER_SCATTER_FIG)
        fig_f.savefig(frontier_fig_path, dpi=150)
        print(f"✓ Frontier scatter: {frontier_fig_path}")

    # --- Per-trajectory frontier scatter (many dots) ---
    if PLOT_FRONTIER_TRAJECTORY_SCATTER:
        traj_points = []  # list of dicts: {'epsilon':..., 'mode':..., 'vrate':..., 'cost':...}
        # Baseline first
        base_summary = os.path.join(RESULTS_DIR, 'v3_summary_drcc_false.csv')
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
        # DRCC runs
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            fpath = os.path.join(RESULTS_DIR, f'v3_summary_drcc_true_epsilon_{tok}.csv')
            if not os.path.exists(fpath):
                continue
            try:
                df_eps = pd.read_csv(fpath)
            except Exception:
                continue
            if 'steps_trafo_over_80pct' not in df_eps.columns:
                continue
            if 'n_steps' not in df_eps.columns:
                continue
            for _, r in df_eps.iterrows():
                try:
                    ns = float(r.get('n_steps', np.nan))
                    st = float(r.get('steps_trafo_over_80pct', np.nan))
                    if ns > 0 and np.isfinite(st):
                        traj_points.append({
                            'epsilon': eps,
                            'mode': 'drcc_true',
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
            grp_valid = grp[grp['trafo_violation_rate_mean'].notna()]
            if not grp_valid.empty:
                pick = grp_valid.sort_values('n_trajectories', ascending=False).iloc[0]
            else:
                pick = grp.sort_values('n_trajectories', ascending=False).iloc[0]
            subset_rows.append(pick)
        mean_df = pd.DataFrame(subset_rows)
        # Gather trajectory cloud points (baseline + DRCC) reusing same approach
        cloud_points: List[Dict] = []
        base_summary = os.path.join(RESULTS_DIR, 'v3_summary_drcc_false.csv')
        if os.path.exists(base_summary):
            try:
                dfb = pd.read_csv(base_summary)
                if {'steps_trafo_over_80pct','n_steps','total_cost_eur'} <= set(dfb.columns):
                    for _, r in dfb.iterrows():
                        ns = float(r.get('n_steps', np.nan))
                        st = float(r.get('steps_trafo_over_80pct', np.nan))
                        if ns > 0 and np.isfinite(st):
                            cloud_points.append({'epsilon': None,'mode':'stochastic','vrate': st/ns,'cost': float(r.get('total_cost_eur', np.nan))})
            except Exception:
                pass
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            fpath = os.path.join(RESULTS_DIR, f'v3_summary_drcc_true_epsilon_{tok}.csv')
            if not os.path.exists(fpath):
                continue
            try:
                df_eps = pd.read_csv(fpath)
            except Exception:
                continue
            if {'steps_trafo_over_80pct','n_steps','total_cost_eur'} - set(df_eps.columns):
                continue
            for _, r in df_eps.iterrows():
                ns = float(r.get('n_steps', np.nan))
                st = float(r.get('steps_trafo_over_80pct', np.nan))
                if ns > 0 and np.isfinite(st):
                    cloud_points.append({'epsilon': eps,'mode':'drcc_true','vrate': st/ns,'cost': float(r.get('total_cost_eur', np.nan))})
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
                preferred = [0.10, 0.20, 0.30]
                present_m = [float(e) for e in sorted(pd.unique(drcc_mean['epsilon'].dropna()))]
                extras_m = [e for e in present_m if e not in preferred]
                order_eps_m = extras_m + [e for e in preferred if e in present_m]
                for e in order_eps_m:
                    subm = drcc_mean[np.isclose(drcc_mean['epsilon'].astype(float), e)]
                    if subm.empty:
                        continue
                    ax_h.scatter(subm['trafo_violation_rate_mean'], subm['mean_cost_eur'],
                                 color=cmap_m(norm_m(e)), s=70, edgecolors='k', linewidths=0.4, zorder=4)
                sm_m = mpl.cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
                sm_m.set_array([])
                fig_h.colorbar(sm_m, ax=ax_h, label='risk level (ε)')
            if not base_mean.empty:
                ax_h.scatter(base_mean['trafo_violation_rate_mean'], base_mean['mean_cost_eur'],
                             marker='o', s=85, c='black', edgecolors='white', linewidths=0.4, zorder=3)
                # Annotate deterministic mean point(s)
                for _, r in base_mean.iterrows():
                    if np.isfinite(r['trafo_violation_rate_mean']) and np.isfinite(r['mean_cost_eur']):
                        _ann3 = ax_h.annotate('deterministic', (r['trafo_violation_rate_mean'], r['mean_cost_eur']),
                                               textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                        _ann3.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
            for _, r in drcc_mean.iterrows():
                if r['epsilon'] is not None and np.isfinite(r['epsilon']):
                    _ann4 = ax_h.annotate(f"ε = {r['epsilon']:.2f}", (r['trafo_violation_rate_mean'], r['mean_cost_eur']),
                                          textcoords='offset points', xytext=(4,4), fontsize=8, color='black')
                    _ann4.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white')])
            ax_h.set_xlabel('Transformer violation rate (trajectory / mean)')
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

    # --- Transformer violation probability per timestep ---
    if PLOT_TRAFO_VIOLATION_TIME_PROFILE:
        # Correct per-timestep probability: for each (sample_id, t) take max loading across trafos; violation if > threshold.
        threshold_pct = violation_threshold_pct
        profiles: List[Tuple[str, np.ndarray]] = []
        t_axis: np.ndarray | None = None
        # Helper to compute profile from a parquet path
        def compute_profile(parquet_path: str):
            try:
                pdf = pd.read_parquet(parquet_path)
            except Exception:
                return None
            must = {'sample_id','t','trafo_index','loading_pct'}
            if not must <= set(pdf.columns):
                return None
            # Max across trafos per (sample_id, t)
            grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
            counts = grp.groupby('t')['sample_id'].nunique()
            viol = grp[grp['loading_pct'] > threshold_pct].groupby('t')['sample_id'].nunique()
            rate_series = (viol / counts).reindex(counts.index).fillna(0.0)
            return counts.index.to_numpy(), rate_series.to_numpy()
        # Baseline (drcc_false)
        base_meta = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
        if os.path.exists(base_meta):
            try:
                with open(base_meta,'r',encoding='utf-8') as f:
                    m = json.load(f)
                rel = m.get('trafo_loading_file')
                if rel:
                    base_pq = os.path.join(RESULTS_DIR, rel)
                    if os.path.exists(base_pq):
                        res = compute_profile(base_pq)
                        if res:
                            t_axis, rate = res
                            profiles.append(('stochastic', rate))
            except Exception as e:
                print(f"[WARN] Baseline trafo profile failed: {e}")
        # DRCC epsilons
        for eps in EPSILONS:
            meta_e = load_meta_for_epsilon(eps)
            rel = meta_e.get('trafo_loading_file') if isinstance(meta_e, dict) else None
            if not rel:
                continue
            pq_path = os.path.join(RESULTS_DIR, rel.replace('/', os.sep))
            if not os.path.exists(pq_path):
                continue
            res = compute_profile(pq_path)
            if not res:
                continue
            t_local, rate = res
            if t_axis is None:
                t_axis = t_local
            else:
                if len(t_local) != len(t_axis):  # simple alignment by truncation
                    min_len = min(len(t_local), len(t_axis))
                    t_axis = t_axis[:min_len]
                    rate = rate[:min_len]
            profiles.append((f"{eps:.2f}", rate))
        if profiles and t_axis is not None:
            # Normalize all lengths
            min_len = min(len(r) for _, r in profiles)
            profiles = [(lab, r[:min_len]) for lab, r in profiles]
            t_axis = t_axis[:min_len]
            fig_tp, ax_tp = plt.subplots(figsize=(10,4.8))
            # baseline first
            for lab, arr in sorted(profiles, key=lambda x: (0 if x[0]=='stochastic' else 1, x[0])):
                if lab == 'stochastic':
                    ax_tp.plot(t_axis, arr, color='black', linestyle='--', linewidth=1.8, label=lab)
                else:
                    ax_tp.plot(t_axis, arr, linewidth=1.2, alpha=0.9, label=f"DRCC, ε={lab}")
            ax_tp.set_xlabel('Timestep index')
            ax_tp.set_ylabel(f"P(any trafo > {int(threshold_pct)}%)")
            ax_tp.set_title('Per-Timestep Transformer Violation Probability')
            ax_tp.grid(alpha=0.3, linewidth=0.5)
            ax_tp.set_ylim(0, 1.0)
            ax_tp.legend(fontsize=8, ncol=3, frameon=False)
            out_tp = os.path.join(RESULTS_DIR, TRAFO_VIOLATION_TIME_PROFILE_FIG)
            fig_tp.tight_layout()
            fig_tp.savefig(out_tp, dpi=160)
            print(f"✓ Transformer violation time profile: {out_tp}")
        else:
            print('[INFO] Skipped transformer violation time profile (no loading parquet data).')

    # --- Transformer violation probability heatmap (cases x timesteps) ---
    if 'profiles' in locals() and profiles and 't_axis' in locals() and t_axis is not None and PLOT_TRAFO_VIOLATION_HEATMAP:
        try:
            # Ensure consistent ordering: deterministic (baseline) first, then 0.30, 0.20, 0.10 (and others afterward)
            labels_present = [lab for lab, _ in profiles]
            desired_order = []
            if 'stochastic' in labels_present:
                desired_order.append('stochastic')
            for e in EPSILONS:  # EPSILONS is [0.30, 0.20, 0.10, 0.05]
                lab = f"{e:.2f}"
                if lab in labels_present:
                    desired_order.append(lab)
            profiles_sorted = sorted(
                profiles,
                key=lambda it: desired_order.index(it[0]) if it[0] in desired_order else 999
            )
            # Align length to min length across rows, reuse earlier normalization if needed
            min_len = min(len(r) for _, r in profiles_sorted)
            mat = np.vstack([r[:min_len] for _, r in profiles_sorted])  # shape: (cases, T)
            # Clip probabilities at 0.5 for visualization
            mat = np.clip(mat, 0.0, 0.5)
            case_labels = [(_display_label_for_case(lab) if lab != 'stochastic' else 'deterministic') for lab, _ in profiles_sorted]
            # Build heatmap
            fig_hm, ax_hm = plt.subplots(figsize=(min_len/10 + 2.5, 0.6*len(case_labels) + 1.8))
            im = ax_hm.imshow(mat, aspect='auto', cmap=WHITE_BLUE_CMAP, vmin=0.0, vmax=0.5, interpolation='nearest')
            ax_hm.set_yticks(np.arange(len(case_labels)))
            ax_hm.set_yticklabels(case_labels)
            ax_hm.set_xlabel('timestep index')
            ax_hm.set_ylabel('optimization case')
            ax_hm.set_title('Chance of transformer overload per timestep')
            # Colorbar on the right
            cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
            cbar.set_label('violation probability (clipped at 0.5)')
            # Optionally thin x ticks for readability on long horizons
            T = mat.shape[1]
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
            pol_path = os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}.csv')
            if os.path.exists(pol_path):
                heat_cases.append((f"{eps:.2f}", pol_path))
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
                # Identify coefficients (numeric columns)
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
            pol_path = os.path.join(RESULTS_DIR, f'policy_coeffs_drcc_true_epsilon_{tok}.csv')
            if os.path.exists(pol_path):
                lambda_cases.append((f"{eps:.2f}", pol_path, eps))
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

    # --- SoC envelope plotting (optional) ---
    if PLOT_SOC_ENVELOPES:
        # Collect envelope files consistent with naming
        cases: List[Tuple[str, str, float | None]] = []  # (label, path, epsilon)
        # Baseline
        soc_base = os.path.join(RESULTS_DIR, 'soc_envelope_drcc_false.csv')
        if os.path.exists(soc_base):
            cases.append((DETERMINISTIC_LABEL, soc_base, None))
        for eps in EPSILONS:
            tok = epsilon_token(eps)
            soc_path = os.path.join(RESULTS_DIR, f'soc_envelope_drcc_true_epsilon_{tok}.csv')
            if os.path.exists(soc_path):
                cases.append((f"{eps:.2f}", soc_path, eps))
        if cases:
            cols = len(cases)
            fig_soc, ax_soc = plt.subplots(1, cols, figsize=(4*cols, 3), constrained_layout=True)
            if cols == 1:
                ax_soc = [ax_soc]
            for ax, (lab, pth, eps) in zip(ax_soc, cases):
                try:
                    df_env = pd.read_csv(pth)
                except Exception:
                    continue
                if not {'soc_p05','soc_p50','soc_p95'}.issubset(df_env.columns):
                    continue
                t = np.arange(len(df_env))
                ax.fill_between(t, df_env['soc_p05'], df_env['soc_p95'], color='#c6dbef', alpha=0.6, label='5–95% band')
                ax.plot(t, df_env['soc_p50'], color='#08519c', linewidth=1.5, label='Median')
                ax.set_ylim(0, 1.02)
                ax.set_title(f"SoC envelope ({_display_label_for_case(lab)})")
                ax.set_xlabel('t step')
                if ax == ax_soc[0]:
                    ax.set_ylabel('SoC fraction')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            soc_fig_path = os.path.join(RESULTS_DIR, SOC_ENV_FIG)
            fig_soc.savefig(soc_fig_path, dpi=150)
            print(f"✓ SoC envelopes figure: {soc_fig_path}")
        else:
            print("[INFO] No SoC envelope files found to plot.")

    if SHOW:
        plt.show()


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
                    meta_path = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
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
                if not os.path.exists(pq):
                    return None
                try:
                    pdf = pd.read_parquet(pq)
                except Exception:
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
                    if not os.path.exists(pq):
                        return np.array([])
                    try:
                        pdf = pd.read_parquet(pq)
                    except Exception:
                        return np.array([])
                    must = {'sample_id','t','trafo_index','loading_pct'}
                    if not must <= set(pdf.columns):
                        return np.array([])
                    grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
                    arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
                    return arr[np.isfinite(arr)]

                det_meta = {}
                try:
                    with open(os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json'),'r',encoding='utf-8') as f:
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
                ax.set_ylabel(f"P(any trafo > {int(OVERLOAD_THRESHOLD_PCT)}%)")
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
                det_sum = os.path.join(RESULTS_DIR, 'v3_summary_drcc_false.csv')
                arr_det_vr = _traj_violation_rates(det_sum)
                # drcc summaries
                arr_drcc_vr = []
                for e in EPSILONS:
                    tok = epsilon_token(e)
                    sp = os.path.join(RESULTS_DIR, f'v3_summary_drcc_true_epsilon_{tok}.csv')
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
                    if not os.path.exists(pq):
                        return None
                    try:
                        return pd.read_parquet(pq)
                    except Exception:
                        return None
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

