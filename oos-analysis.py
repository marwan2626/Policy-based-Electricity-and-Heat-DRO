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
import matplotlib.pyplot as plt
import glob
import re
import math

# === User config ===
RESULTS_DIR = "v3_oos"
EPSILONS: List[float] = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
# Include baseline (k=1, no network tightening) summary as an extra category.
# We now call this 'stochastic' (it still has RT budgets sized by forecast std but no quantile amplification).
INCLUDE_DETERMINISTIC: bool = True
DETERMINISTIC_LABEL: str = "stochastic"  # displayed label for drcc_false (k=1) run
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
FRONTIER_HYBRID_SCATTER_FIG = "frontier_hybrid_scatter.png"
# New: time series of lambda_plus and lambda0 across epsilon cases
PLOT_POLICY_LAMBDA_TIME_SERIES: bool = True
POLICY_LAMBDA_TIME_SERIES_FIG = "policy_lambda_time_series.png"

# Cost model parameters for OOS components
PV_CURT_PRICE_FACTOR = 1.0  # EUR per MWh of curtailed PV is factor * price
BESS_THROUGHPUT_COST_EUR_PER_MWH = 0.0  # cost per MWh of RT BESS throughput (set >0 if you price cycling)


def epsilon_token(eps: float) -> str:
    return f"{eps:.2f}".replace(".", "_")


def load_summary_for_epsilon(eps: float) -> pd.DataFrame:
    token = epsilon_token(eps)
    # New naming pattern includes drcc mode: v3_summary_<mode>_epsilon_<token>.csv
    # We'll search for any file matching that epsilon token; fall back to legacy name.
    legacy = os.path.join(RESULTS_DIR, f"v3_summary_epsilon_{token}.csv")
    if os.path.exists(legacy):
        return pd.read_csv(legacy)
    # Deterministic (drcc_false) file now omits epsilon entirely; allow reuse across eps loop
    det_path = os.path.join(RESULTS_DIR, 'v3_summary_drcc_false.csv')
    if os.path.exists(det_path):
        # Return a copy with an added column to tag it (caller groups by epsilon anyway)
        df_det = pd.read_csv(det_path).copy()
        return df_det
    # Scan directory for pattern
    prefix = f"v3_summary_"
    suffix = f"_epsilon_{token}.csv"
    candidates = [f for f in os.listdir(RESULTS_DIR) if f.startswith(prefix) and f.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"Missing summary for epsilon={eps}: looked for {legacy} or *{suffix}")
    # If multiple (e.g., drcc_true + drcc_false), choose all? For now pick each individually later.
    # Here we just pick the first sorted; multi-mode comparison could be an extension.
    candidates.sort()
    return pd.read_csv(os.path.join(RESULTS_DIR, candidates[0]))


def load_meta_for_epsilon(eps: float) -> Dict:
    token = epsilon_token(eps)
    legacy = os.path.join(RESULTS_DIR, f"v3_meta_epsilon_{token}.json")
    if os.path.exists(legacy):
        try:
            with open(legacy, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    # Deterministic meta (no epsilon) support
    det_meta = os.path.join(RESULTS_DIR, 'v3_meta_drcc_false.json')
    if os.path.exists(det_meta):
        try:
            with open(det_meta, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    # Look for new pattern v3_meta_<mode>_epsilon_<token>.json
    prefix = "v3_meta_"
    suffix = f"_epsilon_{token}.json"
    try:
        candidates = [f for f in os.listdir(RESULTS_DIR) if f.startswith(prefix) and f.endswith(suffix)]
        if candidates:
            candidates.sort()
            with open(os.path.join(RESULTS_DIR, candidates[0]), 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
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

    # Added two more subplots for CVaR90 / CVaR95 of transformer loading severity
    fig, axes = plt.subplots(1, 8, figsize=(48, 4), constrained_layout=True)
    x = np.arange(len(rt_summary))
    width = 0.6

    c_imb = rt_summary['rt_imbalance_cost_mean'].to_numpy()
    c_pv = rt_summary['rt_pv_cost_mean'].to_numpy()
    c_bess = rt_summary['rt_bess_cost_mean'].to_numpy()
    axes[0].bar(x, c_imb, width=width, label='RT Imbalance')
    axes[0].bar(x, c_pv, width=width, bottom=c_imb, label='RT PV Curtail')
    axes[0].bar(x, c_bess, width=width, bottom=c_imb + c_pv, label='RT BESS Cycle')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rt_summary['label'])
    axes[0].set_xlabel('epsilon / mode')
    axes[0].set_ylabel('EUR (mean across samples)')
    axes[0].set_title('RT cost components (stacked)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    c_da = rt_summary['da_import_cost_mean'].to_numpy()
    axes[1].bar(x, c_da, width=width, color='#4c72b0', label='DA Import Cost')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rt_summary['label'])
    axes[1].set_xlabel('epsilon / mode')
    axes[1].set_ylabel('EUR (mean across samples)')
    axes[1].set_title('DA import cost')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

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
    axes[2].set_xticklabels(rt_summary['label'])
    axes[2].set_xlabel('epsilon / mode')
    axes[2].set_ylabel('Steps (sum across trajectories)')
    axes[2].set_title('Transformer loading violations')
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
    axes[3].set_xticklabels(rt_summary['label'])
    axes[3].set_xlabel('epsilon / mode')
    axes[3].set_ylabel('% of total timesteps')
    axes[3].set_title('Transformer violation probability')
    axes[3].grid(axis='y', alpha=0.3)
    axes[3].legend()

    # 5. Transformer Loading Severity CVaR90
    cvar90 = rt_summary.get('trafo_cvar90_loading_pct', pd.Series([np.nan]*len(rt_summary))).to_numpy()
    axes[4].bar(x, cvar90, width=width, color='#c44e52', label='Trafo CVaR90 loading %')
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(rt_summary['label'])
    axes[4].set_xlabel('epsilon / mode')
    axes[4].set_ylabel('Loading %')
    axes[4].set_title('Transformer loading CVaR90')
    axes[4].grid(axis='y', alpha=0.3)
    axes[4].legend()

    # 6. Transformer Loading Severity CVaR95
    cvar95 = rt_summary.get('trafo_cvar95_loading_pct', pd.Series([np.nan]*len(rt_summary))).to_numpy()
    axes[5].bar(x, cvar95, width=width, color='#8172b3', label='Trafo CVaR95 loading %')
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(rt_summary['label'])
    axes[5].set_xlabel('epsilon / mode')
    axes[5].set_ylabel('Loading %')
    axes[5].set_title('Transformer loading CVaR95')
    axes[5].grid(axis='y', alpha=0.3)
    axes[5].legend()
    # 7. Transformer Violation Excess CVaR90 (excess >100%)
    sev_cvar90 = rt_summary.get('trafo_violation_excess_cvar90_pct', pd.Series([np.nan]*len(rt_summary))).to_numpy()
    axes[6].bar(x, sev_cvar90, width=width, color='#4c72b0', label='Trafo Excess CVaR90 (>100%)')
    axes[6].set_xticks(x)
    axes[6].set_xticklabels(rt_summary['label'])
    axes[6].set_xlabel('epsilon / mode')
    axes[6].set_ylabel('Excess % over 100')
    axes[6].set_title('Transformer excess CVaR90')
    axes[6].grid(axis='y', alpha=0.3)
    axes[6].legend()
    # 8. Transformer Violation Excess CVaR95 (excess >100%)
    sev_cvar95 = rt_summary.get('trafo_violation_excess_cvar95_pct', pd.Series([np.nan]*len(rt_summary))).to_numpy()
    axes[7].bar(x, sev_cvar95, width=width, color='#dd8452', label='Trafo Excess CVaR95 (>100%)')
    axes[7].set_xticks(x)
    axes[7].set_xticklabels(rt_summary['label'])
    axes[7].set_xlabel('epsilon / mode')
    axes[7].set_ylabel('Excess % over 100')
    axes[7].set_title('Transformer excess CVaR95')
    axes[7].grid(axis='y', alpha=0.3)
    axes[7].legend()

    out_path = os.path.join(RESULTS_DIR, OUT_FIG)
    fig.savefig(out_path, dpi=150)
    print(f"✓ Overview saved: {out_path}")
    print(f"✓ Summary CSV: {os.path.join(RESULTS_DIR, OUT_CSV)}")

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
        # Color map by epsilon
        if not drcc_df.empty:
            eps_vals = drcc_df['epsilon'].to_numpy(dtype=float)
            norm = plt.Normalize(vmin=np.nanmin(eps_vals), vmax=np.nanmax(eps_vals))
            cmap = plt.cm.viridis
            sc = ax_f.scatter(
                drcc_df['trafo_violation_rate_mean'],
                drcc_df['mean_cost_eur'],
                c=eps_vals, cmap=cmap, s=70, edgecolors='k', linewidths=0.4
            )
            fig_f.colorbar(sc, ax=ax_f, label='epsilon')
        if not base_df.empty:
            ax_f.scatter(
                base_df['trafo_violation_rate_mean'],
                base_df['mean_cost_eur'],
                marker='x', s=80, c='black', linewidths=1.2
            )
        # Annotate epsilon values
        for _, r in drcc_df.iterrows():
            if r['epsilon'] is not None and np.isfinite(r['epsilon']):
                ax_f.annotate(f"{r['epsilon']:.2f}", (r['trafo_violation_rate_mean'], r['mean_cost_eur']), textcoords='offset points', xytext=(4,4), fontsize=8)
        ax_f.set_xlabel('Transformer violation rate (mean)')
        ax_f.set_ylabel('Mean total cost (EUR)')
        ax_f.set_title('Cost–Risk Frontier (Mean vs Violation Rate)')
        ax_f.grid(alpha=0.35)
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([], [], marker='o', linestyle='None', color='black', markersize=7, label='DRCC trajectories'),
            Line2D([], [], marker='x', linestyle='None', color='black', markersize=7, label='stochastic trajectories'),
        ]
    # Legend removed per request
    # ax_f.legend(handles=legend_handles, fontsize=8, frameon=True)
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
            if not drcc_pts.empty:
                eps_vals = drcc_pts['epsilon'].to_numpy()
                norm = plt.Normalize(vmin=np.nanmin(eps_vals), vmax=np.nanmax(eps_vals))
                cmap = plt.cm.plasma
                sc2 = ax_t.scatter(
                    drcc_pts['vrate'], drcc_pts['cost'], c=eps_vals, cmap=cmap,
                    s=25, alpha=0.65, edgecolors='none'
                )
                fig_t.colorbar(sc2, ax=ax_t, label='epsilon')
            if not base_pts.empty:
                ax_t.scatter(
                    base_pts['vrate'], base_pts['cost'], marker='x', s=70, c='black', linewidths=1.0
                )
            ax_t.set_xlabel('Transformer violation rate (trajectory)')
            ax_t.set_ylabel('Trajectory total cost (EUR)')
            ax_t.set_title('Per-Trajectory Cost–Risk Cloud')
            ax_t.grid(alpha=0.35)
            from matplotlib.lines import Line2D
            legend_handles_traj = [
                Line2D([], [], marker='o', linestyle='None', color='black', markersize=6, label='DRCC trajectories'),
                Line2D([], [], marker='x', linestyle='None', color='black', markersize=7, label='stochastic trajectories'),
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
                             s=20, alpha=0.12, edgecolors='none')
            if not base_cloud.empty:
                ax_h.scatter(base_cloud['vrate'], base_cloud['cost'], marker='x', s=30,
                             c='black', alpha=0.12, linewidths=0.6)
            # Mean overlay (reuse style from mean frontier)
            drcc_mean = mean_df[mean_df['mode'] != 'stochastic']
            base_mean = mean_df[mean_df['mode'] == 'stochastic']
            if not drcc_mean.empty:
                eps_mean = drcc_mean['epsilon'].to_numpy(dtype=float)
                norm_m = plt.Normalize(vmin=np.nanmin(eps_mean), vmax=np.nanmax(eps_mean))
                cmap_m = plt.cm.viridis
                sc_m = ax_h.scatter(drcc_mean['trafo_violation_rate_mean'], drcc_mean['mean_cost_eur'],
                                    c=eps_mean, cmap=cmap_m, s=70, edgecolors='k', linewidths=0.4)
                fig_h.colorbar(sc_m, ax=ax_h, label='epsilon')
            if not base_mean.empty:
                ax_h.scatter(base_mean['trafo_violation_rate_mean'], base_mean['mean_cost_eur'],
                             marker='x', s=80, c='black', linewidths=1.2)
            for _, r in drcc_mean.iterrows():
                if r['epsilon'] is not None and np.isfinite(r['epsilon']):
                    ax_h.annotate(f"{r['epsilon']:.2f}", (r['trafo_violation_rate_mean'], r['mean_cost_eur']),
                                  textcoords='offset points', xytext=(4,4), fontsize=8)
            ax_h.set_xlabel('Transformer violation rate (trajectory / mean)')
            ax_h.set_ylabel('Total cost (EUR)')
            ax_h.set_title('Hybrid Cost–Risk Frontier (Cloud + Mean)')
            ax_h.grid(alpha=0.35)
            hybrid_path = os.path.join(RESULTS_DIR, FRONTIER_HYBRID_SCATTER_FIG)
            fig_h.savefig(hybrid_path, dpi=150)
            print(f"✓ Frontier hybrid scatter: {hybrid_path}")
        else:
            print('[INFO] Hybrid frontier scatter skipped (insufficient data).')

    # --- Transformer violation probability per timestep ---
    if PLOT_TRAFO_VIOLATION_TIME_PROFILE:
        # Correct per-timestep probability: for each (sample_id, t) take max loading across trafos; violation if > threshold.
        threshold_pct = 80.0
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
            tok = epsilon_token(eps)
            pq_path = os.path.join(RESULTS_DIR, 'v3_loading', f'trafo_loading_raw_epsilon_{tok}.parquet')
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
                    ax_tp.plot(t_axis, arr, linewidth=1.2, alpha=0.9, label=f"ε={lab}")
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
                for ax_h, (lab, mat_norm, cols) in zip(axes_h, data_mats):
                    im = ax_h.imshow(mat_norm, aspect='auto', cmap='magma', interpolation='nearest')
                    ax_h.set_title(f"Policy ({lab})")
                    ax_h.set_yticks(range(len(cols)))
                    ax_h.set_yticklabels(cols, fontsize=8)
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
                    fig_l.legend(handles, labels, loc='upper center', ncol=min(6, len(labels)), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
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
                ax.set_title(f"SoC envelope ({lab})")
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
    main()

