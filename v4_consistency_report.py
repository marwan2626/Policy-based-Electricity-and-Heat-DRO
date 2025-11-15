import os, math
import pandas as pd
import numpy as np

# Configuration
OUTDIR_BASE = 'v4_oos'
EPSILONS = [0.10, 0.15, 0.20]
INCLUDE_DETERMINISTIC = True  # also include drcc_false baseline
SUMMARY_PATTERN = 'v4_summary_drcc_true_epsilon_{token}.csv'
REPORT_CSV = 'v4_consistency_report.csv'
REPORT_MD = 'v4_consistency_report.md'

# Threshold parameters (relative to peak BESS command per trajectory)
MAE_PEAK_FRAC_THRESHOLD = 0.25   # 25% of peak command
RMSE_PEAK_FRAC_THRESHOLD = 0.30  # 30% of peak command
BESS_COST_REDUCTION_PCT_THRESHOLD = 12.0    # % reduction vs shadow beyond which controller may outperform assumptions
PV_CURT_COST_REDUCTION_PCT_THRESHOLD = 12.0
TOTAL_COST_REDUCTION_PCT_THRESHOLD = 12.0
CORR_THRESHOLD = 0.90

METRIC_COLUMNS = [
    'shadow_bess_cmd_lambda_mae_mw',
    'shadow_bess_cmd_lambda_rmse_mw',
    'shadow_bess_cmd_lambda_corr',
    'shadow_pv_curt_cmd_lambda_mae_mw',
    'shadow_pv_curt_cmd_lambda_rmse_mw',
    'bess_throughput_cost_reduction_pct',
    'pv_curtail_cost_reduction_pct',
    'total_proxy_cost_reduction_pct'
]

records = []

def _add_row(df, eps, mode_label):
    n = len(df)
    if 'recourse_bess_cmd_peak_mw' in df.columns and 'shadow_bess_cmd_lambda_mae_mw' in df.columns:
        peak_arr = df['recourse_bess_cmd_peak_mw'].replace(0, np.nan).to_numpy(dtype=float)
        mae_frac = df.get('shadow_bess_cmd_lambda_mae_mw').to_numpy(dtype=float) / peak_arr
        rmse_frac = df.get('shadow_bess_cmd_lambda_rmse_mw').to_numpy(dtype=float) / peak_arr
    else:
        peak_arr = np.full(n, np.nan)
        mae_frac = np.full(n, np.nan)
        rmse_frac = np.full(n, np.nan)

    def stats(col):
        arr = df[col].to_numpy(dtype=float) if col in df.columns else np.array([])
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return dict(mean=np.nan, median=np.nan, p95=np.nan)
        return dict(mean=float(np.mean(arr)), median=float(np.median(arr)), p95=float(np.percentile(arr, 95)))

    stat_map = {c: stats(c) for c in METRIC_COLUMNS}

    def frac_fail(arr, threshold, mode='gt'):
        arr = np.array(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        if mode == 'gt':
            return float(np.mean(arr > threshold))
        else:
            return float(np.mean(arr < threshold))

    frac_mae_gt = frac_fail(mae_frac, MAE_PEAK_FRAC_THRESHOLD, 'gt')
    frac_rmse_gt = frac_fail(rmse_frac, RMSE_PEAK_FRAC_THRESHOLD, 'gt')
    frac_corr_low = frac_fail(df['shadow_bess_cmd_lambda_corr'] if 'shadow_bess_cmd_lambda_corr' in df.columns else [np.nan], CORR_THRESHOLD, 'lt')
    frac_cost_excess_bess = frac_fail(df['bess_throughput_cost_reduction_pct'] if 'bess_throughput_cost_reduction_pct' in df.columns else [np.nan], BESS_COST_REDUCTION_PCT_THRESHOLD, 'gt')
    frac_cost_excess_pv = frac_fail(df['pv_curtail_cost_reduction_pct'] if 'pv_curtail_cost_reduction_pct' in df.columns else [np.nan], PV_CURT_COST_REDUCTION_PCT_THRESHOLD, 'gt')
    frac_cost_excess_total = frac_fail(df['total_proxy_cost_reduction_pct'] if 'total_proxy_cost_reduction_pct' in df.columns else [np.nan], TOTAL_COST_REDUCTION_PCT_THRESHOLD, 'gt')

    records.append({
        'mode': mode_label,
        'epsilon': eps,
        'n_trajectories': n,
        'mae_frac_mean': float(np.nanmean(mae_frac)) if np.any(np.isfinite(mae_frac)) else np.nan,
        'mae_frac_median': float(np.nanmedian(mae_frac)) if np.any(np.isfinite(mae_frac)) else np.nan,
        'mae_frac_p95': float(np.nanpercentile(mae_frac, 95)) if np.any(np.isfinite(mae_frac)) else np.nan,
        'rmse_frac_mean': float(np.nanmean(rmse_frac)) if np.any(np.isfinite(rmse_frac)) else np.nan,
        'rmse_frac_median': float(np.nanmedian(rmse_frac)) if np.any(np.isfinite(rmse_frac)) else np.nan,
        'rmse_frac_p95': float(np.nanpercentile(rmse_frac, 95)) if np.any(np.isfinite(rmse_frac)) else np.nan,
        'corr_mean': stat_map['shadow_bess_cmd_lambda_corr']['mean'],
        'corr_median': stat_map['shadow_bess_cmd_lambda_corr']['median'],
        'corr_p95': stat_map['shadow_bess_cmd_lambda_corr']['p95'],
        'bess_cost_reduction_mean': stat_map['bess_throughput_cost_reduction_pct']['mean'],
        'pv_curt_cost_reduction_mean': stat_map['pv_curtail_cost_reduction_pct']['mean'],
        'total_cost_reduction_mean': stat_map['total_proxy_cost_reduction_pct']['mean'],
        'frac_mae_frac_gt_threshold': frac_mae_gt,
        'frac_rmse_frac_gt_threshold': frac_rmse_gt,
        'frac_corr_lt_threshold': frac_corr_low,
        'frac_bess_cost_reduction_gt_threshold': frac_cost_excess_bess,
        'frac_pv_curt_cost_reduction_gt_threshold': frac_cost_excess_pv,
        'frac_total_cost_reduction_gt_threshold': frac_cost_excess_total,
    })

for eps in EPSILONS:
    token = f"{eps:.2f}".replace('.', '_')
    fname = SUMMARY_PATTERN.format(token=token)
    path = os.path.join(OUTDIR_BASE, fname)
    if not os.path.exists(path):
        print(f"[WARN] Missing summary for epsilon={eps:.2f} -> {path}; skipping.")
        continue
    df = pd.read_csv(path)
    _add_row(df, eps, 'drcc_true')

if INCLUDE_DETERMINISTIC:
    det_path = os.path.join(OUTDIR_BASE, 'v4_summary_drcc_false.csv')
    if os.path.exists(det_path):
        df_det = pd.read_csv(det_path)
        _add_row(df_det, np.nan, 'drcc_false')
    else:
        print('[WARN] Deterministic summary file not found; skipping drcc_false.')

report_df = pd.DataFrame(records)
report_df.to_csv(REPORT_CSV, index=False)

# Markdown summary
lines = ["# V4 Consistency Report", "", f"Thresholds: MAE/peak <= {MAE_PEAK_FRAC_THRESHOLD:.2f}, RMSE/peak <= {RMSE_PEAK_FRAC_THRESHOLD:.2f}, Corr >= {CORR_THRESHOLD:.2f}, Cost reductions <= {TOTAL_COST_REDUCTION_PCT_THRESHOLD:.1f}% (95th).", ""]
for _, row in report_df.iterrows():
    mode = str(row['mode'])
    eps = row['epsilon']
    title_eps = f"{eps:.2f}" if np.isfinite(eps) else "deterministic"
    lines.append(f"## Mode {mode} | Epsilon {title_eps}")
    lines.append(f"Trajectories: {int(row['n_trajectories'])}")
    lines.append(f"BESS MAE/Peak median={row['mae_frac_median']:.4f} p95={row['mae_frac_p95']:.4f}")
    lines.append(f"BESS RMSE/Peak median={row['rmse_frac_median']:.4f} p95={row['rmse_frac_p95']:.4f}")
    lines.append(f"BESS corr median={row['corr_median']:.4f} p95={row['corr_p95']:.4f}")
    lines.append(f"Mean cost reductions: BESS={row['bess_cost_reduction_mean']:.2f}% PV={row['pv_curt_cost_reduction_mean']:.2f}% Total={row['total_cost_reduction_mean']:.2f}%")
    lines.append(f"Fraction failing thresholds: MAE={row['frac_mae_frac_gt_threshold']:.3f} RMSE={row['frac_rmse_frac_gt_threshold']:.3f} CorrLow={row['frac_corr_lt_threshold']:.3f} TotalCostExcess={row['frac_total_cost_reduction_gt_threshold']:.3f}")
    lines.append("")

with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"[report] Wrote consistency report CSV -> {REPORT_CSV}")
print(f"[report] Wrote consistency markdown -> {REPORT_MD}")
