"""Hybrid analysis script: Uses v4 results but substitutes the v3 aggregated epsilon=0.10 result as the RT OFF variant.

Unorthodox integration: we treat the v3 aggregated epsilon 0.10 run as if it were
v4's rt_off variant for epsilon=0.10 for comparative plots.

We avoid modifying the original oos-analysis.py; this script reuses minimal logic.
"""
import os, json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Base directories (adjust if needed)
V4_DIR = 'v4_oos_agg_gaussian_studentt'  # aggregated v4 we compare
V3_SUB_DIR = 'v3_oos_agg_gaussian_studentt'
RESULTS_DIR = V4_DIR  # primary dir for plotting
HYBRID_TAG = 'rt_off'  # we overwrite this variant using v3 data for epsilon 0.10
TARGET_EPSILON = 0.10

# Output filenames (reuse names from main analysis for consistency)
SOC_ENV_FIG = 'hybrid_soc_envelopes.png'
SOC_FINAL_FIG = 'hybrid_soc_final_envelope.png'
SOC_FINAL_BOXPLOT_FIG = 'hybrid_soc_final_boxplot.png'
SOC_DAILY_BOXPLOT_FIG = 'hybrid_soc_daily_boxplot.png'
OVERVIEW_FIG = 'hybrid_overview.png'

DETERMINISTIC_LABEL = 'deterministic'

def epsilon_token(eps: float) -> str:
    return f"{eps:.2f}".replace('.', '_')

def _rt_display(tag: Optional[str]) -> str:
    if tag == 'rt_on': return 'RT ON'
    if tag == 'rt_off': return 'RT OFF'
    if tag == 'rt_unk': return 'RT ?'
    return ''

# Minimal reader with parquet fallback
def _read_parquet_or_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith('.parquet'):
            if os.path.exists(path):
                try:
                    return pd.read_parquet(path)
                except Exception:
                    csv_path = path[:-8] + 'csv'
                    if os.path.exists(csv_path):
                        return pd.read_csv(csv_path)
                    return None
        if path.lower().endswith('.csv') and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

# Load v4 meta for all epsilon variants we find (rt_on/off/unk)
# plus deterministic baseline; then override epsilon=0.10 rt_off with v3 aggregated meta + summary.

def load_v4_variants() -> List[Tuple[str, str, Dict, pd.DataFrame]]:
    variants: List[Tuple[str,str,Dict,pd.DataFrame]] = []
    # scan for v4 summary files in RESULTS_DIR
    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith('v4_summary_drcc_true_epsilon_'): continue
        if '_rt_' not in fname: continue  # only rt-suffixed
        summary_path = os.path.join(RESULTS_DIR, fname)
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue
        # derive epsilon & rt tag
        base = fname.replace('v4_summary_drcc_true_epsilon_', '')  # token_rt_on.csv
        token = base.split('_rt_')[0]
        rt_tag = 'rt_' + base.split('_rt_')[1].split('.')[0]
        try:
            eps = float(token.replace('_', '.'))
        except Exception:
            continue
        meta_name = f"v4_meta_drcc_true_epsilon_{token}_{rt_tag}.json"
        meta_path = os.path.join(RESULTS_DIR, meta_name)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path,'r',encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        label = f"{eps:.2f} ({_rt_display(rt_tag)})"
        variants.append((label, rt_tag, meta, df))
    return variants

# Deterministic baseline

def load_v4_deterministic() -> List[Tuple[str,str,Dict,pd.DataFrame]]:
    out = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith('v4_summary_drcc_false_'): continue
        if '_rt_' not in fname: continue
        summary_path = os.path.join(RESULTS_DIR, fname)
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue
        rt_tag = 'rt_' + fname.split('_rt_')[1].split('.')[0]
        meta_name = f"v4_meta_drcc_false_{rt_tag}.json"
        meta_path = os.path.join(RESULTS_DIR, meta_name)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path,'r',encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        label = f"{DETERMINISTIC_LABEL} ({_rt_display(rt_tag)})"
        out.append((label, rt_tag, meta, df))
    return out

# Load v3 epsilon 0.10 aggregated (no RT tag originally) and treat as rt_off

def load_v3_epsilon_variant() -> Optional[Tuple[str,str,Dict,pd.DataFrame]]:
    token = epsilon_token(TARGET_EPSILON)
    summary = os.path.join(V3_SUB_DIR, f"v3_summary_drcc_true_epsilon_{token}.csv")
    meta_path = os.path.join(V3_SUB_DIR, f"v3_meta_drcc_true_epsilon_{token}.json")
    if not os.path.exists(summary):
        return None
    try:
        df = pd.read_csv(summary)
    except Exception:
        return None
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path,'r',encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    # inject a faux rt_tag
    rt_tag = HYBRID_TAG
    meta['hybrid_source'] = 'v3_agg_gaussian_studentt'
    meta['rt_tag'] = rt_tag
    label = f"{TARGET_EPSILON:.2f} ({_rt_display(rt_tag)})"
    return (label, rt_tag, meta, df)

# Build a unified variant list with substitution

def build_hybrid_variants() -> List[Tuple[str,str,Dict,pd.DataFrame]]:
    v4 = load_v4_variants()
    det = load_v4_deterministic()
    hybrid_v3 = load_v3_epsilon_variant()
    out: List[Tuple[str,str,Dict,pd.DataFrame]] = []
    # Replace v4 epsilon 0.10 rt_off if present
    for label, rt_tag, meta, df in v4:
        if '0.10' in label and rt_tag == 'rt_off' and hybrid_v3 is not None:
            # substitute
            out.append(hybrid_v3)
        else:
            out.append((label, rt_tag, meta, df))
    out.extend(det)
    # If v4 lacked the rt_off 0.10 variant, append hybrid anyway for visibility
    if hybrid_v3 and not any(('0.10' in l and t=='rt_off') for l,t,_,_ in out):
        out.append(hybrid_v3)
    # Sort by epsilon order with deterministic first
    def base_key(label: str) -> Tuple[int,float,str]:
        if label.startswith(DETERMINISTIC_LABEL):
            return (0, -1.0, label)
        try:
            eps = float(label.split()[0])
            return (1, eps, label)
        except Exception:
            return (2, 999.0, label)
    out.sort(key=lambda x: base_key(x[0]))
    return out

# Simple overview figure: total_cost_eur vs label

def plot_overview(variants: List[Tuple[str,str,Dict,pd.DataFrame]]) -> None:
    labels = []
    costs = []
    for label, rt_tag, meta, df in variants:
        if 'total_cost_eur' in df.columns:
            costs.append(pd.to_numeric(df['total_cost_eur'], errors='coerce').mean())
            labels.append(label + ('*' if meta.get('hybrid_source') else ''))
    if not labels:
        print('[WARN] No cost data for overview.')
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(labels)+2), 4))
    ax.bar(np.arange(len(labels)), costs, color='#3182bd')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean total cost (EUR)')
    ax.set_title('Hybrid Overview (v3 substituted for 0.10 RT OFF) * indicates hybrid source')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, OVERVIEW_FIG), dpi=150)
    print(f"✓ Hybrid overview figure: {os.path.join(RESULTS_DIR, OVERVIEW_FIG)}")

# SoC final + daily boxplots using envelope (if only envelopes) or raw series if available (optional extension)
# For simplicity here we use envelopes for medians and final values; raw substitution could be added similarly.

def locate_soc_envelope(meta: Dict, base_dir: str) -> Optional[pd.DataFrame]:
    rel = meta.get('soc_envelope_file')
    if not rel: return None
    path = os.path.join(base_dir, rel)
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        if {'soc_p50'} <= set(df.columns):
            return df
    except Exception:
        return None
    return None

def plot_soc_envelopes(variants: List[Tuple[str,str,Dict,pd.DataFrame]]) -> None:
    envs: List[Tuple[str,pd.DataFrame]] = []
    for label, rt_tag, meta, df in variants:
        # decide base directory for envelope path: hybrid source uses v3 folder
        base_dir = V3_SUB_DIR if meta.get('hybrid_source') else RESULTS_DIR
        env_df = locate_soc_envelope(meta, base_dir)
        if env_df is not None:
            envs.append((label + ('*' if meta.get('hybrid_source') else ''), env_df))
    if not envs:
        print('[WARN] No SoC envelopes found.')
        return
    cols = len(envs)
    fig, ax_arr = plt.subplots(1, cols, figsize=(4*cols, 3), constrained_layout=True)
    if cols == 1: ax_arr = [ax_arr]
    final_rows = []
    for ax, (lab, env_df) in zip(ax_arr, envs):
        p05 = pd.to_numeric(env_df.get('soc_p05'), errors='coerce')
        p50 = pd.to_numeric(env_df.get('soc_p50'), errors='coerce')
        p95 = pd.to_numeric(env_df.get('soc_p95'), errors='coerce')
        t = np.arange(len(env_df))
        ax.fill_between(t, p05, p95, color='#c6dbef', alpha=0.6, label='5–95%')
        ax.plot(t, p50, color='#08519c', linewidth=1.4, label='Median')
        ax.set_ylim(0, 1.02)
        ax.set_title(f"SoC ({lab})")
        if ax == ax_arr[0]: ax.set_ylabel('SoC frac')
        ax.set_xlabel('t step'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        last = env_df.iloc[-1]
        final_rows.append((lab, float(last.get('soc_p05', np.nan)), float(last.get('soc_p50', np.nan)), float(last.get('soc_p95', np.nan))))
    fig.savefig(os.path.join(RESULTS_DIR, SOC_ENV_FIG), dpi=150)
    print(f"✓ Hybrid SoC envelopes: {os.path.join(RESULTS_DIR, SOC_ENV_FIG)}")
    if final_rows:
        labels = [r[0] for r in final_rows]
        med = np.array([r[2] for r in final_rows])
        lo = np.clip(med - np.array([r[1] for r in final_rows]), 0, None)
        hi = np.clip(np.array([r[3] for r in final_rows]) - med, 0, None)
        fig_f, ax_f = plt.subplots(figsize=(max(6, 0.6*len(final_rows)+2), 4))
        x = np.arange(len(final_rows))
        ax_f.bar(x, med, color='#3182bd', alpha=0.85)
        ax_f.errorbar(x, med, yerr=[lo, hi], fmt='none', ecolor='#08306b', capsize=4)
        ax_f.set_xticks(x); ax_f.set_xticklabels(labels, rotation=45, ha='right')
        ax_f.set_ylabel('Final SoC'); ax_f.set_ylim(0, 1.05); ax_f.grid(axis='y', alpha=0.3)
        fig_f.tight_layout(); fig_f.savefig(os.path.join(RESULTS_DIR, SOC_FINAL_FIG), dpi=150)
        print(f"✓ Hybrid final SoC summary: {os.path.join(RESULTS_DIR, SOC_FINAL_FIG)}")
        # Boxplot final step
        final_vals: List[np.ndarray] = []
        final_labels: List[str] = []
        for lab, env_df in envs:
            if env_df.empty: continue
            last = env_df.iloc[-1]
            if 'soc_p50' in last:
                final_vals.append(np.array([float(last['soc_p50'])]))
                final_labels.append(lab)
        if final_vals:
            fig_b, ax_b = plt.subplots(figsize=(max(6, 0.6*len(final_vals)+2), 4))
            ax_b.boxplot(final_vals, labels=final_labels, showfliers=False, patch_artist=True,
                         boxprops=dict(facecolor='#c6dbef', color='#08519c'), medianprops=dict(color='#08306b'))
            ax_b.set_ylabel('Final SoC'); ax_b.set_ylim(0,1.05); ax_b.grid(axis='y', alpha=0.3)
            plt.setp(ax_b.get_xticklabels(), rotation=45, ha='right')
            fig_b.tight_layout(); fig_b.savefig(os.path.join(RESULTS_DIR, SOC_FINAL_BOXPLOT_FIG), dpi=150)
            print(f"✓ Hybrid final SoC boxplot: {os.path.join(RESULTS_DIR, SOC_FINAL_BOXPLOT_FIG)}")

# Transformer loading comparison for substituted variant

def plot_transformer_loading(variants: List[Tuple[str,str,Dict,pd.DataFrame]]) -> None:
    # For simplicity, show max_trafo_loading_pct distribution per variant
    data = []
    labels = []
    for label, rt_tag, meta, df in variants:
        col = 'max_trafo_loading_pct'
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy()
            if arr.size:
                data.append(arr)
                labels.append(label + ('*' if meta.get('hybrid_source') else ''))
    if not data:
        print('[WARN] No transformer loading data found.')
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.8*len(data)+2), 4))
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#fee6ce', color='#e6550d'), medianprops=dict(color='#a63603'))
    ax.set_ylabel('Max transformer loading (%)')
    ax.set_title('Hybrid Transformer Loading (v3 substitution marked with *)')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'hybrid_trafo_loading_boxplot.png'), dpi=150)
    print(f"✓ Hybrid transformer loading boxplot: {os.path.join(RESULTS_DIR, 'hybrid_trafo_loading_boxplot.png')}")

# Daily SoC boxplot using envelopes median distribution across all timesteps (approximation)

def plot_daily_soc_boxplot(variants: List[Tuple[str,str,Dict,pd.DataFrame]]) -> None:
    vals = []
    labs = []
    for label, rt_tag, meta, df in variants:
        base_dir = V3_SUB_DIR if meta.get('hybrid_source') else RESULTS_DIR
        env_df = locate_soc_envelope(meta, base_dir)
        if env_df is None or env_df.empty: continue
        med = pd.to_numeric(env_df.get('soc_p50'), errors='coerce').dropna().to_numpy()
        if med.size:
            vals.append(med)
            labs.append(label + ('*' if meta.get('hybrid_source') else ''))
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.8*len(vals)+2), 4))
    ax.boxplot(vals, labels=labs, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#e5f5e0', color='#31a354'), medianprops=dict(color='#006d2c'))
    ax.set_ylabel('Median SoC distribution (all timesteps)')
    ax.set_title('Hybrid Whole-Day SoC (median envelope values)')
    ax.set_ylim(0,1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, SOC_DAILY_BOXPLOT_FIG), dpi=150)
    print(f"✓ Hybrid whole-day SoC boxplot: {os.path.join(RESULTS_DIR, SOC_DAILY_BOXPLOT_FIG)}")


def main():
    variants = build_hybrid_variants()
    print(f"[hybrid] Loaded {len(variants)} variants (v3 substitution applied where possible).")
    plot_overview(variants)
    plot_soc_envelopes(variants)
    plot_transformer_loading(variants)
    plot_daily_soc_boxplot(variants)
    print('[hybrid] Complete.')

if __name__ == '__main__':
    main()
