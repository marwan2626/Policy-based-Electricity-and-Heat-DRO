#!/usr/bin/env python3
"""
Generate plots from already exported/analyzed data.

First plot: identical to the oos-overview all-in total cost bar plot (second subplot)
for the aggregated Gaussian + Student-t solution.

It reads the precomputed summary CSV written by oos-analysis in
v4_oos_agg_gaussian_studentt/oos_overview_summary.csv and reproduces the
bar plot with the same labels and % delta annotations vs deterministic
(if present).

Usage:
  python generate_plots.py
  python generate_plots.py --src v4_oos_agg_gaussian_studentt --out oos_overview_total_cost.png
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.colors as mcolors

DEFAULT_SRC_DIR = "v4_oos_agg_gaussian_studentt"
DEFAULT_SUMMARY_NAME = "oos_overview_summary.csv"
DEFAULT_OUT_NAME = "fig_RT-costs.pgf"
DEFAULT_EXPORT_DIR = "export pgf"
VIOLIN_OUT_NAME = "fig_violin.pgf"
THERMAL_STORAGE_OUT_NAME = "fig_thermal_storage_operation_area.pgf"
TRAFO_VIOLATION_OUT_NAME = "fig_Trafo-constraint.pgf"
FINAL_SOC_OUT_NAME = "fig_soc-final.pgf"
OVERLOAD_RT_ON_OFF_OUT_NAME = "fig_overload_energy.pgf"
MAX_TRAFO_EPSILON_OUT_NAME = "fig_trafo_loading_vs_epsilon.pgf"
DA_COST_EPSILON_OUT_NAME = "fig_total_cost_vs_epsilon.pgf"

# Specific injection for transformer violation probability plot (match cost plot naming)
TRAFO_INJECT_LABEL = "Multi-Energy\nCo-optimization"  # matches INJECT_LABEL defined below
TRAFO_INJECT_VALUE = 0.0  # user specified value

# Customization: drop specific label(s) and inject a fixed-cost bar
DROP_LABELS = {"0.15 (RT ON)"}
INJECT_LABEL = "Multi-Energy\nCo-optimization"
INJECT_COST = 684.3104378123563

# Match global style
USE_A4 = True  # Toggle like export_select_pgf_plots
DEFAULT_WIDTH_CM = 13.0 if USE_A4 else 10.89
DEFAULT_FONT_SIZE = 11 if USE_A4 else 8

# Derived sizing constants based on paper size mode
# Figure width and font size are controlled solely by USE_A4
SOC_ASPECT = 0.774  # height = width * aspect (match fig_soc-final/fig_Trafo-constraint)
LINE_ASPECT = 0.6667  # height = width * aspect (2:3) for line plots
BAR_ASPECT = 0.5  # height = width * aspect (1:2) for summary bar plot

# Shared color palette aligned with export_select_pgf_plots
ELECTRIC_BLUE = '#3445a0'
LIGHT_BLUE_FILL = '#9ecae1'
GAS_GREEN = '#3a9d6c'
GAS_GREEN_FILL = '#b2df8a'
HEAT_RED = '#d82e1d'
HEAT_RED_FILL = '#fb6a4a'

def latex_mix_with_white(hex_color, percent):
    """
    percent = 65  -> LaTeX !65
    """
    rgb = mcolors.to_rgb(hex_color)
    a = percent / 100.0
    return tuple(a*c + (1-a) for c in rgb)

def latex_mix_with_black(color, percent, other="black"):
    """
    Matplotlib equivalent of xcolor:
    color!percent!other
    """
    rgb1 = mcolors.to_rgb(color)
    rgb2 = mcolors.to_rgb(other)
    a = percent / 100.0
    return tuple(a*c1 + (1-a)*c2 for c1, c2 in zip(rgb1, rgb2))

GAS_GREEN_65 = latex_mix_with_white(GAS_GREEN, 65)
HEAT_RED_65 = latex_mix_with_white(HEAT_RED, 65)
ELECTRIC_BLUE_65 = latex_mix_with_white(ELECTRIC_BLUE, 65)

GAS_GREEN_80 = latex_mix_with_black(GAS_GREEN, 80)
HEAT_RED_80 = latex_mix_with_black(HEAT_RED, 80)
ELECTRIC_BLUE_80 = latex_mix_with_black(ELECTRIC_BLUE, 80)

def _cm_to_inch(cm: float) -> float:
    return cm / 2.54

def _configure_pgf(texsystem: str = "pdflatex", use_sfmath: bool = True) -> None:
    # Ensure LaTeX renders all text in sans-serif consistently
    preamble_lines = []
    if use_sfmath:
        preamble_lines.append(r"\usepackage{sfmath}")
        preamble_lines.append(r"\renewcommand{\familydefault}{\sfdefault}")
    # Provide no-op definitions for mathtext commands emitted by Matplotlib's PGF
    # This avoids Undefined control sequence errors (e.g., \mathdefault) while preserving appearance.
    preamble_lines.append(r"\providecommand{\mathdefault}[1]{#1}")
    preamble_lines.append(r"\providecommand{\mathregular}[1]{#1}")
    preamble = "\n".join(preamble_lines)
    mpl.rcParams.update({
        "pgf.texsystem": texsystem,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "DejaVu Sans",
            "CMU Sans Serif",
            "Computer Modern Sans Serif",
            "Arial",
        ],
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.formatter.use_mathtext": False,
        "text.latex.preamble": preamble,
        "pgf.preamble": preamble,
        "font.size": DEFAULT_FONT_SIZE,
    })

def _save_pgf(fig: plt.Figure, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    return out_path

def _save_pgf_and_png(fig: plt.Figure, out_path: str, png_dpi: int = 300) -> tuple[str, str]:
    """Save PGF and a sidecar PNG for quick visual debugging."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    base, _ = os.path.splitext(out_path)
    png_path = base + '.png'
    try:
        fig.savefig(png_path, dpi=int(png_dpi), bbox_inches='tight', pad_inches=0.02)
    except Exception as e:
        print(f"[WARN] Failed to save PNG for {out_path}: {e}")
    return out_path, png_path

# Tick formatter helpers to avoid serif math and \mathdefault in PGF
def _force_sans_ticks(ax: plt.Axes, which: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: rf"\textsf{{{v:g}}}"))
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: (lambda s: rf"\textsf{{{s}}}")(
            f"{int(v) if abs(v-round(v))<1e-9 else v:g}"
        )))

def _force_plain_ticks(ax: plt.Axes, which: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v) if abs(v-round(v))<1e-9 else v:g}"))

# Parameters to compute transformer overload energy per-sample from loading parquet
OVERLOAD_THRESHOLD_PCT = float(os.getenv('V4_VIOL_THRESHOLD_PCT', '80.09'))
RATED_TRAFO_MVA = 0.5
STEP_HOURS = 0.25  # 15-minute steps


def _save_png_and_pdf(fig: plt.Figure, out_path: str, dpi_png: int = 150) -> str:
    """Save a figure to PNG at `out_path` and alongside as PDF.

    Returns the PDF path.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # Use tight bounding boxes and minimal padding for crisper exports
    fig.savefig(out_path, dpi=dpi_png, bbox_inches='tight', pad_inches=0.02)
    base, _ = os.path.splitext(out_path)
    pdf_path = base + '.pdf'
    try:
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    except Exception:
        fig.savefig(pdf_path, dpi=dpi_png, bbox_inches='tight', pad_inches=0.02)
    return pdf_path


def load_summary(src_dir: str) -> pd.DataFrame:
    path = os.path.join(src_dir, DEFAULT_SUMMARY_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Summary CSV not found: {path}. Run oos-analysis aggregation or set --src to an existing folder.")
    df = pd.read_csv(path)
    # Ensure required columns exist (use fallbacks if needed)
    if 'all_in_total_cost_mean' not in df.columns:
        # fallback: try to compute from DA + total_rt
        if {'da_total_cost_eur', 'total_rt_cost_mean'} <= set(df.columns):
            try:
                df['all_in_total_cost_mean'] = pd.to_numeric(df['da_total_cost_eur'], errors='coerce') \
                                               + pd.to_numeric(df['total_rt_cost_mean'], errors='coerce')
            except Exception:
                pass
    return df


def _read_parquet_or_csv(path: str) -> pd.DataFrame | None:
    """Try parquet first if extension matches; otherwise CSV fallback.
    Returns DataFrame or None.
    """
    try:
        if os.path.exists(path):
            if path.lower().endswith('.parquet'):
                try:
                    return pd.read_parquet(path)
                except Exception:
                    csv_path = path[:-8] + 'csv'
                    if os.path.exists(csv_path):
                        try:
                            return pd.read_csv(csv_path)
                        except Exception:
                            return None
                    return None
            if path.lower().endswith('.csv'):
                try:
                    return pd.read_csv(path)
                except Exception:
                    return None
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


def _load_loading_distribution_from_meta(meta_path: str, base_dir: str) -> np.ndarray:
    """Load per-(sample_id,t) max transformer loading (%) as 1-D array from meta JSON path.
    Returns empty array if unavailable.
    """
    try:
        if not os.path.exists(meta_path):
            print(f"[WARN] Meta file not found for violin: {meta_path}")
            return np.array([])
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # Try multiple possible keys/structures
        rel = None
        for k in ['trafo_loading_file', 'transformer_loading_file', 'trafo_loading_parquet', 'trafo_loading_csv']:
            if k in meta and meta.get(k):
                rel = meta.get(k)
                break
        # If list of files provided, take the first
        if rel is None and isinstance(meta.get('trafo_loading_files'), list) and meta.get('trafo_loading_files'):
            rel = meta.get('trafo_loading_files')[0]
        if not rel:
            # Fallback: try to infer from epsilon/rt tag in meta filename
            # Expect meta path like v4_meta_drcc_true_epsilon_0_10_rt_on.json
            try:
                name = os.path.basename(meta_path)
                if 'epsilon_' in name and '_rt_' in name:
                    token = name.split('epsilon_', 1)[1].split('.json', 1)[0]
                    base = f"trafo_loading_{token}.parquet"
                    cand = os.path.join(base_dir, base)
                    if not os.path.exists(cand):
                        base = f"trafo_loading_{token}.csv"
                        cand = os.path.join(base_dir, base)
                    rel = cand if os.path.exists(cand) else None
            except Exception:
                rel = None
        if not rel:
            print(f"[WARN] No transformer loading path in meta: {meta_path}")
            return np.array([])
        abs_path = os.path.join(base_dir, str(rel).replace('/', os.sep))
        if not os.path.exists(abs_path):
            if os.path.exists(str(rel)):
                abs_path = str(rel)
            else:
                print(f"[WARN] Transformer loading file not found: {abs_path}")
                return np.array([])
        pdf = _read_parquet_or_csv(abs_path)
        if pdf is None:
            print(f"[WARN] Failed to read loading data (parquet/csv): {abs_path}")
            return np.array([])
        must = {'sample_id','t','trafo_index','loading_pct'}
        if not must <= set(pdf.columns):
            print(f"[WARN] Loading data missing required columns in {abs_path}; found: {list(pdf.columns)[:6]}...")
            return np.array([])
        grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
        arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
        arr = arr[np.isfinite(arr)]
        return arr
    except Exception as e:
        print(f"[WARN] Unexpected error loading violin data from {meta_path}: {e}")
        return np.array([])


def plot_violin_rt_on_vs_off(src_dir: str, out_path: str) -> str:
    """Recreate the focused ε=0.10 RT ON vs RT OFF split-violin plot from aggregated data."""
    # Locate meta files for epsilon 0.10
    meta_on = os.path.join(src_dir, 'v4_meta_drcc_true_epsilon_0_10_rt_on.json')
    meta_off = os.path.join(src_dir, 'v4_meta_drcc_true_epsilon_0_10_rt_off.json')
    vals_on = _load_loading_distribution_from_meta(meta_on, src_dir)
    vals_off = _load_loading_distribution_from_meta(meta_off, src_dir)

    # Prepare figure (match export_select_pgf_plots narrow violin sizing)
    _configure_pgf()
    width_cm = DEFAULT_WIDTH_CM / 1.9
    # Reduce height by 25% while keeping width the same (2.0 -> 1.5)
    fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * 1.2)))
    # Ensure bottom margin is sufficient for multi-line x tick labels
    try:
        fig.subplots_adjust(bottom=0.18)
    except Exception:
        pass
    x0 = 1.0
    if vals_on.size == 0 and vals_off.size == 0:
        ax.text(0.5, 0.5, 'No ε=0.10 RT data', ha='center', va='center', transform=ax.transAxes,
                color='gray')
        # Use a symmetric window and center tick labels in each half
        ax.set_xlim(x0 - 0.5, x0 + 0.5)
        xmin, xmax = ax.get_xlim()
        left_center = (xmin + x0) / 2.0
        right_center = (x0 + xmax) / 2.0
        ax.set_xticks([left_center, right_center])
        ax.set_xticklabels(["No Recourse,\nDRCC $\\varepsilon$ = 0.10", "DRCC $\\varepsilon$ = 0.10"])
    else:
        vals_list = []
        side_tags = []
        if vals_off.size: vals_list.append(vals_off); side_tags.append('off')
        if vals_on.size: vals_list.append(vals_on); side_tags.append('on')
        # Determine y-limits from available data
        stacked = np.concatenate(vals_list) if vals_list else np.array([])
        y_min = float(np.nanmin(stacked)) if stacked.size else 0.0
        y_max = float(np.nanmax(stacked)) if stacked.size else 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = 0.0, 1.0
        pad = 0.02 * (y_max - y_min + 1e-9)
        y_min -= pad; y_max += pad
        # Trim upper axis at 100%
        y_max = min(y_max, 100.0)
        if not np.isfinite(y_min) or y_min >= y_max:
            y_min = max(0.0, y_max - 1.0)
        # Ensure the bottom includes 0 so a '0' tick is visible
        y_min = 0.0
        ax.set_ylim(y_min, y_max)
        # Keep symmetric window and slightly narrower violins
        ax.set_xlim(x0 - 0.5, x0 + 0.5)
        vp = ax.violinplot(vals_list, positions=[x0 for _ in vals_list], showmeans=False, showmedians=False, showextrema=False, widths=0.6)
        from matplotlib.patches import Rectangle, Patch
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        left_clip = Rectangle((xmin, ymin), width=(x0 - xmin), height=(ymax - ymin), transform=ax.transData)
        right_clip = Rectangle((x0, ymin), width=(xmax - x0), height=(ymax - ymin), transform=ax.transData)
        for body, tag in zip(vp['bodies'], side_tags):
            if tag == 'off':
                body.set_facecolor(ELECTRIC_BLUE_65); body.set_edgecolor(ELECTRIC_BLUE_80); body.set_alpha(1.0); body.set_clip_path(left_clip)
            else:
                body.set_facecolor(HEAT_RED_65); body.set_edgecolor(HEAT_RED_80);body.set_alpha(1.0); body.set_clip_path(right_clip)
        # Medians as short horizontal lines
        if vals_off.size:
            m_off = float(np.nanmedian(vals_off)); ax.plot([x0 - 0.21, x0], [m_off, m_off], color=ELECTRIC_BLUE_80, linewidth=2)
        if vals_on.size:
            m_on = float(np.nanmedian(vals_on)); ax.plot([x0, x0 + 0.21], [m_on, m_on], color=HEAT_RED_80, linewidth=2)
        # Place left/right labels centered under respective halves
        left_center = (xmin + x0) / 2.0
        right_center = (x0 + xmax) / 2.0
        ax.set_xticks([left_center, right_center])
        ax.set_xticklabels(["No Recourse,\nDRCC $\\varepsilon$ = 0.10", "DRCC $\\varepsilon$ = 0.10"])
        ax.axvline(x0, color='black', linewidth=0.9, zorder=3)

    ax.set_ylabel('Transformer Loading (%)')
    ax.tick_params(axis='y')
    ax.grid(axis='y', alpha=0.3)
    # Ensure sans/plain numeric ticks to avoid serif/math
    try:
        _force_plain_ticks(ax, which='y')
    except Exception:
        pass
    fig.tight_layout()
    pgf_path, png_path = _save_pgf_and_png(fig, out_path)
    print(f"✓ Violin RT ON vs OFF saved: {pgf_path} (+ PNG)")
    return out_path


def _compute_overload_energy_kwh_per_sample(parquet_path: str,
                                            threshold_pct: float = OVERLOAD_THRESHOLD_PCT,
                                            rated_mva: float = RATED_TRAFO_MVA,
                                            step_hours: float = STEP_HOURS) -> float:
    try:
        pdf = _read_parquet_or_csv(parquet_path)
    except Exception:
        pdf = None
    if pdf is None:
        return float('nan')
    must = {'sample_id', 't', 'trafo_index', 'loading_pct'}
    if not must <= set(pdf.columns):
        return float('nan')
    lp = pd.to_numeric(pdf['loading_pct'], errors='coerce').to_numpy()
    mask = np.isfinite(lp) & (lp > float(threshold_pct))
    if not np.any(mask):
        return 0.0
    excess_pct = lp[mask] - float(threshold_pct)
    excess_mva = (excess_pct / 100.0) * float(rated_mva)
    total_mvah = float(np.sum(excess_mva) * float(step_hours))
    try:
        n_samples = int(pd.to_numeric(pdf['sample_id'], errors='coerce').dropna().nunique())
    except Exception:
        n_samples = 1000
    n_samples = n_samples if n_samples > 0 else 1000
    total_kwh_per_sample = (total_mvah * 1000.0) / float(n_samples)
    return total_kwh_per_sample


def plot_overload_energy_rt_on_vs_off(src_dir: str, out_path: str) -> str:
    """Bar plot comparing total transformer overload energy (kWh per sample)
    for ε=0.10 RT OFF vs RT ON. Matches violin plot dimensions.
    """
    tok = '0_10'
    meta_on = os.path.join(src_dir, f'v4_meta_drcc_true_epsilon_{tok}_rt_on.json')
    meta_off = os.path.join(src_dir, f'v4_meta_drcc_true_epsilon_{tok}_rt_off.json')

    def _value_from_meta(mp: str) -> float:
        try:
            if not os.path.exists(mp):
                return float('nan')
            with open(mp, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            rel = meta.get('trafo_loading_file') if isinstance(meta, dict) else None
            if not rel:
                return float('nan')
            pq = os.path.join(src_dir, rel.replace('/', os.sep))
            if not os.path.exists(pq) and os.path.exists(rel):
                pq = rel
            if not os.path.exists(pq):
                return float('nan')
            return _compute_overload_energy_kwh_per_sample(pq)
        except Exception:
            return float('nan')

    val_off = _value_from_meta(meta_off)
    val_on = _value_from_meta(meta_on)

    # If both missing, try aggregated CSV fallback
    if not (np.isfinite(val_off) or np.isfinite(val_on)):
        csv_path = os.path.join(src_dir, 'overload_energy_compare_010_rt_on_vs_off.csv')
        try:
            if os.path.exists(csv_path):
                cdf = pd.read_csv(csv_path)
                if {'label', 'overload_energy_kwh_per_sample'} <= set(cdf.columns):
                    for _, row in cdf.iterrows():
                        lab = str(row['label']).lower()
                        v = float(pd.to_numeric(pd.Series([row['overload_energy_kwh_per_sample']]), errors='coerce').iloc[0])
                        if 'without' in lab or 'rt off' in lab:
                            val_off = v
                        if 'with' in lab or 'rt on' in lab:
                            val_on = v
        except Exception:
            pass

    # Prepare figure with same size as violin plot (narrow width)
    _configure_pgf()
    width_cm = DEFAULT_WIDTH_CM / 1.9
    # Reduce height by 25% while keeping width the same (2.0 -> 1.5)
    fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * 1.2)))
    labels = []
    values = []
    colors = []
    edges = []
    if np.isfinite(val_off):
        labels.append("No Recourse,\nDRCC $\\varepsilon$ = 0.10")
        values.append(val_off)
        colors.append(ELECTRIC_BLUE_65)
        edges.append(ELECTRIC_BLUE_80)
    if np.isfinite(val_on):
        labels.append("DRCC $\\varepsilon$ = 0.10")
        values.append(val_on)
        colors.append(HEAT_RED_65)
        edges.append(HEAT_RED_80)

    if not values:
        ax.text(0.5, 0.5, 'No ε=0.10 RT data', ha='center', va='center', transform=ax.transAxes,
                color='gray')
        fig.tight_layout()
        pgf_path, png_path = _save_pgf_and_png(fig, out_path)
        print(f"✓ Overload energy RT ON vs OFF saved (empty): {pgf_path} (+ PNG)")
        return out_path

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor=edges, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Overload Energy (KWh)')
    ax.tick_params(axis='y')
    ax.grid(axis='y', alpha=0.3)
    # Add labels inside bars when possible to avoid clipping
    ymax = max(values) if values else 1.0
    pad = 0.02 * ymax
    for rect, val in zip(bars, values):
        h = rect.get_height()
        inside_y = h - pad
        if inside_y > 0:
            ax.text(
                rect.get_x() + rect.get_width()/2,
                inside_y,
                f"{val:.2f}",
                ha='center', va='top', color='black'
            )
        else:
            ax.text(
                rect.get_x() + rect.get_width()/2,
                h + pad,
                f"{val:.2f}",
                ha='center', va='bottom', color='black', clip_on=False
            )
    # Slight headroom to ensure small above-bar labels never clip
    try:
        ax.margins(y=0.05)
    except Exception:
        pass
    # Force sans/plain numeric ticks on y
    try:
        _force_sans_ticks(ax, which='y')
    except Exception:
        pass

    fig.tight_layout()
    pgf_path, png_path = _save_pgf_and_png(fig, out_path)
    plt.close(fig)
    print(f"✓ Overload energy RT ON vs OFF saved: {pgf_path} (+ PNG)")
    return out_path


def plot_max_trafo_loading_vs_epsilon_v2(out_path: str, search_dir: str | None = None) -> str:
    """Two-line plot vs epsilon (x) of max transformer loading (%) (y) from v2 results CSVs.

    - Searches for files named like: `dso_model_v2_results_drcc_true_epsilon_0_XX_rt_on.csv` and `_rt_off.csv`
    - For each file, computes the maximum across all `transformer_*_loading_pct` columns and all timesteps
    - Plots two lines: RT OFF (blue) and RT ON (red). Figure size matches violin (4.0 x 4.2)
    """
    try:
        base = search_dir if search_dir else os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(base):
            base = os.getcwd()

        def _parse_eps(fname: str) -> float | None:
            # Expect token 'epsilon_0_10_' etc.
            s = fname
            if 'epsilon_' not in s:
                return None
            try:
                part = s.split('epsilon_', 1)[1]
                part = part.split('_rt', 1)[0]
                eps_str = part.replace('_', '.')
                return float(eps_str)
            except Exception:
                return None

        def _is_on(fname: str) -> bool:
            return 'rt_on' in fname.lower()

        def _is_off(fname: str) -> bool:
            return 'rt_off' in fname.lower()

        files = [f for f in os.listdir(base) if f.endswith('.csv') and f.startswith('dso_model_v2_results_drcc_true_epsilon_')]
        vals_on: dict[float, float] = {}
        vals_off: dict[float, float] = {}

        for f in files:
            eps = _parse_eps(f)
            if eps is None:
                continue
            full = os.path.join(base, f)
            try:
                df = pd.read_csv(full)
            except Exception:
                continue
            # Identify transformer loading columns
            cols = [c for c in df.columns if ('transformer' in c.lower() and 'loading' in c.lower() and 'pct' in c.lower())]
            if not cols:
                # Fallback to the common single column name
                if 'transformer_0_loading_pct' in df.columns:
                    cols = ['transformer_0_loading_pct']
            if not cols:
                continue
            try:
                sub = df[cols].apply(pd.to_numeric, errors='coerce')
                max_val = float(np.nanmax(sub.to_numpy()))
            except Exception:
                continue
            if not np.isfinite(max_val):
                continue
            if _is_on(f):
                # If duplicate eps appears, keep the maximum
                vals_on[eps] = max(max_val, vals_on.get(eps, -np.inf))
            elif _is_off(f):
                vals_off[eps] = max(max_val, vals_off.get(eps, -np.inf))

        # Prepare data for plotting (descending epsilon order)
        xs_on = sorted(vals_on.keys(), reverse=True)
        ys_on = [vals_on[x] for x in xs_on]
        xs_off = sorted(vals_off.keys(), reverse=True)
        ys_off = [vals_off[x] for x in xs_off]

        # 3:2 aspect ratio (width:height) with fixed height 4.2 -> width 6.3
        xticks = sorted(set(xs_on) | set(xs_off))  # ascending, we'll invert axis next
        _configure_pgf()
        width_cm = DEFAULT_WIDTH_CM
        fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * LINE_ASPECT)))
        if xs_off:
            ax.plot(
                xs_off, ys_off,
                color=ELECTRIC_BLUE, linewidth=2.0,
                marker='o', markerfacecolor='none', markeredgecolor=ELECTRIC_BLUE,
                label='DRCC without Recourse')
        if xs_on:
            ax.plot(
                xs_on, ys_on,
                color=HEAT_RED, linewidth=2.0,
                marker='s', markerfacecolor='none', markeredgecolor=HEAT_RED,
                label='DRCC with Recourse')

        # X ticks as union of epsilons, formatted to two decimals
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.2f}" for x in xticks])
        ax.set_xlabel(r'Allowed Violation Probability ($\varepsilon$)')
        ax.set_ylabel('Max. Transformer Loading (%)')
        ax.tick_params(axis='y')
        ax.grid(axis='y', alpha=0.3)
        # Largest epsilon on the left
        try:
            ax.invert_xaxis()
        except Exception:
            pass
        # Reference threshold line at 80% and focus the y-range to [65, 85]
        ax.axhline(80.0, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.set_ylim(70.0, 85.0)
        ax.legend(loc='best')
        # Plain numeric ticks for both axes to avoid LaTeX math macros
        try:
            _force_plain_ticks(ax, which='both')
        except Exception:
            pass

        fig.tight_layout()
        pgf_path, png_path = _save_pgf_and_png(fig, out_path)
        plt.close(fig)
        print(f"✓ Max transformer loading vs ε (v2) saved: {pgf_path} (+ PNG)")
        return out_path
    except Exception as e:
        # In case of any unexpected error, still create an empty placeholder figure
        _configure_pgf()
        width_cm = DEFAULT_WIDTH_CM
        width_cm = DEFAULT_WIDTH_CM
        fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * SOC_ASPECT)))
        ax.text(0.5, 0.5, f'Plot failed: {e}', ha='center', va='center', transform=ax.transAxes, color='gray')
        fig.tight_layout()
        pdf_path = _save_png_and_pdf(fig, out_path)
        plt.close(fig)
        print(f"[WARN] Max transformer loading vs ε (v2) failed, saved placeholder: {out_path} (PNG) and {pdf_path} (PDF)")
        return out_path


def plot_da_total_cost_vs_epsilon_v2(out_path: str, search_dir: str | None = None) -> str:
    """Two-line plot vs epsilon (x) of DA total cost (EUR) (y) from v2 results CSVs.

    - Searches for files like `dso_model_v2_results_drcc_true_epsilon_0_XX_rt_on.csv` and `_rt_off.csv`
    - Extracts a single DA total cost value per file (last finite value of `da_total_cost_eur`)
    - Plots RT OFF (blue) and RT ON (red) as functions of epsilon
    - Uses 3:2 aspect with fixed height 4.2 in (6.3 x 4.2)
    """
    try:
        base = search_dir if search_dir else os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(base):
            base = os.getcwd()

        def _parse_eps(fname: str) -> float | None:
            if 'epsilon_' not in fname:
                return None
            try:
                part = fname.split('epsilon_', 1)[1]
                part = part.split('_rt', 1)[0]
                return float(part.replace('_', '.'))
            except Exception:
                return None

        def _is_on(fname: str) -> bool:
            return 'rt_on' in fname.lower()

        def _is_off(fname: str) -> bool:
            return 'rt_off' in fname.lower()

        files = [f for f in os.listdir(base) if f.endswith('.csv') and f.startswith('dso_model_v2_results_drcc_true_epsilon_')]
        vals_on: dict[float, float] = {}
        vals_off: dict[float, float] = {}

        for f in files:
            eps = _parse_eps(f)
            if eps is None:
                continue
            full = os.path.join(base, f)
            try:
                df = pd.read_csv(full)
            except Exception:
                continue
            if 'da_total_cost_eur' not in df.columns:
                continue
            try:
                series = pd.to_numeric(df['da_total_cost_eur'], errors='coerce').dropna()
                if series.empty:
                    continue
                cost_val = float(series.iloc[-1])
            except Exception:
                continue
            if _is_on(f):
                vals_on[eps] = cost_val
            elif _is_off(f):
                vals_off[eps] = cost_val

        # Build x/y in descending epsilon order for plotting
        xs_on = sorted(vals_on.keys(), reverse=True)
        ys_on = [vals_on[x] for x in xs_on]
        xs_off = sorted(vals_off.keys(), reverse=True)
        ys_off = [vals_off[x] for x in xs_off]

        xticks = sorted(set(xs_on) | set(xs_off))  # ascending; invert axis for visual
        # Match dimensions with fig_trafo_loading_vs_epsilon.png and fig_RT-costs.png
        _configure_pgf()
        width_cm = DEFAULT_WIDTH_CM
        fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * LINE_ASPECT)))
        if xs_off:
            ax.plot(
                xs_off, ys_off,
                color=ELECTRIC_BLUE, linewidth=2.0,
                marker='o', markerfacecolor='none', markeredgecolor=ELECTRIC_BLUE,
                label='DRCC without Recourse')
        if xs_on:
            ax.plot(
                xs_on, ys_on,
                color=HEAT_RED, linewidth=2.0,
                marker='s', markerfacecolor='none', markeredgecolor=HEAT_RED,
                label='DRCC with Recourse')

        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.2f}" for x in xticks])
        ax.set_xlabel(r'Allowed Violation Probability ($\varepsilon$)')
        ax.set_ylabel('Day-ahead Cost (EUR)')
        ax.tick_params(axis='y')
        ax.grid(axis='y', alpha=0.3)
        try:
            ax.invert_xaxis()
        except Exception:
            pass
        ax.legend(loc='best')
        # Plain numeric ticks for both axes
        try:
            _force_plain_ticks(ax, which='both')
        except Exception:
            pass

        fig.tight_layout()
        pgf_path, png_path = _save_pgf_and_png(fig, out_path)
        plt.close(fig)
        print(f"✓ DA total cost vs ε (v2) saved: {pgf_path} (+ PNG)")
        return out_path
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.text(0.5, 0.5, f'Plot failed: {e}', ha='center', va='center', transform=ax.transAxes, color='gray')
        fig.tight_layout()
        pdf_path = _save_png_and_pdf(fig, out_path)
        plt.close(fig)
        print(f"[WARN] DA total cost vs ε (v2) failed, saved placeholder: {out_path} (PNG) and {pdf_path} (PDF)")
        return out_path


def plot_thermal_storage_operation_area(results_csv: str, out_path: str) -> str:
    """Duplicate thermal storage operation area plot (originally PGF) using Matplotlib PNG.

    Reads `q_storage_kw` from results CSV (default fully_coordinated_model_results.csv) and, if available,
    overlays electricity price from `market_prices_15min.csv` (column `price_EUR_MWh`).
    Styling matches existing figures: Times New Roman font, area fills:
        Charging (>=0): green (standard) alpha 0.30
        Discharging (<=0): red (standard) alpha 0.30
    Secondary y-axis (right) for price in blue (standard matplotlib) including spine.
    """
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    df = pd.read_csv(results_csv)
    if 'q_storage_kw' not in df.columns:
        raise KeyError("Column 'q_storage_kw' not in results CSV")
    ts = pd.to_numeric(df['q_storage_kw'], errors='coerce').fillna(0.0).to_numpy()
    x = np.arange(1, len(ts)+1)

    # Attempt to load aligned market price (mirror PGF logic): 24h window starting 2023-01-10
    price = None
    try:
        # Build canonical 24h time window (96 steps of 15min) matching fully coordinated model period
        start_dt = pd.to_datetime('2023-01-10 00:00:00')
        duration_hours = 24
        end_dt = start_dt + pd.Timedelta(hours=duration_hours) - pd.Timedelta(minutes=15)
        # Read market price file
        if os.path.exists('market_prices_15min.csv'):
            pdf = pd.read_csv('market_prices_15min.csv')
            if 'datetime' in pdf.columns and 'price_EUR_MWh' in pdf.columns:
                pdf['datetime'] = pd.to_datetime(pdf['datetime'])
                pdf = pdf.set_index('datetime')
                window = pdf.loc[start_dt:end_dt]
                if not window.empty:
                    price_series = pd.to_numeric(window['price_EUR_MWh'], errors='coerce')
                    # Ensure length matches ts (pad with last value if shorter, trim if longer)
                    vals = price_series.to_numpy()
                    if vals.size < ts.size:
                        pad_val = float(vals[-1]) if vals.size else 0.0
                        vals = np.concatenate([vals, np.full(ts.size - vals.size, pad_val)])
                    price = vals[:ts.size]
    except Exception:
        price = None
    # Fallback: direct column from results CSV if alignment failed
    if price is None:
        for cname in ['electricity_price_eur_mwh','price_EUR_MWh','price','price_eur_mwh']:
            if cname in df.columns:
                price_series = pd.to_numeric(df[cname], errors='coerce').fillna(method='ffill').fillna(method='bfill')
                vals = price_series.to_numpy()
                if vals.size < ts.size:
                    pad_val = float(vals[-1]) if vals.size else 0.0
                    vals = np.concatenate([vals, np.full(ts.size - vals.size, pad_val)])
                price = vals[:ts.size]
                break

    # Colors aligned with export_select_pgf_plots / oos-analysis palette
    electric_blue = ELECTRIC_BLUE
    green_fill = GAS_GREEN_FILL
    green_edge = GAS_GREEN
    red_fill = HEAT_RED_FILL
    red_edge = HEAT_RED

    # Match line plots: 3:2 aspect, fixed height 4.2 in
    _configure_pgf()
    width_cm = DEFAULT_WIDTH_CM
    fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * LINE_ASPECT)))
    ax.plot(x, ts, color='black', linewidth=1.0, zorder=3)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.fill_between(x, ts, 0, where=ts>=0, facecolor=GAS_GREEN_65, interpolate=True, zorder=1)
    ax.fill_between(x, ts, 0, where=ts<=0, facecolor=HEAT_RED_65, interpolate=True, zorder=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Thermal Power (kW)')
    ax.tick_params(axis='y')
    ax.tick_params(axis='x')
    ax.set_xlim(0, x[-1])
    ax.margins(x=0)
    ax.grid(True, axis='y', color='lightgray', alpha=0.6, linewidth=0.6)
    # Fixed left axis ticks and padding as requested
    left_ticks = np.array([-200, -100, 0, 100, 200], dtype=float)
    left_min = left_ticks[0]
    left_max = left_ticks[-1]
    left_range = left_max - left_min
    left_pad = 0.10 * left_range  # 10% padding
    ax.set_ylim(left_min - left_pad, left_max + left_pad)
    ax.set_yticks(left_ticks)

    # Add annotations: 'Charging' (x=36, y=200) with two arrows 45° down;
    # and 'Discharging' (x=50, y=-200) with two arrows 45° up.
    try:
        # Charging annotation
        charge_x, charge_y = 32, 200
        ax.text(charge_x, charge_y, 'Charging', color='black',
                ha='center', va='bottom')
        for dx, dy in [(17, -17), (-17, -17)]:
            ax.annotate('', xy=(charge_x + dx, charge_y + dy), xytext=(charge_x, charge_y),
                        arrowprops=dict(arrowstyle='->', color='black', linewidth=1.2))

        # Discharging annotation
        discharge_x, discharge_y = 52, -200
        ax.text(discharge_x, discharge_y, 'Discharging', color='black',
                ha='center', va='top')
        for dx, dy in [(17, 17), (-17, 17)]:
            ax.annotate('', xy=(discharge_x + dx, discharge_y + dy), xytext=(discharge_x, discharge_y),
                        arrowprops=dict(arrowstyle='->', color='black', linewidth=1.2))
    except Exception:
        pass

    ax2 = ax.twinx()
    ax2.patch.set_alpha(0.0)
    if price is not None:
        # Plot price and enforce fixed right axis ticks with same relative padding (5%).
        ax2.plot(x, price, color=electric_blue, linewidth=1.6, zorder=4)
        right_ticks = np.array([85, 100, 115, 130, 145], dtype=float)
        r_min = right_ticks[0]
        r_max = right_ticks[-1]
        r_range = r_max - r_min
        r_pad = 0.10 * r_range  # 10% padding
        ax2.set_ylim(r_min - r_pad, r_max + r_pad)
        ax2.set_yticks(right_ticks)
        try:
            from matplotlib.ticker import FormatStrFormatter
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        except Exception:
            pass
    ax2.set_ylabel('Electricity price (EUR/MWh)', color=electric_blue)
    ax2.tick_params(axis='y', colors=electric_blue)
    try:
        ax2.spines['right'].set_color(electric_blue)
    except Exception:
        pass
    ax2.grid(False)
    # Enforce sans-serif ticks on both axes; right y in sans
    try:
        _force_sans_ticks(ax, which='both')
        _force_sans_ticks(ax2, which='y')
    except Exception:
        pass

    # X ticks at 24-step intervals including 0
    try:
        ticks_24 = np.arange(0, x[-1]+1, 24)
        ax.set_xticks(ticks_24)
    except Exception:
        pass

    fig.tight_layout()
    pgf_path, png_path = _save_pgf_and_png(fig, out_path)
    plt.close(fig)
    print(f"✓ Thermal storage plot saved: {pgf_path} (+ PNG)")
    return out_path


def plot_all_in_total_cost(df: pd.DataFrame, out_path: str) -> str:
    # Sort by plot_order if available to match overview ordering
    if 'plot_order' in df.columns:
        dfp = df.sort_values('plot_order').reset_index(drop=True)
    else:
        dfp = df.copy().reset_index(drop=True)

    # Drop requested labels if present
    if 'label' in dfp.columns and len(DROP_LABELS) > 0:
        dfp = dfp[~dfp['label'].astype(str).isin(DROP_LABELS)].reset_index(drop=True)

    labels = dfp['label'].astype(str).tolist() if 'label' in dfp.columns else [str(i) for i in range(len(dfp))]
    # Use precomputed all-in column if present, else compute fallback again here
    if 'all_in_total_cost_mean' in dfp.columns:
        c_allin = pd.to_numeric(dfp['all_in_total_cost_mean'], errors='coerce').to_numpy()
    else:
        da = pd.to_numeric(dfp.get('da_total_cost_eur', pd.Series([np.nan]*len(dfp))), errors='coerce').fillna(0.0)
        rt = pd.to_numeric(dfp.get('total_rt_cost_mean', pd.Series([np.nan]*len(dfp))), errors='coerce').fillna(0.0)
        c_allin = (da + rt).to_numpy()

    # Inject the fixed-cost bar at the end
    labels.append(INJECT_LABEL)
    try:
        c_allin = np.concatenate([c_allin, np.array([float(INJECT_COST)], dtype=float)])
    except Exception:
        # Fallback in unlikely case of dtype issues
        c_allin = np.array(list(c_allin) + [float(INJECT_COST)], dtype=float)

    # Sort bars by height (descending)
    try:
        order = np.argsort(-np.asarray(c_allin, dtype=float))
        labels = [labels[i] for i in order]
        c_allin = np.asarray(c_allin, dtype=float)[order]
    except Exception:
        pass

    # Build display labels per requested naming scheme
    def _display_label(raw: str) -> str:
        if raw == INJECT_LABEL:
            return raw
        if raw.lower() == 'deterministic':
            return raw
        s = str(raw)
        # Expect forms like "0.10 (RT ON)" or "0.10 (RT OFF)"
        try:
            base = s.split()[0]
            float(base)  # validate epsilon part
            if '(RT ON' in s:
                return rf"DRCC $\varepsilon$ = {base}"
            if '(RT OFF' in s:
                return f"No Recourse,\nDRCC $\\varepsilon$ = {base}"
            # legacy unsuffixed
            return rf"DRCC $\varepsilon$ = {base}"
        except Exception:
            return raw

    display_labels = [_display_label(l) for l in labels]

    x = np.arange(len(labels))
    # Size proportional to case count (slightly wider for readability)
    fig_width = max(6.5, 1.35 * len(labels) + 2.5)
    _configure_pgf()
    width_cm = DEFAULT_WIDTH_CM
    fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * BAR_ASPECT)))
    bars = ax.bar(x, c_allin, width=0.35, color=ELECTRIC_BLUE_65, edgecolor=ELECTRIC_BLUE_80)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0)
    ax.set_xlabel('Optimization Model')
    ax.set_ylabel('Total Cost (EUR)')
    ax.tick_params(axis='y')
    # Explicit y-axis headroom for bar-top labels (5%)
    try:
        ymax = float(np.nanmax(c_allin)) if len(c_allin) else 1.0
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0.0, ymax * 1.10)
    except Exception:
        pass
    # No title to mirror oos-analysis (title commented out there)
    ax.grid(axis='y', alpha=0.3)
    # Sans-serif numeric y ticks
    try:
        _force_sans_ticks(ax, which='y')
    except Exception:
        pass

    # Add % difference labels (placed like overload energy labels)
    try:
        # Determine baseline (prefer injected Multi-Energy Co-optimization; else deterministic)
        base_idx = None
        base_label = None
        for i, lab in enumerate(labels):
            if isinstance(lab, str) and lab == INJECT_LABEL:
                base_idx = i; base_label = INJECT_LABEL; break
        if base_idx is None:
            for i, lab in enumerate(labels):
                if isinstance(lab, str) and lab.lower() == 'deterministic':
                    base_idx = i; base_label = 'deterministic'; break

        if base_idx is not None and np.isfinite(c_allin[base_idx]) and c_allin[base_idx] != 0.0:
            base_val = float(c_allin[base_idx])
            ymax = float(np.nanmax(c_allin)) if len(c_allin) else 1.0
            pad = 0.02 * ymax
            for rect, val, lab in zip(bars, c_allin, labels):
                if not np.isfinite(val):
                    continue
                if isinstance(lab, str) and lab == base_label:
                    # Skip annotating the baseline bar itself
                    continue
                pct = (val - base_val) / base_val * 100.0
                sgn = '+' if pct >= 0 else ''
                txt = f"{sgn}{pct:.1f}\\%"
                h = rect.get_height()
                inside_y = h - pad
                if inside_y > 0:
                    ax.text(
                        rect.get_x() + rect.get_width()/2,
                        h + pad,
                        txt,
                        ha='center', va='bottom', color='black'
                    )
                else:
                    ax.text(
                        rect.get_x() + rect.get_width()/2,
                        h + pad,
                        txt,
                        ha='center', va='bottom', color='black', clip_on=False
                    )
    except Exception:
        pass

    # Add a bit of top padding for bar labels
    try:
        ax.margins(y=0.05)
    except Exception:
        pass
    fig.tight_layout()
    pgf_path, png_path = _save_pgf_and_png(fig, out_path)
    print(f"✓ Total cost plot saved: {pgf_path} (+ PNG)")
    return out_path


def plot_trafo_violation_probability(df: pd.DataFrame, out_path: str) -> str:
    """Bar plot for transformer violation probability (%), mirroring style of total cost plot.

    Drops 0.15 (RT ON) and injects a 'Co-optimization' bar with value 0, then sorts bars descending.
    """
    if 'plot_order' in df.columns:
        dfp = df.sort_values('plot_order').reset_index(drop=True)
    else:
        dfp = df.copy().reset_index(drop=True)

    if 'label' in dfp.columns and len(DROP_LABELS) > 0:
        dfp = dfp[~dfp['label'].astype(str).isin(DROP_LABELS)].reset_index(drop=True)

    labels = dfp['label'].astype(str).tolist() if 'label' in dfp.columns else [str(i) for i in range(len(dfp))]
    if 'trafo_violation_probability_pct' in dfp.columns:
        vals = pd.to_numeric(dfp['trafo_violation_probability_pct'], errors='coerce').fillna(0.0).to_numpy()
    else:
        # Fallback: attempt from steps ratio if columns available
        if {'trafo_steps','horizon_timesteps'} <= set(dfp.columns):
            num = pd.to_numeric(dfp['trafo_steps'], errors='coerce').fillna(0.0)
            den = pd.to_numeric(dfp['horizon_timesteps'], errors='coerce').replace(0, np.nan)
            ratio = (num / den * 100.0).fillna(0.0)
            vals = ratio.to_numpy()
        else:
            vals = np.zeros(len(dfp), dtype=float)

    # Inject custom bar
    labels.append(TRAFO_INJECT_LABEL)
    try:
        vals = np.concatenate([vals, np.array([float(TRAFO_INJECT_VALUE)], dtype=float)])
    except Exception:
        vals = np.array(list(vals) + [float(TRAFO_INJECT_VALUE)], dtype=float)

    # Sort descending
    try:
        order = np.argsort(-np.asarray(vals, dtype=float))
        labels = [labels[i] for i in order]
        vals = np.asarray(vals, dtype=float)[order]
    except Exception:
        pass

    def _display_label(raw: str) -> str:
        if raw == TRAFO_INJECT_LABEL:
            return raw
        if raw.lower() == 'deterministic':
            return raw
        s = str(raw)
        try:
            base = s.split()[0]
            float(base)
            if '(RT ON' in s:
                return rf"DRCC $\varepsilon$ = {base}"
            if '(RT OFF' in s:
                return f"No Recourse,\nDRCC $\\varepsilon$ = {base}"
            return rf"DRCC \\$\varepsilon$ = {base}"
        except Exception:
            return raw

    display_labels = [_display_label(l) for l in labels]
    x = np.arange(len(labels))
    fig_width = max(6.5, 1.35 * len(labels) + 2.5)
    width_cm = DEFAULT_WIDTH_CM
    fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * LINE_ASPECT)))
    bars = ax.bar(x, vals, width=0.35, color=HEAT_RED_65, edgecolor=HEAT_RED_80)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0)
    ax.set_xlabel('Optimization Model')
    ax.set_ylabel('Transformer Overload Probability (%)')
    ax.tick_params(axis='y')
    ax.grid(axis='y', alpha=0.3)
    try:
        _force_sans_ticks(ax, which='y')
    except Exception:
        pass

    fig.tight_layout()
    pgf_path, png_path = _save_pgf_and_png(fig, out_path)
    print(f"✓ Transformer violation probability plot saved: {pgf_path} (+ PNG)")
    return out_path


def _extract_final_soc_median(label: str, src_dir: str) -> float | None:
    """Extract final timestep median SOC (soc_p50) given a label.
    Returns float or None if unavailable.
    """
    try:
        s = str(label)
        # Deterministic case: no envelope file, treat as missing
        if s.lower() == 'deterministic':
            return None
        parts = s.split()
        if not parts:
            return None
        eps = parts[0]
        try:
            float(eps)
        except Exception:
            return None
        rt_tag = None
        if '(RT' in s:
            if 'RT ON' in s:
                rt_tag = 'rt_on'
            elif 'RT OFF' in s:
                rt_tag = 'rt_off'
        if rt_tag is None:
            return None
        # Filenames use underscore in epsilon: e.g. 0.10 -> 0_10
        eps_token = eps.replace('.', '_')
        fname = f"soc_envelope_drcc_true_epsilon_{eps_token}_{rt_tag}.csv"
        path = os.path.join(src_dir, fname)
        if not os.path.exists(path):
            return None
        pdf = pd.read_csv(path)
        if 'soc_p50' not in pdf.columns:
            return None
        if 'timestamp' in pdf.columns:
            try:
                pdf['timestamp'] = pd.to_datetime(pdf['timestamp'], errors='coerce')
                pdf = pdf.sort_values('timestamp')
            except Exception:
                pass
        series = pd.to_numeric(pdf['soc_p50'], errors='coerce')
        series = series[np.isfinite(series)]
        if series.empty:
            return None
        return float(series.iloc[-1])
    except Exception:
        return None


def plot_final_bess_soc_median(df: pd.DataFrame, out_path: str, src_dir: str) -> str:
    """Bar plot of median battery SOC at final timestep for each case.

    Drops 0.15 (RT ON) and injects Multi-Energy Co-optimization with value 0.5.
    Bars sorted descending; no error bars.
    """
    if 'plot_order' in df.columns:
        dfp = df.sort_values('plot_order').reset_index(drop=True)
    else:
        dfp = df.copy().reset_index(drop=True)

    if 'label' in dfp.columns and len(DROP_LABELS) > 0:
        dfp = dfp[~dfp['label'].astype(str).isin(DROP_LABELS)].reset_index(drop=True)

    labels = dfp['label'].astype(str).tolist() if 'label' in dfp.columns else [str(i) for i in range(len(dfp))]
    vals_list = []
    for lab in labels:
        v = _extract_final_soc_median(lab, src_dir)
        if v is None or not np.isfinite(v):
            vals_list.append(np.nan)
        else:
            vals_list.append(v)
    vals = np.array(vals_list, dtype=float)
    # Inject custom bar
    labels.append(INJECT_LABEL)
    try:
        vals = np.concatenate([vals, np.array([0.5], dtype=float)])
    except Exception:
        vals = np.array(list(vals) + [0.5], dtype=float)

    # Sort descending (treat NaN as very small)
    try:
        sortable = np.where(np.isfinite(vals), vals, -1.0)
        order = np.argsort(-sortable)
        labels = [labels[i] for i in order]
        vals = vals[order]
    except Exception:
        pass

    def _display_label(raw: str) -> str:
        if raw == INJECT_LABEL:
            return raw
        if isinstance(raw, str) and raw.lower() == 'deterministic':
            return raw
        s = str(raw)
        try:
            base = s.split()[0]
            float(base)
            if '(RT ON' in s:
                return rf"DRCC $\varepsilon$ = {base}"
            if '(RT OFF' in s:
                return f"No Recourse,\nDRCC $\\varepsilon$ = {base}"
            return rf"DRCC \\$\varepsilon$ = {base}"
        except Exception:
            return raw

    display_labels = [_display_label(l) for l in labels]
    x = np.arange(len(labels))
    fig_width = max(6.5, 1.35 * len(labels) + 2.5)
    width_cm = DEFAULT_WIDTH_CM
    fig, ax = plt.subplots(figsize=(_cm_to_inch(width_cm), _cm_to_inch(width_cm * BAR_ASPECT)))
    bars = ax.bar(x, vals, width=0.35, color=GAS_GREEN_65, edgecolor=GAS_GREEN_80)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0)
    ax.set_xlabel('Optimization Model')
    ax.set_ylabel('Median Final BESS SOC')
    ax.tick_params(axis='y')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # Reference line at SOC 0.5
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.0, alpha=0.9)
    try:
        _force_sans_ticks(ax, which='y')
    except Exception:
        pass

    fig.tight_layout()
    pgf_path, png_path = _save_pgf_and_png(fig, out_path)
    print(f"✓ Final battery SOC median plot saved: {pgf_path} (+ PNG)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots from aggregated OOS results.")
    ap.add_argument('--src', default=DEFAULT_SRC_DIR, help="Source directory containing oos_overview_summary.csv")
    ap.add_argument('--out', default=None, help="Output image path (overrides export dir/name if provided)")
    ap.add_argument('--export-dir', default=DEFAULT_EXPORT_DIR, help="Directory to write exported figures (default: 'export figures')")
    args = ap.parse_args()

    src_dir = args.src
    # Determine output path: explicit --out wins; else export-dir/DEFAULT_OUT_NAME
    if args.out:
        out_path = args.out
    else:
        out_dir = args.export_dir or DEFAULT_EXPORT_DIR
        out_path = os.path.join(out_dir, DEFAULT_OUT_NAME)

    df = load_summary(src_dir)
    plot_all_in_total_cost(df, out_path)

    # Transformer violation probability plot
    trafo_out = os.path.join(os.path.dirname(out_path), TRAFO_VIOLATION_OUT_NAME)
    try:
        plot_trafo_violation_probability(df, trafo_out)
    except Exception as e:
        print(f"[WARN] Transformer violation plot skipped: {e}")

    # Final battery SOC median plot
    final_soc_out = os.path.join(os.path.dirname(out_path), FINAL_SOC_OUT_NAME)
    try:
        plot_final_bess_soc_median(df, final_soc_out, src_dir)
    except Exception as e:
        print(f"[WARN] Final SOC plot skipped: {e}")

    # Also produce focused ε=0.10 RT ON vs RT OFF split-violin into export folder
    violin_out = args.out
    if not violin_out:
        out_dir = args.export_dir or DEFAULT_EXPORT_DIR
        violin_out = os.path.join(out_dir, VIOLIN_OUT_NAME)
    else:
        # If a single --out was provided for the bar chart, still place violin next to it using default name
        out_dir = os.path.dirname(args.out) or (args.export_dir or DEFAULT_EXPORT_DIR)
        violin_out = os.path.join(out_dir, VIOLIN_OUT_NAME)
    plot_violin_rt_on_vs_off(src_dir, violin_out)

    # Thermal storage area plot (replicated)
    thermal_out = os.path.join(out_dir if 'out_dir' in locals() else (args.export_dir or DEFAULT_EXPORT_DIR), THERMAL_STORAGE_OUT_NAME)
    try:
        plot_thermal_storage_operation_area('fully_coordinated_model_results.csv', thermal_out)
    except Exception as e:
        print(f"[WARN] Thermal storage plot skipped: {e}")

    # Overload energy comparison: ε=0.10 RT ON vs RT OFF (same size as violin)
    overload_out = os.path.join(out_dir if 'out_dir' in locals() else (args.export_dir or DEFAULT_EXPORT_DIR), OVERLOAD_RT_ON_OFF_OUT_NAME)
    try:
        plot_overload_energy_rt_on_vs_off(src_dir, overload_out)
    except Exception as e:
        print(f"[WARN] Overload energy RT ON vs OFF plot skipped: {e}")

    # Max transformer loading vs epsilon (v2 results): two-line plot
    max_trafo_out = os.path.join(out_dir if 'out_dir' in locals() else (args.export_dir or DEFAULT_EXPORT_DIR), MAX_TRAFO_EPSILON_OUT_NAME)
    try:
        plot_max_trafo_loading_vs_epsilon_v2(max_trafo_out)
    except Exception as e:
        print(f"[WARN] Max. transformer loading vs ε (v2) plot skipped: {e}")

    # DA total cost vs epsilon (v2 results): two-line plot
    da_cost_out = os.path.join(out_dir if 'out_dir' in locals() else (args.export_dir or DEFAULT_EXPORT_DIR), DA_COST_EPSILON_OUT_NAME)
    try:
        plot_da_total_cost_vs_epsilon_v2(da_cost_out)
    except Exception as e:
        print(f"[WARN] DA total cost vs ε (v2) plot skipped: {e}")


if __name__ == '__main__':
    main()
