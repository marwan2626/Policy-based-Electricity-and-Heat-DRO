"""
Export selected figures to PGF for LaTeX inclusion, using consistent sans-serif fonts
and fixed sizing. Includes:
- Thermal storage operation area (recreated from results CSV)
- OOS Student-t outputs rendered from PNG: overload_energy_compare, soc_envelopes,
  trafo_violation_heatmap, violin_compare, frontier_hybrid_scatter
- PV & ambient temperature uncertainty bands (mean ±1σ) from Gaussian samples: pv_temp_uncertainty

Default figure width: 10.89 cm; aspect ratio: 0.5
LaTeX PGF configuration matches export_thermal_storage_pgf.py (sans-serif, xcolor, brand colors).

Usage examples (PowerShell):
    # Export everything
    python ./export_select_pgf_plots.py --all

    # Export only OOS images
    python ./export_select_pgf_plots.py --overload-energy --soc-envelopes --trafo-violation-heatmap --violin-compare --frontier-hybrid-scatter

    # Export PV & temperature uncertainty only
    python ./export_select_pgf_plots.py --pv-temp-uncertainty

    # Export only thermal storage with aligned market price (defaults)
    python ./export_select_pgf_plots.py --ts
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Dict, List, Tuple
from pathlib import Path
import shutil
import json
import glob
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Prefer using the proven TS exporter from export_thermal_storage_pgf.py for identical output
try:
    from export_thermal_storage_pgf import (
        export_thermal_storage_pgf as export_ts_ref,
    )
except Exception:
    export_ts_ref = None

# --------------------- Paper Size Toggle ---------------------
# Set USE_A4 = True to switch to A4 sizing (13.0 cm width, font size 12).
# When False (A5 mode), defaults remain width 10.89 cm and font size 8.
USE_A4 = True  # Toggle this flag

# Derived sizing constants based on paper size mode
if USE_A4:
    DEFAULT_WIDTH_CM = 13.0
    DEFAULT_FONT_SIZE = 11
else:
    DEFAULT_WIDTH_CM = 10.89
    DEFAULT_FONT_SIZE = 8

# --------------------- Constants ---------------------
DEFAULT_ASPECT = 0.5
# Per-plot default height ratios (aspect = height/width). Adjust here to tune each plot individually.
DEFAULT_ASPECT_TS = DEFAULT_ASPECT
DEFAULT_ASPECT_OVERLOAD = DEFAULT_ASPECT
DEFAULT_ASPECT_SOC = 0.8
DEFAULT_ASPECT_HEATMAP = DEFAULT_ASPECT
DEFAULT_ASPECT_VIOLIN = 2.0
DEFAULT_ASPECT_FRONTIER = 1.0
DEFAULT_ASPECT_PVTEMP = 0.8
DEFAULT_ASPECT_LOADS = 0.5
DEFAULT_ASPECT_BEV = 0.5
DEFAULT_ASPECT_THERMAL = 0.8  # stacked hotwater / heating profiles
DEFAULT_MAX_TICKS_X = 6
DEFAULT_MAX_TICKS_Y = 5
DEFAULT_MAX_CLOUD_POINTS = 4000
# Cloud scatter marker sizes (can be overridden via CLI)
DEFAULT_CLOUD_MARKER_SIZE_DRCC = 30
DEFAULT_CLOUD_MARKER_SIZE_BASE = 30
BAR_WIDTH = 0.35  # Uniform bar width for all bar plots
RESULTS_DIR = "v3_oos_agg_gaussian_studentt"  # Updated default OOS results directory
# Central export directory for PGF outputs
EXPORT_PGF_DIR = "export pgf"
PNG_DPI = 300

# ===================== USER CONFIG (edit and Run) =====================
# Set RUN_WITH_INLINE_CONFIG = True and edit INLINE_CONFIG to run without CLI.
RUN_WITH_INLINE_CONFIG = True
INLINE_CONFIG: Dict[str, object] = {
    # Which plots to export: use ["all"] or any subset of
    # ["ts", "overload", "soc", "trafo", "violin", "frontier", "pvtemp", "loads", "bev", "thermal"]
    "selections": ["all"],
    # Override results directory for OOS plots; None uses module-level RESULTS_DIR
    "results_dir": None,

    # PGF / sizing
    "texsystem": "pdflatex",  # or "xelatex", "lualatex"
    "sfmath": False,
    "width_cm": DEFAULT_WIDTH_CM,
    # Global aspect applies if set; per-plot overrides below take precedence
    "aspect": None,
    "aspect_ts": None,
    "aspect_overload": None,
    "aspect_soc": None,
    "aspect_heatmap": None,
    "aspect_violin": None,
    "aspect_frontier": None,
    "aspect_pvtemp": None,
    "aspect_loads": None,
    "aspect_bev": None,
    "aspect_thermal": None,
    # Optional violin width; if None defaults to width_cm/2
    "width_cm_violin": None,
    # Ticks
    "max_ticks_x": DEFAULT_MAX_TICKS_X,
    "max_ticks_y": DEFAULT_MAX_TICKS_Y,
    # PNG quality
    "png_dpi": 300,

    # Thermal storage specific
    "ts": {
        "input": "fully_coordinated_model_results.csv",
        "price_csv": None,
        "price_col": None,
        "start_date": "2023-01-10 00:00:00",
        "duration_hours": 24,
        "no_market_price": False,
    },
}

# When exporting plots we also write a PNG alongside the PGF for quick visual debugging.
def _save_dual(
    fig: plt.Figure,
    pgf_path: str,
    png_dpi: Optional[int] = None,
    tight: bool = True,
    pad_inches: float = 0.02,
) -> Tuple[str, str]:
    d = os.path.dirname(pgf_path)
    if d and d not in ("", "."):
        os.makedirs(d, exist_ok=True)
    stem, _ = os.path.splitext(pgf_path)
    png_path = stem + ".png"
    # Save PGF first (primary artifact) with tight bounding box to minimize white margins
    save_kwargs = {}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = float(pad_inches)
    fig.savefig(pgf_path, **save_kwargs)
    # Then PNG for inspection (best effort)
    try:
        dpi = int(png_dpi) if png_dpi is not None else int(PNG_DPI)
        fig.savefig(png_path, dpi=dpi, **save_kwargs)
    except Exception as e:
        print(f"Warning: failed to save PNG for {pgf_path}: {e}")
    return pgf_path, png_path

# --------------------- Helpers ---------------------

def _cm_to_inch(cm: float) -> float:
    return cm / 2.54


def _configure_pgf(
    enable: bool = True,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
) -> None:
    r"""Configure Matplotlib PGF output with LaTeX enabled and safe preamble.

    Important: avoid \usepackage or global font redefinitions in the PGF preamble to
    prevent option clashes when \input into larger LaTeX docs. We enforce sans-serif
    via \textsf wrappers on labels and tick formatters instead of preamble hacks.
    """
    if not enable:
        return

    # Keep the PGF/LaTeX preamble minimal to avoid breaking the parent document.
    # We deliberately avoid adding \usepackage{...} or \renewcommand here.
    preamble = ""

    mpl.rcParams.update(
        {
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
        }
    )


def _force_sans_ticks(ax: plt.Axes, which: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter
    if which in ("y", "both"):
        yfmt = FuncFormatter(lambda v, pos: rf"\textsf{{{v:g}}}")
        ax.yaxis.set_major_formatter(yfmt)
    if which in ("x", "both"):
        xfmt = FuncFormatter(lambda v, pos: rf"\textsf{{{int(v)}}}")
        ax.xaxis.set_major_formatter(xfmt)


def _force_plain_ticks(ax: plt.Axes, which: str = "both", percent_x: bool = False, percent_y: bool = False) -> None:
    """Force tick labels to plain text (no math, no LaTeX macros) to avoid \mathdefault in PGF.

    - percent_x/percent_y: format as percentage strings without using PercentFormatter.
    """
    from matplotlib.ticker import FuncFormatter
    if which in ("y", "both"):
        if percent_y:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    if which in ("x", "both"):
        if percent_x:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v) if abs(v-round(v))<1e-9 else v:g}"))


def _apply_max_ticks(ax: plt.Axes, max_ticks_x: Optional[int] = None, max_ticks_y: Optional[int] = None, integer_x: bool = False, integer_y: bool = False) -> None:
    from matplotlib.ticker import MaxNLocator
    if max_ticks_x and max_ticks_x > 0:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks_x, integer=integer_x))
    if max_ticks_y and max_ticks_y > 0:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks_y, integer=integer_y))


# --------------------- TS price alignment (copied) ---------------------

def _build_time_index_from_vdi(start_date: str, duration_hours: int) -> pd.DatetimeIndex:
    profiles_path = os.path.join("vdi_profiles", "all_house_profiles.csv")
    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"Required file not found: {profiles_path}")

    profiles_df = pd.read_csv(profiles_path, index_col=0)
    profiles_df.index = pd.to_datetime(profiles_df.index)

    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + pd.Timedelta(hours=int(duration_hours)) - pd.Timedelta(minutes=15)

    if start_dt < profiles_df.index.min():
        raise ValueError(
            f"Start date {start_dt} is before earliest profile data {profiles_df.index.min()}"
        )
    if end_dt > profiles_df.index.max():
        raise ValueError(
            f"End date {end_dt} is after latest profile data {profiles_df.index.max()}"
        )

    dhn_window = profiles_df.loc[start_dt:end_dt]
    if dhn_window.empty:
        raise ValueError(f"No DHN data between {start_dt} and {end_dt}")
    return dhn_window.index


def _load_aligned_market_price(time_index: pd.DatetimeIndex) -> Optional[np.ndarray]:
    try:
        price_df = pd.read_csv("market_prices_15min.csv")
        if "datetime" not in price_df.columns:
            raise KeyError("'datetime' column not found in market_prices_15min.csv")
        price_df["datetime"] = pd.to_datetime(price_df["datetime"])
        price_df.set_index("datetime", inplace=True)

        if "price_EUR_MWh" not in price_df.columns:
            print("Warning: 'price_EUR_MWh' column not found in market_prices_15min.csv")
            return None

        aligned = price_df.reindex(time_index)["price_EUR_MWh"]
        if aligned.isnull().any():
            aligned = aligned.ffill().bfill()
        if aligned.isnull().any():
            print("Warning: market_prices_15min.csv has no overlap with requested time window")
            return None
        return aligned.to_numpy()
    except FileNotFoundError:
        print("Warning: market_prices_15min.csv not found")
        return None
    except Exception as e:
        print(f"Warning: failed to load/align market_prices_15min.csv: {e}")
        return None


# --------------------- TS plot (copied + colors) ---------------------

def export_thermal_storage_pgf(
    input_csv: str = "fully_coordinated_model_results.csv",
    output_path: str = os.path.join(EXPORT_PGF_DIR, "thermal_storage_operation_area.pgf"),
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    show: bool = False,
    price_csv: Optional[str] = None,
    price_col: Optional[str] = None,
    start_date: str = "2023-01-10 00:00:00",
    duration_hours: int = 24,
    disable_market_price: bool = False,
) -> str:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Results CSV not found: {input_csv}")

    # Load results CSV
    df = pd.read_csv(input_csv)
    if "q_storage_kw" not in df.columns:
        raise KeyError(
            "Column 'q_storage_kw' not found in results CSV. Available columns: "
            + ", ".join(df.columns)
        )

    ts_series = df["q_storage_kw"].to_numpy()
    x_ts = np.arange(len(ts_series))

    # Prefer aligned market price
    price_series: Optional[np.ndarray] = None
    if not disable_market_price:
        try:
            time_index = _build_time_index_from_vdi(start_date, int(duration_hours))
            price_series = _load_aligned_market_price(time_index)
            if price_series is not None:
                if len(price_series) < len(df):
                    pad_val = float(price_series[-1]) if len(price_series) > 0 else 0.0
                    price_series = np.concatenate(
                        [price_series, np.full(len(df) - len(price_series), pad_val)]
                    )
                price_series = price_series[: len(df)]
        except Exception as e:
            print(f"Note: falling back from aligned market price: {e}")

    # Fallbacks
    if price_series is None:
        for c in [
            "price_EUR_MWh",
            "electricity_price",
            "electricity_price_eur_mwh",
            "price_eur_mwh",
        ]:
            if c in df.columns:
                price_series = pd.to_numeric(df[c], errors="coerce").to_numpy()
                break
    if price_series is None and price_csv:
        try:
            pdf = pd.read_csv(price_csv)
            col = price_col or next(
                (k for k in [
                    "price_EUR_MWh",
                    "electricity_price",
                    "electricity_price_eur_mwh",
                    "price_eur_mwh",
                    "price",
                ] if k in pdf.columns),
                None,
            )
            if col:
                price_series = pd.to_numeric(pdf[col], errors="coerce").to_numpy()
        except Exception:
            price_series = None

    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect

    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        # Reserve space for right-side epsilon legend and axis labels
        try:
            fig.subplots_adjust(left=0.14, right=0.86, bottom=0.18, top=0.95)
        except Exception:
            pass
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        try:
            fig.subplots_adjust(right=0.88)
        except Exception:
            pass

        gas_green = (58/255.0, 157/255.0, 108/255.0)
        heat_red = (216/255.0, 46/255.0, 29/255.0)
        electric_blue = (52/255.0, 69/255.0, 160/255.0)

        ax.plot(x_ts, ts_series, color="black", linewidth=1.8, label="TS power (kW)", zorder=3)
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.fill_between(
            x_ts, ts_series, 0, where=(ts_series >= 0), facecolor=gas_green, alpha=0.65, interpolate=True, zorder=1
        )
        ax.fill_between(
            x_ts, ts_series, 0, where=(ts_series <= 0), facecolor=heat_red, alpha=0.65, interpolate=True, zorder=1
        )

        ax2 = ax.twinx()
        ax2.set_zorder(2)
        ax2.patch.set_alpha(0.0)
        ax2.tick_params(axis="y", colors=electric_blue, right=True, labelright=True)
        ax2.yaxis.label.set_color(electric_blue)
        if "right" in ax2.spines:
            ax2.spines["right"].set_color(electric_blue)
            ax2.spines["right"].set_linewidth(1.0)
        ax2.set_ylabel("Electricity price (EUR/MWh)")

        if price_series is not None:
            ax2.plot(x_ts, price_series, color=electric_blue, linewidth=1.6, zorder=4)
            try:
                from matplotlib.ticker import MaxNLocator
                left_tick_count = len(ax.get_yticks())
                if left_tick_count > 0:
                    ax2.yaxis.set_major_locator(MaxNLocator(nbins=left_tick_count))
            except Exception:
                pass
        else:
            try:
                ax2.set_ylim(ax.get_ylim())
            except Exception:
                pass
        ax2.grid(False)

        _force_sans_ticks(ax, which="both")
        _force_sans_ticks(ax2, which="y")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Power (kW)")
        # Fixed time ticks at 0,24,...,96; ensure last label (e.g., 96) is visible by extending x-limits
        try:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            total_steps = int(len(x_ts))
            # Always include 96 label if series length is >= 96
            base_ticks = [0, 24, 48, 72, 96]
            ticks = [t for t in base_ticks if t <= total_steps]
            if ticks:
                # Extend axis to include the final step label when it equals the length
                ax.set_xlim(0, max(total_steps, max(ticks)))
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        except Exception:
            pass

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
        for name, spine in ax2.spines.items():
            spine.set_visible(True)
            if name != "right":
                spine.set_linewidth(1.0)
                spine.set_color("black")

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="lightgray", alpha=0.6, linewidth=0.6)

        pgf_out, png_out = _save_dual(fig, output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return pgf_out


def export_bev_profile_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "bev_car8_profile.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT_BEV,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
    start_date: str = "2023-01-10 00:00:00",
    duration_hours: int = 24,
    bev_csv: str = os.path.join("LadeprofileBEV", "bev_2023_power_first100.csv"),
    car_cols: Optional[List[str]] = None,
) -> str:
    """Export BEV charging power profile(s) for selected car(s) over the aligned time window.

    - Reads 15-min power from bev_csv (columns: datetime, car_1..car_100)
    - Aligns window to [start_date, start_date+duration_hours)
    - If multiple cars are provided, plots them as stacked subplots sharing the x-axis
    - Fixed x-ticks at 0,24,48,72,96; legend inside upper-left per subplot
    """
    if not os.path.exists(bev_csv):
        raise FileNotFoundError(f"BEV CSV not found: {bev_csv}")
    df = pd.read_csv(bev_csv)
    if "datetime" not in df.columns:
        raise KeyError("Column 'datetime' not found in BEV CSV")
    # Default to one car if not provided
    if not car_cols or len(car_cols) == 0:
        car_cols = ["car_8"]
    # Validate requested columns
    missing = [c for c in car_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Requested columns missing in BEV CSV: " + ", ".join(missing) +
            ". Example available: " + ", ".join([c for c in df.columns if c.startswith('car_')][:10]) + "..."
        )
    # Parse time and set index
    df["datetime"] = pd.to_datetime(df["datetime"])  # expect 15-min steps
    df.set_index("datetime", inplace=True)
    # Align window
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + pd.Timedelta(hours=int(duration_hours)) - pd.Timedelta(minutes=15)
    window = df.loc[start_dt:end_dt]
    if window.empty:
        raise ValueError("No BEV data in requested time window")
    # Build values per requested car
    series_list: List[np.ndarray] = []
    for c in car_cols:
        series_list.append(pd.to_numeric(window[c], errors="coerce").fillna(0.0).to_numpy())
    n_steps = len(window)
    x = np.arange(n_steps)
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        # Single axes overlay of all requested cars
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        palette = ['#3445A0', '#3A9D6C', '#D82E1D', '#9467BD', '#FF7F0E']
        for idx, (vals, colname) in enumerate(zip(series_list, car_cols)):
            color = palette[idx % len(palette)]
            ax.plot(x, vals, color=color, linewidth=1.2, label=colname.replace('_', '\\_'))
        ax.set_ylabel("Charging Power (kW)")
        ax.set_xlabel("Time Step")
        ax.grid(alpha=0.3)
        _force_plain_ticks(ax, which="y")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        # Remove default data margins so the plot hugs the axes (reduces internal whitespace)
        try:
            ax.margins(x=0)
        except Exception:
            pass
        # Fixed time ticks at 0,24,48,72,96 when within range
        try:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            base_ticks = [0, 24, 48, 72, 96]
            # Include the final label (e.g., 96) by allowing t == n_steps and extending xlim
            ticks = [t for t in base_ticks if (t < n_steps) or (t == n_steps)]
            if ticks:
                max_tick = max(ticks)
                ax.set_xlim(0, max(n_steps - 1, max_tick))
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        except Exception:
            pass
        # Tighten layout padding further to minimize surrounding whitespace
        try:
            fig.tight_layout(pad=0.1)
        except Exception:
            fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


def export_hotwater_heating_profiles_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "hotwater_heating_profiles.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT_THERMAL,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
    start_date: str = "2023-01-10 00:00:00",
    duration_hours: int = 24,
    profiles_csv: str = os.path.join("vdi_profiles", "all_house_profiles.csv"),
    houses: Optional[List[int]] = None,
) -> str:
    """Export stacked hot water (top) and heating (bottom) demand for given houses.

    Each panel overlays the selected houses' profiles over the aligned time window
    defined by [start_date, start_date + duration_hours). 15-minute resolution assumed.

    Parameters
    ----------
    houses : list of integer house/load IDs (e.g. [23,41]). Defaults to [23,41].
    profiles_csv : CSV containing columns like LV4_101_Load_<id>_hotwater / _heating.
    aspect : stacked figure height/width ratio (default 0.8 similar to pv/temperature).
    """
    if not os.path.exists(profiles_csv):
        raise FileNotFoundError(f"Profiles CSV not found: {profiles_csv}")
    df = pd.read_csv(profiles_csv, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        raise ValueError("Failed to parse datetime index in house profiles CSV for thermal profiles")
    if not houses:
        houses = [23, 40]
    # Determine window
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + pd.Timedelta(hours=int(duration_hours)) - pd.Timedelta(minutes=15)
    window = df.loc[start_dt:end_dt]
    if window.empty:
        raise ValueError("No data in requested time window for thermal profiles")
    # Build column lists; silently skip houses without required columns
    hot_cols: List[str] = []
    heat_cols: List[str] = []
    for hid in houses:
        hot = f"LV4_101_Load_{hid}_hotwater"
        heat = f"LV4_101_Load_{hid}_heating"
        if hot in window.columns:
            hot_cols.append(hot)
        else:
            print(f"Warning: missing hotwater column for house {hid} -> {hot}")
        if heat in window.columns:
            heat_cols.append(heat)
        else:
            print(f"Warning: missing heating column for house {hid} -> {heat}")
    if not hot_cols and not heat_cols:
        raise KeyError("No hotwater or heating columns found for requested houses")
    # Prepare numeric arrays
    hot_df = window[hot_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0) if hot_cols else pd.DataFrame(index=window.index)
    heat_df = window[heat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0) if heat_cols else pd.DataFrame(index=window.index)
    n_steps = len(window)
    x = np.arange(n_steps)
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        fig, axes = plt.subplots(2, 1, figsize=(width_in, height_in), sharex=True)
        hot_ax, heat_ax = axes
        palette = ['#3445A0', '#3A9D6C', '#D82E1D', '#9467BD', '#FF7F0E']
        # Build friendly label map: 23 -> SFH, 41 (or 40) -> MFH; fallback to Load <id>
        def _label_for_col(col: str) -> str:
            m = re.search(r"LV4_101_Load_(\d+)_(hotwater|heating)$", col)
            if not m:
                return col.replace('_', '\\_')
            hid = int(m.group(1))
            if hid == 23:
                return "SFH"
            if hid in (40, 41):
                return "MFH"
            return f"Load {hid}"

        # Top: hot water
        if hot_cols:
            for i, col in enumerate(hot_cols):
                color = palette[i % len(palette)]
                label = _label_for_col(col)
                hot_ax.plot(x, hot_df[col].to_numpy(), color=color, linewidth=1.2, label=label)
            try:
                hot_ax.legend(loc='upper left', frameon=False, fontsize=8)
            except Exception:
                pass
        else:
            hot_ax.text(0.5, 0.5, 'No hotwater columns', ha='center', va='center', transform=hot_ax.transAxes, fontsize=8, color='gray')
        hot_ax.set_ylabel('DHW Demand (kW)')
        hot_ax.grid(alpha=0.3)
        _force_plain_ticks(hot_ax, which="y")
        # Bottom: heating
        if heat_cols:
            for i, col in enumerate(heat_cols):
                color = palette[i % len(palette)]
                label = _label_for_col(col)
                heat_ax.plot(x, heat_df[col].to_numpy(), color=color, linewidth=1.2, label=label)
            try:
                heat_ax.legend(loc='upper left', frameon=False, fontsize=8)
            except Exception:
                pass
        else:
            heat_ax.text(0.5, 0.5, 'No heating columns', ha='center', va='center', transform=heat_ax.transAxes, fontsize=8, color='gray')
        heat_ax.set_ylabel('Heating Demand (kW)')
        heat_ax.set_xlabel('Time Step')
        heat_ax.grid(alpha=0.3)
        _force_plain_ticks(heat_ax, which="both")
        # Fixed time ticks 0,24,...,96; ensure final label visibility (extend xlim)
        try:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            base_ticks = [0, 24, 48, 72, 96]
            ticks = [t for t in base_ticks if t <= (n_steps - 1) or t == n_steps]
            if ticks:
                max_tick = max(ticks)
                heat_ax.set_xlim(0, max(n_steps - 1, max_tick))
                heat_ax.xaxis.set_major_locator(FixedLocator(ticks))
                heat_ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        except Exception:
            pass
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
        fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out
# --------------------- Generic image -> PGF exporter ---------------------

def save_image_as_pgf(
    image_path: str,
    output_path: str,
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
) -> str:
    """Create a stable PGF wrapper that includes the source PNG directly.

    This avoids Matplotlib PGF's companion "-img0.png" files and makes LaTeX
    includes robust. The function copies the input PNG next to the PGF with the
    same stem and references it via \pgfimage at the requested width.
    """
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_pgf = Path(output_path)
    out_dir = out_pgf.parent if out_pgf.parent.as_posix() not in ("", ".") else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy PNG alongside PGF with predictable name: <stem>.png
    copied_png = out_dir / f"{out_pgf.stem}.png"
    try:
        if src.resolve() != copied_png.resolve():
            shutil.copyfile(src, copied_png)
    except Exception as e:
        raise RuntimeError(f"Failed to copy image to {copied_png}: {e}")

    width_spec = f"{width_cm:.2f}cm"
    label = out_pgf.stem.replace('-', '_')

    pgf_code = (
        "% Auto-generated PGF wrapper for raster image; do not edit\n"
        "\\begingroup\n"
        "\\makeatletter\n"
        f"\\pgfdeclareimage[width={width_spec}]{{{label}}}{{{copied_png.name}}}\n"
        f"\\pgfuseimage{{{label}}}\n"
        "\\makeatother\n"
        "\\endgroup\n"
    )

    try:
        out_pgf.write_text(pgf_code, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write PGF file {out_pgf}: {e}")

    return str(out_pgf)


# --------------------- OOS plotting helpers (data loading) ---------------------

def _epsilon_token(eps: float) -> str:
    return f"{eps:.2f}".replace(".", "_")


def _load_oos_meta(eps: Optional[float]) -> Dict:
    if eps is None:
        meta_path = Path(RESULTS_DIR) / "v3_meta_drcc_false.json"
    else:
        meta_path = Path(RESULTS_DIR) / f"v3_meta_drcc_true_epsilon_{_epsilon_token(eps)}.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _compute_overload_energy_from_parquet(parquet_path: Path, threshold_pct: float = 80.0,
                                          rated_trafo_mva: float = 0.5, step_hours: float = 0.25) -> float:
    import pandas as pd
    try:
        pdf = pd.read_parquet(parquet_path)
    except Exception:
        return float('nan')
    must = {'sample_id','t','trafo_index','loading_pct'}
    if not must <= set(pdf.columns):
        return float('nan')
    lp = pd.to_numeric(pdf['loading_pct'], errors='coerce').to_numpy()
    mask = np.isfinite(lp) & (lp > threshold_pct)
    if not np.any(mask):
        return 0.0
    excess_pct = lp[mask] - threshold_pct
    excess_mva = (excess_pct / 100.0) * rated_trafo_mva
    total_mvah = float(np.sum(excess_mva) * step_hours)
    try:
        n_samples = int(pd.to_numeric(pdf['sample_id'], errors='coerce').dropna().nunique())
    except Exception:
        n_samples = 1000
    n_samples = n_samples if n_samples > 0 else 1000
    total_kwh_per_sample = (total_mvah * 1000.0) / float(n_samples)
    return total_kwh_per_sample


def _load_trafo_profile(parquet_path: Path, threshold_pct: float = 80.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    import pandas as pd
    try:
        pdf = pd.read_parquet(parquet_path)
    except Exception:
        return None
    must = {'sample_id','t','trafo_index','loading_pct'}
    if not must <= set(pdf.columns):
        return None
    grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
    counts = grp.groupby('t')['sample_id'].nunique()
    viol = grp[grp['loading_pct'] > threshold_pct].groupby('t')['sample_id'].nunique()
    rate_series = (viol / counts).reindex(counts.index).fillna(0.0)
    return counts.index.to_numpy(), rate_series.to_numpy()


def _load_flat_loading_distribution(parquet_path: Path) -> np.ndarray:
    import pandas as pd
    try:
        pdf = pd.read_parquet(parquet_path)
    except Exception:
        return np.array([])
    must = {'sample_id','t','trafo_index','loading_pct'}
    if not must <= set(pdf.columns):
        return np.array([])
    grp = pdf.groupby(['sample_id','t'])['loading_pct'].max().reset_index()
    arr = pd.to_numeric(grp['loading_pct'], errors='coerce').to_numpy()
    return arr[np.isfinite(arr)]


# --------------------- OOS plots (native Matplotlib -> PGF) ---------------------

def export_overload_energy_compare_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "overload_energy_compare_det_vs_010.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
) -> str:
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    det_meta = _load_oos_meta(None)
    eps_meta = _load_oos_meta(0.10)
    det_pq = det_meta.get('trafo_loading_file')
    eps_pq = eps_meta.get('trafo_loading_file')
    det_val = float('nan')
    eps_val = float('nan')
    if det_pq:
        det_val = _compute_overload_energy_from_parquet(Path(RESULTS_DIR) / det_pq)
    if eps_pq:
        eps_val = _compute_overload_energy_from_parquet(Path(RESULTS_DIR) / eps_pq)
    labels = ['Deterministic', r'DRCC $\varepsilon$=0.10']
    values = [det_val if np.isfinite(det_val) else 0.0,
              eps_val if np.isfinite(eps_val) else 0.0]
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, width=BAR_WIDTH, color="#3445A0", alpha=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Total transformer overload energy [kWh]')
        ax.grid(axis='y', alpha=0.3)
        _apply_max_ticks(ax, max_ticks_x=max_ticks_x, max_ticks_y=max_ticks_y, integer_x=True, integer_y=False)
        # Only format y ticks numerically so custom LaTeX x labels remain
        _force_plain_ticks(ax, which="y")
        for rect, val in zip(bars, values):
            y = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2, y + max(0.01*y, 0.01), f"{val:.2f}", ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


def export_soc_envelopes_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "soc_envelopes.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
) -> str:
    # Only include Deterministic and DRCC, eps=0.10 stacked vertically (deterministic on top)
    cases: List[Tuple[str, Path]] = []
    det_path = Path(RESULTS_DIR) / 'soc_envelope_drcc_false.csv'
    eps010_path = Path(RESULTS_DIR) / f"soc_envelope_drcc_true_epsilon_{_epsilon_token(0.10)}.csv"
    if det_path.exists():
        cases.append(('Deterministic', det_path))
    if eps010_path.exists():
        cases.append(('DRCC, $\\varepsilon$=0.10', eps010_path))
    if not cases:
        raise FileNotFoundError("No SoC envelope CSVs found for deterministic or DRCC epsilon 0.10 under v3_oos_studentt")
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        nrows = len(cases)
        fig, axes = plt.subplots(nrows, 1, figsize=(width_in, height_in), squeeze=False, sharex=True)
        axes = axes[:, 0]
        for i, (lab, path) in enumerate(cases):
            ax = axes[i]
            df = pd.read_csv(path)
            if not {'soc_p05','soc_p50','soc_p95'} <= set(df.columns):
                ax.text(0.5, 0.5, 'Invalid SoC envelope CSV', ha='center', va='center', transform=ax.transAxes, color='red')
                continue
            t = np.arange(len(df))
            ax.fill_between(t, df['soc_p05'], df['soc_p95'], color='#3445A0', alpha=0.4)
            ax.plot(t, df['soc_p50'], color='#3445A0', linewidth=1.4)
            ax.set_ylim(0.2, 0.8)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_title(lab)
            if i == nrows - 1:
                ax.set_xlabel('Time Step')
            ax.set_ylabel('BESS Capacity')
            ax.grid(alpha=0.3)
            _apply_max_ticks(ax, max_ticks_x=max_ticks_x, max_ticks_y=max_ticks_y, integer_x=True, integer_y=False)
            # Apply fixed time ticks (0,24,...,96) after generic locator to ensure they stick;
            # expand x-limits so the final label (e.g., 96) is visible.
            if i == nrows - 1:
                try:
                    from matplotlib.ticker import FixedLocator, FixedFormatter
                    total_steps = int(len(t))
                    base_ticks = [0, 24, 48, 72, 96]
                    ticks = [tt for tt in base_ticks if tt <= total_steps]
                    if ticks:
                        ax.set_xlim(0, max(total_steps, max(ticks)))
                        ax.xaxis.set_major_locator(FixedLocator(ticks))
                        ax.xaxis.set_major_formatter(FixedFormatter([str(tt) for tt in ticks]))
                except Exception:
                    pass
            # Enforce exact y ticks as requested
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            _force_plain_ticks(ax, which="both")
        fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


def export_trafo_violation_heatmap_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "trafo_violation_heatmap.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
    legend_method: str = "none",
    cbar_width_cm: Optional[float] = None,
) -> str:
    # Build profiles for baseline + epsilons present
    profiles: List[Tuple[str, np.ndarray]] = []
    t_axis: Optional[np.ndarray] = None
    # Collect available series
    base_rate: Optional[np.ndarray] = None
    eps_rates: Dict[float, np.ndarray] = {}
    # Baseline
    base_meta = _load_oos_meta(None)
    rel = base_meta.get('trafo_loading_file')
    if rel:
        res = _load_trafo_profile(Path(RESULTS_DIR) / rel, threshold_pct=80.0)
        if res:
            t_axis, base_rate = res
    # DRCC eps candidates
    for eps in [0.30, 0.20, 0.10, 0.05]:
        meta = _load_oos_meta(eps)
        rel = meta.get('trafo_loading_file') if isinstance(meta, dict) else None
        if not rel:
            continue
        res = _load_trafo_profile(Path(RESULTS_DIR) / rel, threshold_pct=80.0)
        if not res:
            continue
        t_local, rate = res
        if t_axis is None:
            t_axis = t_local
        # Align rate length later after ordering
        eps_rates[eps] = rate
    if (base_rate is None) and (not eps_rates) or t_axis is None:
        raise RuntimeError("No transformer loading parquet data available to build heatmap")
    # Build ordered list: Deterministic, 0.30, 0.20, 0.10 (only include available)
    ordered_profiles: List[Tuple[str, np.ndarray]] = []
    if base_rate is not None:
        ordered_profiles.append(("Deterministic", base_rate))
    for eps in [0.30, 0.20, 0.10]:
        if eps in eps_rates:
            # Show case name directly on y-axis as requested: "DRCC, $\varepsilon$=0.XX"
            ordered_profiles.append((rf"DRCC, $\varepsilon$={eps:.2f}", eps_rates[eps]))
    # Align lengths across all rows
    min_len = min(len(r) for _, r in ordered_profiles)
    mat = np.vstack([r[:min_len] for _, r in ordered_profiles])
    mat = np.clip(mat, 0.0, 0.5)
    labels = [lab for lab, _ in ordered_profiles]
    # If using a pgfplots-based smooth colorbar, reserve space for the colorbar by
    # rendering the core plot narrower. We'll re-assemble in a wrapper PGF.
    cbar_width_cm_eff = float(cbar_width_cm) if cbar_width_cm is not None else max(0.35, width_cm * 0.12)
    total_width_cm = float(width_cm)
    if legend_method == "pgfplots_inline":
        core_width_cm = max(0.0, total_width_cm - cbar_width_cm_eff)
    else:
        core_width_cm = total_width_cm
    width_in = _cm_to_inch(core_width_cm)
    height_in = width_in * aspect if core_width_cm > 0 else _cm_to_inch(total_width_cm) * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
    # Use pcolormesh to avoid external raster image files in PGF
        ny, nx = mat.shape
        x_edges = np.arange(nx+1)
        y_edges = np.arange(ny+1)
        # Colormap: white at 0, brand blue (#3445A0) at 0.5 and above
        try:
            from matplotlib.colors import LinearSegmentedColormap
            brand_blue = '#3445A0'
            cmap = LinearSegmentedColormap.from_list('white_to_brandblue', ['#FFFFFF', brand_blue])
        except Exception:
            cmap = mpl.cm.Blues
        quad = ax.pcolormesh(x_edges, y_edges, mat, cmap=cmap, vmin=0.0, vmax=0.5, shading='flat')
        # Limit ticks manually: choose evenly spaced columns and cases
        ny, nx = mat.shape
        xt_count = max(2, min(max_ticks_x, nx))
        xt_pos = np.linspace(0, nx-1, num=xt_count, dtype=int)
        ax.set_xticks(xt_pos + 0.5)
        ax.set_xticklabels([str(int(i)) for i in xt_pos])
        yt_count = max(1, min(max_ticks_y, ny))
        yt_pos = np.linspace(0, ny-1, num=yt_count, dtype=int)
        ax.set_yticks(yt_pos + 0.5)
        ax.set_yticklabels([labels[i] for i in yt_pos])
        # Ensure visual order has row 0 at the top (Deterministic top, eps=0.10 bottom)
        try:
            ax.invert_yaxis()
        except Exception:
            pass
        ax.set_xlabel('Time Step')
        # Replace automatic ticks with fixed time ticks at 0,24,...,96 (labels), positioned at cell centers.
        # Ensure the final label (e.g., 96) is present by clamping its position to the last cell center.
        try:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            total_steps = int(nx)
            base_ticks = [0, 24, 48, 72, 96]
            ticks = [t for t in base_ticks if t <= total_steps]
            if ticks:
                # Clamp label positions to the last cell center (nx - 0.5)
                positions = [min(t + 0.5, nx - 0.5) for t in ticks]
                ax.set_xlim(0, nx)
                ax.xaxis.set_major_locator(FixedLocator(positions))
                ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        except Exception:
            pass
        ax.set_ylabel('Optimization Case')
    # Do not override y tick formatter so manual string labels (case names) stay intact
        # Legend handling
        # - "discrete": draw manual rectangle stack
        # - "pgfplots_inline": will be handled after saving (inline pgfplots colorbar)
        # - "none": no legend/colorbar drawn at all (core heatmap only)
        if legend_method == "discrete":
            try:
                from matplotlib.patches import Rectangle
                legend_ax = ax.inset_axes([1.02, 0.1, 0.03, 0.8], transform=ax.transAxes)
                legend_ax.set_axis_off()
                levels = np.linspace(0.0, 0.5, 6)
                h = 1.0 / (len(levels)-1)
                for i in range(len(levels)-1):
                    c = cmap((levels[i] - 0.0) / (0.5 - 0.0))
                    legend_ax.add_patch(Rectangle((0, i*h), 1.0, h, transform=legend_ax.transAxes, color=c, ec='none'))
                legend_ax.text(1.2, 0.0, '0.0', transform=legend_ax.transAxes, va='center', fontsize=8)
                legend_ax.text(1.2, 1.0, '0.5', transform=legend_ax.transAxes, va='center', fontsize=8)
                ax.text(1.06, 0.92, 'violation probability', transform=ax.transAxes, rotation=90, fontsize=8)
            except Exception:
                pass
        fig.tight_layout()

        # Save either directly (discrete legend) or as a core + wrapper (pgfplots colorbar)
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)

    # legend_method "none": no post-processing
    if legend_method == "none":
        return pgf_out
    return pgf_out


# (Removed previous wrapper helper; inline method now used)


def export_violin_compare_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "violin_compare_det_vs_010.pgf"),
    width_cm: float = 5.3,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
) -> str:
    # Load distributions from parquet
    det_meta = _load_oos_meta(None)
    eps_meta = _load_oos_meta(0.10)
    det_pq = det_meta.get('trafo_loading_file')
    eps_pq = eps_meta.get('trafo_loading_file')
    if not det_pq and not eps_pq:
        raise RuntimeError("Missing transformer loading parquet paths for violin compare")
    left_vals = _load_flat_loading_distribution(Path(RESULTS_DIR) / det_pq) if det_pq else np.array([])
    right_vals = _load_flat_loading_distribution(Path(RESULTS_DIR) / eps_pq) if eps_pq else np.array([])
    left_vals = left_vals[np.isfinite(left_vals)] if left_vals.size else np.array([])
    right_vals = right_vals[np.isfinite(right_vals)] if right_vals.size else np.array([])
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        x0 = 1.0
        if not left_vals.size and not right_vals.size:
            ax.text(0.5, 0.5, 'No transformer loading data', ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
        else:
            vals_list = []
            side_tags = []
            if left_vals.size:
                vals_list.append(left_vals)
                side_tags.append('left')
            if right_vals.size:
                vals_list.append(right_vals)
                side_tags.append('right')
            positions = [x0 for _ in vals_list]
            vp = ax.violinplot(vals_list, positions=positions, showmeans=False, showmedians=False, showextrema=False)
            from matplotlib.patches import Rectangle, Patch
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            left_clip = Rectangle((xmin, ymin), width=(x0 - xmin), height=(ymax - ymin), transform=ax.transData)
            right_clip = Rectangle((x0, ymin), width=(xmax - x0), height=(ymax - ymin), transform=ax.transData)
            for body, tag in zip(vp['bodies'], side_tags):
                if tag == 'left':
                    body.set_facecolor('#3A9D6C'); body.set_edgecolor('#3A9D6C'); body.set_alpha(0.4); body.set_clip_path(left_clip); body.set_linewidth(0.8)
                else:
                    body.set_facecolor('#3445A0'); body.set_edgecolor('#3445A0'); body.set_alpha(0.4); body.set_clip_path(right_clip); body.set_linewidth(0.8)
            # Median + min/max ticks (min/max: solid, half-length, touching central spine)
            line_half = 0.075  # half-length for min/max lines
            if left_vals.size:
                m_left = float(np.nanmedian(left_vals))
                ax.plot([x0 - 0.21, x0], [m_left, m_left], color='#3A9D6C', linewidth=1.0)
                min_left = float(np.nanmin(left_vals))
                max_left = float(np.nanmax(left_vals))
                ax.plot([x0 - line_half, x0], [min_left, min_left], color='#3A9D6C', linewidth=0.8, alpha=0.9)
                ax.plot([x0 - line_half, x0], [max_left, max_left], color='#3A9D6C', linewidth=0.8, alpha=0.9)
            if right_vals.size:
                m_right = float(np.nanmedian(right_vals))
                ax.plot([x0, x0 + 0.21], [m_right, m_right], color='#3445A0', linewidth=1.0)
                min_right = float(np.nanmin(right_vals))
                max_right = float(np.nanmax(right_vals))
                ax.plot([x0, x0 + line_half], [min_right, min_right], color='#3445A0', linewidth=0.8, alpha=0.9)
                ax.plot([x0, x0 + line_half], [max_right, max_right], color='#3445A0', linewidth=0.8, alpha=0.9)
            # Add deterministic and DRCC labels left/right of central spine
            left_x = x0 - 0.21
            right_x = x0 + 0.21
            ax.set_xlim(x0 - 0.4, x0 + 0.4)
            ax.set_xticks([left_x, right_x])
            ax.set_xticklabels(['Deterministic', r'DRCC $\varepsilon$=0.10'])
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            ax.set_ylabel('Transformer loading [%]'); ax.grid(axis='y', alpha=0.3)
            # Apply only Y tick constraints; leave X to our fixed category labels
            _apply_max_ticks(ax, max_ticks_x=None, max_ticks_y=max_ticks_y, integer_x=False, integer_y=False)
            ax.axvline(x0, color='black', linewidth=0.9, alpha=0.8, zorder=3)
            # Ensure axis spines are not thicker than 1.0
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
        # Only format y ticks so LaTeX x-category labels stay intact
        _force_plain_ticks(ax, which="y")
        # Ensure grid lines don't exceed 1.0 width
        try:
            ax.grid(True, axis='y', alpha=0.3, linewidth=0.8)
        except Exception:
            pass
        fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


def export_frontier_hybrid_scatter_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "frontier_hybrid_scatter.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
    max_cloud_points: int = DEFAULT_MAX_CLOUD_POINTS,
    cloud_marker_size_drcc: int = DEFAULT_CLOUD_MARKER_SIZE_DRCC,
    cloud_marker_size_base: int = DEFAULT_CLOUD_MARKER_SIZE_BASE,
    show_epsilon_legend: bool = False,
) -> str:
    # Load mean frontier from frontier_summary.csv if present; else build from v3_summary_*.csv
    summary_csv = Path(RESULTS_DIR) / 'frontier_summary.csv'
    frontier_df = None
    if summary_csv.exists():
        try:
            frontier_df = pd.read_csv(summary_csv)
        except Exception:
            frontier_df = None
    if frontier_df is None or frontier_df.empty:
        # Build from available summaries
        rows: List[Dict[str, object]] = []
        for path in glob.glob(str(Path(RESULTS_DIR) / 'v3_summary_*.csv')):
            try:
                df_sum = pd.read_csv(path)
            except Exception:
                continue
            if df_sum.empty or 'total_cost_eur' not in df_sum.columns:
                continue
            fname = os.path.basename(path)
            if 'drcc_false_epsilon_' in fname:
                continue
            if 'drcc_false' in fname:
                mode = 'stochastic'; eps_val = None
            else:
                m = re.search(r'_epsilon_([0-9]+_[0-9]+)', fname)
                eps_val = float(m.group(1).replace('_','.')) if m else None
                mode = 'drcc_true'
            mean_cost = float(pd.to_numeric(df_sum['total_cost_eur'], errors='coerce').dropna().mean())
            if {'steps_trafo_over_80pct','n_steps'} <= set(df_sum.columns):
                rates = []
                for _, r in df_sum.iterrows():
                    ns = float(r.get('n_steps', np.nan)); st = float(r.get('steps_trafo_over_80pct', np.nan))
                    if ns > 0 and np.isfinite(st):
                        rates.append(st/ns)
                vrate = float(np.mean(rates)) if rates else float('nan')
            else:
                vrate = float('nan')
            rows.append({'mode': mode, 'epsilon': eps_val, 'mean_cost_eur': mean_cost, 'trafo_violation_rate_mean': vrate, 'n_trajectories': len(df_sum)})
        frontier_df = pd.DataFrame(rows)
    # Build cloud from per-trajectory summaries
    cloud_points = []
    base_path = Path(RESULTS_DIR) / 'v3_summary_drcc_false.csv'
    if base_path.exists():
        try:
            dfb = pd.read_csv(base_path)
            if {'steps_trafo_over_80pct','n_steps','total_cost_eur'} <= set(dfb.columns):
                for _, r in dfb.iterrows():
                    ns = float(r.get('n_steps', np.nan)); st = float(r.get('steps_trafo_over_80pct', np.nan))
                    if ns > 0 and np.isfinite(st):
                        cloud_points.append({'epsilon': None,'mode':'stochastic','vrate': st/ns,'cost': float(r.get('total_cost_eur', np.nan))})
        except Exception:
            pass
    for path in glob.glob(str(Path(RESULTS_DIR) / 'v3_summary_drcc_true_epsilon_*.csv')):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        m = re.search(r'_epsilon_([0-9]+_[0-9]+)\.csv$', path)
        eps_val = float(m.group(1).replace('_','.')) if m else None
        if {'steps_trafo_over_80pct','n_steps','total_cost_eur'} - set(df.columns):
            continue
        for _, r in df.iterrows():
            ns = float(r.get('n_steps', np.nan)); st = float(r.get('steps_trafo_over_80pct', np.nan))
            if ns > 0 and np.isfinite(st):
                cloud_points.append({'epsilon': eps_val,'mode':'drcc_true','vrate': st/ns,'cost': float(r.get('total_cost_eur', np.nan))})
    cloud_df = pd.DataFrame(cloud_points) if cloud_points else None
    if frontier_df is None or frontier_df.empty:
        raise RuntimeError("No frontier data available to plot")
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        # Cloud first (downsample to keep PGF size manageable for LaTeX)
        if cloud_df is not None and not cloud_df.empty:
            drcc_cloud = cloud_df[cloud_df['mode'] == 'drcc_true']
            base_cloud = cloud_df[cloud_df['mode'] == 'stochastic']
            # Helper: evenly-spaced subsample without randomness
            def _linspace_sample(df, n):
                try:
                    n = int(n)
                except Exception:
                    n = 0
                if n <= 0:
                    return df.iloc[0:0]
                if len(df) <= n:
                    return df
                idxs = np.linspace(0, len(df)-1, num=n, dtype=int)
                return df.iloc[idxs]
            if not drcc_cloud.empty:
                # Subsample DRCC cloud to at most max_cloud_points
                drcc_s = _linspace_sample(drcc_cloud.reset_index(drop=True), max_cloud_points)
                eps_vals = drcc_s['epsilon'].to_numpy(dtype=float)
                norm_c = plt.Normalize(vmin=np.nanmin(eps_vals), vmax=np.nanmax(eps_vals))
                cmap_c = plt.cm.viridis
                ax.scatter(drcc_s['vrate'], drcc_s['cost'], c=eps_vals, cmap=cmap_c,
                           s=float(cloud_marker_size_drcc), alpha=0.12, edgecolors='none', zorder=2)
            if not base_cloud.empty:
                base_cap = max(200, max_cloud_points // 10)
                base_s = _linspace_sample(base_cloud.reset_index(drop=True), base_cap)
                ax.scatter(base_s['vrate'], base_s['cost'], marker='o', s=float(cloud_marker_size_base), c='black', alpha=0.15, edgecolors='none', zorder=1)
        # Mean overlay
        base_mean = frontier_df[frontier_df['mode'] == 'stochastic']
        drcc_mean = frontier_df[frontier_df['mode'] != 'stochastic']
        if not drcc_mean.empty:
            eps_mean = drcc_mean['epsilon'].to_numpy(dtype=float)
            vmin_m, vmax_m = float(np.nanmin(eps_mean)), float(np.nanmax(eps_mean))
            norm_m = plt.Normalize(vmin=vmin_m, vmax=vmax_m)
            cmap_m = plt.cm.viridis
            for e in sorted(pd.unique(drcc_mean['epsilon'].dropna())):
                subm = drcc_mean[np.isclose(drcc_mean['epsilon'].astype(float), e)]
                ax.scatter(subm['trafo_violation_rate_mean'], subm['mean_cost_eur'], color=cmap_m(norm_m(e)), s=70, edgecolors='k', linewidths=0.4, zorder=4)
                # Label the optimization case on the mean point (LaTeX epsilon)
                try:
                    xv = float(subm['trafo_violation_rate_mean'].iloc[0])
                    yv = float(subm['mean_cost_eur'].iloc[0])
                    ax.text(
                        xv + 0.005,
                        yv,
                        rf"$\varepsilon={e:.2f}$",
                        fontsize=8,
                        va='center',
                        ha='left',
                        zorder=10,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='square,pad=0.12')
                    )
                except Exception:
                    pass
            # Omit colorbar to avoid rasterized gradient in PGF; encode epsilon by color only
        if not base_mean.empty:
            ax.scatter(base_mean['trafo_violation_rate_mean'], base_mean['mean_cost_eur'], marker='o', s=85, c='black', edgecolors='white', linewidths=0.4, zorder=3)
            try:
                xv = float(base_mean['trafo_violation_rate_mean'].iloc[0])
                yv = float(base_mean['mean_cost_eur'].iloc[0])
                ax.text(
                    xv + 0.005,
                    yv,
                    "Deterministic",
                    fontsize=8,
                    va='center',
                    ha='left',
                    zorder=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='square,pad=0.12')
                )
            except Exception:
                pass
        ax.set_xlabel('Transformer violation rate (trajectory / mean)')
        ax.set_ylabel('Total cost (EUR)')
        ax.grid(alpha=0.35)
        _apply_max_ticks(ax, max_ticks_x=max_ticks_x, max_ticks_y=max_ticks_y, integer_x=False, integer_y=False)
        _force_plain_ticks(ax, which="both", percent_x=True)
        # Optional epsilon legend on the right (hidden by default)
        if show_epsilon_legend:
            try:
                unique_eps = sorted(pd.unique(drcc_mean['epsilon'].dropna())) if not drcc_mean.empty else []
                if unique_eps:
                    leg_ax = ax.inset_axes([1.02, 0.12, 0.035, 0.76], transform=ax.transAxes)
                    leg_ax.set_axis_off()
                    # Draw discrete gradient from min to max epsilon (vector-only)
                    e_min, e_max = float(min(unique_eps)), float(max(unique_eps))
                    n_bins = 100
                    edges = np.linspace(e_min, e_max, n_bins+1)
                    for i in range(n_bins):
                        e_mid = 0.5*(edges[i] + edges[i+1])
                        c = plt.cm.viridis((e_mid - e_min) / (e_max - e_min if e_max > e_min else 1.0))
                        y0 = i / n_bins
                        leg_ax.add_patch(plt.Rectangle((0, y0), 1.0, 1.0/n_bins, transform=leg_ax.transAxes, color=c, ec='none'))
                    # Label min/max
                    leg_ax.text(1.15, 0.0, f"{e_min:.2f}", transform=leg_ax.transAxes, va='center', fontsize=8)
                    leg_ax.text(1.15, 1.0, f"{e_max:.2f}", transform=leg_ax.transAxes, va='center', fontsize=8)
                    ax.text(1.08, 0.50, r"risk level ($\varepsilon$)", transform=ax.transAxes, rotation=90, fontsize=8, va='center', ha='center')
            except Exception:
                pass

        d = os.path.dirname(output_path)
        if d and d not in ("", "."):
            os.makedirs(d, exist_ok=True)
        # Save PGF and a PNG sidecar for quick inspection
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


def export_pv_temperature_uncertainty_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "pv_temperature_uncertainty.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
    pv_samples_csv: str = os.path.join("samples", "samples_pv_gaussian.csv"),
    temp_samples_csv: str = os.path.join("samples", "samples_temperature_c_gaussian.csv"),
) -> str:
    """Export stacked uncertainty bands for PV generation (aggregate MW) and ambient temperature (°C).

    Data expectation:
        pv_samples_csv: columns => timestamp,sample_id,<pv_bus_*_mw ...>
        temp_samples_csv: columns => timestamp,sample_id,temperature_c

    For PV: total PV per sample & timestamp is sum over all pv_bus_* columns.
    For temperature: direct use of temperature_c.
    We compute mean and standard deviation across sample_id for each timestamp.
    Plot mean line (solid) and shaded +/-1σ band (semi-transparent) for each series.
    """
    if not os.path.exists(pv_samples_csv):
        raise FileNotFoundError(f"PV samples CSV not found: {pv_samples_csv}")
    if not os.path.exists(temp_samples_csv):
        raise FileNotFoundError(f"Temperature samples CSV not found: {temp_samples_csv}")

    # Load PV samples
    pv_df = pd.read_csv(pv_samples_csv)
    must_cols = {"timestamp", "sample_id"}
    if not must_cols <= set(pv_df.columns):
        raise KeyError(f"PV samples missing required columns {must_cols}. Found: {pv_df.columns.tolist()}")
    pv_bus_cols = [c for c in pv_df.columns if c.startswith("pv_bus_") and c.endswith("_mw")]
    if not pv_bus_cols:
        raise KeyError("No pv_bus_*_mw columns found in PV samples CSV")
    # Convert timestamp to ordered categorical -> integer time steps
    pv_df['timestamp'] = pd.to_datetime(pv_df['timestamp'])
    pv_df.sort_values('timestamp', inplace=True)
    # Total PV per row (MW)
    pv_vals = pv_df[pv_bus_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    pv_df['total_pv_mw'] = pv_vals.sum(axis=1)
    # Normalization: divide each sample's total PV by the global maximum (capacity proxy)
    capacity_total = float(pv_df['total_pv_mw'].max())
    if capacity_total > 0:
        pv_df['total_pv_pu'] = pv_df['total_pv_mw'] / capacity_total
        pv_metric_col = 'total_pv_pu'
        pv_ylabel = 'Normalized Power'
    else:
        pv_metric_col = 'total_pv_mw'
        pv_ylabel = 'Total PV (MW)'
    # Group by timestamp -> aggregate across samples using normalized metric if available
    pv_grp = pv_df.groupby('timestamp')[pv_metric_col]
    pv_mean = pv_grp.mean().to_numpy()
    pv_std = pv_grp.std(ddof=0).to_numpy()  # population std (consistent for large samples)
    ts_index = pv_grp.mean().index.to_numpy()

    # Load temperature samples
    t_df = pd.read_csv(temp_samples_csv)
    if not must_cols | {"temperature_c"} <= set(t_df.columns):
        raise KeyError("Temperature samples CSV missing required columns 'timestamp','sample_id','temperature_c'")
    t_df['timestamp'] = pd.to_datetime(t_df['timestamp'])
    t_df.sort_values('timestamp', inplace=True)
    t_grp = t_df.groupby('timestamp')['temperature_c']
    t_mean = t_grp.mean().to_numpy()
    t_std = t_grp.std(ddof=0).to_numpy()
    t_ts_index = t_grp.mean().index.to_numpy()

    # Align lengths if they differ (truncate to shortest to maintain shared x-axis)
    common_len = min(len(ts_index), len(t_ts_index))
    ts_index = ts_index[:common_len]
    pv_mean, pv_std = pv_mean[:common_len], pv_std[:common_len]
    t_mean, t_std = t_mean[:common_len], t_std[:common_len]
    x_steps = np.arange(common_len)

    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        # Two stacked subplots sharing x-axis
        fig, axes = plt.subplots(2, 1, figsize=(width_in, height_in), sharex=True)
        pv_ax, temp_ax = axes
        # PV plot
        pv_color = '#FFA500'  # brand blue for PV
        pv_ax.fill_between(x_steps, pv_mean - pv_std, pv_mean + pv_std, facecolor=pv_color, alpha=0.35, linewidth=0.0)
        pv_ax.plot(x_steps, pv_mean, color=pv_color, linewidth=1.4)
        pv_ax.set_ylabel(pv_ylabel)
        pv_ax.grid(alpha=0.3)
        _apply_max_ticks(pv_ax, max_ticks_x=max_ticks_x, max_ticks_y=max_ticks_y, integer_x=True, integer_y=False)
        _force_plain_ticks(pv_ax, which="y")  # x handled globally

        # Temperature plot
        temp_color = '#D82E1D'  # distinct orange for temperature
        temp_ax.fill_between(x_steps, t_mean - t_std, t_mean + t_std, facecolor=temp_color, alpha=0.35, linewidth=0.0)
        temp_ax.plot(x_steps, t_mean, color=temp_color, linewidth=1.4)
        temp_ax.set_ylabel('Temperature (°C)')
        temp_ax.set_xlabel('Time Step')
        temp_ax.grid(alpha=0.3)
        _apply_max_ticks(temp_ax, max_ticks_x=max_ticks_x, max_ticks_y=max_ticks_y, integer_x=True, integer_y=False)
        _force_plain_ticks(temp_ax, which="both")

        # Replace automatic x ticks with fixed 0,24,48,72,96 if within range; ensure last tick visible by extending xlim
        try:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            base_ticks = [0, 24, 48, 72, 96]
            ticks = [t for t in base_ticks if t < common_len or (t == common_len)]
            if ticks:
                # if final tick equals common_len add axis extension so label shows
                max_tick = max(ticks)
                temp_ax.set_xlim(0, max(common_len - 1, max_tick))
                temp_ax.xaxis.set_major_locator(FixedLocator(ticks))
                temp_ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
                # hide tick markers but keep labels if desired? (follow prior style) -> keep markers for time series
        except Exception:
            pass

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
        fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


def export_house_load_profiles_pgf(
    output_path: str = os.path.join(EXPORT_PGF_DIR, "house_load_profiles.pgf"),
    width_cm: float = DEFAULT_WIDTH_CM,
    aspect: float = DEFAULT_ASPECT_LOADS,
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    max_ticks_x: int = DEFAULT_MAX_TICKS_X,
    max_ticks_y: int = DEFAULT_MAX_TICKS_Y,
    start_date: str = "2023-01-10 00:00:00",
    duration_hours: int = 24,
    profiles_csv: str = os.path.join("vdi_profiles", "all_house_profiles.csv"),
    load_cols: Optional[List[str]] = None,
) -> str:
    """Export electric load profiles for selected houses over an aligned time window.

    Parameters
    ----------
    load_cols : list of column names to plot. Defaults to ['LV4.101_Load_40','LV4.101_Load_6'] if available.
    duration_hours : number of hours (15-min resolution assumed) starting at start_date.
    We reuse alignment logic from _build_time_index_from_vdi.
    """
    if not os.path.exists(profiles_csv):
        raise FileNotFoundError(f"Profiles CSV not found: {profiles_csv}")
    df = pd.read_csv(profiles_csv, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        raise ValueError("Failed to parse datetime index in house profiles CSV")
    # Determine window
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + pd.Timedelta(hours=int(duration_hours)) - pd.Timedelta(minutes=15)
    window = df.loc[start_dt:end_dt]
    if window.empty:
        raise ValueError("No data in requested time window for load profiles")
    # Resolve requested columns to actual electricity columns in the CSV
    def _resolve_load_columns(req: Optional[List[str]], cols: List[str]) -> List[str]:
        all_cols = list(cols)
        # Prefer explicit electricity channels
        elec_cols = [c for c in all_cols if c.lower().endswith('_electricity')]
        if not req:
            # Smart default to two example houses if present; else first two electricity cols
            preferred = ['LV4_101_Load_40_electricity', 'LV4_101_Load_23_electricity']
            found = [c for c in preferred if c in all_cols]
            if len(found) < 2 and elec_cols:
                # Fill with first electricity columns deterministically
                for c in elec_cols:
                    if c not in found:
                        found.append(c)
                    if len(found) >= 2:
                        break
            return found
        resolved: List[str] = []
        for r in req:
            if r in all_cols:
                resolved.append(r)
                continue
            # Build common variations
            base = r.replace('.', '_')
            cand1 = base if base.endswith('_electricity') else base + '_electricity'
            cand2 = base + '_electric'
            # Try exact matches
            for c in (cand1, cand2):
                if c in all_cols:
                    resolved.append(c)
                    break
            else:
                # Fuzzy: any column containing the load id and electricity
                tokens = [t for t in base.split('_') if t]
                def _looks_like(col: str) -> bool:
                    low = col.lower()
                    return (tokens[-1].lower() in low) and ('electric' in low)
                matches = [c for c in all_cols if _looks_like(c)]
                if matches:
                    resolved.append(matches[0])
        return resolved

    candidate_cols = _resolve_load_columns(load_cols, list(window.columns))
    if not candidate_cols:
        # As ultimate fallback, take the first two electricity columns if they exist
        fallback_elec = [c for c in window.columns if c.lower().endswith('_electricity')]
        candidate_cols = fallback_elec[:2]
    if not candidate_cols:
        raise KeyError("No electricity load columns found in profiles CSV (expected *_electricity)")
    # Build time step axis (0..N-1) for plotting with integer ticks
    # Use resolved candidate_cols (NOT original load_cols which may be None or unmapped)
    values = window[candidate_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    n_steps = len(values)
    x = np.arange(n_steps)
    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        # Plot each load profile
        palette = ['#3445A0', '#3A9D6C', '#D82E1D', '#9467BD', '#FF7F0E']
        label_map = {
            'LV4_101_Load_40_electricity': 'MFH',
            'LV4_101_Load_23_electricity': 'SFH',
        }
        for i, col in enumerate(candidate_cols):
            color = palette[i % len(palette)]
            lab = label_map.get(col, col.replace('_', '\\_'))
            ax.plot(x, values[col].to_numpy(), label=lab, linewidth=1.2, color=color)
        ax.set_ylabel('Electric Load (kW)')
        ax.set_xlabel('Time Step')
        ax.grid(alpha=0.3)
        # Fixed time ticks at 0,24,48,72,96 if present
        try:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            base_ticks = [0, 24, 48, 72, 96]
            ticks = [t for t in base_ticks if t <= (n_steps - 1)]
            if ticks:
                ax.set_xlim(0, max(n_steps - 1, max(ticks)))
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        except Exception:
            pass
        _force_plain_ticks(ax, which="y")
        # Legend inside, upper left
        try:
            ax.legend(loc='upper left', frameon=False, fontsize=8)
        except Exception:
            pass
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        fig.tight_layout()
        pgf_out, png_out = _save_dual(fig, output_path)
        plt.close(fig)
    return pgf_out


# --------------------- CLI ---------------------

def _export_selected(
    selections: Sequence[str],
    width_cm: float,
    aspects: Dict[str, float],
    texsystem: str,
    use_sfmath: bool,
    ts_args: dict,
    max_ticks_x: int,
    max_ticks_y: int,
    cloud_marker_size_drcc: Optional[int] = None,
    cloud_marker_size_base: Optional[int] = None,
    violin_width_cm: Optional[float] = None,
    png_dpi: Optional[int] = None,
) -> None:
    # common kwargs for OOS exporters (width passed per-plot below)
    common = dict(texsystem=texsystem, use_sfmath=use_sfmath,
                  max_ticks_x=max_ticks_x, max_ticks_y=max_ticks_y)

    if "ts" in selections or "all" in selections:
        # Always use local exporter to guarantee PNG alongside PGF
        export_thermal_storage_pgf(
            width_cm=width_cm,
            aspect=aspects.get("ts", DEFAULT_ASPECT_TS),
            texsystem=texsystem,
            use_sfmath=use_sfmath,
            **ts_args,
        )
        print(f"Saved: {os.path.join(EXPORT_PGF_DIR, 'thermal_storage_operation_area.pgf')} (+ PNG)")

    if "overload" in selections or "all" in selections:
        # Use same narrower width and height (aspect) as violin plot so visual style aligns
        narrow_w = violin_width_cm if violin_width_cm is not None else (width_cm / 2.0)
        overload_aspect = aspects.get("violin", DEFAULT_ASPECT_VIOLIN)
        out = export_overload_energy_compare_pgf(width_cm=narrow_w, aspect=overload_aspect, **common)
        print(f"Saved (narrow overload size): {out} (+ PNG)")

    if "soc" in selections or "all" in selections:
        out = export_soc_envelopes_pgf(width_cm=width_cm, aspect=aspects.get("soc", DEFAULT_ASPECT_SOC), **common)
        print(f"Saved: {out} (+ PNG)")

    if "trafo" in selections or "all" in selections:
        try:
            # Make heatmap 1.5 cm narrower than the global/default width (user request)
            heatmap_width_cm = max(1.0, float(width_cm) - 1.5)
            out = export_trafo_violation_heatmap_pgf(width_cm=heatmap_width_cm, aspect=aspects.get("trafo", DEFAULT_ASPECT_HEATMAP), **common)
            print(f"Saved: {out} (+ PNG)")
        except Exception as e:
            print(f"Skipping transformer violation heatmap: {e}")

    if "violin" in selections or "all" in selections:
        vw = violin_width_cm if violin_width_cm is not None else (width_cm / 2.0)
        out = export_violin_compare_pgf(width_cm=vw, aspect=aspects.get("violin", DEFAULT_ASPECT_VIOLIN), **common)
        print(f"Saved: {out} (+ PNG)")

    if "frontier" in selections or "all" in selections:
        out = export_frontier_hybrid_scatter_pgf(
            width_cm=width_cm,
            aspect=aspects.get("frontier", DEFAULT_ASPECT_FRONTIER),
            **common,
            cloud_marker_size_drcc=(cloud_marker_size_drcc if cloud_marker_size_drcc is not None else DEFAULT_CLOUD_MARKER_SIZE_DRCC),
            cloud_marker_size_base=(cloud_marker_size_base if cloud_marker_size_base is not None else DEFAULT_CLOUD_MARKER_SIZE_BASE),
        )
        print(f"Saved: {out} (+ PNG)")

    if "pvtemp" in selections or "all" in selections:
        out = export_pv_temperature_uncertainty_pgf(width_cm=width_cm, aspect=aspects.get("pvtemp", DEFAULT_ASPECT), **common)
        print(f"Saved: {out} (+ PNG)")

    if "loads" in selections or "all" in selections:
        # Use TS alignment window for consistency
        start_date = ts_args.get("start_date", "2023-01-10 00:00:00")
        duration_hours = ts_args.get("duration_hours", 24)
        out = export_house_load_profiles_pgf(width_cm=width_cm,
                                             aspect=aspects.get("loads", DEFAULT_ASPECT_LOADS),
                                             texsystem=texsystem,
                                             use_sfmath=use_sfmath,
                                             start_date=start_date,
                                             duration_hours=int(duration_hours))
        print(f"Saved: {out} (+ PNG)")

    if "bev" in selections or "all" in selections:
        # Align to the same TS window
        start_date = ts_args.get("start_date", "2023-01-10 00:00:00")
        duration_hours = ts_args.get("duration_hours", 24)
        out = export_bev_profile_pgf(width_cm=width_cm,
                                     aspect=aspects.get("bev", DEFAULT_ASPECT_BEV),
                                     texsystem=texsystem,
                                     use_sfmath=use_sfmath,
                                     start_date=start_date,
                                     duration_hours=int(duration_hours),
                                     car_cols=["car_2", "car_9"],
                                     output_path=os.path.join(EXPORT_PGF_DIR, "bev_two_cars_profile.pgf"))
        print(f"Saved: {out} (+ PNG)")

    if "thermal" in selections or "all" in selections:
        start_date = ts_args.get("start_date", "2023-01-10 00:00:00")
        duration_hours = ts_args.get("duration_hours", 24)
        out = export_hotwater_heating_profiles_pgf(width_cm=width_cm,
                                                   aspect=aspects.get("thermal", DEFAULT_ASPECT_THERMAL),
                                                   texsystem=texsystem,
                                                   use_sfmath=use_sfmath,
                                                   start_date=start_date,
                                                   duration_hours=int(duration_hours))
        print(f"Saved: {out} (+ PNG)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    # If inline config is enabled, bypass CLI parsing entirely
    if RUN_WITH_INLINE_CONFIG:
        cfg = INLINE_CONFIG
        selections = list(cfg.get("selections", [])) or ["all"]
        custom_dir = cfg.get("results_dir")
        if custom_dir:
            global RESULTS_DIR
            RESULTS_DIR = str(custom_dir)
        width_cm = float(cfg.get("width_cm", DEFAULT_WIDTH_CM))
        # Build aspects using helper logic similar to CLI
        def _pick(per_plot: Optional[float], per_plot_default: float) -> float:
            global_aspect = cfg.get("aspect", None)
            if per_plot is not None:
                return float(per_plot)
            if global_aspect is not None:
                return float(global_aspect)
            return float(per_plot_default)
        aspects = {
            "ts": _pick(cfg.get("aspect_ts"), DEFAULT_ASPECT_TS),
            "overload": _pick(cfg.get("aspect_overload"), DEFAULT_ASPECT_OVERLOAD),
            "soc": _pick(cfg.get("aspect_soc"), DEFAULT_ASPECT_SOC),
            "trafo": _pick(cfg.get("aspect_heatmap"), DEFAULT_ASPECT_HEATMAP),
            "violin": _pick(cfg.get("aspect_violin"), DEFAULT_ASPECT_VIOLIN),
            "frontier": _pick(cfg.get("aspect_frontier"), DEFAULT_ASPECT_FRONTIER),
            "pvtemp": _pick(cfg.get("aspect_pvtemp"), DEFAULT_ASPECT_PVTEMP),
            "loads": _pick(cfg.get("aspect_loads"), DEFAULT_ASPECT_LOADS),
            "bev": _pick(cfg.get("aspect_bev"), DEFAULT_ASPECT_BEV),
            "thermal": _pick(cfg.get("aspect_thermal"), DEFAULT_ASPECT_THERMAL),
        }
        ts_cfg = cfg.get("ts", {})
        ts_args = dict(
            input_csv=ts_cfg.get("input", "fully_coordinated_model_results.csv"),
            output_path=os.path.join(EXPORT_PGF_DIR, "thermal_storage_operation_area.pgf"),
            price_csv=ts_cfg.get("price_csv", None),
            price_col=ts_cfg.get("price_col", None),
            start_date=ts_cfg.get("start_date", "2023-01-10 00:00:00"),
            duration_hours=int(ts_cfg.get("duration_hours", 24)),
            disable_market_price=bool(ts_cfg.get("no_market_price", False)),
            show=False,
        )
        _export_selected(
            selections=selections,
            width_cm=width_cm,
            aspects=aspects,
            texsystem=str(cfg.get("texsystem", "pdflatex")),
            use_sfmath=bool(cfg.get("sfmath", False)),
            ts_args=ts_args,
            max_ticks_x=int(cfg.get("max_ticks_x", DEFAULT_MAX_TICKS_X)),
            max_ticks_y=int(cfg.get("max_ticks_y", DEFAULT_MAX_TICKS_Y)),
            cloud_marker_size_drcc=None,
            cloud_marker_size_base=None,
            violin_width_cm=cfg.get("width_cm_violin", None),
            png_dpi=int(cfg.get("png_dpi", PNG_DPI)),
        )
        return 0

    p = argparse.ArgumentParser(description="Export selected plots to PGF (TS + OOS Student-t images)")
    # Selection flags
    p.add_argument("--all", action="store_true", help="Export all supported plots")
    p.add_argument("--ts", action="store_true", help="Export thermal storage operation area")
    p.add_argument("--overload-energy", dest="overload", action="store_true", help="Export overload_energy_compare (Student-t)")
    p.add_argument("--soc-envelopes", dest="soc", action="store_true", help="Export soc_envelopes (Student-t)")
    p.add_argument("--trafo-violation-heatmap", dest="trafo", action="store_true", help="Export trafo_violation_heatmap (Student-t)")
    p.add_argument("--violin-compare", dest="violin", action="store_true", help="Export violin_compare (Student-t)")
    p.add_argument("--frontier-hybrid-scatter", dest="frontier", action="store_true", help="Export frontier_hybrid_scatter (Student-t)")
    p.add_argument("--pv-temp-uncertainty", dest="pvtemp", action="store_true", help="Export PV & ambient temperature uncertainty bands")
    p.add_argument("--load-profiles", dest="loads", action="store_true", help="Export selected house electric load profiles")
    p.add_argument("--bev", dest="bev", action="store_true", help="Export one BEV charging profile (default car_8)")
    p.add_argument("--thermal-profiles", dest="thermal", action="store_true", help="Export stacked hotwater + heating demand profiles")

    # PGF and sizing
    p.add_argument("--texsystem", default="pdflatex", choices=["pdflatex", "xelatex", "lualatex"], help="LaTeX engine")
    p.add_argument("--sfmath", action="store_true", help="Enable sfmath package for sans-serif math in LaTeX")
    p.add_argument("--width-cm", type=float, default=DEFAULT_WIDTH_CM, help="Figure width in centimeters")
    # Per-plot width overrides
    p.add_argument("--width-cm-violin", type=float, default=None, help="Width in centimeters for violin compare plot (defaults to half of global width)")
    # Global aspect (applies to all unless individual overrides are given). If omitted, per-plot defaults are used.
    p.add_argument("--aspect", type=float, default=None, help="Figure height as ratio of width (global)")
    # Per-plot aspect overrides
    p.add_argument("--aspect-ts", type=float, default=None, help="Aspect for thermal storage plot")
    p.add_argument("--aspect-overload", type=float, default=None, help="Aspect for overload energy compare plot")
    p.add_argument("--aspect-soc", type=float, default=None, help="Aspect for SoC envelopes plot")
    p.add_argument("--aspect-heatmap", type=float, default=None, help="Aspect for transformer violation heatmap")
    p.add_argument("--aspect-violin", type=float, default=None, help="Aspect for violin compare plot")
    p.add_argument("--aspect-frontier", type=float, default=None, help="Aspect for frontier hybrid scatter plot")
    p.add_argument("--aspect-pvtemp", type=float, default=None, help="Aspect for PV & temperature uncertainty plot")
    # Frontier scatter visual knobs
    p.add_argument("--cloud-s-drcc", type=int, default=None, help="Marker size for DRCC cloud points in frontier plot")
    p.add_argument("--cloud-s-base", type=int, default=None, help="Marker size for baseline cloud points in frontier plot")
    p.add_argument("--max-ticks-x", type=int, default=DEFAULT_MAX_TICKS_X, help="Max number of x-axis major ticks")
    p.add_argument("--max-ticks-y", type=int, default=DEFAULT_MAX_TICKS_Y, help="Max number of y-axis major ticks")

    # TS-specific
    p.add_argument("--input", default="fully_coordinated_model_results.csv", help="Results CSV for TS plot")
    p.add_argument("--price-csv", default=None, help="Optional CSV with electricity price series for TS plot")
    p.add_argument("--price-col", default=None, help="Column name to use from --price-csv")
    p.add_argument("--start-date", default="2023-01-10 00:00:00", help="Start datetime for alignment (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--duration-hours", type=int, default=24, help="Duration (hours) for alignment window")
    p.add_argument("--no-market-price", action="store_true", dest="no_market_price", help="Disable loading market_prices_15min.csv for TS plot")

    args = p.parse_args(argv)

    selections = []
    if args.all or not any([args.ts, args.overload, args.soc, args.trafo, args.violin, args.frontier, args.pvtemp, args.loads, args.bev, args.thermal]):
        selections.append("all")
    else:
        if args.ts: selections.append("ts")
        if args.overload: selections.append("overload")
        if args.soc: selections.append("soc")
        if args.trafo: selections.append("trafo")
        if args.violin: selections.append("violin")
        if args.frontier: selections.append("frontier")
        if args.pvtemp: selections.append("pvtemp")
        if args.loads: selections.append("loads")
        if args.bev: selections.append("bev")
        if args.thermal: selections.append("thermal")

    ts_args = dict(
        input_csv=args.input,
        output_path=os.path.join(EXPORT_PGF_DIR, "thermal_storage_operation_area.pgf"),
        price_csv=args.price_csv,
        price_col=args.price_col,
        start_date=args.start_date,
        duration_hours=args.duration_hours,
        disable_market_price=args.no_market_price,
        show=False,
    )

    def _pick(per_plot: Optional[float], per_plot_default: float) -> float:
        if per_plot is not None:
            return float(per_plot)
        if args.aspect is not None:
            return float(args.aspect)
        return float(per_plot_default)

    aspects = {
        "ts": _pick(args.aspect_ts, DEFAULT_ASPECT_TS),
        "overload": _pick(args.aspect_overload, DEFAULT_ASPECT_OVERLOAD),
        "soc": _pick(args.aspect_soc, DEFAULT_ASPECT_SOC),
        "trafo": _pick(args.aspect_heatmap, DEFAULT_ASPECT_HEATMAP),
        "violin": _pick(args.aspect_violin, DEFAULT_ASPECT_VIOLIN),
        "frontier": _pick(args.aspect_frontier, DEFAULT_ASPECT_FRONTIER),
        "pvtemp": _pick(args.aspect_pvtemp, DEFAULT_ASPECT_PVTEMP),
        "loads": _pick(args.aspect, DEFAULT_ASPECT_LOADS),
        "bev": _pick(args.aspect, DEFAULT_ASPECT_BEV),
        "thermal": _pick(args.aspect, DEFAULT_ASPECT_THERMAL),
    }

    _export_selected(
        selections=selections,
        width_cm=args.width_cm,
        aspects=aspects,
        texsystem=args.texsystem,
        use_sfmath=args.sfmath,
        ts_args=ts_args,
        max_ticks_x=args.max_ticks_x,
        max_ticks_y=args.max_ticks_y,
        cloud_marker_size_drcc=args.cloud_s_drcc,
        cloud_marker_size_base=args.cloud_s_base,
        violin_width_cm=args.width_cm_violin,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
