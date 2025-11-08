"""
Export Thermal Storage Operation Area plot to PGF for LaTeX inclusion.

This script reads the optimization results CSV (default: fully_coordinated_model_results.csv)
and recreates the thermal storage operation area plot, exporting it as a .pgf file
with LaTeX/PGF settings matching sans-serif font preferences.

All PGF output now defaults into the subfolder "export pgf" to keep the project root clean.
The folder will be created automatically if it does not exist.

Additionally, a PNG is exported by default to the same folder (with .png suffix) for quick debugging.
You can tighten the figure's surrounding whitespace at export time using bbox_inches='tight' with a configurable pad.

Usage (PowerShell):
    python .\export_thermal_storage_pgf.py

Optional arguments:
    --input <path>         Path to results CSV (default: fully_coordinated_model_results.csv)
    --output <path>        Output PGF path (default: export pgf/thermal_storage_operation_area.pgf)
    --texsystem <name>     LaTeX engine (pdflatex|xelatex|lualatex). Default: pdflatex
    --width-cm <num>       Figure width in cm (default: 10.89)
    --aspect <num>         Height as ratio of width (default: 0.5)
    --show                 Display the plot interactively in addition to saving PGF
    --start-date <str>     Start datetime for alignment (YYYY-MM-DD HH:MM:SS). Default: 2023-01-10 00:00:00
    --duration-hours <int> Duration in hours for alignment window. Default: 24
    --no-market-price      Disable loading market_prices_15min.csv; fall back to CSV columns/--price-csv
    --no-png               Disable saving a PNG alongside the PGF (default saves PNG)
    --png-output <path>    Optional PNG output path (default: PGF path with .png extension)
    --png-dpi <int>        PNG DPI (default: 300)
    --tight-export         Use bbox_inches='tight' when saving (reduces external margins)
    --pad-inches <float>   Padding (in inches) used with --tight-export (default: 0.02)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# --------------------- Helpers ---------------------

def _cm_to_inch(cm: float) -> float:
    return cm / 2.54


def _configure_pgf(
    enable: bool = True,
    texsystem: str = "pdflatex",
    use_sfmath: bool = True,
) -> None:
    """Configure Matplotlib to export using PGF/LaTeX with sans-serif fonts.

    If `enable` is False, this is a no-op.
    """
    if not enable:
        return
    preamble_lines = [
        r"\usepackage{xcolor}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        # Custom brand colors for LaTeX/PGF usage
        r"\definecolor{gasGreen}{RGB}{58,157,108}",
        r"\definecolor{heatRed}{RGB}{216,46,29}",
        r"\definecolor{electricBlue}{RGB}{52,69,160}",
        r"\definecolor{solarOrange}{HTML}{FFA500}",
    ]
    if use_sfmath:
        preamble_lines.append(r"\usepackage{sfmath}")
    preamble = "\n".join(preamble_lines)

    mpl.rcParams.update(
        {
            "pgf.texsystem": texsystem,
            # Use sans-serif fonts
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
            # Ensure tick labels are not forced into math (which would pick serif math fonts)
            "axes.formatter.use_mathtext": False,
            # Make LaTeX default to sans-serif; optionally enable sfmath
            "text.latex.preamble": preamble,
            "pgf.preamble": preamble,
            "font.size": 8,
        }
    )


def _pick_price_series(df: pd.DataFrame) -> Optional[np.ndarray]:
    candidates = [
        "price_EUR_MWh",
        "electricity_price",
        "electricity_price_eur_mwh",
        "price_eur_mwh",
    ]
    for c in candidates:
        if c in df.columns:
            return df[c].to_numpy()
    return None


def _load_price_from_csv(price_csv: str, price_col: Optional[str] = None) -> Optional[np.ndarray]:
    try:
        pdf = pd.read_csv(price_csv)
        # If a specific column is requested and present
        if price_col and price_col in pdf.columns:
            s = pd.to_numeric(pdf[price_col], errors="coerce").to_numpy()
            return s
        # Otherwise, try common names
        for c in [
            "price_EUR_MWh",
            "electricity_price",
            "electricity_price_eur_mwh",
            "price_eur_mwh",
            "price",
        ]:
            if c in pdf.columns:
                s = pd.to_numeric(pdf[c], errors="coerce").to_numpy()
                return s
        # Fallback: first numeric column
        for c in pdf.columns:
            if pd.api.types.is_numeric_dtype(pdf[c]):
                return pd.to_numeric(pdf[c], errors="coerce").to_numpy()
    except Exception:
        return None
    return None


def _force_sans_ticks(ax: plt.Axes, which: str = "y") -> None:
    """Force tick labels to render as sans-serif text (not math) by wrapping with \textsf{...}."""
    from matplotlib.ticker import FuncFormatter

    if which in ("y", "both"):
        yfmt = FuncFormatter(lambda v, pos: rf"\textsf{{{v:g}}}")
        ax.yaxis.set_major_formatter(yfmt)
    if which in ("x", "both"):
        xfmt = FuncFormatter(lambda v, pos: rf"\textsf{{{int(v)}}}")
        ax.xaxis.set_major_formatter(xfmt)


def _build_time_index_from_vdi(start_date: str, duration_hours: int) -> pd.DatetimeIndex:
    """Recreate the canonical DHN time_index like fully_coordinated_model.py.

    Reads vdi_profiles/all_house_profiles.csv (index is datetime), slices the
    requested window [start_date, start_date + duration - 15min], and returns
    the DatetimeIndex of that window. Raises ValueError if out-of-bounds or empty.
    """
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
    """Load market_prices_15min.csv and align price_EUR_MWh to given time_index.

    Reindexes onto the provided time_index and applies ffill/bfill. Returns None
    if loading fails, column missing, or no overlap causes persistent NaNs.
    """
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


# --------------------- Plotter ---------------------

def export_thermal_storage_pgf(
    input_csv: str = "fully_coordinated_model_results.csv",
    output_path: str = os.path.join("export pgf", "thermal_storage_operation_area.pgf"),
    texsystem: str = "pdflatex",
    use_sfmath: bool = False,
    width_cm: float = 10.89,
    aspect: float = 0.5,
    show: bool = False,
    price_csv: Optional[str] = None,
    price_col: Optional[str] = None,
    start_date: str = "2023-01-10 00:00:00",
    duration_hours: int = 24,
    disable_market_price: bool = False,
    save_png: bool = True,
    png_output: Optional[str] = None,
    png_dpi: int = 300,
    tight_export: bool = False,
    pad_inches: float = 0.02,
) -> str:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Results CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "q_storage_kw" not in df.columns:
        raise KeyError(
            "Column 'q_storage_kw' not found in results CSV. Available columns: "
            + ", ".join(df.columns)
        )

    ts_series = df["q_storage_kw"].to_numpy()
    # Use 1-based indexing for presentation so the final step (e.g., 96) is shown explicitly
    x_ts = np.arange(1, len(ts_series) + 1)

    # Prefer market price aligned to the canonical DHN time_index, like the model
    price_series: Optional[np.ndarray] = None
    if not disable_market_price:
        try:
            time_index = _build_time_index_from_vdi(start_date, int(duration_hours))
            price_series = _load_aligned_market_price(time_index)
            if price_series is not None:
                # Ensure same length as results
                if len(price_series) < len(df):
                    pad_val = float(price_series[-1]) if len(price_series) > 0 else 0.0
                    price_series = np.concatenate(
                        [price_series, np.full(len(df) - len(price_series), pad_val)]
                    )
                price_series = price_series[: len(df)]
        except Exception as e:
            # If alignment fails for any reason, fall back to CSV/columns
            print(f"Note: falling back from aligned market price: {e}")

    # Fallbacks: results CSV columns, or user-provided price CSV
    if price_series is None:
        price_series = _pick_price_series(df)
    if price_series is None and price_csv:
        price_series = _load_price_from_csv(price_csv, price_col)
    if price_series is not None:
        # Pad/trim to match length
        if price_series.ndim == 0:
            price_series = np.full(len(x_ts), float(price_series))
        if len(price_series) < len(x_ts):
            pad_val = float(price_series[-1]) if len(price_series) > 0 else 0.0
            price_series = np.concatenate(
                [np.asarray(price_series), np.full(len(x_ts) - len(price_series), pad_val)]
            )
        price_series = np.asarray(price_series)[: len(x_ts)]

    width_in = _cm_to_inch(width_cm)
    height_in = width_in * aspect

    # Ensure parent directory exists for output
    out_dir = Path(output_path).parent
    if str(out_dir) not in ("", "."):
        out_dir.mkdir(parents=True, exist_ok=True)

    # Apply PGF configuration only within this context so it doesn't affect others
    with mpl.rc_context():
        _configure_pgf(enable=True, texsystem=texsystem, use_sfmath=use_sfmath)

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        # Leave room on the right for secondary y-axis label
        try:
            # Leave room on the right for secondary y-axis label and at the bottom for x-label
            fig.subplots_adjust(right=0.88, bottom=0.22)
        except Exception:
            pass

        # Brand color tuples for Matplotlib rendering
        gas_green = (58/255.0, 157/255.0, 108/255.0)
        heat_red = (216/255.0, 46/255.0, 29/255.0)
        electric_blue = (52/255.0, 69/255.0, 160/255.0)
        # solar_orange = (1.0, 0.6470588, 0.0)  # Not used in this figure

        # Plot TS power
        ax.plot(x_ts, ts_series, color="black", linewidth=1.0, label="TS power (kW)", zorder=3)
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        # Green for charging (>=0), Red for discharging (<=0)
        ax.fill_between(
            x_ts, ts_series, 0, where=(ts_series >= 0), facecolor=gas_green, alpha=0.65, interpolate=True, zorder=1
        )
        ax.fill_between(
            x_ts, ts_series, 0, where=(ts_series <= 0), facecolor=heat_red, alpha=0.65, interpolate=True, zorder=1
        )

        # Secondary right y-axis (always create), styled in red; no legend
        ax2 = ax.twinx()
        ax2.set_zorder(2)
        ax2.patch.set_alpha(0.0)  # transparent background for overlay
        # Ensure right ticks/labels are visible and red
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
            # Mirror left y-limits so ticks show even without series
            try:
                ax2.set_ylim(ax.get_ylim())
            except Exception:
                pass
        ax2.grid(False)

        # Force sans-serif tick labels on both axes
        _force_sans_ticks(ax, which="both")
        _force_sans_ticks(ax2, which="y")

        # No legend

        # Force desired x ticks at 24-step intervals (e.g., 24, 48, 72, 96)
        try:
            import numpy as _np
            from matplotlib.ticker import FixedLocator
            # Include 0 on the x-axis and then multiples of 24 up to the final step
            ticks_24 = _np.arange(0, len(x_ts) + 1, 24)
            if ticks_24.size > 0:
                ax.xaxis.set_major_locator(FixedLocator(ticks_24))
        except Exception:
            pass

        ax.set_xlabel("Time Step", labelpad=6)
        ax.set_ylabel("Power (kW)")

        # Ensure the full integer range is shown (e.g., 1..96) and no margins hide the last tick
        try:
            # Extend left limit to 0 so the 0 tick is visible while keeping last step visible
            ax.set_xlim(0, x_ts[-1])
            ax.margins(x=0)
        except Exception:
            pass

        # Frame spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
        for spine in ax2.spines.values():
            spine.set_visible(True)
            # right spine color already set to red; keep others black
            if spine is not ax2.spines.get("right"):
                spine.set_linewidth(1.0)
                spine.set_color("black")

        # Light gray y-grid behind shading
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="lightgray", alpha=0.6, linewidth=0.6)

        # Save without extra tightening to avoid clipping twin axis labels
        # Save PGF with optional tight bbox
        if tight_export:
            fig.savefig(output_path, bbox_inches="tight", pad_inches=pad_inches)
        else:
            fig.savefig(output_path)
        # Optionally also export PNG for debugging
        if save_png:
            png_path = png_output if png_output else os.path.splitext(output_path)[0] + ".png"
            try:
                if tight_export:
                    fig.savefig(png_path, dpi=png_dpi, format="png", bbox_inches="tight", pad_inches=pad_inches)
                else:
                    fig.savefig(png_path, dpi=png_dpi, format="png")
            except Exception as _e:
                print(f"Warning: failed to save PNG '{png_path}': {_e}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export thermal storage operation area plot to PGF.")
    p.add_argument("--input", default="fully_coordinated_model_results.csv", help="Path to results CSV")
    p.add_argument("--output", default=os.path.join("export pgf", "thermal_storage_operation_area.pgf"), help="Output PGF file path (will auto-create folder if needed)")
    p.add_argument(
        "--texsystem",
        default="pdflatex",
        choices=["pdflatex", "xelatex", "lualatex"],
        help="LaTeX engine to target",
    )
    p.add_argument("--sfmath", action="store_true", help="Enable sfmath package for sans-serif math in LaTeX")
    p.add_argument("--width-cm", type=float, default=10.89, dest="width_cm", help="Figure width in centimeters")
    p.add_argument("--aspect", type=float, default=0.5, help="Figure height as a ratio of width")
    p.add_argument("--show", action="store_true", help="Display the plot interactively")
    p.add_argument("--price-csv", default=None, help="Optional CSV file with electricity price series")
    p.add_argument("--price-col", default=None, help="Column name to use from --price-csv (auto-detect if omitted)")
    p.add_argument("--start-date", default="2023-01-10 00:00:00", help="Start datetime for alignment (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--duration-hours", type=int, default=24, help="Duration (hours) for alignment window")
    p.add_argument("--no-market-price", action="store_true", dest="no_market_price", help="Disable loading market_prices_15min.csv; use CSV columns/--price-csv only")
    p.add_argument("--no-png", action="store_true", dest="no_png", help="Disable saving PNG alongside PGF")
    p.add_argument("--png-output", default=None, help="Explicit PNG output path (default: PGF path with .png)")
    p.add_argument("--png-dpi", type=int, default=300, help="PNG DPI (default 300)")
    p.add_argument("--tight-export", action="store_true", dest="tight_export", help="Use bbox_inches='tight' to reduce margins")
    p.add_argument("--pad-inches", type=float, default=0.02, help="Padding in inches when using --tight-export")

    args = p.parse_args(argv)

    try:
        out = export_thermal_storage_pgf(
            input_csv=args.input,
            output_path=args.output,
            texsystem=args.texsystem,
            use_sfmath=args.sfmath,
            width_cm=args.width_cm,
            aspect=args.aspect,
            show=args.show,
            price_csv=args.price_csv,
            price_col=args.price_col,
            start_date=args.start_date,
            duration_hours=args.duration_hours,
            disable_market_price=args.no_market_price,
            save_png=not args.no_png,
            png_output=args.png_output,
            png_dpi=args.png_dpi,
            tight_export=args.tight_export,
            pad_inches=args.pad_inches,
        )
        print(f"Saved PGF to: {out}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
