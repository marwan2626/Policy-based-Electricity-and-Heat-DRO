"""
DHN Heat Pump Multi-Period Optimization using Gurobi

This script extends the single-period optimization to handle multiple time periods
with varying loads based on the heatpumpPrognosis.csv load profile.

Key features:
- Loads scaled according to normalized meanP profile from CSV
- Heat pump power optimized for each time period
- Time-coupled constraints (optional storage, ramping constraints)
- Minimizes total heat pump energy consumption over all periods

Control variables: Heat pump heat input Q_hp[t] for each time period t
Objective: Minimize total heat pump energy consumption
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle as pkl
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Set plotting style
plt.style.use('seaborn-v0_8')

def run_optimization(start_date=None, duration_hours=None):
    """
    Run the optimization for a specific time period.
    
    Args:
        start_date (str, optional): Start date in format 'YYYY-MM-DD HH:mm:ss'
        duration_hours (int, optional): Number of hours to optimize for
        
    Returns:
        dict: Optimization results
    """
    global START_DATE, DURATION_HOURS
    
    if start_date is not None:
        START_DATE = start_date
    if duration_hours is not None:
        DURATION_HOURS = duration_hours
        
    print(f"Running optimization from {START_DATE} for {DURATION_HOURS} hours")
    
    # The rest of the optimization code follows...

# CSV-based network data loading
NETWORK_DIR = "extracted_network_data"
print(f"Loading network data from CSV files in: {NETWORK_DIR}")

# Load network component data from CSV files
nodes_df = pd.read_csv(os.path.join(NETWORK_DIR, "heating_nodes.csv"))
pipes_df = pd.read_csv(os.path.join(NETWORK_DIR, "heating_pipes.csv"))
hex_df = pd.read_csv(os.path.join(NETWORK_DIR, "heating_heat_exchangers.csv"))
boiler_df = pd.read_csv(os.path.join(NETWORK_DIR, "heating_gas_boiler.csv"))
flow_control_df = pd.read_csv(os.path.join(NETWORK_DIR, "heating_flow_control.csv"))
print(f"Loaded {len(nodes_df)} nodes, {len(pipes_df)} pipes, {len(hex_df)} heat exchangers, {len(boiler_df)} boilers/pumps, {len(flow_control_df)} flow controls")

####################################################################################################
# LOAD PROFILE PROCESSING

print(f"\n" + "="*80)
print("LOADING AND PROCESSING LOAD PROFILE")
print("="*80)

# Default optimization parameters
START_DATE = "2023-01-10 00:00:00"  # Can be modified when running the optimization
DURATION_HOURS = 24  # Can be modified when running the optimization

# ======================= USER CONFIG (edit here) =======================
"""Top-level feature flags (default values before any CLI/env overrides).

We keep these defaults expressive, but immediately below we implement a clean
override block so that users can reliably switch features off without editing
the file. Previously the runtime override for ENABLE_RT_POLICIES was nested
inside an unrelated conditional which made it hard to disable (indentation
bug). That prevented isolation debugging of numeric issues. This patch fixes
that and also introduces an environment/CLI hook for DRCC tightening.
"""
# Real-time (second-stage) policy coefficients in day-ahead model (robust affine policy proxies)
ENABLE_RT_POLICIES = True  # <--- EDIT ME (base default)

# Allow command-line or environment to override ENABLE_RT_POLICIES.
# Set to False to lock the above value regardless of CLI/env.
ALLOW_RT_FLAG_RUNTIME_OVERRIDE = True
ENABLE_DRCC_RT_BUDGETS = True  # <--- EDIT ME (uses PV/temperature std to size D+ / D-)
DRCC_EPSILON = 0.05            # chance violation level; k = sqrt((1-eps)/eps)

# DRCC-based network tightening (transformers, lines, voltages)
ENABLE_DRCC_NETWORK_TIGHTENING = True  # <--- EDIT ME (base default)
# You can selectively toggle sub-components when master is ON
DRCC_TIGHTEN_TRAFO = True
DRCC_TIGHTEN_LINES = True
DRCC_TIGHTEN_VOLTAGES = True
## Base deterministic constraint enforcement (always-on network physics caps)
ENFORCE_BASE_VOLT_LIMITS = True   # if False, voltage magnitude limits are skipped entirely (debug only)
ENFORCE_BASE_LINE_LIMITS = True   # if False, line thermal limits are skipped (debug only)
ENFORCE_BASE_TRAFO_LIMITS = True  # if False, transformer thermal limits are skipped (debug only)
PV_STD_FROM_CSV = True         # try to read pv std from pv_profiles_output.csv
PV_RELATIVE_STD = 0.20         # fallback relative std of PV availability (fraction of avail)
PV_STD_CORRELATION = 1.00      # used only if constructing from per-bus stds (not CSV aggregate)
HP_FULLY_CORRELATED = True     # temperature is common across HPs
RHO_TEMP_AVG = 0             # 0=independent, 1=fully correlated within day
# Capacity buy-back pricing (EUR per MW-hour of purchased connection reduction)
C_CAP_EUR_PER_MW = 0.01  # <--- EDIT ME (capacity price); set 0 to disable economic impact
C_SHED_EUR_PER_MW_H = 100000.0

# =====================================================================
## Debug flags for aggregated flexibility mechanism
DEBUG_EXTRACT_IIS = True              # If model infeasible, extract IIS
DEBUG_PRINT_FLEX_DIAGNOSTICS = True   # Print per-period cap diagnostics (first few + min/max)
RELAX_AGG_CONN_CAP = False            # (Removed) kept for backward compatibility; no slack will be created
MAX_FLEX_DIAG_PERIODS = 8             # How many early periods to print diagnostics
SHOW_BASELINE_FLEX_FIG = True         # If True, create a standalone figure with baseline (deterministic) flexible loads per bus
## End debug flags

# --- Heat Pump predictor and DRCC constants (single source of truth) ---
# Affine predictor (Baseline + Deviation normalized by Pmax, with daily T_avg)
HP_PRED_PMAX = 0.30           # MW
HP_PRED_TBASE_C = 10.0        # °C for HDD
HP_COEFF_B0 = 0.331877
HP_COEFF_BHDD = 0.015908
HP_COEFF_BPI = -0.000492
HP_COEFF_BTAV = -0.014595
HP_COEFF_A1 = 0.013139
HP_COEFF_A2 = -0.006540

# DRCC/uncertainty sizing uses same sensitivities as predictor
HP_DRCC_PMAX = HP_PRED_PMAX
HP_DRCC_BTAV = HP_COEFF_BTAV
HP_DRCC_BHDD = HP_COEFF_BHDD

# Residual uncertainty for HP predictor (normalized units, multiply by HP_PRED_PMAX to get MW)
HP_INCLUDE_RESIDUAL = True
HP_RESIDUAL_SIGMA_NORM = 0.01616 # std of predictor residual in normalized y_dev units
HP_RESIDUAL_CORRELATION = 0.0     # 0=independent across HP buses, 1=fully correlated (not used in current RSS)

# Allow runtime override of START_DATE/DURATION_HOURS via CLI args or environment
try:
    import argparse, os, sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--start-date', dest='start_date', help='Override START_DATE (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--duration-hours', dest='duration_hours', type=int, help='Override DURATION_HOURS (int)')
    parser.add_argument('--enable-rt-policies', dest='enable_rt_policies', action='store_true', help='Enable robust real-time policy coefficients in the day-ahead model')
    parser.add_argument('--disable-rt-policies', dest='disable_rt_policies', action='store_true', help='Force-disable RT policies regardless of defaults')
    parser.add_argument('--enable-drcc-tightening', dest='enable_drcc_tight', action='store_true', help='Enable DRCC network tightening')
    parser.add_argument('--disable-drcc-tightening', dest='disable_drcc_tight', action='store_true', help='Disable DRCC network tightening')
    args, _ = parser.parse_known_args()

    # Window overrides
    if args.start_date:
        START_DATE = args.start_date
    elif os.environ.get('CMES_START_DATE'):
        START_DATE = os.environ.get('CMES_START_DATE')

    if args.duration_hours is not None:
        DURATION_HOURS = args.duration_hours
    elif os.environ.get('CMES_DURATION_HOURS'):
        try:
            DURATION_HOURS = int(os.environ.get('CMES_DURATION_HOURS'))
        except Exception:
            pass

    # Feature overrides (priority: explicit CLI flag > env var > file default)
    if ALLOW_RT_FLAG_RUNTIME_OVERRIDE:
        if args.disable_rt_policies:
            ENABLE_RT_POLICIES = False
        elif args.enable_rt_policies:
            ENABLE_RT_POLICIES = True
        else:
            # Environment variable (set to 1/true/on to enable, 0/false/off to disable)
            env_rt = os.environ.get('CMES_ENABLE_RT_POLICIES')
            if env_rt is not None:
                ENABLE_RT_POLICIES = str(env_rt).strip().lower() in ('1','true','yes','on')

    # DRCC tightening overrides via CLI / env (independent of ALLOW_RT_FLAG_RUNTIME_OVERRIDE)
    if args.disable_drcc_tight:
        ENABLE_DRCC_NETWORK_TIGHTENING = False
    elif args.enable_drcc_tight:
        ENABLE_DRCC_NETWORK_TIGHTENING = True
    else:
        env_drcc = os.environ.get('CMES_ENABLE_DRCC_TIGHTENING')
        if env_drcc is not None:
            ENABLE_DRCC_NETWORK_TIGHTENING = str(env_drcc).strip().lower() in ('1','true','yes','on')
except Exception as _e:
    print(f"[WARN] Runtime override parsing failed: {_e}. Using file defaults.")

# Load the VDI profiles with heating and hot water loads
print("Loading VDI profiles from 'vdi_profiles/all_house_profiles.csv'...")
profiles_df = pd.read_csv('vdi_profiles/all_house_profiles.csv', index_col=0)
profiles_df.index = pd.to_datetime(profiles_df.index)

# Build the DHN canonical time index for the requested window
start_dt = pd.to_datetime(START_DATE)
end_dt = start_dt + pd.Timedelta(hours=DURATION_HOURS) - pd.Timedelta(minutes=15)

if start_dt < profiles_df.index.min():
    raise ValueError(f"Start date {start_dt} is before the earliest available data {profiles_df.index.min()}")
if end_dt > profiles_df.index.max():
    raise ValueError(f"End date {end_dt} is after the latest available data {profiles_df.index.max()}")

# Reindex to the DHN time grid (this enforces exact alignment and will surface missing timestamps)
dhn_window = profiles_df.loc[start_dt:end_dt]
if dhn_window.empty:
    raise ValueError(f"No DHN data between {start_dt} and {end_dt}")
time_index = dhn_window.index
window_df = profiles_df.reindex(time_index)

# Any NaNs in the reindexed window are fatal (strict mode)
nan_cols = window_df.columns[window_df.isnull().any()].tolist()
if nan_cols:
    raise ValueError(f"Profile CSV contains NaN values after reindexing to requested window. Columns with any NaN: {nan_cols}")

NUM_PERIODS = len(time_index)
print(f"Selected time window: {time_index[0]} to {time_index[-1]}")
print(f"Number of time steps: {NUM_PERIODS}")

print("\nMapping profiles to network components (exact name matching, strict)")
import re

def normalize_name(s):
    return s.replace('.', '_').replace('-', '_').lower()

# Group profile columns by their base name (strip trailing suffix)
suffix_pattern = re.compile(r'[_\.-]?(heating|hotwater|electricity|electric|elec)$', flags=re.I)
profile_parts = {}
for col in window_df.columns:
    m = suffix_pattern.search(col)
    if not m:
        continue
    suffix = m.group(1).lower()
    base = col[:m.start()]
    profile_parts.setdefault(base, {})[suffix] = window_df[col].to_numpy()

# Load electrical loads CSV for matching
electrical_loads_df = pd.read_csv(os.path.join(NETWORK_DIR, "electrical_loads.csv"))

# Build normalization maps for hex names and electrical load names
hex_names = set(hex_df['name'].astype(str).values)
hex_norm_map = {normalize_name(n): n for n in hex_names}
el_names = electrical_loads_df['name'].astype(str).fillna('').values
el_time_series = {}
el_norm_map = {normalize_name(n): n for n in el_names}

# Containers for assigned time series (kW)
# Deterministic mapping: map each network electrical load name to the exact CSV column '<base>_electricity'
# Try both '.' and '_' variants of the base name to match common formatting differences.
for el_orig in el_names:
    if not el_orig:
        continue
    el_norm = normalize_name(el_orig)
    # Candidate CSV column bases to check (original normalized, with '.' replaced by '_' and vice-versa)
    candidate_bases = [el_norm, el_norm.replace('.', '_'), el_norm.replace('_', '.')]
    found = False
    for cb in candidate_bases:
        # CSV column names in window_df may be like '<base>_electricity' (base uses underscores) or '<base>.electricity'
        candidate_col1 = cb + '_electricity'
        candidate_col2 = cb + '.electricity'
        # window_df columns are the original header names — normalize them for comparison
        cols_norm = {normalize_name(c): c for c in window_df.columns}
        if normalize_name(candidate_col1) in cols_norm:
            col_name = cols_norm[normalize_name(candidate_col1)]
            el_time_series[el_orig] = window_df[col_name].to_numpy()
            #print(f"Assigned electrical profile -> {el_orig} (CSV column='{col_name}')")
            found = True
            break
        if normalize_name(candidate_col2) in cols_norm:
            col_name = cols_norm[normalize_name(candidate_col2)]
            el_time_series[el_orig] = window_df[col_name].to_numpy()
            #print(f"Assigned electrical profile -> {el_orig} (CSV column='{col_name}')")
            found = True
            break
    if not found:
        # No exact CSV match — assign zero-series and warn once
        el_time_series[el_orig] = np.zeros(NUM_PERIODS)
        print(f"Warning: no CSV electricity column found for electrical load '{el_orig}'. Assigned zero-series.")

####################################################################################################
# AMBIENT TEMPERATURE PROFILE PROCESSING

print(f"\n" + "="*80)
print("LOADING AMBIENT TEMPERATURE PROFILE")
print("="*80)

# Load ambient temperature data from new complete dataset
temp_df = pd.read_csv('temperature_data_complete.csv')
temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
temp_df.set_index('datetime', inplace=True)

# Filter temperature data for the optimization window
temp_window = temp_df.loc[start_dt:end_dt]
temp_values_k = temp_window['temperature_K'].values
temp_mean_k = temp_window['temperature_mean_K'].values
temp_std_k = temp_window['temperature_std_K'].values

# Convert to Celsius for display
temp_values_c = temp_values_k - 273.15

print(f"Loaded temperature data: {len(temp_values_c)} periods")
print(f"Temperature range: {temp_values_c.min():.1f}°C to {temp_values_c.max():.1f}°C")
print(f"Temperature mean: {temp_values_c.mean():.1f}°C")

# Align temperature profile with load profile
temp_profile_c = temp_values_c[:NUM_PERIODS]  # Use first NUM_PERIODS entries
temp_profile_k = temp_profile_c + 273.15  # Convert to Kelvin

print(f"Using {len(temp_profile_c)} temperature periods for optimization")
print(f"Temperature profile preview: {temp_profile_c[:5]} ... {temp_profile_c[-5:]} °C")
print(f"Temperature in Kelvin: {temp_profile_k[:5]} ... {temp_profile_k[-5:]} K")

# Precompute uncertainty metrics from temperature std for HP DRCC sizing
try:
    # Aligned std array (K) added later in load_input_data_from_csv; build placeholders now
    # We'll overwrite these after load_input_data_from_csv is called in __main__.
    globals()['sigma_Tavg_by_day'] = {}
    globals()['Var_HDD_by_t'] = {}
except Exception:
    pass

# Verify alignment
# print(f"\nDATA ALIGNMENT VERIFICATION (first 5 periods):")
# for i in range(min(5, NUM_PERIODS)):
#     print(f"Period {i+1}: Load factor = {load_profile[i]:.3f}, Ambient temp = {temp_profile_c[i]:.1f}°C")

####################################################################################################
# IMPORT EXACT FUNCTIONS FROM PANDAPIPES_SIMPLE.PY


# --- Assign time series from VDI profiles to heat exchangers and electrical loads ---
print("Assigning mapped time series to network components...")
# hex_time_series and el_time_series are created earlier during strict CSV parsing
try:
    el_time_series   # noqa: F821
except NameError:
    # If they don't exist, the strict loader should have errored earlier
    raise RuntimeError("Internal error: expected mapped time series (el_time_series) not found. Ensure profile mapping ran before network assembly.")

# For electrical loads, build a mapping from electrical load name -> p_mw time series (kW)
electrical_time_series = {}
for el_name, series in el_time_series.items():
    if len(series) != NUM_PERIODS:
        raise ValueError(f"Electrical series length mismatch for {el_name}: expected {NUM_PERIODS}, got {len(series)}")
    electrical_time_series[el_name] = series

print(f"Assigned {len(electrical_time_series)} electrical series.")

# --- Additional: assign BEV profiles to BEV loads (if present) ---
# Load additional BEV load definitions and sequentially map columns from the
# BEV profiles CSV (e.g. LadeprofileBEV/bev_2023_power_first100.csv) to each
# BEV load in extracted_network_data/electrical_BEV.csv by order.
try:
    bev_loads_path = os.path.join(NETWORK_DIR, "electrical_BEV.csv")
    bev_profiles_path = os.path.join('LadeprofileBEV', 'bev_2023_power_first100.csv')
    if os.path.exists(bev_loads_path) and os.path.exists(bev_profiles_path):
        bev_loads_df = pd.read_csv(bev_loads_path)
        if bev_loads_df.empty:
            print(f"BEV loads CSV found but empty: {bev_loads_path}")
        else:
            # Read BEV profiles, parse datetime, and align to the selected time window
            bev_profiles_df = pd.read_csv(bev_profiles_path, parse_dates=['datetime'], index_col='datetime')
            # Reindex to the DHN time window used earlier
            try:
                bev_window = bev_profiles_df.reindex(time_index)
            except Exception:
                bev_window = bev_profiles_df.loc[start_dt:end_dt]

            # Ensure there's at least one profile column
            profile_cols = [c for c in bev_window.columns if str(c).lower().startswith('car') or str(c).lower().startswith('bev')]
            if not profile_cols:
                # Fall back to all columns except index
                profile_cols = list(bev_window.columns)

            if len(profile_cols) == 0:
                print(f"Warning: no BEV profile columns found in {bev_profiles_path}")
            else:
                # Iterate BEV loads in file order and assign profiles sequentially
                col_iter = iter(profile_cols)
                assigned = 0
                for _, bev_row in bev_loads_df.iterrows():
                    load_name = str(bev_row.get('name', '')).strip()
                    if not load_name:
                        continue
                    try:
                        col = next(col_iter)
                    except StopIteration:
                        break  # no more profile columns to assign

                    # Extract the series for this profile column and align to time_index
                    try:
                        series = bev_window[col].to_numpy()
                    except Exception:
                        # If direct extraction fails, try reindexing per-index
                        s = bev_window[col] if col in bev_window.columns else pd.Series([], index=time_index)
                        s = s.reindex(time_index).fillna(0.0)
                        series = s.to_numpy()

                    # Ensure correct length: pad/truncate if necessary
                    if len(series) != NUM_PERIODS:
                        s = pd.Series(series, index=bev_window.index[:len(series)])
                        s = s.reindex(time_index).fillna(0.0)
                        series = s.to_numpy()

                    # Store in the global electrical_time_series mapping (kW expected)
                    if load_name in electrical_time_series:
                        print(f"Note: Overwriting existing electrical series for '{load_name}' with BEV profile column '{col}'")
                    electrical_time_series[load_name] = series
                    assigned += 1
                    #print(f"Assigned BEV profile -> {load_name} (BEV column='{col}')")

                #print(f"BEV profile assignment complete: {assigned} BEV loads assigned from {len(profile_cols)} profile columns.")
except Exception as _e:
    print(f"Warning: failed to assign BEV profiles: {_e}")

####################################################################################################
# PLOTTING FUNCTIONS (defined early to be available for use)

def create_comprehensive_plots(results_df, hp_power_values, ambient_temps_c=None, storage_soc_values=None, slack_power_values=None, non_flexible_load_p=None, flexible_load_p=None, electricity_price=None):
    """Create comprehensive plots of the optimization results

    Parameters added:
    - electricity_price: optional array-like of electricity prices (e.g. EUR/MWh) aligned to the time axis
    """
    
    # Create time axis for plots (assuming hourly data for a week)
    hours = np.arange(len(hp_power_values))
    time_labels = [f"Hour {h}" for h in hours]
    
    # Set up the figure with multiple subplots - increased size for additional plot
    fig = plt.figure(figsize=(20, 28))
    
    # 1. BESS Energy Over Time (replaces Heat Pump + Slack subplot)
    plt.subplot(7, 2, 1)
    # Strict behavior: use per-bus BESS energy columns from results_df (same pattern as PV/line plots).
    bess_cols = [col for col in results_df.columns if col.lower().startswith('bess_energy_bus_')]
    if not bess_cols:
        raise ValueError("create_comprehensive_plots requires per-bus BESS energy columns in results_df named 'bess_energy_bus_{bus}_mwh'. None found.")

    # Determine a safe y-axis limit (use the max of all bess columns)
    try:
        overall_max = max(results_df[col].max() for col in bess_cols)
    except Exception:
        overall_max = None

    for col in bess_cols:
        arr = results_df[col].to_numpy()
        # Clip negative values for plotting only (do not modify underlying data)
        arr_plot = np.clip(arr, 0.0, None)
        plt.plot(hours, arr_plot, linewidth=1.5, label=col)

    # If we successfully computed an overall maximum, set y-limit slightly above it
    if overall_max is not None and np.isfinite(overall_max):
        plt.ylim(0.0, max(1e-6, overall_max) * 1.1)
    plt.title('BESS Stored Energy Over Time (per bus)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour')
    plt.ylabel('Energy (MWh)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Electrical Grid Import/Export Over Time
    plt.subplot(7, 2, 2)
    
    # Get electrical grid data from results_df
    import_cols = [col for col in results_df.columns if 'ext_grid_import_mw' in col.lower()]
    export_cols = [col for col in results_df.columns if 'ext_grid_export_mw' in col.lower()]
    net_cols = [col for col in results_df.columns if 'net_grid_power_mw' in col.lower()]
    
    if import_cols and export_cols:
        # Confirm which columns are being used (these are populated from ext_grid_import_p/_export_p results)
        chosen_import_col = import_cols[0]
        chosen_export_col = export_cols[0]
        import_power = results_df[chosen_import_col].values * 1000  # Convert MW to kW
        export_power = results_df[chosen_export_col].values * 1000  # Convert MW to kW
        
        plt.plot(hours, import_power, 'r-', linewidth=2, label='Grid Import', alpha=0.8)
        plt.plot(hours, export_power, 'g-', linewidth=2, label='Grid Export', alpha=0.8)
        
        # Fill areas to show import vs export
        plt.fill_between(hours, 0, import_power, alpha=0.3, color='red', label='Import Area')
        plt.fill_between(hours, 0, export_power, alpha=0.3, color='green', label='Export Area')
        
        plt.title('Electrical Grid Power Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Ambient Temperature Over Time
    plt.subplot(7, 2, 3)
    if ambient_temps_c is not None:
        ax = plt.gca()
        ax.plot(hours, ambient_temps_c, 'g-', linewidth=2, label='Ambient Temperature')
        ax.set_title('Ambient Temperature Profile Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
        # Plot COP on the right y-axis if results_df contains 'cop_t'
        try:
            if 'cop_t' in results_df.columns:
                ax_twin = ax.twinx()
                ax_twin.plot(hours, results_df['cop_t'].values, 'orange', linewidth=2, label='Heat Pump COP')
                ax_twin.set_ylabel('COP')
                # Combine legends
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
                #ylimit
                ax_twin.set_ylim(2, max(2.0, results_df['cop_t'].max() * 1.1))
        except Exception:
            pass
    
    # 4. Heat Pump Electrical Power Over Time
    plt.subplot(7, 2, 4)
    
    # Extract heat pump electrical power data
    hp_elec_cols = [col for col in results_df.columns if 'p_hp_mw' in col.lower()]
    
    if hp_elec_cols:
        # Calculate total electrical power from all heat pump buses
        total_hp_elec_power = results_df[hp_elec_cols].sum(axis=1) * 1000  # Convert MW to kW
        
        # Plot both thermal and electrical power for comparison
        plt.plot(hours, hp_power_values, 'r-', linewidth=2, label='Thermal Power (kW)')
        plt.plot(hours, total_hp_elec_power, 'b-', linewidth=2, label='Electrical Power (kW)')
        
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('Heat Pump Thermal vs Electrical Power', fontsize=14, fontweight='bold')
        
        # Add some statistics in text box
        avg_thermal = np.mean(hp_power_values)
        avg_electrical = np.mean(total_hp_elec_power)
        avg_cop = avg_thermal / avg_electrical if avg_electrical > 0 else 0
        plt.text(0.05, 0.95, f'Avg Thermal: {avg_thermal:.1f} kW\nAvg Electrical: {avg_electrical:.1f} kW\nAvg COP: {avg_cop:.2f}', 
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No Heat Pump Electrical Power Data Available', 
                 transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
        plt.title('Heat Pump Thermal vs Electrical Power', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.grid(True, alpha=0.3)
    
    # 5. Transformer Loading Over Time (replaces HP vs Ambient Temperature)
    plt.subplot(7, 2, 5)
    # Look for transformer loading columns in the results dataframe. Expected pattern: 'transformer_<idx>_loading_pct' or similar
    transformer_cols = [col for col in results_df.columns if 'transformer' in col.lower() and 'loading' in col.lower()]
    if transformer_cols:
        # Plot up to the first 10 transformers for clarity
        for i, col in enumerate(transformer_cols[:10]):
            plt.plot(hours, results_df[col].values, alpha=0.8, label=f'{col}')
        plt.title('Transformer Loading (%) (first 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Loading (%)')
        plt.grid(True, alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # No transformer data found in results_df — show helpful message
        plt.text(0.5, 0.5, 'No transformer loading data available in results', 
                 transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
        plt.title('Transformer Loading (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Loading (%)')
        plt.grid(True, alpha=0.3)
    
    # 6. Electricity Price Over Time (replaces HP Power Distribution)
    plt.subplot(7, 2, 6)
    price_series = None

    # Try to find price within results_df using common column names
    price_col_candidates = [c for c in results_df.columns if 'price' in c.lower() or 'electricity_price' in c.lower()]
    if 'price_EUR_MWh' in results_df.columns:
        price_series = results_df['price_EUR_MWh'].values
    elif 'electricity_price' in results_df.columns:
        price_series = results_df['electricity_price'].values
    elif 'electricity_price_eur_mwh' in results_df.columns:
        price_series = results_df['electricity_price_eur_mwh'].values
    elif 'price_eur_mwh' in results_df.columns:
        price_series = results_df['price_eur_mwh'].values
    elif electricity_price is not None:
        try:
            price_series = np.asarray(electricity_price)
        except Exception:
            price_series = None
    else:
        # Try fallback to module-level global if available
        price_series = globals().get('electricity_price', None)
        if price_series is not None:
            try:
                price_series = np.asarray(price_series)
            except Exception:
                price_series = None

    if price_series is not None and len(price_series) > 0:
        # Align/truncate to hours length
        price_to_plot = price_series[:len(hours)]
        plt.plot(hours, price_to_plot, 'm-', linewidth=2, label='Electricity Price')
        plt.title('Electricity Price Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        # Try to infer units from values (common default EUR/MWh in this codebase)
        plt.ylabel('Price (EUR/MWh)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No electricity price data available', 
                 transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
        plt.title('Electricity Price Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
    
    # 7. Junction Temperatures Overview - Supply Temperatures
    plt.subplot(7, 2, 7)
    supply_cols = [col for col in results_df.columns if 'supply' in col.lower() and 'temp' in col.lower()]
    if supply_cols:
        supply_temps = results_df[supply_cols].values
        for i, col in enumerate(supply_cols[:10]):  # Show first 10 for clarity
            plt.plot(hours, results_df[col], alpha=0.7, label=f'Junction {i+1}')
        plt.title('Supply Junction Temperatures (First 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 8. Baseline (time-synchronized) Flexible Load per Bus (kW) (replaces Return Junction Temperatures)
    plt.subplot(7, 2, 8)
    try:
        baseline_flex_dict = globals().get('flexible_time_synchronized_loads_P', {})
        if baseline_flex_dict:
            all_buses = set()
            for tmap in baseline_flex_dict.values():
                all_buses.update(tmap.keys())
            buses_sorted = sorted(list(all_buses))
            color_map = plt.cm.tab20(np.linspace(0, 1, max(1, len(buses_sorted))))
            T = len(hours)
            for i, bus in enumerate(buses_sorted):
                series_kw = []
                for t in range(T):
                    val_mw = 0.0
                    if t in baseline_flex_dict and bus in baseline_flex_dict[t]:
                        val_mw = float(baseline_flex_dict[t][bus])
                    series_kw.append(val_mw * 1000.0)
                alpha = 0.85 if len(buses_sorted) <= 25 else 0.5
                lw = 1.2 if len(buses_sorted) <= 25 else 0.8
                label = f"Bus {bus}" if len(buses_sorted) <= 15 else None
                plt.plot(hours, series_kw, linewidth=lw, alpha=alpha, color=color_map[i % len(color_map)], label=label)
            plt.title('Baseline Flexible Load per Bus (kW)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)
            if len(buses_sorted) <= 15:
                plt.legend(fontsize=7, loc='upper right')
        else:
            plt.text(0.5, 0.5, 'No baseline flexible load data', transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
            plt.title('Baseline Flexible Load per Bus (kW)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)
    except Exception as _baseline_subplot_err:
        plt.text(0.5, 0.5, f'Baseline flex plot error: {_baseline_subplot_err}', transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)
        plt.title('Baseline Flexible Load per Bus (error)', fontsize=14, fontweight='bold')

    # 9. Electrical Bus Voltages Over Time (restored)
    plt.subplot(7, 2, 9)
    voltage_cols = [col for col in results_df.columns if 'voltage_pu' in col.lower()]
    if voltage_cols:
        for i, col in enumerate(voltage_cols):
            try:
                # Attempt to extract bus number between first and second underscore if pattern matches 'bus_<id>_voltage_pu'
                parts = col.split('_')
                if len(parts) >= 3 and parts[0] == 'bus' and parts[-2] == 'voltage' and parts[-1] == 'pu':
                    bus_number = parts[1]
                elif len(parts) >= 2 and parts[0] == 'bus':
                    bus_number = parts[1]
                else:
                    bus_number = str(i)
            except Exception:
                bus_number = str(i)
            plt.plot(hours, results_df[col], alpha=0.7, linewidth=1.0, label=f'Bus {bus_number}')
        plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Nominal (1.0 p.u.)')
        plt.title('Electrical Bus Voltages Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Voltage (p.u.)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.85, 1.15)
        if len(voltage_cols) <= 15:
            plt.legend(fontsize=7, loc='upper right')
    else:
        plt.text(0.5, 0.5, 'No Voltage Data', transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.title('Electrical Bus Voltages Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Voltage (p.u.)')
        plt.grid(True, alpha=0.3)
    
    # 10. Flexible Load Active Power per Bus (REPLACES Thermal Storage SOC plot)
    plt.subplot(7, 2, 10)
    try:
        flex_bus_cols = [c for c in results_df.columns if c.startswith('flex_load_bus_') and c.endswith('_mw')]
        if flex_bus_cols:
            cols_sorted = sorted(flex_bus_cols, key=lambda x: int(x.split('_')[3]))
            color_map = plt.cm.tab20(np.linspace(0, 1, max(1, len(cols_sorted))))
            for i, col in enumerate(cols_sorted):
                series_kw = results_df[col].values * 1000.0
                alpha = 0.85 if len(cols_sorted) <= 25 else 0.5
                lw = 1.2 if len(cols_sorted) <= 25 else 0.8
                label = f"Bus {col.split('_')[3]}" if len(cols_sorted) <= 20 else None
                plt.plot(hours, series_kw, linewidth=lw, alpha=alpha, color=color_map[i % len(color_map)], label=label)
            plt.title('Flexible Load per Bus (kW)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)
            if len(cols_sorted) <= 20:
                plt.legend(fontsize=7, loc='upper right')
        else:
            plt.text(0.5, 0.5, 'No Flexible Load Data', transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
            plt.title('Flexible Load per Bus (kW)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)
    except Exception as _flex_plot_err:
        plt.text(0.5, 0.5, f'Flex plot error: {_flex_plot_err}', transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)
        plt.title('Flexible Load per Bus (error)', fontsize=14, fontweight='bold')
    
    # 11. Non-Flexible Load Active Power (per-bus traces)
    plt.subplot(7, 2, 11)
    # If the caller provided per-period load dicts (t -> {bus: p_mw}), combine flexible and non-flexible and plot each bus
    per_bus = {}
    T = len(hp_power_values)

    # Helper to add entries from a dict-of-dicts (t -> {bus: mw}) into per_bus (kW)
    def _accumulate_loads(source_dict):
        if not source_dict:
            return
        for t in range(T):
            if t in source_dict:
                for bus, mw in source_dict[t].items():
                    per_bus.setdefault(bus, [0.0] * T)
                    per_bus[bus][t] += mw * 1000.0  # convert MW -> kW and accumulate

    # Accumulate non-flexible and flexible loads (if present)
    _accumulate_loads(non_flexible_load_p)
    _accumulate_loads(flexible_load_p)

    if per_bus:
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(per_bus))))
        for i, (bus, series_kw) in enumerate(sorted(per_bus.items(), key=lambda x: x[0])):
            plt.plot(hours, series_kw, linewidth=1.2, alpha=0.8, color=colors[i % len(colors)])

        plt.title('Load Active Power per Bus (flexible + non-flexible)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.grid(True, alpha=0.3)
        # legend intentionally omitted per user request
    else:
        # Fallback: keep the original daily energy consumption bars if non_flexible data not provided
        if len(hp_power_values) >= 24:
            daily_pattern = []
            for day in range(min(7, len(hp_power_values)//24)):
                start_idx = day * 24
                end_idx = min((day + 1) * 24, len(hp_power_values))
                daily_energy = np.sum(hp_power_values[start_idx:end_idx])
                daily_pattern.append(daily_energy)

            days = [f'Day {i+1}' for i in range(len(daily_pattern))]
            plt.bar(days, daily_pattern, alpha=0.7, color='orange')
            plt.title('Daily Energy Consumption', fontsize=14, fontweight='bold')
            plt.xlabel('Day')
            plt.ylabel('Energy (kWh)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
    
    # 12. All Line Loading Percentage
    plt.subplot(7, 2, 12)
    
    # Extract line loading data
    line_loading_cols = [col for col in results_df.columns if 'line_' in col.lower() and 'loading_pct' in col.lower()]
    
    if line_loading_cols:
        # Sort lines by their line number for consistent ordering
        line_data = []
        for col in line_loading_cols:
            line_number = int(col.split('_')[1])  # Extract line number from column name
            max_loading = results_df[col].max()
            line_data.append((line_number, max_loading, col))
        
        # Sort by line number
        line_data.sort(key=lambda x: x[0])
        
        # Plot all lines with different colors
        num_lines = len(line_data)
        colors = plt.cm.tab20(np.linspace(0, 1, num_lines))  # Use more colors for all lines
        
        for i, (line_num, max_load, col_name) in enumerate(line_data):
            plt.plot(hours, results_df[col_name], color=colors[i], linewidth=1.0, 
                    alpha=0.7, label=f'Line {line_num}')
        
        # Add loading limit reference lines
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.8, linewidth=2, label='100% Limit')
        plt.axhline(y=80, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='80% Warning')
        
        plt.title(f'All Line Loading Percentage Over Time ({num_lines} lines)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Loading (%)')
        plt.grid(True, alpha=0.3)
        
        # # Only show legend if not too many lines
        # if num_lines <= 15:
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        # else:
        #     # For many lines, just show the limits in legend
        #     plt.legend(['100% Limit', '80% Warning'], loc='upper right', fontsize=8)
            
        plt.ylim(0, max(110, max([max_load for _, max_load, _ in line_data]) + 5))
        
    else:
        plt.text(0.5, 0.5, 'No Line Loading Data Available', 
                 transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
        plt.title('All Line Loading Percentage Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Loading (%)')
        plt.grid(True, alpha=0.3)
    
    # 13. PV Generation per Bus (kW) ONLY (no totals, no availability)
    plt.subplot(7, 2, 13)
    try:
        pv_gen_bus_cols = [c for c in results_df.columns if c.startswith('pv_gen_bus_') and c.endswith('_mw')]
        if pv_gen_bus_cols:
            cols_sorted = sorted(pv_gen_bus_cols, key=lambda x: int(x.split('_')[3]))
            color_map = plt.cm.tab20(np.linspace(0, 1, min(len(cols_sorted), 20)))
            for i, col in enumerate(cols_sorted):
                series_kw = results_df[col].values * 1000.0
                color = color_map[i % len(color_map)]
                alpha = 0.85 if len(cols_sorted) <= 20 else 0.5
                lw = 1.1 if len(cols_sorted) <= 20 else 0.8
                label = col if len(cols_sorted) <= 20 else None
                plt.plot(hours, series_kw, linewidth=lw, alpha=alpha, color=color, label=label)
            plt.title('PV Generation per Bus (kW)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)
            if len(cols_sorted) <= 20:
                plt.legend(fontsize=7, ncol=1, loc='upper right')
        else:
            # Fallback to global pv_gen_results if DataFrame columns absent
            pv_ts = globals().get('pv_gen_results', {})
            if pv_ts:
                T = len(hours)
                per_bus = {}
                for t in range(T):
                    if t in pv_ts:
                        for bus, mw in pv_ts[t].items():
                            per_bus.setdefault(bus, [0.0]*T)
                            per_bus[bus][t] = mw * 1000.0
                colors = plt.cm.tab20(np.linspace(0,1,len(per_bus)))
                for i, (bus, series) in enumerate(sorted(per_bus.items(), key=lambda x: x[0])):
                    plt.plot(hours, series, linewidth=1.0, alpha=0.8, color=colors[i%len(colors)], label=f'Bus {bus}' if len(per_bus) <= 20 else None)
                plt.title('PV Generation per Bus (kW)', fontsize=14, fontweight='bold')
                plt.xlabel('Hour')
                plt.ylabel('Power (kW)')
                plt.grid(True, alpha=0.3)
                if len(per_bus) <= 20:
                    plt.legend(fontsize=7, loc='upper right')
            else:
                plt.text(0.5, 0.5, 'No PV data available', transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
                plt.title('PV Generation per Bus', fontsize=14, fontweight='bold')
    except Exception as _pv_plot_err:
        plt.text(0.5, 0.5, f'PV plot error: {_pv_plot_err}', transform=plt.gca().transAxes, ha='center', va='center', fontsize=10)
        plt.title('PV Generation per Bus (error)', fontsize=14, fontweight='bold')
    
    # 14. Heat Demand Coverage Stacked Area Plot
    plt.subplot(7, 2, 14)
    
    # Calculate heat sources for stacked area plot - corrected for thermal storage charging
    storage_discharge_power = []
    storage_charge_power = []
    hp_demand_coverage = []
    slack_demand_coverage = []
    
    for i in range(len(hp_power_values)):
        # Check if we have storage data
        if storage_soc_values is not None and i < len(results_df):
            q_storage_kw = results_df.iloc[i]['q_storage_kw'] if 'q_storage_kw' in results_df.columns else 0
            # Storage discharge (negative Q_storage as positive contribution)
            storage_discharge_power.append(max(0, -q_storage_kw))
            # Storage charge (positive Q_storage)
            storage_charge_power.append(max(0, q_storage_kw))
        else:
            storage_discharge_power.append(0)
            storage_charge_power.append(0)
        
        # Heat pump contribution to demand coverage = total HP power - any charging to storage
        hp_contribution = hp_power_values[i] - storage_charge_power[i]
        hp_demand_coverage.append(max(0, hp_contribution))
        
        # Slack contribution to demand coverage = total slack power - any remaining charging
        remaining_charge = storage_charge_power[i] - min(storage_charge_power[i], hp_power_values[i])
        slack_contribution = (slack_power_values[i] if slack_power_values is not None else 0) - remaining_charge
        slack_demand_coverage.append(max(0, slack_contribution))
    
    # Prepare data for stacked area plot (actual demand coverage)
    hp_power = hp_demand_coverage
    storage_power = storage_discharge_power
    slack_power = slack_demand_coverage
    
    # Create stacked area plot
    plt.stackplot(hours, hp_power, storage_power, slack_power,
                  labels=['Heat Pump (to Demand)', 'Thermal Storage (Discharge)', 'Slack Heat Source (to Demand)'],
                  colors=['skyblue', 'orange', 'red'],
                  alpha=0.7)
    
    plt.title('Actual Heat Demand Coverage Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text with key statistics
    total_hp = np.sum(hp_power)
    total_storage = np.sum(storage_power)
    total_slack = np.sum(slack_power)
    total_supply = total_hp + total_storage + total_slack
    
    if total_supply > 0:
        hp_fraction = total_hp / total_supply * 100
        storage_fraction = total_storage / total_supply * 100
        slack_fraction = total_slack / total_supply * 100
        
        stats_text = f'HP: {hp_fraction:.1f}%\nTS: {storage_fraction:.1f}%\nSlack: {slack_fraction:.1f}%'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dso_model_v2_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Optional standalone baseline flexible load figure
    if bool(globals().get('SHOW_BASELINE_FLEX_FIG', False)):
        try:
            baseline_flex_dict = globals().get('flexible_time_synchronized_loads_P', {})
            if baseline_flex_dict:
                fig_bl, ax_bl = plt.subplots(figsize=(14,6))
                all_buses = set()
                for tmap in baseline_flex_dict.values():
                    all_buses.update(tmap.keys())
                buses_sorted = sorted(list(all_buses))
                color_map = plt.cm.tab20(np.linspace(0,1,max(1,len(buses_sorted))))
                T = len(hours)
                for i, bus in enumerate(buses_sorted):
                    series_kw = []
                    for t in range(T):
                        val_mw = 0.0
                        if t in baseline_flex_dict and bus in baseline_flex_dict[t]:
                            val_mw = float(baseline_flex_dict[t][bus])
                        series_kw.append(val_mw * 1000.0)
                    ax_bl.plot(hours, series_kw, linewidth=1.0, alpha=0.8, color=color_map[i%len(color_map)], label=f'Bus {bus}' if len(buses_sorted) <= 20 else None)
                ax_bl.set_title('Baseline Flexible Load per Bus (kW)', fontsize=14, fontweight='bold')
                ax_bl.set_xlabel('Hour')
                ax_bl.set_ylabel('Power (kW)')
                ax_bl.grid(True, alpha=0.3)
                if len(buses_sorted) <= 20:
                    ax_bl.legend(fontsize=8, ncol=2)
                fig_bl.tight_layout()
                fig_bl.savefig('baseline_flexible_loads.png', dpi=300, bbox_inches='tight')
                plt.show()
        except Exception as _baseline_fig_err:
            print(f"[WARN] Could not produce baseline flexible load figure: {_baseline_fig_err}")
    
    # Create additional detailed plots
    create_detailed_junction_plots(results_df)
    

def create_detailed_junction_plots(results_df):
    """Create detailed plots showing all junction temperatures"""
    
    # Get all temperature columns
    temp_cols = [col for col in results_df.columns if 'temp' in col.lower()]
    supply_cols = [col for col in temp_cols if 'supply' in col.lower()]
    return_cols = [col for col in temp_cols if 'return' in col.lower()]
    
    hours = np.arange(len(results_df))
    
    if supply_cols:
        # Plot all supply temperatures
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Supply temperatures
        for col in supply_cols:
            ax1.plot(hours, results_df[col], alpha=0.6, linewidth=1)
        
        ax1.set_title('All Supply Junction Temperatures Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Temperature (°C)')
        ax1.grid(True, alpha=0.3)
        
        # Return temperatures
        for col in return_cols:
            ax2.plot(hours, results_df[col], alpha=0.6, linewidth=1)
        
        ax2.set_title('All Return Junction Temperatures Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('DHN_all_junction_temperatures.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("All plots have been generated and saved!")

####################################################################################################
# MULTI-PERIOD GUROBI OPTIMIZATION
####################################################################################################


def save_optim_results(results, filename):
    try:
        with open(filename, 'wb') as file:
            pkl.dump(results, file)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")


def create_network_from_csv():
    """Create a pandapower-like network structure from CSV files."""
    
    # Load CSV files
    network_data_path = "extracted_network_data"
    
    electrical_buses = pd.read_csv(os.path.join(network_data_path, "electrical_buses.csv"))
    electrical_lines = pd.read_csv(os.path.join(network_data_path, "electrical_lines.csv"))
    electrical_loads = pd.read_csv(os.path.join(network_data_path, "electrical_loads.csv"))
    electrical_pv_systems = pd.read_csv(os.path.join(network_data_path, "electrical_pv_systems.csv"))
    electrical_bess = pd.read_csv(os.path.join(network_data_path, "electrical_bess.csv"))
    electrical_transformers = pd.read_csv(os.path.join(network_data_path, "electrical_transformers.csv"))
    electrical_external_grids = pd.read_csv(os.path.join(network_data_path, "electrical_external_grids.csv"))
    
    print(f"Loaded network data:")
    print(f"  - Buses: {len(electrical_buses)}")
    print(f"  - Lines: {len(electrical_lines)}")
    print(f"  - Transformers: {len(electrical_transformers)}")
    print(f"  - Loads: {len(electrical_loads)}")
    print(f"  - PV Systems: {len(electrical_pv_systems)}")
    print(f"  - External Grids: {len(electrical_external_grids)}")
    
    # Create a simple class to hold network data
    class NetworkFromCSV:
        def __init__(self):
            # System base power
            self.sn_mva = 100.0  # Default base MVA
            
            # Create bus DataFrame - use bus_id as index
            self.bus = pd.DataFrame({
                'vn_kv': electrical_buses['vn_kv'],
                'name': electrical_buses['name'],
                'type': electrical_buses['type'],
                'zone': electrical_buses.get('zone', ''),
                'in_service': electrical_buses.get('in_service', True)
            })
            self.bus.index = electrical_buses['bus_id'].values
            
            # Create line DataFrame 
            self.line = pd.DataFrame({
                'from_bus': electrical_lines['from_bus'],
                'to_bus': electrical_lines['to_bus'],
                'length_km': electrical_lines['length_km'],
                'r_ohm_per_km': electrical_lines['r_ohm_per_km'],
                'x_ohm_per_km': electrical_lines['x_ohm_per_km'],
                'c_nf_per_km': electrical_lines.get('c_nf_per_km', 0),
                'max_i_ka': electrical_lines['max_i_ka'],
                'name': electrical_lines['name'],
                'in_service': electrical_lines.get('in_service', True)
            })
            
            # Create transformer DataFrame
            self.trafo = pd.DataFrame({
                'hv_bus': electrical_transformers['hv_bus'],
                'lv_bus': electrical_transformers['lv_bus'],
                'sn_mva': electrical_transformers['sn_mva'],
                'vn_hv_kv': electrical_transformers['vn_hv_kv'],
                'vn_lv_kv': electrical_transformers['vn_lv_kv'],
                'vk_percent': electrical_transformers['vk_percent'],
                'vkr_percent': electrical_transformers['vkr_percent'],
                'name': electrical_transformers['name'],
                'in_service': electrical_transformers.get('in_service', True)
            })
            
            # Create load DataFrame
            # Default values for missing columns
            default_p_mw = 0.001  # 1 kW default
            default_q_mvar = 0.0
            default_controllable = False
            
            self.load = pd.DataFrame({
                'bus': electrical_loads['bus'],  # Column is 'bus', not 'bus_id'
                'p_mw': electrical_loads.get('p_mw', default_p_mw),
                'q_mvar': electrical_loads.get('q_mvar', default_q_mvar),
                'controllable': electrical_loads.get('controllable', default_controllable),
                'name': electrical_loads.get('name', 'Load'),
                'in_service': electrical_loads.get('in_service', True)
            })
            
            # Create sgen (PV systems) DataFrame
            self.sgen = pd.DataFrame({
                'bus': electrical_pv_systems['bus_id'],
                'p_mw': electrical_pv_systems['capacity_kw'] / 1000.0,  # Convert kW to MW
                'q_mvar': electrical_pv_systems.get('q_mvar', 0.0),
                'name': electrical_pv_systems.get('name', 'PV'),
                'in_service': electrical_pv_systems.get('in_service', True),
                'type': electrical_pv_systems.get('type', 'PV')
            })


            # Also include BESS entries as static generators so that downstream code
            # that expects `net.sgen` (with columns like 'bus' and 'p_mw') will see
            # battery capacity as available dispatchable generation.
            bess_sgen = pd.DataFrame()
            try:
                if electrical_bess is not None and len(electrical_bess) > 0:
                    # Determine the power column for BESS (prefer max_p_mw, then p_mw, then kW variants)
                    if 'max_p_mw' in electrical_bess.columns:
                        p_mw_series = electrical_bess['max_p_mw'].astype(float)
                    elif 'p_mw' in electrical_bess.columns:
                        p_mw_series = electrical_bess['p_mw'].astype(float)
                    elif 'max_p_kw' in electrical_bess.columns:
                        p_mw_series = electrical_bess['max_p_kw'].astype(float) / 1000.0
                    elif 'p_kw' in electrical_bess.columns:
                        p_mw_series = electrical_bess['p_kw'].astype(float) / 1000.0
                    else:
                        # Fallback to zero power if no known column exists
                        p_mw_series = pd.Series(0.0, index=electrical_bess.index)

                    bess_sgen = pd.DataFrame({
                        'bus': electrical_bess['bus'],
                        'p_mw': p_mw_series,
                        'q_mvar': electrical_bess.get('q_mvar', 0.0),
                        'name': electrical_bess.get('name', 'BESS'),
                        'in_service': electrical_bess.get('in_service', True),
                        'type': electrical_bess.get('type', 'BESS')
                    })
                    # Cast bus indices to integers when possible
                    try:
                        bess_sgen['bus'] = bess_sgen['bus'].astype(int)
                    except Exception:
                        pass
                else:
                    bess_sgen = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'name', 'in_service', 'type'])
            except Exception:
                # If reading BESS failed for any reason, continue with PV-only sgen
                bess_sgen = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'name', 'in_service', 'type'])

            # Concatenate PV sgens and BESS-derived sgens into a single sgen DataFrame
            if bess_sgen is not None and len(bess_sgen) > 0:
                # Ensure both dataframes have the same columns before concat
                for col in ['bus', 'p_mw', 'q_mvar', 'name', 'in_service', 'type']:
                    if col not in self.sgen.columns:
                        self.sgen[col] = None
                    if col not in bess_sgen.columns:
                        bess_sgen[col] = None

                self.sgen = pd.concat([self.sgen, bess_sgen], ignore_index=True, sort=False)
            else:
                # No BESS to add; keep PV-only sgen
                self.sgen = self.sgen.reset_index(drop=True)
            
            # Create external grid DataFrame
            self.ext_grid = pd.DataFrame({
                'bus': electrical_external_grids['bus'],
                'vm_pu': electrical_external_grids.get('vm_pu', 1.0),
                'va_degree': electrical_external_grids.get('va_degree', 0.0),
                'name': electrical_external_grids.get('name', 'External Grid'),
                'in_service': electrical_external_grids.get('in_service', True)
            })
            
            # Create empty controller DataFrame (no controllers from CSV)
            self.controller = pd.DataFrame(columns=['object'])
            
    return NetworkFromCSV()


def calculate_z_matrix(net):
    """Calculate the impedance matrix (Z) for all lines and transformers."""
    num_branches = len(net.line) + len(net.trafo)  # Total branches
    Z = np.zeros(num_branches, dtype=complex)  # Initialize impedance vector
    
    # System base values
    base_MVA = net.sn_mva  

    # Process lines
    for i, line in enumerate(net.line.itertuples()):
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Base impedance for per-unit conversion
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Use "from bus" voltage as reference
        Z_base = base_voltage ** 2 / base_MVA
        
        # Compute per-unit impedance
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        Z[i] = r_pu + 1j * x_pu  # Store complex impedance

    # Process transformers
    for j, trafo in enumerate(net.trafo.itertuples(), start=len(net.line)):
        trafo_base_mva = trafo.sn_mva  # Transformer rated MVA
        system_base_mva = base_MVA  # System base MVA
        
        # Compute per-unit impedance
        z_pu = (trafo.vk_percent / 100) * (system_base_mva / trafo_base_mva)
        r_pu = (trafo.vkr_percent / 100) * (system_base_mva / trafo_base_mva)
        x_pu = np.sqrt(z_pu**2 - r_pu**2)  # Solve for reactance
        
        Z[j] = r_pu + 1j * x_pu  # Store complex impedance
    
    return Z

def compute_incidence_matrix(net):
    """Constructs the correct incidence matrix A for the network."""
    A_bus = np.zeros((len(net.bus), len(net.line) + len(net.trafo)))  # Extend for transformers

    # Process lines
    for idx, line in net.line.iterrows():
        A_bus[int(line.from_bus), idx] = 1   # Sending end
        A_bus[int(line.to_bus), idx] = -1    # Receiving end

    # Process transformers
    for idx, trafo in net.trafo.iterrows():
        trafo_index = len(net.line) + idx  # Continue indexing after lines
        A_bus[int(trafo.hv_bus), trafo_index] = 1    # High-voltage side
        A_bus[int(trafo.lv_bus), trafo_index] = -1   # Low-voltage side

    # Save for debugging
    #A_df = pd.DataFrame(A_bus, index=net.bus.index, columns=list(net.line.index) + list(net.trafo.index))
    #A_df.to_csv("fixed_incidence_matrix_A.csv")
    #print("Fixed incidence matrix saved as 'fixed_incidence_matrix_A.csv'.")

    return A_bus

def calculate_gbus_matrix(net):
    """Calculate the conductance matrix (Gbus) for LDF-LC."""
    num_buses = len(net.bus)
    Gbus = np.zeros((num_buses, num_buses))  # Initialize Gbus matrix
    
    base_MVA = net.sn_mva  # System base MVA
    
    # Add line resistances
    for line in net.line.itertuples():
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Convert resistance to per-unit (R_pu = R_ohm / (Base Voltage^2 / Base MVA))
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Base voltage for the from bus
        Z_base = base_voltage ** 2 / net.sn_mva  # Calculate base impedance
        Y_base = 1 / Z_base  # Calculate base admittance
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        
        Y_series = 1 / (r_pu + 1j * x_pu)  # Series admittance
        #print(f"Y_series: {Y_series}")
        G_pu = Y_series.real  # Conductance in per-unit
        #print(f"G_pu: {G_pu}")
        
        # Gbus off-diagonal elements
        Gbus[from_bus, to_bus] -= G_pu 
        Gbus[to_bus, from_bus] -= G_pu 
        
        # Gbus diagonal elements
        Gbus[from_bus, from_bus] += G_pu 
        Gbus[to_bus, to_bus] += G_pu 
    
    # Add transformer resistances
    for trafo in net.trafo.itertuples():
        hv_bus = trafo.hv_bus
        lv_bus = trafo.lv_bus
        
        trafo_base_mva = trafo.sn_mva  # Extract transformer base MVA
        system_base_mva = net.sn_mva  # System base MVA


        # Compute correct transformer impedance values
        z_pu = (trafo.vk_percent / 100)*(system_base_mva/trafo_base_mva)  # Total impedance in per-unit
        #print(f"z_pu: {z_pu}")

        r_pu = (trafo.vkr_percent / 100)*(system_base_mva/trafo_base_mva)  # Resistance in per-unit
        #print(f"r_pu: {r_pu}")

        x_pu = np.sqrt(z_pu**2 - r_pu**2)  # Reactance computed from Z and R
        #print(f"x_pu: {x_pu}")

        G_pu = r_pu / (r_pu**2 + x_pu**2)  # Conductance in per-unit

        # Gbus off-diagonal elements
        Gbus[hv_bus, lv_bus] -= G_pu 
        Gbus[lv_bus, hv_bus] -= G_pu 
    
        # Gbus diagonal elements
        Gbus[hv_bus, hv_bus] += G_pu 
        Gbus[lv_bus, lv_bus] += G_pu 

    return Gbus


def calculate_bbus_matrix(net):
    """Calculate the susceptance matrix (Bbus) for LDF-LC."""
    num_buses = len(net.bus)
    Bbus = np.zeros((num_buses, num_buses))  # Initialize Bbus matrix
    
    base_MVA = net.sn_mva  # System base MVA
    
    # Add line reactances
    for line in net.line.itertuples():
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Convert reactance to per-unit (X_pu = X_ohm / (Base Voltage^2 / Base MVA))
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Base voltage for the from bus
        Z_base = base_voltage ** 2 / net.sn_mva  # Calculate base impedance
        Y_base = 1 / Z_base  # Calculate base admittance
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        
        Y_series = 1 / (r_pu + 1j * x_pu)  # Series admittance
        #print(f"Y_series: {Y_series}")
        B_pu = Y_series.imag  # Susceptance in per-unit

        
        # Bbus off-diagonal elements
        Bbus[from_bus, to_bus] -= B_pu 
        Bbus[to_bus, from_bus] -= B_pu 
        
        # Bbus diagonal elements
        Bbus[from_bus, from_bus] += B_pu 
        Bbus[to_bus, to_bus] += B_pu 
    
    # Add transformer reactances
    for trafo in net.trafo.itertuples():
        hv_bus = trafo.hv_bus
        lv_bus = trafo.lv_bus
        
        trafo_base_mva = trafo.sn_mva  # Extract transformer base MVA
        system_base_mva = net.sn_mva  # System base MVA


        # Compute correct transformer impedance values
        z_pu = (trafo.vk_percent / 100)*(system_base_mva/trafo_base_mva)  # Total impedance in per-unit
        #print(f"z_pu: {z_pu}")

        r_pu = (trafo.vkr_percent / 100)*(system_base_mva/trafo_base_mva)  # Resistance in per-unit
        #print(f"r_pu: {r_pu}")

        x_pu = np.sqrt(z_pu**2 - r_pu**2)  # Reactance computed from Z and R
        #print(f"x_pu: {x_pu}")

        B_pu = -x_pu / (r_pu**2 + x_pu**2)  # Conductance in per-unit
    
        # Bbus off-diagonal elements
        Bbus[hv_bus, lv_bus] -= B_pu
        Bbus[lv_bus, hv_bus] -= B_pu
    
        # Bbus diagonal elements
        Bbus[hv_bus, hv_bus] += B_pu
        Bbus[lv_bus, lv_bus] += B_pu
    
    return Bbus


def accumulate_downstream_power(A, P_mw, Q_mw, net, downstream_map):
    """Accumulates downstream power flows using the correct downstream node mapping."""
    num_buses = A.shape[0]

    # Initialize accumulated power with the nodal values
    P_accumulated = P_mw.copy()
    Q_accumulated = Q_mw.copy()

    # Identify slack bus
    slack_bus_index = net.ext_grid.bus.iloc[0]

    # Traverse each bus and accumulate power from its downstream buses
    for bus in range(num_buses):
        if bus == slack_bus_index:
            continue  # Skip slack bus

        for child_bus in downstream_map[bus]:  # All downstream buses
            P_accumulated[bus] += P_mw[child_bus]
            Q_accumulated[bus] += Q_mw[child_bus]

    return P_accumulated, Q_accumulated



def compute_downstream_nodes(A, net):
    """Returns a mapping of each bus to its downstream nodes and prints the hierarchy."""
    num_buses, num_branches = A.shape
    graph = nx.DiGraph()

    # Construct directed graph from incidence matrix
    for branch_idx in range(num_branches):
        from_bus = np.where(A[:, branch_idx] == 1)[0][0]
        to_bus = np.where(A[:, branch_idx] == -1)[0][0]
        graph.add_edge(from_bus, to_bus)

    # Perform BFS from the slack bus
    slack_bus = net.ext_grid.bus.iloc[0]
    downstream_map = {bus: [] for bus in range(num_buses)}

    for bus in graph.nodes:
        if bus == slack_bus:
            continue
        predecessors = list(nx.ancestors(graph, bus))
        for pred in predecessors:
            downstream_map[pred].append(bus)

    # Print hierarchy
    #print("\n--- Downstream Node Hierarchy ---")
    #for bus, children in downstream_map.items():
    #    print(f"Bus {bus}: {children}")

    return downstream_map

def compute_Ybus(Gbus, Bbus):
    """Computes the admittance matrix Ybus = Gbus + jBbus."""
    return Gbus + 1j * Bbus

base_df = pd.read_csv("hp_baseline_profile.csv")
base_map = {(int(r.weekday), int(r.hour), int(r.minute)): float(r.P_base_MW)
            for r in base_df.itertuples(index=False)}

def baseline_lookup(dt):
    key = (dt.weekday(), dt.hour, dt.minute)
    # fallback to global mean if slot missing
    return base_map.get(key, float(base_df["P_base_MW"].mean()))


def solve_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb):
    #variance_net= gd.setup_grid_IAS_variance(season)
    #var_results = calculate_variance_propagation(time_steps, variance_net)

    #k_epsilon = np.sqrt((1 - par.epsilon) / par.epsilon)
    target_supply_temp_K = 85+273.15  # Target supply temperature in Kelvin
    HP_PMAX_MW = 0.30  # Max heat pump power in MW 

    pd.set_option('display.precision', 10)
    model = gp.Model("opf_with_ldf_lc")

    # ------------------------------------------------------------------
    # Build daily mean temperature lookup used by the HP predictor:
    # daily_mean_temp_for_dt[date] -> mean ambient temperature (Celsius)
    # Expectation: T_amb is an array-like of temperatures in Kelvin aligned
    # with the global `time_index` (pandas.DatetimeIndex). We compute per-day
    # means in Celsius and store as a dict keyed by date objects.
    try:
        temps_k = np.asarray(T_amb, dtype=float)
    except Exception:
        # fallback: try converting elements one-by-one
        temps_k = np.array([float(x) for x in T_amb])

    temps_c = temps_k - 273.15

    # Try to align with global `time_index` if available and length matches
    try:
        if 'time_index' in globals() and len(time_index) == len(temps_c):
            idx = pd.to_datetime(time_index)
        else:
            # Fallback: create a generic hourly index starting today
            idx = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(temps_c), freq='H')
    except Exception:
        idx = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(temps_c), freq='H')

    temps_series = pd.Series(temps_c, index=idx)
    # Group by date (datetime.date) and compute mean; result is a Series with date keys
    daily_mean_temp_for_dt = temps_series.groupby(temps_series.index.date).mean().to_dict()
    # ------------------------------------------------------------------

    try:
        base_df = pd.read_csv("hp_baseline_profile.csv")
        base_map = {
            (int(r.weekday), int(r.hour), int(r.minute)): float(r.P_base_MW)
            for r in base_df.itertuples(index=False)
        }
        global_base_mean = float(base_df["P_base_MW"].mean())
    except FileNotFoundError:
        # Safe fallback: flat baseline = 0
        base_map = {}
        global_base_mean = 0.0
        print("⚠️  hp_baseline_profile.csv not found. Using zero baseline.")

    def baseline_lookup(dt: pd.Timestamp) -> float:
        """Return baseline MW for (weekday, hour, minute); fallback to mean."""
        return base_map.get((dt.weekday(), dt.hour, dt.minute), global_base_mean)


    bess_eff = 0.95  # Round-trip efficiency
    bess_initial_soc = 0.5  # Initial state of charge as a percentage of capacity
    bess_capacity_mwh = 0.1  # BESS capacity in MWh
    bess_cost_per_mwh = 5.1 # Cost per MWh of BESS capacity
    # Baseline (intercept) BESS throughput cost (EUR/MWh) for p0 channel
    c_base_bess = 1.5
    # Extract transformer capacity in MW (assuming sn_mva is in MVA)
    transformer_capacity_mw = net.trafo['sn_mva'].values[0]
    #print(f"Transformer Capacity: {transformer_capacity_mw}")


    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables+
    bess_energy_vars = {}  # Store BESS state of charge decision variables
    bess_charge_vars = {}  # Store BESS charging power decision variables
    bess_discharge_vars = {}  # Store BESS discharging power decision variables
    ext_grid_import_P_vars = {}  # Store external grid import power decision variables
    ext_grid_import_Q_vars = {}  # Store external grid import power decision variables
    ext_grid_export_P_vars = {}  # Store external grid export power decision variables
    ext_grid_export_Q_vars = {}  # Store external grid export power decision variables
    V_vars = {}  # Store voltage angle decision variables (radians)
    curtailment_vars = {} # Store decision variables for curtailment
    flexible_load_P_vars = {}  # flexible load variables
    flexible_load_Q_vars = {}  # flexible load variables
    flex_curtail_P_vars = {} # Store flexible load curtailment variables
    p_hp_vars = {}  # Heat pump load variables indexed by bus and timestep
    P_branch_vars = {}  # Store line power flow decision variables
    Q_branch_vars = {}  # Store line power flow decision variables
    Line_loading_vars = {}  # Store line loading decision variables
    P_trafo_vars = {}  # Store transformer loading decision variables
    Q_trafo_vars = {}  # Store transformer loading decision variables
    S_trafo_vars = {}  # Store transformer loading decision variables
    transformer_loading_vars = {}  # Store transformer loading percentage decision variables 
    transformer_loading_perc_vars = {}  # Store transformer loading percentage decision variables 
    shed_vars = {} 

    slack_bus_index = net.ext_grid.bus.iloc[0]

    Z = calculate_z_matrix(net)
    Gbus = calculate_gbus_matrix(net)
    Bbus = calculate_bbus_matrix(net)
    Ybus = compute_Ybus(Gbus, Bbus)
    A = compute_incidence_matrix(net)
    downstream_map = compute_downstream_nodes(A, net)
    # Get correct downstream mappings
    downstream_map = compute_downstream_nodes(A, net)
    # Retrieve all controllers from net

    Ybus_reduced = np.delete(np.delete(Ybus, slack_bus_index, axis=0), slack_bus_index, axis=1)


    # Dictionaries to store results
    pv_gen_results = {}
    bess_charge_results = {}
    bess_discharge_results = {}
    bess_energy_results = {}
    flexible_load_P_results = {}
    flexible_load_Q_results = {}
    non_flexible_load_P_results = {}
    non_flexible_load_Q_results = {}
    p_hp_results = {}
    load_Q_results = {}
    ext_grid_import_P_results = {}
    ext_grid_import_Q_results = {}
    ext_grid_export_P_results = {}
    ext_grid_export_Q_results = {}
    flex_curtail_P_results = {}
    V_results = {}
    line_pl_results = {}
    line_ql_results = {}
    line_current_results = {}
    line_loading_results = {}
    transformer_loading_results = {}

    # Robust recourse summaries (initialized early)
    D_plus_max = {}
    D_minus_max = {}
    pv_avail_sum_by_t = {}
    hp_pred_nominal = {}
    sum_nonflex_by_t = {}
    sum_flex_by_t = {}
    # Policy variable dicts
    y_cap_vars = {}
    gamma0_vars = {}
    gamma_plus_vars = {}
    chi0_vars = {}
    chi_minus_vars = {}
    lambda0_vars = {}
    lambda_plus_vars = {}
    lambda_minus_vars = {}
    rho_plus0_vars = {}
    rho_plus1_vars = {}
    rho_minus0_vars = {}
    rho_minus1_vars = {}
    z_dis_vars = {}
    z_ch_vars = {}
    # Baseline intercept channel (added to make lambda0 physically meaningful)
    p0_dis_vars = {}
    p0_ch_vars = {}
    # Robust SoC extreme envelope trajectories (down/up under full deviations)
    E_down_vars = {}
    E_up_vars = {}
    ycap_var = None  # single y_cap for this model


    # Temporary dictionary to store updated load values per time step
    flexible_time_synchronized_loads_P = {t: {} for t in time_steps}
    flexible_time_synchronized_loads_Q = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads_P = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads_Q = {t: {} for t in time_steps}

    # Attach early (empty) to globals so later code that queries it never gets None
    globals()['flexible_time_synchronized_loads_P'] = flexible_time_synchronized_loads_P
    globals()['flexible_time_synchronized_loads_Q'] = flexible_time_synchronized_loads_Q
    globals()['non_flexible_time_synchronized_loads_P'] = non_flexible_time_synchronized_loads_P
    globals()['non_flexible_time_synchronized_loads_Q'] = non_flexible_time_synchronized_loads_Q



    # Add BEV loads from CSV into net.load (if present) so they behave like electrical_loads.csv entries
    try:
        bev_csv_path = os.path.join('extracted_network_data', 'electrical_BEV.csv')
        if os.path.exists(bev_csv_path):
            bev_df = pd.read_csv(bev_csv_path)
            bev_rows = []
            for _, r in bev_df.iterrows():
                # tolerate common column names
                name = r.get('name') if 'name' in r.index else (r.get('Name') if 'Name' in r.index else None)
                if name is None:
                    continue
                # bus may be int-like
                try:
                    bus = int(r.get('bus'))
                except Exception:
                    # skip invalid bus
                    continue
                # p_mw may be provided in MW or kW ('p_mw' or 'p_kw')
                p_mw = 0.0
                if 'p_mw' in r.index and not pd.isna(r.get('p_mw')):
                    p_mw = float(r.get('p_mw'))
                elif 'p_kw' in r.index and not pd.isna(r.get('p_kw')):
                    p_mw = float(r.get('p_kw')) / 1000.0
                # reactive power
                q_mvar = 0.0
                if 'q_mvar' in r.index and not pd.isna(r.get('q_mvar')):
                    q_mvar = float(r.get('q_mvar'))
                # controllable flag
                controllable = True
                if 'controllable' in r.index and not pd.isna(r.get('controllable')):
                    try:
                        controllable = bool(r.get('controllable'))
                    except Exception:
                        controllable = str(r.get('controllable')).strip().lower() in ('1','true','yes')
                in_service = True
                if 'in_service' in r.index and not pd.isna(r.get('in_service')):
                    try:
                        in_service = bool(r.get('in_service'))
                    except Exception:
                        in_service = str(r.get('in_service')).strip().lower() in ('1','true','yes')

                bev_rows.append({'bus': bus, 'p_mw': p_mw, 'q_mvar': q_mvar, 'controllable': controllable, 'name': name, 'in_service': in_service})

            if bev_rows:
                bev_loads_df = pd.DataFrame(bev_rows)
                # avoid adding duplicate names already present in net.load
                existing_names = set(net.load['name'].astype(str).tolist()) if 'name' in net.load.columns else set()
                bev_loads_df = bev_loads_df[~bev_loads_df['name'].astype(str).isin(existing_names)]
                if not bev_loads_df.empty:
                    # append rows to net.load, preserve index semantics
                    net.load = pd.concat([net.load, bev_loads_df], ignore_index=True)
                    #print(f"Added {len(bev_loads_df)} BEV loads to net.load from '{bev_csv_path}'")
    except Exception as _e:
        print(f"Warning: failed to append BEV loads to net.load: {_e}")

    # Identify buses with flexible loads
    flexible_load_buses = list(set(net.load[net.load['controllable'] == True].bus.values))
    non_flexible_load_buses = list(set(net.load[net.load['controllable'] == False].bus.values))
    # Identify buses with heat pump loads (loads with names starting with 'HP')
    hp_load_buses = list(set(net.load[net.load['name'].str.startswith('HP', na=False)].bus.values))
    # Power factor settings: households (non-flexible loads) and heat pumps
    # Inductive power factor (lagging): specify as positive pf < 1.0
    pf_household = 0.98  # households (non-flexible)
    pf_heatpump = 0.99   # heat pumps
    # Precompute Q-factor (tan(arccos(pf))) for MW->MVar scaling
    qfactor_household = float(np.tan(np.arccos(pf_household)))
    qfactor_heatpump = float(np.tan(np.arccos(pf_heatpump)))
    #print(f"Flexible load buses: {flexible_load_buses}")
    #print(f"Heat pump load buses: {hp_load_buses}")

    # Diagnostic: check non-flexible electrical loads have mapped time series
    import difflib
    # electrical_time_series is created earlier during profile mapping and stored in globals
    electrical_ts_keys = set(globals().get('electrical_time_series', {}).keys())

    # Collect non-flexible load names (exclude HP loads)
    non_flexible_load_names = []
    for load in net.load.itertuples():
        if not getattr(load, 'controllable', False):
            name = getattr(load, 'name', '')
            if name and not str(name).upper().startswith('HP'):
                non_flexible_load_names.append(name)

    missing_elec = [n for n in non_flexible_load_names if n not in electrical_ts_keys]
    if missing_elec:
        print('\n⚠️  Diagnostic: non-flexible electrical loads without mapped time series (will use zero-series):')
        for n in missing_elec:
            suggestions = difflib.get_close_matches(n, list(electrical_ts_keys), n=5, cutoff=0.5)
            print(f"  - {n}  -> suggestions: {suggestions}")
        print("\nNote: Unmatched electrical loads will be assigned a zero time series. To avoid this, ensure matching columns exist in 'vdi_profiles/all_house_profiles.csv' or adjust load names in 'extracted_network_data/electrical_loads.csv'.\n")

    # Track which electrical loads we've already warned about to avoid repeating warnings each timestep
    _warned_missing_electrical = set()

    # Compute a static aggregate BESS power capacity (MW) from net.sgen for RT proxy bounds
    try:
        if 'type' in net.sgen.columns:
            _bess_mask_static = net.sgen['type'].astype(str).str.contains('BESS', na=False)
            total_bess_pmax = float(net.sgen.loc[_bess_mask_static, 'p_mw'].sum())
        else:
            total_bess_pmax = 0.0
    except Exception:
        total_bess_pmax = 0.0

    # Determine PV and BESS buses/constants once (static over horizon)
    if 'type' in net.sgen.columns:
        pv_mask_static = net.sgen['type'].astype(str).str.contains('PV', na=False)
        bess_mask_static = net.sgen['type'].astype(str).str.contains('BESS', na=False)
    else:
        pv_mask_static = pd.Series([True] * len(net.sgen), index=net.sgen.index)
        bess_mask_static = pd.Series([False] * len(net.sgen), index=net.sgen.index)

    pv_buses = sorted(set(net.sgen.loc[pv_mask_static, 'bus'].astype(int).values))
    bess_buses = sorted(set(net.sgen.loc[bess_mask_static, 'bus'].astype(int).values))
    base_pv_bus_limits = {bus: float(net.sgen.loc[(net.sgen['bus'] == bus) & pv_mask_static, 'p_mw'].sum()) for bus in pv_buses}
    base_bess_bus_limits = {bus: float(net.sgen.loc[(net.sgen['bus'] == bus) & bess_mask_static, 'p_mw'].sum()) for bus in bess_buses}

    # net.load details suppressed to avoid verbose output
    for t in time_steps:
        dt = time_index[t]                  # pandas Timestamp for this slot
        P_base = baseline_lookup(dt)

        # features
        T_C_t = float(T_amb[t]) - 273.15
        price_t = float(electricity_price[t])
        HDD_t = max(0.0, 10.0 - T_C_t)      # use the Tbase you chose
        T_avg_d = daily_mean_temp_for_dt[dt.date()]  # provide this array/dict
        tod = dt.hour + dt.minute/60.0
        sin24 = np.sin(2*np.pi*tod/24.0)
        cos24 = np.cos(2*np.pi*tod/24.0)

        # Debug: print the full net.load table only for the first timestep to avoid huge output
        if t == time_steps[0]:
            try:
                print('\nDEBUG: net.load (first timestep) - showing all load rows:')
                # use to_string to ensure full DataFrame printing
                print(net.load.to_string())
                print('\nDEBUG: net.sgen (first timestep) - showing all sgen rows:')
                print(net.sgen.to_string())
            except Exception:
                # Fallback: lightweight repr to avoid exceptions
                try:
                    print('\nDEBUG: net.load (first timestep):', repr(net.load))
                except Exception:
                    print('\nDEBUG: net.load (first timestep): <unprintable>')

        # HP predictor: baseline + deviation using shared constants
        y_dev = (
            HP_COEFF_B0
            + HP_COEFF_BHDD * HDD_t
            + HP_COEFF_BPI * price_t
            + HP_COEFF_BTAV * T_avg_d
            + HP_COEFF_A1 * sin24
            + HP_COEFF_A2 * cos24
        )
        P_t = P_base + HP_PRED_PMAX * y_dev
        P_t = min(max(P_t, 0.0), HP_PRED_PMAX)      # clip to [0, Pmax]

        # Track nominal HP aggregate for budgets for every timestep
        hp_pred_nominal[t] = float(P_t * len(hp_load_buses)) if len(hp_load_buses) > 0 else 0.0

    # Precompute k_epsilon (quantile amplification) and aggregate sigma when DRCC tightening is enabled.
    # Option 1 semantics: if tightening disabled, keep RT budgets but neutralize chance amplification with k=1.
    use_net_drcc = bool(ENABLE_DRCC_NETWORK_TIGHTENING)
    k_epsilon = 1.0  # default neutral scaling when DRCC inactive
    k_source = "neutral"
    sigma_net = {t: 0.0 for t in time_steps}
    if use_net_drcc:
        try:
            eps = float(DRCC_EPSILON)
            if eps <= 0 or eps >= 1:
                eps = 0.05
            k_epsilon = float(np.sqrt((1.0 - eps) / eps))
            k_source = "drcc_quantile"
        except Exception:
            k_epsilon = float(np.sqrt((1.0 - 0.05) / 0.05))
            k_source = "drcc_quantile_fallback"

        # Needed std inputs prepared in main()
        const_pv_std = globals().get('const_pv_std', np.zeros(len(time_steps)))
        T_amb_std = globals().get('T_amb_std', np.zeros(len(time_steps)))

        # PV installed capacity across PV buses (MW)
        if len(pv_buses) > 0:
            pv_installed_mw = float(sum(base_pv_bus_limits.get(b, 0.0) for b in pv_buses))
        else:
            pv_installed_mw = 0.0

        # HP predictor sensitivity wrt daily mean temperature (same used earlier)
        hp_temp_sens = abs(HP_PRED_PMAX * HP_COEFF_BTAV)  # MW per K

        for t in time_steps:
            sigma_pv = float(const_pv_std[t]) * pv_installed_mw
            sigma_hp = hp_temp_sens * float(T_amb_std[t])
            sigma_net[t] = float(np.sqrt(sigma_pv**2 + sigma_hp**2))
    else:
        # When not using DRCC tightening, we still may use std-based RT budgets elsewhere with k=1.
        # No network sigma aggregation needed for tightening margins.
        pass

    # Add variables for each time step
    for t in time_steps:
        if not net.controller.empty:
            for _, controller in net.controller.iterrows():
                controller.object.time_step(net, time=t)


        # Initialize dictionaries for time-synchronized loads
        flexible_time_synchronized_loads_P[t] = {}
        flexible_time_synchronized_loads_Q[t] = {}
        non_flexible_time_synchronized_loads_P[t] = {}
        non_flexible_time_synchronized_loads_Q[t] = {}

        # # Debug: print the full net.load table only for the first timestep to avoid huge output
        # if t == time_steps[0]:
        #     try:
        #         print('\nDEBUG: net.load (first timestep) - showing all load rows:')
        #         # use to_string to ensure full DataFrame printing
        #         print(net.load.to_string())
        #         print('\nDEBUG: net.sgen (first timestep) - showing all sgen rows:')
        #         print(net.sgen.to_string())
        #     except Exception:
        #         # Fallback: lightweight repr to avoid exceptions
        #         try:
        #             print('\nDEBUG: net.load (first timestep):', repr(net.load))
        #         except Exception:
        #             print('\nDEBUG: net.load (first timestep): <unprintable>')

        # Iterate over all loads
        for load in net.load.itertuples():
            bus = load.bus
            load_name = getattr(load, 'name', '')
            
            # Check if this is a heat pump load (name starts with 'HP')
            if load_name.startswith('HP'):
                # Heat pump loads are identified but their power will be determined by Q_hp and COP
                # No time-synchronized load processing needed for heat pumps
                pass
            elif load.controllable:
                # Flexible load: map its time series (same behavior as non-flexible loads)
                mapped_series = None
                if load_name and 'electrical_time_series' in globals():
                    if load_name in electrical_time_series:
                        # series is in kW per timestep -> convert to MW (same scaling as non-flexible)
                        mapped_series = np.array(electrical_time_series[load_name]) / 1000.0

                if mapped_series is None:
                    # Fallback: assign zero-series for unmatched electrical load
                    print(f"Flexible load '{load_name}' has no mapped time series, assigning zero-series.")
                    mapped_series = np.zeros(NUM_PERIODS)
                    # Only warn once per load name to avoid verbose output
                    if load_name not in _warned_missing_electrical:
                        print(f"Warning: no mapped electrical series for flexible load '{load_name}'. Using zero-series.")
                        _warned_missing_electrical.add(load_name)

                # Use mapped value for this timestep
                if t < len(mapped_series):
                    load_p_mw_t = float(mapped_series[t])
                else:
                    load_p_mw_t = 0.0

                flexible_time_synchronized_loads_P[t][bus] = (
                    flexible_time_synchronized_loads_P[t].get(bus, 0.0) + load_p_mw_t
                )
                # Compute reactive power from active power using household power factor (inductive)
                q_mvar_t = load_p_mw_t * qfactor_household
                flexible_time_synchronized_loads_Q[t][bus] = (
                    flexible_time_synchronized_loads_Q[t].get(bus, 0.0) + q_mvar_t
                )
        # ...existing code for flexible load mapping...
            else:
                # Non-flexible load - time-varying profile must be provided by electrical_time_series
                load_p_mw = load.p_mw  # Base peak power from network definition (MW)
                load_q_mvar = load.q_mvar  # Base reactive power

                # Determine load name and check for a mapped time series
                load_name = getattr(load, 'name', '')
                mapped_series = None
                if load_name and 'electrical_time_series' in globals():
                    # electrical_time_series keys are original names (string), match exactly
                    if load_name in electrical_time_series:
                        # series is in kW per timestep -> convert to MW
                        mapped_series = np.array(electrical_time_series[load_name])*10 / 1000.0 #scaling factor of 10 applied here

                if mapped_series is None:
                    # Fallback: assign zero-series for unmatched electrical load
                    mapped_series = np.zeros(NUM_PERIODS)
                    # Only warn once per load name to avoid verbose output
                    if load_name not in _warned_missing_electrical:
                        print(f"Warning: no mapped electrical series for '{load_name}'. Using zero-series.")
                        _warned_missing_electrical.add(load_name)

                # Use mapped value for this timestep
                if t < len(mapped_series):
                    load_p_mw_t = mapped_series[t]
                else:
                    # If series is too short, use 0 for missing timesteps
                    load_p_mw_t = 0.0

                non_flexible_time_synchronized_loads_P[t][bus] = (
                    non_flexible_time_synchronized_loads_P[t].get(bus, 0.0) + load_p_mw_t
                )
                # Compute reactive power from active power using household power factor (inductive)
                # load_p_mw_t is in MW, Q will be in MVar ~ MW * tan(arccos(pf))
                q_mvar_t = load_p_mw_t * qfactor_household
                non_flexible_time_synchronized_loads_Q[t][bus] = (
                    non_flexible_time_synchronized_loads_Q[t].get(bus, 0.0) + q_mvar_t
                )


        # Ensure all buses have an entry, even if no loads are connected
        for bus in net.bus.index:
            if bus not in flexible_time_synchronized_loads_P[t]:
                flexible_time_synchronized_loads_P[t][bus] = 0.0
            if bus not in flexible_time_synchronized_loads_Q[t]:
                flexible_time_synchronized_loads_Q[t][bus] = 0.0
            if bus not in non_flexible_time_synchronized_loads_P[t]:
                non_flexible_time_synchronized_loads_P[t][bus] = 0.0
            if bus not in non_flexible_time_synchronized_loads_Q[t]:
                non_flexible_time_synchronized_loads_Q[t][bus] = 0.0

        # After finishing this timestep, optionally store cumulative stats for debugging
        try:
            sum_flex = sum(flexible_time_synchronized_loads_P[t].values())
            sum_nonflex = sum(non_flexible_time_synchronized_loads_P[t].values())
            if t == 0:
                print(f"[DEBUG FLEX BUILD] t={t} total baseline flexible MW = {sum_flex:.6f} | non-flexible MW = {sum_nonflex:.6f}")
            elif t in (int(len(time_steps)/2), len(time_steps)-25):
                print(f"[DEBUG FLEX BUILD] t={t} total baseline flexible MW = {sum_flex:.6f} | non-flexible MW = {sum_nonflex:.6f}")
        except Exception:
            pass

    # Final debug summary for baseline flexible loads
    try:
        flex_totals = [sum(flexible_time_synchronized_loads_P[t].values()) for t in time_steps]
        if len(flex_totals) > 0:
            print(f"[DEBUG FLEX BUILD] Baseline flexible load totals across horizon: min={min(flex_totals):.6f} MW max={max(flex_totals):.6f} MW")
        else:
            print("[DEBUG FLEX BUILD] No entries in baseline flexible loads (flexible_time_synchronized_loads_P is empty).")
    except Exception as _e_dbg:
        print(f"[DEBUG FLEX BUILD] Failed computing flex totals summary: {_e_dbg}")

    # Reassign to globals in case references changed
    globals()['flexible_time_synchronized_loads_P'] = flexible_time_synchronized_loads_P

    # ------------------------------------------------------------------
    # CORRECT PER-TIMESTEP VARIABLE CREATION (replaces earlier broken single-t block)
    # ------------------------------------------------------------------
    for t in time_steps:
        # PV availability (MW) this timestep per bus
        pv_bus_limits_t = {bus: float(base_pv_bus_limits.get(bus, 0.0)) * float(const_pv[t]) for bus in pv_buses}
        if len(pv_buses) > 0:
            pv_gen_vars[t] = model.addVars(pv_buses, lb=0, ub=pv_bus_limits_t, name=f'pv_gen_{t}')
            curtailment_vars[t] = model.addVars(pv_buses, lb=0, ub=pv_bus_limits_t, name=f'curtailment_{t}')
            for bus in pv_buses:
                avail = pv_bus_limits_t[bus]
                model.addConstr(curtailment_vars[t][bus] == avail - pv_gen_vars[t][bus], name=f'curtailment_constraint_{t}_{bus}')
            pv_avail_sum_by_t[t] = float(sum(pv_bus_limits_t.values()))

        if t == 0:
            print(f"[INIT VARS] Created PV/BESS/ext-grid/flex/HP vars for first timestep (pv_buses={pv_buses}, bess_buses={bess_buses})")

        # BESS variables (charge/discharge/energy trajectory)
        if len(bess_buses) > 0:
            bess_charge_vars[t] = model.addVars(bess_buses, lb=0, ub=base_bess_bus_limits, name=f'bess_charge_{t}')
            bess_discharge_vars[t] = model.addVars(bess_buses, lb=0, ub=base_bess_bus_limits, name=f'bess_discharge_{t}')
            bess_energy_vars[t] = model.addVars(bess_buses, lb=0, ub=bess_capacity_mwh, name=f'bess_energy_{t}')
            if t == time_steps[0]:
                for bus in bess_buses:
                    model.addConstr(bess_energy_vars[t][bus] == bess_initial_soc * bess_capacity_mwh, name=f'bess_energy_initial_{t}_{bus}')
            else:
                for bus in bess_buses:
                    prev_t = t - 1
                    if prev_t in bess_energy_vars and bus in bess_energy_vars[prev_t]:
                        model.addConstr(
                            bess_energy_vars[t][bus] == bess_energy_vars[prev_t][bus] + bess_charge_vars[t][bus] * bess_eff - bess_discharge_vars[t][bus] / bess_eff,
                            name=f'bess_energy_update_{t}_{bus}'
                        )
                    else:
                        # Fallback to initial SOC if previous missing (should not occur once fixed)
                        model.addConstr(
                            bess_energy_vars[t][bus] == bess_initial_soc * bess_capacity_mwh,
                            name=f'bess_energy_update_fallback_{t}_{bus}'
                        )
                if t == time_steps[-1]:
                    for bus in bess_buses:
                        # Enforce cyclical SOC
                        model.addConstr(bess_energy_vars[t][bus] == bess_energy_vars[time_steps[0]][bus], name=f'bess_energy_cyclical_{t}_{bus}')

        # External grid (import/export) per timestep
        ext_grid_import_P_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_P_{t}')
        ext_grid_import_Q_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_Q_{t}')
        ext_grid_export_P_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_P_{t}')
        ext_grid_export_Q_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_Q_{t}')

        # Flexible load decision vars
        flexible_load_P_vars[t] = model.addVars(flexible_load_buses, lb=0, name=f'flexible_load_P_{t}') if len(flexible_load_buses) > 0 else {}
        flexible_load_Q_vars[t] = model.addVars(flexible_load_buses, name=f'flexible_load_Q_{t}') if len(flexible_load_buses) > 0 else {}
        shed_vars[t] = model.addVars(flexible_load_buses, lb=0, name=f'shed_{t}') if len(flexible_load_buses) > 0 else {}

        # Heat pump decision vars
        if len(hp_load_buses) > 0:
            p_hp_vars[t] = model.addVars(hp_load_buses, lb=0, ub=HP_PMAX_MW, name=f'p_hp_{t}')
            if t == time_steps[0]:
                print(f"  Created p_hp_vars for buses: {hp_load_buses}")

    # ------------------------------------------------------------------
    # Aggregated flexible connection capacity buy-back variable (ycap)
    # ------------------------------------------------------------------
    if len(flexible_load_buses) > 0:
        total_conn_cap_MW = 11.0/1000.0 * len(flexible_load_buses)  # 11 kW per connection -> MW
        try:
            ycap_var = model.addVar(lb=0.0, ub=total_conn_cap_MW, name="ycap")
            print(f"Added aggregated connection capacity variable ycap with upper bound {total_conn_cap_MW:.4f} MW")
        except Exception as _e:
            print(f"Warning: failed to add ycap variable: {_e}")
            ycap_var = None
    else:
        ycap_var = None
                            
       
    
    non_slack_buses = [bus for bus in net.bus.index if bus != slack_bus_index]

    
    V_vars = model.addVars(time_steps, net.bus.index, name="V")
    V_reduced_vars = model.addVars(time_steps, non_slack_buses, name="V_reduced")
    # Define external grid net variables once (indexed by time)
    ext_grid_P_net = model.addVars(time_steps, lb=-GRB.INFINITY, name="P_net")
    ext_grid_Q_net = model.addVars(time_steps, lb=-GRB.INFINITY, name="Q_net")
    # Set slack bus voltage to 1.0 p.u. at all time steps
    for t in time_steps:
        model.addConstr(V_vars[t, slack_bus_index] == 1.0, name=f"slack_voltage_fixed_{t}")

    P_branch_vars = model.addVars(time_steps, net.line.index, lb=-GRB.INFINITY, name="P_branch")
    Q_branch_vars = model.addVars(time_steps, net.line.index, lb=-GRB.INFINITY, name="Q_branch")
    S_branch_vars = model.addVars(time_steps, net.line.index, lb=0, name="S_branch")

    P_trafo_vars = model.addVars(time_steps, net.trafo.index, lb=-GRB.INFINITY, name="P_trafo")
    Q_trafo_vars = model.addVars(time_steps, net.trafo.index, lb=-GRB.INFINITY, name="Q_trafo")
    #S_trafo_vars = model.addVars(time_steps, net.line.index, lb=0, name="S_trafo")  

    #Transformer loading percentage
    transformer_loading_perc_vars = model.addVars(time_steps, net.trafo.index, lb=0, name="Trafo_loading_percent")
    #Line_loading_vars = model.addVars(time_steps, net.line.index, name="Line_loading")
    # Initialize as a structured dictionary of linear expressions
    Line_loading_expr = {}
    # Define expressions for each time step and line
    for t in time_steps:
        for line_idx in net.line.index:
            Line_loading_expr[t, line_idx] = gp.LinExpr()  # Properly initialize each entry
    S_branch_approx_expr = {}
    for t in time_steps:
        for line_idx in net.line.index:
            S_branch_approx_expr[t, line_idx] = gp.LinExpr()



    # Accumulated power at each bus
    P_accumulated_vars = model.addVars(time_steps, net.bus.index, lb=-GRB.INFINITY, name="P_accumulated")
    Q_accumulated_vars = model.addVars(time_steps, net.bus.index, lb=-GRB.INFINITY, name="Q_accumulated")

    # (Removed obsolete safeguard for late creation of flexible load vars after loop repair)

    # Container for per-timestep proportional scaling factor (0..1)
    scale_frac_vars = {}

    # Add power balance and load flow constraints for each time step
    for t in time_steps:
        # ------------------------------------------------------------------
        # Flexible load modeling (reworked):
        #   Let f_t in [0,1] be proportional served fraction of baseline.
        #   For each flexible bus b: P_tb = baseline_tb * f_t
        #   Aggregate relation: f_t * sum_baseline_t + ycap_var + shed_agg_t == sum_baseline_t
        #   (ycap_var absorbs capacity buy-back first; shed_agg_t captures extra shedding beyond buy-back).
        #   This ensures per-bus load stays at baseline when ycap_var = shed_agg_t = 0 and only scales uniformly when curtailment needed.
        # ------------------------------------------------------------------
        if len(flexible_load_buses) > 0:
            # Single aggregate shedding variable (tupledict with one key to reuse existing cost logic)
            shed_t = model.addVars([0], lb=0.0, name=f"shed_{t}")  # shed_t[0] represents aggregate shedding MW
            shed_vars[t] = shed_t

            # Served fraction variable
            scale_frac_vars[t] = model.addVar(lb=0.0, ub=1.0, name=f"flex_scale_{t}")

            # Precompute baseline sums and enforce proportional allocation
            sum_baseline_t = float(sum(float(flexible_time_synchronized_loads_P[t].get(b, 0.0)) for b in flexible_load_buses))
            # Avoid division by zero if baseline is zero (rare): fix scale=1 and shedding=0
            if sum_baseline_t <= 1e-9:
                model.addConstr(scale_frac_vars[t] == 1.0, name=f"flex_scale_zero_baseline_{t}")
                model.addConstr(shed_t[0] == 0.0, name=f"flex_shed_zero_baseline_{t}")
            for bus in flexible_load_buses:
                baseline_tb = float(flexible_time_synchronized_loads_P[t].get(bus, 0.0))
                # Proportional served load
                model.addConstr(
                    flexible_load_P_vars[t][bus] == baseline_tb * scale_frac_vars[t],
                    name=f"flex_prop_t{t}_b{bus}"
                )
                # Reactive set to zero (can be extended later)
                model.addConstr(flexible_load_Q_vars[t][bus] == 0.0, name=f"flex_Q_pf_t{t}_b{bus}")

            # Aggregate balance: served + ycap + shed == baseline
            if sum_baseline_t > 1e-9:
                model.addConstr(
                    scale_frac_vars[t] * sum_baseline_t + (ycap_var if ('ycap_var' in locals() and ycap_var is not None) else 0.0) + shed_t[0]
                    == sum_baseline_t,
                    name=f"flex_agg_balance_t{t}"
                )

            if bool(DEBUG_PRINT_FLEX_DIAGNOSTICS):
                globals().setdefault('diag_sum_flex', {})[t] = scale_frac_vars[t] * sum_baseline_t
                globals().setdefault('diag_cap_rhs', {})[t] = sum_baseline_t  # for reference; actual curtail via (1-scale)*baseline
                    


        # Power injection vector P
        P_injected = {bus: gp.LinExpr() for bus in net.bus.index}
        Q_injected = {bus: gp.LinExpr() for bus in net.bus.index}

        for i, bus in enumerate(net.bus.index):
            if bus in net.load.bus.values:
                if bus in flexible_load_buses:
                    # Use the flexible load variable for controllable loads
                    P_injected[bus] -= flexible_load_P_vars[t][bus]
                    Q_injected[bus] -= flexible_load_Q_vars[t][bus]

                if bus in non_flexible_load_buses:
                    # For non-flexible loads, use the time-synchronized load
                    P_injected[bus] -= non_flexible_time_synchronized_loads_P[t].get(bus, 0.0)
                    # non_flexible_time_synchronized_loads_Q already computed using household PF (MVar)
                    Q_injected[bus] -= non_flexible_time_synchronized_loads_Q[t].get(bus, 0.0)

                if bus in hp_load_buses and t in p_hp_vars and bus in p_hp_vars[t]:
                    # For heat pump loads, use the heat pump variable
                    P_injected[bus] -= p_hp_vars[t][bus]
                    # Add reactive consumption for heat pump using its PF (inductive)
                    # p_hp_vars is MW (since other P are in MW), convert to MVar via qfactor_heatpump capacitive
                    Q_injected[bus] += p_hp_vars[t][bus] * qfactor_heatpump

            if len(pv_buses) > 0 and bus in pv_buses and t in pv_gen_vars and bus in pv_gen_vars[t]:
                # Only add PV generation if the bus has PV variable created
                P_injected[bus] += pv_gen_vars[t][bus]

            if len(bess_buses) > 0 and bus in bess_buses:
                # BESS charging is negative injection, discharging is positive injection
                if t in bess_charge_vars and bus in bess_charge_vars[t]:
                    P_injected[bus] -= bess_charge_vars[t][bus]
                if t in bess_discharge_vars and bus in bess_discharge_vars[t]:
                    P_injected[bus] += bess_discharge_vars[t][bus]
                # Assume BESS operates at unity power factor (no reactive power)

        # Add aggregated baseline intercept injection at a representative BESS bus (assumption: first bus)
        if ENABLE_RT_POLICIES and len(bess_buses) > 0 and t in p0_dis_vars and t in p0_ch_vars:
            rep_bus = bess_buses[0]
            if rep_bus in P_injected:
                P_injected[rep_bus] += (p0_dis_vars[t] - p0_ch_vars[t])

        model.update()

        #for bus in net.bus.index:
            #print(f"Time step {t}, Bus {bus}: Power injected (MW) = {P_injected[bus]}")

        # Convert P_injected to per unit
        P_pu = {bus: P_injected[bus] / net.sn_mva for bus in net.bus.index}
        Q_pu = {bus: Q_injected[bus] / net.sn_mva for bus in net.bus.index}

        # Compute the reduced impedance matrices
        Zbus_reduced = np.linalg.inv(Ybus_reduced)
        R = np.real(Zbus_reduced)
        X = np.imag(Zbus_reduced)

        # Build per-bus active/reactive power uncertainty for DRCC tightening using PV/HP info
        sigmaP_MW_by_bus = None
        sigmaQ_MVar_by_bus = None
        sigmaP_pu_vec = None
        if ENABLE_DRCC_NETWORK_TIGHTENING and (k_epsilon is not None):
            const_pv_std_arr = globals().get('const_pv_std', np.zeros(len(time_steps)))
            T_amb_std_arr = globals().get('T_amb_std', np.zeros(len(time_steps)))
            # Combined HP std per timestep (MW): includes daily-average and HDD variance
            try:
                sigma_Tavg_by_day = globals().get('sigma_Tavg_by_day', {})
                Var_HDD_by_t = globals().get('Var_HDD_by_t', {})
            except Exception:
                sigma_Tavg_by_day = {}
                Var_HDD_by_t = {}
            Pmax_HP = HP_DRCC_PMAX
            bTav_loc = HP_DRCC_BTAV
            bHDD_loc = HP_DRCC_BHDD
            sigmaP_MW_by_bus = {bus: 0.0 for bus in net.bus.index}
            sigmaQ_MVar_by_bus = {bus: 0.0 for bus in net.bus.index}
            # PV deviations per PV bus
            for bus in pv_buses:
                sigmaP_MW_by_bus[bus] += float(const_pv_std_arr[t]) * float(base_pv_bus_limits.get(bus, 0.0))
                # PV assumed unity PF here => no reactive deviation contribution
            # HP deviations spread across HP load buses
            if len(hp_load_buses) > 0:
                # Per-timestep HP std combining T_avg and HDD variance
                try:
                    dt_local = time_index[t]
                    sigma_Tavg_d = float(sigma_Tavg_by_day.get(dt_local.date(), 0.0))
                    var_HDD_t = float(Var_HDD_by_t.get(t, 0.0))
                    sigma_HP_temp = float(np.sqrt(max(0.0, (Pmax_HP*bTav_loc*sigma_Tavg_d)**2 + (Pmax_HP*bHDD_loc)**2 * var_HDD_t)))
                except Exception:
                    sigma_HP_temp = abs(Pmax_HP * bTav_loc) * float(T_amb_std_arr[t])
                # Add residual in quadrature if enabled
                if bool(HP_INCLUDE_RESIDUAL):
                    sigma_hp_resid = float(HP_PRED_PMAX * HP_RESIDUAL_SIGMA_NORM)
                    sigma_HP_t = float(np.sqrt(max(0.0, sigma_HP_temp**2 + sigma_hp_resid**2)))
                else:
                    sigma_HP_t = sigma_HP_temp
                # Distribute equally across HP buses
                split = sigma_HP_t / max(1, len(hp_load_buses))
                per_bus_hp_sigma = split
                for bus in hp_load_buses:
                    sigmaP_MW_by_bus[bus] += per_bus_hp_sigma
                    # Map HP P sigma to Q sigma using fixed PF (inductive): Q = P * tan(arccos(pf_HP))
                    sigmaQ_MVar_by_bus[bus] += per_bus_hp_sigma * qfactor_heatpump
            # Per-unit vector aligned to non_slack_buses order
            sigmaP_pu_vec = np.array([sigmaP_MW_by_bus[bus] / net.sn_mva for bus in non_slack_buses], dtype=float)

        # Define voltage magnitude constraints using correct indexing
        for i, bus in enumerate(non_slack_buses):
            model.addConstr(
                V_reduced_vars[t, bus] == 1 +
                 2* (gp.quicksum(R[i, j] * P_pu[non_slack_buses[j]] for j in range(len(non_slack_buses))) +
                gp.quicksum(X[i, j] * Q_pu[non_slack_buses[j]] for j in range(len(non_slack_buses)))),
                name=f"voltage_magnitude_{t}_{bus}"
        )

        # Map V_reduced_vars to V_vars for non-slack buses
        for i, bus in enumerate(non_slack_buses):
                model.addConstr(V_vars[t, bus] == V_reduced_vars[t, bus], name=f"voltage_assignment_{t}_{bus}")

                # Base voltage band (squared magnitude) used before
                base_v_min = 0.70**2
                base_v_max = 1.50**2

                # Compute tightened voltage band if DRCC tightening enabled; else fall back to base
                if ENABLE_DRCC_NETWORK_TIGHTENING and DRCC_TIGHTEN_VOLTAGES and (k_epsilon is not None) and (sigmaP_pu_vec is not None):
                    R_row = np.array([R[i, j] for j in range(len(non_slack_buses))], dtype=float)
                    X_row = np.array([X[i, j] for j in range(len(non_slack_buses))], dtype=float)
                    # Split into PV P and HP P/Q contributions using earlier sigma maps if present
                    try:
                        rho_pv = float(PV_STD_CORRELATION)
                    except Exception:
                        rho_pv = 0.0
                    rho_pv = max(0.0, min(1.0, rho_pv))

                    # Build per-unit vectors per source if available, else fallback to combined RSS
                    if 'sigmaP_MW_by_bus' in locals():
                        # Recompute per-source per-unit arrays
                        const_pv_std_arr = globals().get('const_pv_std', np.zeros(len(time_steps)))
                        T_amb_std_arr = globals().get('T_amb_std', np.zeros(len(time_steps)))
                        sigma_Tavg_by_day = globals().get('sigma_Tavg_by_day', {})
                        Var_HDD_by_t = globals().get('Var_HDD_by_t', {})
                        Pmax_HP = HP_DRCC_PMAX; bTav_loc = HP_DRCC_BTAV; bHDD_loc = HP_DRCC_BHDD
                        sigmaP_PV_pu = np.zeros(len(non_slack_buses))
                        sigmaP_HP_pu = np.zeros(len(non_slack_buses))
                        sigmaQ_HP_pu = np.zeros(len(non_slack_buses))
                        # PV per-bus
                        for j, b in enumerate(non_slack_buses):
                            if b in pv_buses:
                                sigmaP_PV_pu[j] = (float(const_pv_std_arr[t]) * float(base_pv_bus_limits.get(b, 0.0))) / net.sn_mva
                        # HP per-bus (even spread)
                        if len(hp_load_buses) > 0:
                            try:
                                dt_local = time_index[t]
                                sigma_Tavg_d = float(sigma_Tavg_by_day.get(dt_local.date(), 0.0))
                                var_HDD_t = float(Var_HDD_by_t.get(t, 0.0))
                                sigma_HP_temp = float(np.sqrt(max(0.0, (Pmax_HP*bTav_loc*sigma_Tavg_d)**2 + (Pmax_HP*bHDD_loc)**2 * var_HDD_t)))
                            except Exception:
                                sigma_HP_temp = abs(Pmax_HP * bTav_loc) * float(T_amb_std_arr[t])
                            sigma_hp_resid = float(HP_PRED_PMAX * HP_RESIDUAL_SIGMA_NORM) if bool(HP_INCLUDE_RESIDUAL) else 0.0
                            sigma_HP_t = float(np.sqrt(max(0.0, sigma_HP_temp**2 + sigma_hp_resid**2)))
                            per_bus_hp_sigma = sigma_HP_t / max(1, len(hp_load_buses))
                            for j, b in enumerate(non_slack_buses):
                                if b in hp_load_buses:
                                    sigmaP_HP_pu[j] = per_bus_hp_sigma / net.sn_mva
                                    sigmaQ_HP_pu[j] = (per_bus_hp_sigma * qfactor_heatpump) / net.sn_mva
                        # PV contribution (equicorrelated aggregation)
                        wPV = R_row * sigmaP_PV_pu
                        var_PV = (1.0 - rho_pv) * float(np.sum(wPV**2)) + rho_pv * float(np.sum(wPV))**2
                        # HP P contribution
                        wHP_P = R_row * sigmaP_HP_pu
                        var_HP_P = float(np.sum(wHP_P))**2 if HP_FULLY_CORRELATED else float(np.sum(wHP_P**2))
                        # HP Q contribution
                        wHP_Q = X_row * sigmaQ_HP_pu
                        var_HP_Q = float(np.sum(wHP_Q))**2 if HP_FULLY_CORRELATED else float(np.sum(wHP_Q**2))
                        delta_v = float(2.0 * k_epsilon * np.sqrt(max(0.0, var_PV + var_HP_P + var_HP_Q)))
                    else:
                        # Fallback to the earlier combined RSS behavior
                        if 'sigmaQ_MVar_by_bus' in locals() and sigmaQ_MVar_by_bus is not None:
                            sigmaQ_pu_vec = np.array([sigmaQ_MVar_by_bus.get(b, 0.0) / net.sn_mva for b in non_slack_buses], dtype=float)
                        else:
                            sigmaQ_pu_vec = np.zeros_like(sigmaP_pu_vec)
                        delta_v = float(2.0 * k_epsilon * np.sqrt(np.sum((R_row * sigmaP_pu_vec) ** 2) + np.sum((X_row * sigmaQ_pu_vec) ** 2)))
                    tight_v_min = max(0.0, base_v_min + delta_v)
                    tight_v_max = max(0.0, base_v_max - delta_v)
                else:
                    tight_v_min = base_v_min
                    tight_v_max = base_v_max

                # Always enforce base band (unless explicitly disabled); apply tightening only to the limit values
                if ENFORCE_BASE_VOLT_LIMITS:
                    vmin_enf = tight_v_min if (ENABLE_DRCC_NETWORK_TIGHTENING and DRCC_TIGHTEN_VOLTAGES) else base_v_min
                    vmax_enf = tight_v_max if (ENABLE_DRCC_NETWORK_TIGHTENING and DRCC_TIGHTEN_VOLTAGES) else base_v_max
                    model.addConstr(V_vars[t, bus] >= vmin_enf, name=f"voltage_min_{t}_{bus}")
                    model.addConstr(V_vars[t, bus] <= vmax_enf, name=f"voltage_max_{t}_{bus}")
        

        # External grid power balance at slack bus
        model.addConstr(ext_grid_P_net[t] == ext_grid_import_P_vars[t] - ext_grid_export_P_vars[t])
        model.addConstr(ext_grid_Q_net[t] == ext_grid_import_Q_vars[t] - ext_grid_export_Q_vars[t])

        # Accumulate power for each bus (excluding slack)
        sorted_buses = sorted(net.bus.index, key=lambda bus: len(downstream_map[bus]))  # Sort from leaves to root
        #print(f"Sorted Buses: {sorted_buses}")

        for bus in sorted_buses:
            if bus != slack_bus_index:
                #print(f"Bus {bus}: P_accumulated = P_injected[{bus}] + sum of downstream buses {downstream_map[bus]}")
                # Ensure it starts with its own injection
                model.addConstr(
                    P_accumulated_vars[t, bus] == P_injected[bus] + 
                    gp.quicksum(P_injected[child_bus] for child_bus in downstream_map[bus]),
                    name=f"P_accumulated_{t}_{bus}"
                )
                model.addConstr(
                    Q_accumulated_vars[t, bus] == Q_injected[bus] + 
                    gp.quicksum(Q_injected[child_bus] for child_bus in downstream_map[bus]),
                    name=f"Q_accumulated_{t}_{bus}"
                )


    #Line power flow and loading constraints (with the corrected expression)
    for t in time_steps:

        for line in net.line.itertuples():
            line_idx = line.Index  # Extract correct index
            from_bus = line.from_bus
            to_bus = line.to_bus

            # Compute Sending-End Power
            model.addConstr(
                P_branch_vars[t, line_idx] == P_accumulated_vars[t, to_bus],
                name=f"P_send_calc_{line_idx}"
            )

            model.addConstr(
                Q_branch_vars[t, line_idx] == Q_accumulated_vars[t, to_bus],
                name=f"Q_send_calc_{line_idx}"
            )

            S_rated_line = np.sqrt(3) * line.max_i_ka * net.bus.at[from_bus, 'vn_kv']
            if ENABLE_DRCC_NETWORK_TIGHTENING and DRCC_TIGHTEN_LINES and (k_epsilon is not None):
                # Correlation-aware aggregation over downstream set
                const_pv_std_arr = globals().get('const_pv_std', np.zeros(len(time_steps)))
                T_amb_std_arr = globals().get('T_amb_std', np.zeros(len(time_steps)))
                sigma_Tavg_by_day = globals().get('sigma_Tavg_by_day', {})
                Var_HDD_by_t = globals().get('Var_HDD_by_t', {})
                Pmax_HP = HP_DRCC_PMAX; bTav_loc = HP_DRCC_BTAV; bHDD_loc = HP_DRCC_BHDD
                rho_pv = max(0.0, min(1.0, float(PV_STD_CORRELATION))) if 'PV_STD_CORRELATION' in globals() else 0.0
                downstream_set = set(downstream_map[to_bus]) | {to_bus}
                # Build per-bus sigmas
                sigmaP_PV = [float(const_pv_std_arr[t]) * float(base_pv_bus_limits.get(b, 0.0)) if b in pv_buses else 0.0 for b in downstream_set]
                if len(hp_load_buses) > 0:
                    try:
                        dt_local = time_index[t]
                        sigma_Tavg_d = float(sigma_Tavg_by_day.get(dt_local.date(), 0.0))
                        var_HDD_t = float(Var_HDD_by_t.get(t, 0.0))
                        sigma_HP_temp = float(np.sqrt(max(0.0, (Pmax_HP*bTav_loc*sigma_Tavg_d)**2 + (Pmax_HP*bHDD_loc)**2 * var_HDD_t)))
                    except Exception:
                        sigma_HP_temp = abs(Pmax_HP * bTav_loc) * float(T_amb_std_arr[t])
                    sigma_hp_resid = float(HP_PRED_PMAX * HP_RESIDUAL_SIGMA_NORM) if bool(HP_INCLUDE_RESIDUAL) else 0.0
                    sigma_HP_t = float(np.sqrt(max(0.0, sigma_HP_temp**2 + sigma_hp_resid**2)))
                    per_bus_hp_sigma = sigma_HP_t / max(1, len(hp_load_buses))
                    sigmaP_HP = [per_bus_hp_sigma if b in hp_load_buses else 0.0 for b in downstream_set]
                    sigmaQ_HP = [per_bus_hp_sigma * qfactor_heatpump if b in hp_load_buses else 0.0 for b in downstream_set]
                else:
                    sigmaP_HP = [0.0 for _ in downstream_set]
                    sigmaQ_HP = [0.0 for _ in downstream_set]

                sumPV = float(np.sum(sigmaP_PV)); sumsqPV = float(np.sum(np.array(sigmaP_PV)**2))
                varPV = (1.0 - rho_pv) * sumsqPV + rho_pv * (sumPV ** 2)
                stdPV = float(np.sqrt(max(0.0, varPV)))
                if HP_FULLY_CORRELATED:
                    stdHP_P = float(np.sum(sigmaP_HP)); stdHP_Q = float(np.sum(sigmaQ_HP))
                else:
                    stdHP_P = float(np.sqrt(np.sum(np.array(sigmaP_HP)**2)))
                    stdHP_Q = float(np.sqrt(np.sum(np.array(sigmaQ_HP)**2)))

                sigmaP_branch = float(np.sqrt(stdPV**2 + stdHP_P**2))
                sigmaQ_branch = float(stdHP_Q)
                sigmaS_branch = float(np.sqrt(sigmaP_branch**2 + sigmaQ_branch**2))
                S_branch_limit = 0.8 * S_rated_line - k_epsilon * sigmaS_branch
                S_branch_limit = max(0.0, S_branch_limit)
            else:
                S_branch_limit = 0.8 * S_rated_line

            # Always enforce base thermal limit unless debugging flag disables it
            if ENFORCE_BASE_LINE_LIMITS:
                # If tightening inactive, S_branch_limit already equals base; if active, it's tightened.
                model.addQConstr(
                    P_branch_vars[t, line_idx]*P_branch_vars[t, line_idx] +
                    Q_branch_vars[t, line_idx]*Q_branch_vars[t, line_idx]
                    <= (S_branch_limit**2),
                    name=f"S_branch_limit_{t}_{line_idx}"
                )

            # Define line rating based on voltage and current limits

            #model.addConstr(S_branch_vars[t, line_idx] <= (0.8*S_rated_line)-tight_line_limit, name=f"S_branch_limit_{t}_{line_idx}")

            #model.addConstr(Line_loading_vars[t, line_idx] == (S_branch_vars[t, line_idx] / S_rated_line) * 100, name=f"line_loading_{t}_{line_idx}")


        # Transformer loading constraints
        for trafo in net.trafo.itertuples():
            trafo_idx = trafo.Index
            lv_bus = trafo.lv_bus
            hv_bus = trafo.hv_bus

            # Transformer HV-side power flow
            model.addConstr(P_trafo_vars[t, trafo_idx] == P_accumulated_vars[t, lv_bus])
            model.addConstr(Q_trafo_vars[t, trafo_idx] == Q_accumulated_vars[t, lv_bus])

            # Compute transformer loading percentage
            S_rated = net.trafo.sn_mva.iloc[trafo_idx]
        
            if ENABLE_DRCC_NETWORK_TIGHTENING and DRCC_TIGHTEN_TRAFO and (k_epsilon is not None):
                # Correlation-aware aggregation over LV downstream set
                const_pv_std_arr = globals().get('const_pv_std', np.zeros(len(time_steps)))
                T_amb_std_arr = globals().get('T_amb_std', np.zeros(len(time_steps)))
                sigma_Tavg_by_day = globals().get('sigma_Tavg_by_day', {})
                Var_HDD_by_t = globals().get('Var_HDD_by_t', {})
                Pmax_HP = HP_DRCC_PMAX; bTav_loc = HP_DRCC_BTAV; bHDD_loc = HP_DRCC_BHDD
                rho_pv = max(0.0, min(1.0, float(PV_STD_CORRELATION))) if 'PV_STD_CORRELATION' in globals() else 0.0
                downstream_set = set(downstream_map[lv_bus]) | {lv_bus}
                sigmaP_PV = [float(const_pv_std_arr[t]) * float(base_pv_bus_limits.get(b, 0.0)) if b in pv_buses else 0.0 for b in downstream_set]
                if len(hp_load_buses) > 0:
                    try:
                        dt_local = time_index[t]
                        sigma_Tavg_d = float(sigma_Tavg_by_day.get(dt_local.date(), 0.0))
                        var_HDD_t = float(Var_HDD_by_t.get(t, 0.0))
                        sigma_HP_temp = float(np.sqrt(max(0.0, (Pmax_HP*bTav_loc*sigma_Tavg_d)**2 + (Pmax_HP*bHDD_loc)**2 * var_HDD_t)))
                    except Exception:
                        sigma_HP_temp = abs(Pmax_HP * bTav_loc) * float(T_amb_std_arr[t])
                    sigma_hp_resid = float(HP_PRED_PMAX * HP_RESIDUAL_SIGMA_NORM) if bool(HP_INCLUDE_RESIDUAL) else 0.0
                    sigma_HP_t = float(np.sqrt(max(0.0, sigma_HP_temp**2 + sigma_hp_resid**2)))
                    per_bus_hp_sigma = sigma_HP_t / max(1, len(hp_load_buses))
                    sigmaP_HP = [per_bus_hp_sigma if b in hp_load_buses else 0.0 for b in downstream_set]
                    sigmaQ_HP = [per_bus_hp_sigma * qfactor_heatpump if b in hp_load_buses else 0.0 for b in downstream_set]
                else:
                    sigmaP_HP = [0.0 for _ in downstream_set]
                    sigmaQ_HP = [0.0 for _ in downstream_set]

                sumPV = float(np.sum(sigmaP_PV)); sumsqPV = float(np.sum(np.array(sigmaP_PV)**2))
                varPV = (1.0 - rho_pv) * sumsqPV + rho_pv * (sumPV ** 2)
                stdPV = float(np.sqrt(max(0.0, varPV)))
                if HP_FULLY_CORRELATED:
                    stdHP_P = float(np.sum(sigmaP_HP)); stdHP_Q = float(np.sum(sigmaQ_HP))
                else:
                    stdHP_P = float(np.sqrt(np.sum(np.array(sigmaP_HP)**2)))
                    stdHP_Q = float(np.sqrt(np.sum(np.array(sigmaQ_HP)**2)))

                sigmaP_trafo = float(np.sqrt(stdPV**2 + stdHP_P**2))
                sigmaQ_trafo = float(stdHP_Q)
                sigmaS_trafo = float(np.sqrt(sigmaP_trafo**2 + sigmaQ_trafo**2))
                S_limit = 0.8*S_rated - k_epsilon * sigmaS_trafo
                S_limit = max(0.0, S_limit)
            else:
                S_limit = 0.8*S_rated

            # Always enforce base transformer limit unless debugging flag disables it
            if ENFORCE_BASE_TRAFO_LIMITS:
                model.addQConstr(
                    P_trafo_vars[t, trafo_idx]*P_trafo_vars[t, trafo_idx] +
                    Q_trafo_vars[t, trafo_idx]*Q_trafo_vars[t, trafo_idx]
                    <= (S_limit**2),
                    name=f"S_trafo_limit_{t}_{trafo_idx}"
                )

        # Link external grid net power to the negative sum of transformer flows so that
        # imports/exports are consistent with the power leaving/entering the network
        # through transformers. Negative sign because P_trafo is defined as power
        # flowing from the LV side towards the HV/external grid in this formulation.
        model.addConstr(
            ext_grid_P_net[t] == -gp.quicksum(P_trafo_vars[t, i] for i in net.trafo.index),
            name=f"ext_grid_P_balance_{t}"
        )
        model.addConstr(
            ext_grid_Q_net[t] == -gp.quicksum(Q_trafo_vars[t, i] for i in net.trafo.index),
            name=f"ext_grid_Q_balance_{t}"
        )

    print(f"adding coupling constraint between dhn and electrical network...")
    # Include correlation knobs in the flags printout
    try:
        _rho_pv_txt = f"{float(PV_STD_CORRELATION):.2f}"
    except Exception:
        _rho_pv_txt = str(PV_STD_CORRELATION)
    _hp_corr_mode = 'full' if HP_FULLY_CORRELATED else 'rss'
    print(
        f"Flags — RT:{ENABLE_RT_POLICIES} | DRCC budgets:{ENABLE_DRCC_RT_BUDGETS} | DRCC network tighten:{ENABLE_DRCC_NETWORK_TIGHTENING} "
        f"[tighten trafo={DRCC_TIGHTEN_TRAFO}, lines={DRCC_TIGHTEN_LINES}, volts={DRCC_TIGHTEN_VOLTAGES}] "
        f"[base enforce trafo={ENFORCE_BASE_TRAFO_LIMITS}, lines={ENFORCE_BASE_LINE_LIMITS}, volts={ENFORCE_BASE_VOLT_LIMITS}] | "
        f"rho_PV={_rho_pv_txt}, HP_corr={_hp_corr_mode}"
    )
    # Coupling constraint: Electrical power consumption of heat pump
    # Parameters for COP formula (fixed per user)
    DELTA_THETA = 2.1966   # ΔΘ in K (updated per user)
    ETA_C0 = 0.6           # η^{C0} (Carnot efficiency fraction) - fixed

    # Precompute COP per timestep using the supply temperature (target_supply_temp_K)
    # COP_t = ETA_C0 * (theta_s + DELTA_THETA) / (theta_s - theta_amb_t + 2*DELTA_THETA)
    cop_profile = []
    # Coupling: replace removed heating-network coupling with an affine predictor

    print("Using baseline+deviation (normalized by Pmax, with daily T_avg) predictor for HP (deterministic).")

    # Coeffs from learn_affine.py "Baseline + Deviation (normalized by Pmax, with daily T_avg) — Ridge"
    Pmax = HP_PRED_PMAX
    Tbase = HP_PRED_TBASE_C # °C (from the fit)
    b0   = HP_COEFF_B0
    bHDD = HP_COEFF_BHDD
    bpi  = HP_COEFF_BPI
    bTav = HP_COEFF_BTAV
    a1   = HP_COEFF_A1
    a2   = HP_COEFF_A2

    for t in time_steps:
        dt = time_index[t]  # pandas Timestamp
        P_base = baseline_lookup(dt)

        # features
        T_C_t   = float(T_amb[t]) - 273.15
        price_t = float(electricity_price[t])
        HDD_t   = max(0.0, Tbase - T_C_t)
        T_avg_d = daily_mean_temp_for_dt[dt.date()]

        tod   = dt.hour + dt.minute/60.0
        sin24 = np.sin(2*np.pi*tod/24.0)
        cos24 = np.cos(2*np.pi*tod/24.0)

        # deviation fraction y_t, then map to MW and clip to [0, Pmax]
        y_dev   = b0 + bHDD*HDD_t + bpi*price_t + bTav*T_avg_d + a1*sin24 + a2*cos24
        P_t     = P_base + Pmax * y_dev
        P_t     = min(max(P_t, 0.0), Pmax)

        # assign predicted HP power at every HP bus (same value per bus unless you want to split)
        if len(hp_load_buses) > 0:
            for bus in hp_load_buses:
                model.addConstr(p_hp_vars[t][bus] == P_t, name=f'predicted_p_hp_t{t}_bus{bus}')
        if t < 3:
            print(f"  t={t}: P_base={P_base:.4f} MW, y_dev={y_dev:.4f} -> HP={P_t:.4f} MW")

    
    # NOTE: legacy per-period flexible curtailment variables removed; capacity mechanism now governs flexibility via ycap.

    if ENABLE_RT_POLICIES:
        # --- Robust no-scenario affine recourse: budgets, variables, constraints, and proxy costs ---
        alpha_plus = 0.10
        alpha_minus = 0.10
        # Buy-back activation removed: no RT capacity pricing
        cap_price_factor = 0.0
        imb_up_factor = 1.3
        imb_dn_factor = 1.3
        pv_curt_price_factor = 1.0
        bess_rt_price_per_mw = 5.0

        try:
            if len(time_index) >= 2:
                dt_hours = max(1/60.0, (time_index[1] - time_index[0]).total_seconds() / 3600.0)
            else:
                dt_hours = 1.0
        except Exception:
            dt_hours = 1.0

        # Summaries needed for budgets
        for t in time_steps:
            sum_nonflex_by_t[t] = float(sum(non_flexible_time_synchronized_loads_P[t].values())) if t in non_flexible_time_synchronized_loads_P else 0.0
            sum_flex_by_t[t] = float(sum(flexible_time_synchronized_loads_P[t].values())) if t in flexible_time_synchronized_loads_P else 0.0

        # Optional DRCC-derived budgets using moment info (per-time-step std)
        use_drcc = bool(ENABLE_DRCC_RT_BUDGETS)
        if use_drcc:
            # DRCC RT budgets active. If network tightening is OFF we still keep budgets but neutralize amplification (k_e=1.0)
            if bool(ENABLE_DRCC_NETWORK_TIGHTENING):
                # Chance parameter to k factor: k = sqrt((1-eps)/eps)
                try:
                    eps = float(DRCC_EPSILON)
                    k_e = np.sqrt((1.0 - eps) / max(eps, 1e-6))
                except Exception:
                    k_e = np.sqrt((1.0 - 0.05) / 0.05)
            else:
                k_e = 1.0  # Neutral scaling in deterministic (no tightening) mode
            # Retrieve aligned std arrays prepared in the calling scope
            const_pv_std = globals().get('const_pv_std', np.zeros(len(time_steps)))
            T_amb_std = globals().get('T_amb_std', np.zeros(len(time_steps)))
            try:
                print(f"[INFO] RT budget mode={'quantile_drcc' if ENABLE_DRCC_NETWORK_TIGHTENING else 'std_only_k1'} | k_e={k_e:.4f}")
            except Exception:
                pass

        for t in time_steps:
            base_demand_t = float(sum_nonflex_by_t.get(t, 0.0) + sum_flex_by_t.get(t, 0.0) + hp_pred_nominal.get(t, 0.0))
            base_pv_t = float(pv_avail_sum_by_t.get(t, 0.0))

            if use_drcc:
                # PV std in MW: std of normalized factor * installed PV (MW)
                pv_installed_mw = float(base_pv_bus_limits.get(pv_buses[0], 0.0)) if len(pv_buses) == 1 else float(sum(base_pv_bus_limits.get(b, 0.0) for b in pv_buses))
                sigma_pv = float(const_pv_std[t]) * pv_installed_mw
                # HP std from combined model: daily-average and HDD variance
                sigma_Tavg_by_day = globals().get('sigma_Tavg_by_day', {})
                Var_HDD_by_t = globals().get('Var_HDD_by_t', {})
                Pmax_HP = HP_DRCC_PMAX; bTav_loc = HP_DRCC_BTAV; bHDD_loc = HP_DRCC_BHDD
                try:
                    dt_local = time_index[t]
                    sigma_Tavg_d = float(sigma_Tavg_by_day.get(dt_local.date(), 0.0))
                    var_HDD_t = float(Var_HDD_by_t.get(t, 0.0))
                    sigma_hp_temp = float(np.sqrt(max(0.0, (Pmax_HP*bTav_loc*sigma_Tavg_d)**2 + (Pmax_HP*bHDD_loc)**2 * var_HDD_t)))
                except Exception:
                    sigma_hp_temp = abs(Pmax_HP * bTav_loc) * float(T_amb_std[t])
                # Add residual sigma (normalized * Pmax) in quadrature if enabled
                if bool(HP_INCLUDE_RESIDUAL):
                    sigma_hp_resid = float(HP_PRED_PMAX * HP_RESIDUAL_SIGMA_NORM)
                    sigma_hp = float(np.sqrt(max(0.0, sigma_hp_temp**2 + sigma_hp_resid**2)))
                else:
                    sigma_hp = sigma_hp_temp
                # Aggregate std assuming independence
                sigma_tot = np.sqrt(max(0.0, sigma_pv**2 + sigma_hp**2))
                # Set symmetric budgets as k * sigma
                D_plus_max[t] = k_e * sigma_tot
                D_minus_max[t] = k_e * sigma_tot
            else:
                D_plus_max[t] = alpha_plus * (base_demand_t + base_pv_t)
                D_minus_max[t] = alpha_minus * (base_demand_t + base_pv_t)

            # Buy-back activation disabled: remove y_cap and gamma variables
            chi0_vars[t] = model.addVar(lb=0.0, name=f'chi0_{t}')
            chi_minus_vars[t] = model.addVar(lb=0.0, name=f'chi_minus_{t}')
            # Baseline intercept BESS power decomposition (discharge / charge) non-negative
            p0_dis_vars[t] = model.addVar(lb=0.0, ub=total_bess_pmax if 'total_bess_pmax' in locals() else 0.0, name=f'p0_dis_{t}')
            p0_ch_vars[t] = model.addVar(lb=0.0, ub=total_bess_pmax if 'total_bess_pmax' in locals() else 0.0, name=f'p0_ch_{t}')
            # Keep lambda0 for continuity in downstream usage but link to p0_dis - p0_ch
            lambda0_vars[t] = model.addVar(lb=- (total_bess_pmax if 'total_bess_pmax' in locals() else 0.0), ub=(total_bess_pmax if 'total_bess_pmax' in locals() else 0.0), name=f'lambda0_{t}')
            lambda_plus_vars[t] = model.addVar(lb=0.0, name=f'lambda_plus_{t}')
            lambda_minus_vars[t] = model.addVar(lb=0.0, name=f'lambda_minus_{t}')
            rho_plus0_vars[t] = model.addVar(lb=0.0, name=f'rho_plus0_{t}')
            rho_plus1_vars[t] = model.addVar(lb=0.0, name=f'rho_plus1_{t}')
            rho_minus0_vars[t] = model.addVar(lb=0.0, name=f'rho_minus0_{t}')
            rho_minus1_vars[t] = model.addVar(lb=0.0, name=f'rho_minus1_{t}')
            z_dis_vars[t] = model.addVar(lb=0.0, ub=total_bess_pmax if 'total_bess_pmax' in locals() else 0.0, name=f'z_dis_{t}')
            z_ch_vars[t] = model.addVar(lb=0.0, ub=total_bess_pmax if 'total_bess_pmax' in locals() else 0.0, name=f'z_ch_{t}')

            # Link intercept variable to decomposition
            model.addConstr(p0_dis_vars[t] - p0_ch_vars[t] == lambda0_vars[t], name=f'lambda0_link_t{t}')
            # Instantaneous power caps (baseline + deviation cannot exceed physical rating)
            if 'total_bess_pmax' in locals() and total_bess_pmax > 0.0:
                model.addConstr(p0_dis_vars[t] + z_dis_vars[t] <= total_bess_pmax, name=f'bess_cap_dis_t{t}')
                model.addConstr(p0_ch_vars[t] + z_ch_vars[t] <= total_bess_pmax, name=f'bess_cap_ch_t{t}')

            # Robust SoC extreme trajectories (simple affine envelope) only if energy known
            if 'bess_capacity_mwh' in locals() and 'bess_initial_soc' in locals():
                total_bess_energy = float(bess_capacity_mwh) * (len(bess_buses) if 'bess_buses' in locals() else 1)
                if total_bess_energy > 0.0:
                    E_down_vars[t] = model.addVar(lb=0.0, ub=total_bess_energy, name=f'E_down_{t}')
                    E_up_vars[t] = model.addVar(lb=0.0, ub=total_bess_energy, name=f'E_up_{t}')
                    if t == time_steps[0]:
                        model.addConstr(E_down_vars[t] == bess_initial_soc * total_bess_energy, name=f'Edown_init')
                        model.addConstr(E_up_vars[t] == bess_initial_soc * total_bess_energy, name=f'Eup_init')
                    else:
                        # Down path: assume worst-case discharge deviation realized
                        model.addConstr(
                            E_down_vars[t] == E_down_vars[t-1] + (p0_ch_vars[t]*bess_eff - p0_dis_vars[t]/bess_eff - z_dis_vars[t]/bess_eff)*dt_hours,
                            name=f'Edown_dyn_t{t}'
                        )
                        # Up path: assume worst-case charge deviation realized
                        model.addConstr(
                            E_up_vars[t] == E_up_vars[t-1] + (p0_ch_vars[t]*bess_eff - p0_dis_vars[t]/bess_eff + z_ch_vars[t]*bess_eff)*dt_hours,
                            name=f'Eup_dyn_t{t}'
                        )
                    # Feasibility relationship
                    model.addConstr(E_down_vars[t] <= E_up_vars[t], name=f'Eenv_order_t{t}')

            model.addConstr(
                # No buy-back activation: cover deficit with BESS proxies and imbalance proxies only
                lambda_plus_vars[t] * D_plus_max[t]
                + rho_plus0_vars[t] + rho_plus1_vars[t] * D_plus_max[t]
                >= D_plus_max[t],
                name=f'coverage_deficit_t{t}'
            )

            model.addConstr(
                chi0_vars[t] + chi_minus_vars[t] * D_minus_max[t]
                + lambda_minus_vars[t] * D_minus_max[t]
                + rho_minus0_vars[t] + rho_minus1_vars[t] * D_minus_max[t]
                >= D_minus_max[t],
                name=f'coverage_surplus_t{t}'
            )
            model.addConstr(chi0_vars[t] + chi_minus_vars[t] * D_minus_max[t] <= pv_avail_sum_by_t.get(t, 0.0), name=f'pv_curt_leq_avail_t{t}')
            model.addConstr(lambda_plus_vars[t] * D_plus_max[t] <= z_dis_vars[t], name=f'bess_deficit_proxy_t{t}')
            model.addConstr(lambda_minus_vars[t] * D_minus_max[t] <= z_ch_vars[t], name=f'bess_surplus_proxy_t{t}')

        if 'total_bess_pmax' in locals() and total_bess_pmax > 0.0:
            try:
                total_bess_energy = float(bess_capacity_mwh) * len(bess_buses) if 'bess_buses' in locals() else float(bess_capacity_mwh)
            except Exception:
                total_bess_energy = 0.0
            if total_bess_energy > 0.0:
                model.addConstr(gp.quicksum(z_dis_vars[t] * dt_hours for t in time_steps) <= total_bess_energy, name='rt_bess_dis_energy_cap')
                model.addConstr(gp.quicksum(z_ch_vars[t] * dt_hours for t in time_steps) <= total_bess_energy, name='rt_bess_ch_energy_cap')

    # Determine timestep duration in hours for energy-based costs
    try:
        if 'time_index' in globals() and len(time_index) >= 2:
            dt_hours = max(1/60.0, (time_index[1] - time_index[0]).total_seconds() / 3600.0)
        else:
            dt_hours = 1.0
    except Exception:
        dt_hours = 1.0

    # Build objective components (original + robust policy proxies)
    print(f"\nStarting multi-period optimization...")
    print(f"COST-BASED Objective: Minimize total operational cost over {NUM_PERIODS} periods")
    # Convert MW to MWh using dt_hours when multiplying by prices in EUR/MWh
    electricity_cost =  gp.quicksum(electricity_price[t] * (ext_grid_import_P_vars[t] + ext_grid_export_P_vars[t]) * dt_hours for t in time_steps)
    bess_cost = gp.quicksum(bess_cost_per_mwh * (bess_charge_vars[t][bus] + bess_discharge_vars[t][bus]) * dt_hours for bus in bess_buses for t in time_steps) if len(bess_buses) > 0 else 0
    # Baseline intercept throughput cost (only if RT policies active and baseline vars exist)
    if ENABLE_RT_POLICIES:
        baseline_bess_cost = gp.quicksum(c_base_bess * (p0_dis_vars[t] + p0_ch_vars[t]) * dt_hours for t in time_steps if t in p0_dis_vars)
    else:
        baseline_bess_cost = 0
    pv_curtail_cost = gp.quicksum(electricity_price[t] * curtailment_vars[t][bus] * dt_hours for bus in pv_buses for t in time_steps) if len(pv_buses) > 0 else 0
    flex_capacity_cost = C_CAP_EUR_PER_MW * ycap_var
    # Shedding cost: shed_vars[t] is a tupledict (per-bus shedding vars) — must sum its values.
    # Previous code attempted: float * tupledict -> TypeError. We expand explicitly.
    if C_SHED_EUR_PER_MW_H != 0 and len(shed_vars) > 0:
        shed_cost = gp.quicksum(
            C_SHED_EUR_PER_MW_H * v * dt_hours
            for t in time_steps
            for v in (shed_vars[t].values() if isinstance(shed_vars[t], gp.tupledict) else (
                shed_vars[t].values() if hasattr(shed_vars[t], 'values') else []))
        )
    else:
        shed_cost = 0

    # New: first-stage capacity and RT proxy costs for robust policies (only if enabled)
    if ENABLE_RT_POLICIES:
        # No RT capacity cost (buy-back removed)
        cap_cost = 0
        imb_proxy_cost = gp.quicksum(
            (imb_up_factor * electricity_price[t]) * (rho_plus0_vars[t] + rho_plus1_vars[t] * D_plus_max[t]) * dt_hours
            + (imb_dn_factor * electricity_price[t]) * (rho_minus0_vars[t] + rho_minus1_vars[t] * D_minus_max[t]) * dt_hours
            for t in time_steps
        )
        pv_curt_proxy_cost = gp.quicksum((pv_curt_price_factor * electricity_price[t]) * (chi0_vars[t] + chi_minus_vars[t] * D_minus_max[t]) * dt_hours for t in time_steps)
        bess_rt_proxy_cost = gp.quicksum(bess_rt_price_per_mw * (z_dis_vars[t] + z_ch_vars[t]) * dt_hours for t in time_steps)
    else:
        cap_cost = 0
        imb_proxy_cost = 0
        pv_curt_proxy_cost = 0
        bess_rt_proxy_cost = 0

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = electricity_cost + bess_cost + baseline_bess_cost + flex_capacity_cost + pv_curtail_cost + shed_cost
    total_cost += cap_cost + imb_proxy_cost + pv_curt_proxy_cost + bess_rt_proxy_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # -------------------- SOLVER PARAM STRATEGY (pass 1) --------------------
    # Previous code disabled presolve; that often hurts numerics. We allow an env override to keep it off.
    if os.environ.get('CMES_PRESOLVE_OFF', '0') == '1':
        model.setParam('Presolve', 0)
    else:
        model.setParam('Presolve', 2)  # full presolve
    # Light initial numeric stabilization (heavier tweaks only if needed in fallback)
    model.setParam('NumericFocus', int(os.environ.get('CMES_NUMERIC_FOCUS', '0')))  # user can raise to 1..3
    # Allow barrier (default) to try first; no forced Method unless user supplies one
    if os.environ.get('CMES_FORCE_BARRIER', '0') == '1':
        model.setParam('Method', 2)  # 2 = barrier
    # Optional: quiet output
    if os.environ.get('CMES_GUROBI_SILENT', '0') == '1':
        model.setParam('OutputFlag', 0)

    model.update()

    # Optimize the model
    model.optimize()

    # -------------------- NUMERICAL RECOVERY / SECOND PASS --------------------
    # If numerical trouble or ambiguous infeasible/unbounded, attempt a recovery pass with
    # stronger scaling and homogeneous self-dual barrier, producing diagnostic artifacts.
    recovery_attempted = False
    def _write_diagnostics(tag: str):
        try:
            model.write(f'debug_model_{tag}.lp')
        except Exception:
            pass
        try:
            model.write(f'debug_model_{tag}.mps')
        except Exception:
            pass

    troubled_statuses = {getattr(GRB, 'NUMERIC', 12), getattr(GRB, 'INF_OR_UNBD', 4), getattr(GRB, 'UNBOUNDED', 5)}
    if model.status in troubled_statuses and bool(int(os.environ.get('CMES_ENABLE_RECOVERY', '1'))):
        print(f"[RECOVERY] Detected problematic status ({model.status}); launching numerical recovery pass...")
        _write_diagnostics('pass1')
        recovery_attempted = True
        # Strengthen numeric settings
        model.setParam('InfUnbdInfo', 1)
        model.setParam('DualReductions', 0)
        model.setParam('BarHomogeneous', 1)   # homogeneous self-dual barrier
        model.setParam('NumericFocus', 3)
        model.setParam('ScaleFlag', 2)        # more aggressive scaling
        model.setParam('Presolve', 2)
        model.setParam('Method', 2)           # barrier
        model.setParam('Crossover', 0)        # stay in barrier space (reduce instability)
        try:
            model.reset()  # clear previous solve state while keeping model
        except Exception:
            pass
        model.optimize()

    # A third pass if still numeric trouble: escalate BarHomogeneous
    if recovery_attempted and model.status in troubled_statuses:
        print(f"[RECOVERY-2] Still problematic (status {model.status}); escalating homogeneous barrier level 2...")
        _write_diagnostics('pass2')
        try:
            model.setParam('BarHomogeneous', 2)
            model.reset()
        except Exception:
            pass
        model.optimize()

    # If still ambiguous infeasible/unbounded, force dual reductions off & try simplex as last resort
    if recovery_attempted and model.status in troubled_statuses:
        print(f"[RECOVERY-3] Status {model.status} persists; trying primal simplex fallback...")
        _write_diagnostics('pass3')
        try:
            model.setParam('Method', 0)  # primal simplex
            model.setParam('BarHomogeneous', 0)
            model.reset()
            model.optimize()
        except Exception as _e_final:
            print(f"[RECOVERY-3] Fallback attempt raised {_e_final}")

    # If infeasible and IIS debugging enabled, extract IIS details early
    if model.status == GRB.INFEASIBLE and bool(DEBUG_EXTRACT_IIS):
        print("\n[DIAG] Model infeasible — extracting IIS ...")
        try:
            model.computeIIS()
            viol_cons = []
            for c in model.getConstrs():
                if c.IISConstr:
                    viol_cons.append(c.ConstrName)
            viol_quads = []
            for qc in model.getQConstrs():
                if qc.IISQConstr:
                    viol_quads.append(qc.QCName)
            viol_bounds = []
            for v in model.getVars():
                if v.IISLB or v.IISUB:
                    viol_bounds.append(v.VarName)
            print(f"[DIAG] IIS constraints count={len(viol_cons)} quads={len(viol_quads)} bounds={len(viol_bounds)}")
            if len(viol_cons) > 0:
                print("[DIAG] Constraints in IIS (first 50):")
                for name in viol_cons[:50]:
                    print("   -", name)
            if len(viol_bounds) > 0:
                print("[DIAG] Vars with IIS bounds (first 50):")
                    
                for name in viol_bounds[:50]:
                    print("   -", name)
            # Save IIS list to a text file for deeper inspection
            try:
                with open('iis_diagnostics.txt','w') as f:
                    f.write('IIS Constraints:\n')
                    for n in viol_cons: f.write(n+'\n')
                    f.write('\nIIS Quad Constraints:\n')
                    for n in viol_quads: f.write(n+'\n')
                    f.write('\nIIS Variable Bounds:\n')
                    for n in viol_bounds: f.write(n+'\n')
                print('[DIAG] IIS written to iis_diagnostics.txt')
            except Exception as _e:
                print(f"[DIAG] Failed writing IIS file: {_e}")
        except Exception as _e:
            print(f"[DIAG] IIS extraction failed: {_e}")

    # Check if optimization was successful
    if model.status == GRB.OPTIMAL:
        print(f"OPF Optimal Objective Value: {model.ObjVal}")
        #print("\n--- Debugging P_abs and P_branch Values ---\n")
        # Post-solve quick summary (as requested): epsilon, kappa, max transformer and line loading
        # Determine epsilon/kappa display according to tightening semantics
        if ENABLE_DRCC_NETWORK_TIGHTENING:
            try:
                eps_val = float(DRCC_EPSILON) if 'DRCC_EPSILON' in globals() else None
            except Exception:
                eps_val = None
            try:
                if eps_val is not None and 0.0 < eps_val < 1.0:
                    kappa_val = float(np.sqrt((1.0 - eps_val) / eps_val))
                else:
                    kappa_val = None
            except Exception:
                kappa_val = None
            mode_tag = "quantile_drcc"
        else:
            # Neutral deterministic mode: suppress epsilon and force kappa=1
            eps_val = None
            kappa_val = 1.0
            mode_tag = "std_only_k1"

        # Compute maximum transformer loading (%) across all periods/transformers
        max_trafo_loading_pct = None
        try:
            if len(net.trafo.index) > 0:
                max_trafo_loading_pct = 0.0
                for t in time_steps:
                    for trafo in net.trafo.itertuples():
                        idx = trafo.Index
                        s_mva = float(trafo.sn_mva)
                        p = float(P_trafo_vars[t, idx].X)
                        q = float(Q_trafo_vars[t, idx].X)
                        loading_pct = (np.sqrt(p*p + q*q) / max(1e-9, s_mva)) * 100.0
                        if loading_pct > max_trafo_loading_pct:
                            max_trafo_loading_pct = loading_pct
        except Exception:
            max_trafo_loading_pct = None

        # Compute maximum line loading (%) across all periods/lines
        max_line_loading_pct = None
        try:
            if len(net.line.index) > 0:
                max_line_loading_pct = 0.0
                for t in time_steps:
                    for line in net.line.itertuples():
                        idx = line.Index
                        from_bus = int(line.from_bus)
                        vn_kv = float(net.bus.at[from_bus, 'vn_kv'])
                        imax_ka = float(line.max_i_ka)
                        p = float(P_branch_vars[t, idx].X)
                        q = float(Q_branch_vars[t, idx].X)
                        v_pu = float(V_vars[t, from_bus].X)
                        s_mag = np.sqrt(p*p + q*q)
                        denom = (np.sqrt(3.0) * max(1e-6, v_pu) * vn_kv * max(1e-9, imax_ka))
                        loading_pct = (s_mag / denom) * 100.0
                        if loading_pct > max_line_loading_pct:
                            max_line_loading_pct = loading_pct
        except Exception:
            max_line_loading_pct = None

        # Print the summary line before the cost breakdown
        try:
            if ENABLE_DRCC_NETWORK_TIGHTENING:
                eps_text = f"epsilon={eps_val:.4f}" if eps_val is not None else "epsilon=N/A"
                kap_text = f"kappa={kappa_val:.4f}" if kappa_val is not None else "kappa=N/A"
            else:
                eps_text = "epsilon=--"
                kap_text = f"kappa={kappa_val:.4f}"  # will be 1.0
        except Exception:
            eps_text = "epsilon=--" if not ENABLE_DRCC_NETWORK_TIGHTENING else (f"epsilon={eps_val}" if eps_val is not None else "epsilon=N/A")
            kap_text = f"kappa={kappa_val}" if kappa_val is not None else "kappa=N/A"

        try:
            trafo_text = f"max_trafo_loading={max_trafo_loading_pct:.2f}%" if max_trafo_loading_pct is not None else "max_trafo_loading=N/A"
            line_text = f"max_line_loading={max_line_loading_pct:.2f}%" if max_line_loading_pct is not None else "max_line_loading=N/A"
        except Exception:
            trafo_text = f"max_trafo_loading={max_trafo_loading_pct}%" if max_trafo_loading_pct is not None else "max_trafo_loading=N/A"
            line_text = f"max_line_loading={max_line_loading_pct}%" if max_line_loading_pct is not None else "max_line_loading=N/A"

        print(f"{eps_text} | {kap_text} | mode={mode_tag} | {trafo_text} | {line_text}")

        # Extract optimized values for PV generation, external grid power, loads, and theta
        for t in time_steps:
            pv_gen_results[t] = {bus: pv_gen_vars[t][bus].x for bus in pv_buses}
            bess_charge_results[t] = {bus: bess_charge_vars[t][bus].x for bus in bess_buses}
            bess_discharge_results[t] = {bus: bess_discharge_vars[t][bus].x for bus in bess_buses}
            bess_energy_results[t] = {bus: bess_energy_vars[t][bus].x for bus in bess_buses}
            ext_grid_import_P_results[t] = ext_grid_import_P_vars[t].x
            ext_grid_import_Q_results[t] = ext_grid_import_Q_vars[t].x
            ext_grid_export_P_results[t] = ext_grid_export_P_vars[t].x
            ext_grid_export_Q_results[t] = ext_grid_export_Q_vars[t].x
            # Legacy per-period curtailment removed; set to 0 for backward compatibility
            flex_curtail_P_results[t] = 0.0
            V_results[t] = {bus: V_vars[t, bus].x for bus in net.bus.index}

            # Extract load results as **flat dictionaries**
            flexible_load_P_results[t] = {bus: flexible_load_P_vars[t][bus].x for bus in flexible_load_buses}
            flexible_load_Q_results[t] = {bus: flexible_load_Q_vars[t][bus].x for bus in flexible_load_buses}
            non_flexible_load_P_results[t] = {bus: non_flexible_time_synchronized_loads_P[t][bus] for bus in non_flexible_load_buses}
            non_flexible_load_Q_results[t] = {bus: non_flexible_time_synchronized_loads_Q[t][bus] for bus in non_flexible_load_buses}
            # Heat pump load results
            if len(hp_load_buses) > 0:
                p_hp_results[t] = {bus: p_hp_vars[t][bus].x for bus in hp_load_buses}
            else:
                p_hp_results[t] = {}

            transformer_loading_results[t] = {
                trafo_idx: (
                    np.sqrt(
                        P_trafo_vars[t, trafo_idx].x ** 2 + Q_trafo_vars[t, trafo_idx].x ** 2
                    ) / net.trafo.at[trafo_idx, 'sn_mva']
                ) * 100
                for trafo_idx in net.trafo.index
            }

            line_pl_results[t] = {
                line_idx: -1 * P_branch_vars[t, line_idx].x for line_idx in net.line.index
            }
            line_ql_results[t] = {
                line_idx: -1 * Q_branch_vars[t, line_idx].x for line_idx in net.line.index
            }

            line_loading_results[t] = {
                line_idx: (
                    (
                    np.sqrt(P_branch_vars[t, line_idx].x ** 2 + Q_branch_vars[t, line_idx].x ** 2) /
                    (np.sqrt(3) * V_results[t][net.line.at[line_idx, 'from_bus']] * net.bus.at[net.line.at[line_idx, 'from_bus'], 'vn_kv'])
                    ) / net.line.at[line_idx, 'max_i_ka']
                ) * 100
                for line_idx in net.line.index
            }

            line_current_results[t] = {
                line_idx: (
                    np.sqrt(P_branch_vars[t, line_idx].x ** 2 + Q_branch_vars[t, line_idx].x ** 2) /
                    (np.sqrt(3) * V_results[t][net.line.at[line_idx, 'from_bus']] * net.bus.at[net.line.at[line_idx, 'from_bus'], 'vn_kv'])
                )
                for line_idx in net.line.index
            }

            

        # Return results in a structured format
        results = {
            'pv_gen': pv_gen_results,
            'bess_charge': bess_charge_results,
            'bess_discharge': bess_discharge_results,
            'bess_energy': bess_energy_results,
            'flexible_load_p': flexible_load_P_results,
            'flexible_load_q': flexible_load_Q_results,
            'non_flexible_load_p': non_flexible_load_P_results,
            'non_flexible_load_q': non_flexible_load_Q_results,
            'p_hp': p_hp_results,
            'ext_grid_import_p': ext_grid_import_P_results,
            'ext_grid_import_q': ext_grid_import_Q_results,
            'ext_grid_export_p': ext_grid_export_P_results,
            'ext_grid_export_q': ext_grid_export_Q_results,
            'flex_curtail_p': flex_curtail_P_results,
            'voltage': V_results, 
            'line_P': line_pl_results,
            'line_Q': line_ql_results,
            'line_current': line_current_results,
            'line_loading': line_loading_results,
            'transformer_loading': transformer_loading_results,
        }

        # # Save the results to a file
        #     rs.save_optim_results(results, "drcc_results.pkl")

        # Compute and print cost breakdown from the solved variables/results
        try:
            # Electricity cost (uses import+export as in objective)
            electricity_cost_value = sum(electricity_price[t] * (ext_grid_import_P_results.get(t, 0.0) + ext_grid_export_P_results.get(t, 0.0)) * dt_hours for t in time_steps)
        except Exception:
            electricity_cost_value = None

        try:
            if len(bess_buses) > 0:
                bess_cost_value = sum(bess_cost_per_mwh * (sum(bess_charge_results[t].values()) + sum(bess_discharge_results[t].values())) * dt_hours for t in time_steps)
            else:
                bess_cost_value = 0.0
        except Exception:
            bess_cost_value = None

        try:
            flex_curtail_cost_value = 0.0  # legacy metric (now handled by capacity cost)
        except Exception:
            flex_curtail_cost_value = None

        try:
            pv_curtail_cost_value = 0.0
            if len(pv_buses) > 0:
                for t in time_steps:
                    pv_curtail_cost_value += electricity_price[t] * sum(curtailment_vars[t][bus].x for bus in pv_buses) * dt_hours
        except Exception:
            pv_curtail_cost_value = None

        # Compute a base total (without RT proxies) and, if RT is enabled, a grand total including proxies
        base_total_value = None
        grand_total_value = None
        try:
            if None not in (electricity_cost_value, bess_cost_value, flex_curtail_cost_value, pv_curtail_cost_value):
                base_total_value = electricity_cost_value + bess_cost_value + flex_curtail_cost_value + pv_curtail_cost_value
            else:
                base_total_value = None
        except Exception:
            base_total_value = None

        # Derive flexible connection capacity (y_cap) cost and shedding cost
        try:
            capacity_cost_value = C_CAP_EUR_PER_MW * (ycap_var.X if ('ycap_var' in locals() and ycap_var is not None) else 0.0) * dt_hours * len(time_steps)
        except Exception:
            capacity_cost_value = None
        # Aggregate shedding over all buses/time (power MW per period -> energy MWh via dt_hours in cost)
        try:
            total_shed_power_sum = 0.0  # sum of MW over periods (will multiply by dt_hours for energy-based cost)
            for t in time_steps:
                if t in shed_vars and hasattr(shed_vars[t], 'values'):
                    for v in shed_vars[t].values():
                        total_shed_power_sum += getattr(v, 'X', 0.0)
            shed_cost_value = C_SHED_EUR_PER_MW_H * total_shed_power_sum * dt_hours
            avg_shed_mw = total_shed_power_sum / len(time_steps) if len(time_steps) else 0.0
        except Exception:
            shed_cost_value = None
            total_shed_power_sum = None
            avg_shed_mw = None

        print("\nCOST BREAKDOWN:")
        print(f"  electricity_cost = {electricity_cost_value}")
        print(f"  bess_cost = {bess_cost_value}")
        print(f"  flex_capacity_cost (y_cap) = {capacity_cost_value}")
        print(f"  shed_cost = {shed_cost_value}")
        try:
            if 'ycurt_vars' in locals() and ycurt_vars:
                total_curt = sum(v.X for v in ycurt_vars.values())
                avg_curt = total_curt / len(ycurt_vars)
            else:
                total_curt = 0.0
                avg_curt = 0.0
        except Exception:
            total_curt = None
            avg_curt = None
        print(f"  total_curtailment_mw = {total_curt}")
        print(f"  avg_curtailment_mw = {avg_curt}")
        print(f"  total_shed_power_sum_mw_periods = {total_shed_power_sum}")
        print(f"  avg_shed_mw = {avg_shed_mw}")
        print(f"  pv_curtail_cost = {pv_curtail_cost_value}")
        if ENABLE_RT_POLICIES:
            try:
                imb_proxy_cost_value = sum(
                    (imb_up_factor * float(electricity_price[t])) * (rho_plus0_vars[t].x + rho_plus1_vars[t].x * D_plus_max[t]) * dt_hours
                    + (imb_dn_factor * float(electricity_price[t])) * (rho_minus0_vars[t].x + rho_minus1_vars[t].x * D_minus_max[t]) * dt_hours
                    for t in time_steps
                )
            except Exception:
                imb_proxy_cost_value = None
            try:
                pv_curt_proxy_cost_value = sum((pv_curt_price_factor * float(electricity_price[t])) * (chi0_vars[t].x + chi_minus_vars[t].x * D_minus_max[t]) * dt_hours for t in time_steps)
            except Exception:
                pv_curt_proxy_cost_value = None
            try:
                bess_rt_proxy_cost_value = sum(bess_rt_price_per_mw * (z_dis_vars[t].x + z_ch_vars[t].x) * dt_hours for t in time_steps)
            except Exception:
                bess_rt_proxy_cost_value = None
            print(f"  imb_proxy_cost (RT imbalance proxies) = {imb_proxy_cost_value}")
            print(f"  pv_curt_proxy_cost (RT PV curtail proxies) = {pv_curt_proxy_cost_value}")
            print(f"  bess_rt_proxy_cost (RT BESS proxies) = {bess_rt_proxy_cost_value}")
            # If we have a base total, also print a grand total including RT proxies
            try:
                if base_total_value is not None and None not in (imb_proxy_cost_value, pv_curt_proxy_cost_value, bess_rt_proxy_cost_value):
                    grand_total_value = base_total_value + imb_proxy_cost_value + pv_curt_proxy_cost_value + bess_rt_proxy_cost_value
            except Exception:
                grand_total_value = None

        # Recompute base total including new capacity and shedding cost components if available
        try:
            components = [electricity_cost_value, bess_cost_value, pv_curtail_cost_value]
            # Only add if not None
            if capacity_cost_value is not None:
                components.append(capacity_cost_value)
            if shed_cost_value is not None:
                components.append(shed_cost_value)
            # (legacy flex_curtail_cost_value intentionally omitted)
            base_total_recomputed = sum(c for c in components if c is not None)
        except Exception:
            base_total_recomputed = base_total_value  # fallback
        if ENABLE_RT_POLICIES:
            # Recompute grand total including RT proxies based on the updated base (which now includes capacity & shedding)
            try:
                grand_total_recomputed = None
                if base_total_recomputed is not None and \
                   'imb_proxy_cost_value' in locals() and 'pv_curt_proxy_cost_value' in locals() and 'bess_rt_proxy_cost_value' in locals() and \
                   None not in (imb_proxy_cost_value, pv_curt_proxy_cost_value, bess_rt_proxy_cost_value):
                    grand_total_recomputed = base_total_recomputed + imb_proxy_cost_value + pv_curt_proxy_cost_value + bess_rt_proxy_cost_value
            except Exception:
                grand_total_recomputed = None
            print(f"  total_cost_base (no RT proxies) = {base_total_recomputed}")
            print(f"  total_cost_with_rt_proxies = {grand_total_recomputed if grand_total_recomputed is not None else 'N/A'}")
        else:
            print(f"  total_cost (components sum or model.ObjVal) = {base_total_recomputed if base_total_recomputed is not None else model.ObjVal}")
        try:
            print(f"  model.ObjVal = {model.ObjVal}")
        except Exception:
            pass
        # Diagnostics for flexible capacity cap (print first few periods)
        if bool(DEBUG_PRINT_FLEX_DIAGNOSTICS) and 'diag_sum_flex' in globals() and 'diag_cap_rhs' in globals():
            try:
                print("\n[DIAG] Aggregated flexible capacity usage (first periods):")
                for t in sorted(diag_sum_flex.keys())[:MAX_FLEX_DIAG_PERIODS]:
                    rhs_val = (11.0/1000.0 * len(flexible_load_buses)) - (ycap_var.X if ("ycap_var" in locals() and ycap_var is not None) else 0.0)
                    print(f"   t={t} sum_flex≈{diag_sum_flex[t].getValue():.5f} MW  cap_rhs≈{rhs_val:.5f} MW")
                # Provide min/max across horizon
                sum_vals = [diag_sum_flex[t].getValue() for t in diag_sum_flex]
                print(f"[DIAG] sum_flex_min={min(sum_vals):.5f} max={max(sum_vals):.5f}")
            except Exception as _e:
                print(f"[DIAG] Flex diagnostics failed: {_e}")
        print(f"✓ MULTI-PERIOD OPTIMIZATION SUCCESSFUL!")
        print(f"="*80)
        
        
        results_data = {
            'period': [t+1 for t in time_steps],
        }

        # Add timestamp column for exact time alignment in downstream simulators
        try:
            results_data['timestamp'] = [str(pd.to_datetime(time_index[t])) for t in time_steps]
        except Exception:
            pass

        # Add COP profile (per-timestep) to results if available
        try:
            # cop_profile was computed earlier as a list indexed by time_steps
            results_data['cop_t'] = [cop_profile[t] for t in time_steps]
        except Exception:
            # Fallback to ones if cop_profile not available
            results_data['cop_t'] = [1.0 for _ in time_steps]

        # Add all bus voltages from electrical network results (per-bus columns)
        for bus in net.bus.index:
            voltage_col_name = f"bus_{bus}_voltage_pu"
            # results['voltage'] is structured as {t: {bus: V_value}}
            results_data[voltage_col_name] = [results['voltage'][t][bus] for t in time_steps]

        # Add line loading results (percentage) as per-line columns
        for line_idx in net.line.index:
            line_loading_col_name = f"line_{line_idx}_loading_pct"
            results_data[line_loading_col_name] = [results['line_loading'][t][line_idx] for t in time_steps]


        # Add transformer loading results (percentage) so plotting can access them
        # transformer_loading_results: { t: { trafo_idx: loading_pct } }
        if 'transformer_loading' in results:
            for trafo_idx in net.trafo.index:
                trafo_col_name = f"transformer_{trafo_idx}_loading_pct"
                results_data[trafo_col_name] = [results['transformer_loading'][t][trafo_idx] for t in time_steps]
        
    # Add electrical grid import/export data
        results_data['ext_grid_import_mw'] = [results['ext_grid_import_p'][t] for t in time_steps]
        results_data['ext_grid_export_mw'] = [results['ext_grid_export_p'][t] for t in time_steps]
        results_data['net_grid_power_mw'] = [results['ext_grid_import_p'][t] - results['ext_grid_export_p'][t] for t in time_steps]
        # Add transformer signed P (sum of P_trafo; signed as saved in model). Positive means P flowing into trafo (matching P_trafo_vars definitions)
        if 'transformer_loading' in results:
            # Signed P as stored in model (may be negative depending on flow direction)
            results_data['transformer_signed_p_mw'] = [sum(P_trafo_vars[t, trafo_idx].x for trafo_idx in net.trafo.index) for t in time_steps]
            # Use the same sign convention as net_grid_power_mw for plotting: positive = import from grid
            # net_grid_power_mw is defined as ext_grid_import - ext_grid_export (import positive)
            # transformer_signed_p_mw is P_accumulated at LV side; to compare directly with net grid import, invert sign if needed.
            # We'll set transformer_grid_mw to net_grid_power_mw so plots use the same import-positive convention.
            results_data['transformer_grid_mw'] = [results['ext_grid_import_p'][t] - results['ext_grid_export_p'][t] for t in time_steps]
        else:
            results_data['transformer_signed_p_mw'] = [0.0 for t in time_steps]
            results_data['transformer_grid_mw'] = [0.0 for t in time_steps]
        # Add electricity price and ambient temperature for self-contained cost/simulation
        # Export electricity price unconditionally (must align with time_steps)
        results_data['electricity_price_eur_mwh'] = [float(electricity_price[t]) for t in time_steps]
        try:
            results_data['ambient_temp_c'] = [float(temp_profile_c[t]) for t in time_steps]
        except Exception:
            pass

        # Total HP electrical MW (sum across buses)
        results_data['hp_elec_mw'] = [sum(results['p_hp'][t].values()) if t in results['p_hp'] else 0.0 for t in time_steps]
        # Gross load MW (flexible + non-flexible + hp)
        results_data['gross_load_mw'] = [
            (sum(results['flexible_load_p'][t].values()) if t in results.get('flexible_load_p', {}) else 0.0) +
            (sum(results['non_flexible_load_p'][t].values()) if t in results.get('non_flexible_load_p', {}) else 0.0) +
            (sum(results['p_hp'][t].values()) if t in results.get('p_hp', {}) else 0.0)
            for t in time_steps
        ]

        # Export per-bus schedules needed for v3 OOS simulation
        # PV generation per bus (DA schedule)
        try:
            if 'pv_gen' in results and len(pv_buses) > 0:
                for bus in pv_buses:
                    col = f'pv_gen_bus_{bus}_mw'
                    results_data[col] = [results['pv_gen'][t].get(bus, 0.0) if t in results['pv_gen'] else 0.0 for t in time_steps]
        except Exception:
            pass
        # PV available per bus (mean forecast without DA curtailment)
        try:
            if len(pv_buses) > 0:
                for bus in pv_buses:
                    col = f'pv_avail_bus_{bus}_mw'
                    results_data[col] = [float(base_pv_bus_limits.get(bus, 0.0)) * float(const_pv[t]) for t in time_steps]
        except Exception:
            pass
        # Flexible and non-flexible loads per bus (P and Q)
        try:
            for bus in net.bus.index:
                # P components
                results_data[f'load_flex_p_bus_{bus}_mw'] = [results.get('flexible_load_p', {}).get(t, {}).get(bus, 0.0) for t in time_steps]
                results_data[f'load_nonflex_p_bus_{bus}_mw'] = [results.get('non_flexible_load_p', {}).get(t, {}).get(bus, 0.0) for t in time_steps]
                # Q components
                results_data[f'load_flex_q_bus_{bus}_mvar'] = [results.get('flexible_load_q', {}).get(t, {}).get(bus, 0.0) for t in time_steps]
                results_data[f'load_nonflex_q_bus_{bus}_mvar'] = [results.get('non_flexible_load_q', {}).get(t, {}).get(bus, 0.0) for t in time_steps]
        except Exception:
            pass
        # HP electrical consumption per bus (MW)
        try:
            for bus in net.bus.index:
                results_data[f'hp_elec_bus_{bus}_mw'] = [results.get('p_hp', {}).get(t, {}).get(bus, 0.0) for t in time_steps]
        except Exception:
            pass
        
        # Expand BESS results into explicit per-bus columns (units: MWh for energy, MW for power)
        if 'bess_energy' in results and len(bess_buses) > 0:
            for bus in bess_buses:
                col_name_e = f'bess_energy_bus_{bus}_mwh'
                results_data[col_name_e] = [results['bess_energy'][t].get(bus, 0.0) if t in results['bess_energy'] else 0.0 for t in time_steps]
            # Add total BESS capacity (MWh) for reference
            try:
                total_bess_capacity = float(bess_capacity_mwh) * len(bess_buses)
            except Exception:
                total_bess_capacity = 0.0
            results_data['bess_total_capacity_mwh'] = [total_bess_capacity for _ in time_steps]
        if 'bess_charge' in results and len(bess_buses) > 0:
            for bus in bess_buses:
                col_name_c = f'bess_charge_bus_{bus}_mw'
                results_data[col_name_c] = [results['bess_charge'][t].get(bus, 0.0) if t in results['bess_charge'] else 0.0 for t in time_steps]
        if 'bess_discharge' in results and len(bess_buses) > 0:
            for bus in bess_buses:
                col_name_d = f'bess_discharge_bus_{bus}_mw'
                results_data[col_name_d] = [results['bess_discharge'][t].get(bus, 0.0) if t in results['bess_discharge'] else 0.0 for t in time_steps]

        results_df = pd.DataFrame(results_data)
        # Attach metadata columns for full self-containment (repeat per row)
        try:
            # DRCC / RT policy flags
            results_df['meta_enable_rt_policies'] = [bool(ENABLE_RT_POLICIES)] * len(results_df)
            results_df['meta_enable_drcc_rt_budgets'] = [bool(ENABLE_DRCC_RT_BUDGETS)] * len(results_df)
            results_df['meta_enable_drcc_network_tightening'] = [bool(ENABLE_DRCC_NETWORK_TIGHTENING)] * len(results_df)
            results_df['meta_drcc_tighten_trafo'] = [bool(DRCC_TIGHTEN_TRAFO)] * len(results_df)
            results_df['meta_drcc_tighten_lines'] = [bool(DRCC_TIGHTEN_LINES)] * len(results_df)
            results_df['meta_drcc_tighten_voltages'] = [bool(DRCC_TIGHTEN_VOLTAGES)] * len(results_df)
            try:
                network_tightening_active_flag = bool(
                    ENABLE_DRCC_NETWORK_TIGHTENING and (DRCC_TIGHTEN_TRAFO or DRCC_TIGHTEN_LINES or DRCC_TIGHTEN_VOLTAGES)
                )
            except Exception:
                network_tightening_active_flag = False
            results_df['meta_network_tightening_active'] = [network_tightening_active_flag] * len(results_df)
            results_df['meta_rt_budgets_active'] = [bool(ENABLE_DRCC_RT_BUDGETS)] * len(results_df)
            results_df['meta_enforce_base_trafo_limits'] = [bool(ENFORCE_BASE_TRAFO_LIMITS)] * len(results_df)
            results_df['meta_enforce_base_line_limits'] = [bool(ENFORCE_BASE_LINE_LIMITS)] * len(results_df)
            results_df['meta_enforce_base_volt_limits'] = [bool(ENFORCE_BASE_VOLT_LIMITS)] * len(results_df)
            # DRCC parameters
            # Epsilon/k metadata: blank them when tightening disabled (we neutralize k to 1 internally)
            if bool(ENABLE_DRCC_NETWORK_TIGHTENING):
                try:
                    eps_val = float(DRCC_EPSILON)
                except Exception:
                    eps_val = np.nan
                results_df['meta_drcc_epsilon'] = [eps_val] * len(results_df)
                try:
                    k_eps = float(np.sqrt((1.0 - eps_val) / eps_val)) if (eps_val is not None and eps_val > 0.0 and eps_val < 1.0) else np.nan
                except Exception:
                    k_eps = np.nan
                results_df['meta_drcc_k_epsilon'] = [k_eps] * len(results_df)
            else:
                results_df['meta_drcc_epsilon'] = ["" for _ in range(len(results_df))]
                results_df['meta_drcc_k_epsilon'] = ["" for _ in range(len(results_df))]
            # Record how RT budgets were sized (quantile or neutral)
            try:
                results_df['meta_rt_budget_mode'] = ["quantile_drcc" if ENABLE_DRCC_NETWORK_TIGHTENING else "std_only_k1"] * len(results_df)
            except Exception:
                pass
            # PV uncertainty handling
            results_df['meta_pv_std_from_csv'] = [bool(PV_STD_FROM_CSV)] * len(results_df)
            try:
                results_df['meta_pv_relative_std'] = [float(PV_RELATIVE_STD)] * len(results_df) if 'PV_RELATIVE_STD' in globals() else [np.nan] * len(results_df)
            except Exception:
                results_df['meta_pv_relative_std'] = [np.nan] * len(results_df)
            try:
                results_df['meta_pv_std_correlation'] = [float(PV_STD_CORRELATION)] * len(results_df)
            except Exception:
                results_df['meta_pv_std_correlation'] = [np.nan] * len(results_df)
            # HP predictor and uncertainty
            results_df['meta_hp_fully_correlated'] = [bool(HP_FULLY_CORRELATED)] * len(results_df)
            try:
                results_df['meta_rho_temp_avg'] = [float(RHO_TEMP_AVG)] * len(results_df)
            except Exception:
                results_df['meta_rho_temp_avg'] = [np.nan] * len(results_df)
            results_df['meta_hp_include_residual'] = [bool(HP_INCLUDE_RESIDUAL)] * len(results_df)
            try:
                results_df['meta_hp_residual_sigma_norm'] = [float(HP_RESIDUAL_SIGMA_NORM)] * len(results_df)
            except Exception:
                results_df['meta_hp_residual_sigma_norm'] = [np.nan] * len(results_df)
            try:
                results_df['meta_hp_pred_pmax_mw'] = [float(HP_PRED_PMAX)] * len(results_df)
            except Exception:
                results_df['meta_hp_pred_pmax_mw'] = [np.nan] * len(results_df)
            # HP coefficients
            for name, val in (
                ('meta_hp_coeff_b0', 'HP_COEFF_B0'),
                ('meta_hp_coeff_bhdd', 'HP_COEFF_BHDD'),
                ('meta_hp_coeff_bpi', 'HP_COEFF_BPI'),
                ('meta_hp_coeff_btav', 'HP_COEFF_BTAV'),
                ('meta_hp_coeff_a1', 'HP_COEFF_A1'),
                ('meta_hp_coeff_a2', 'HP_COEFF_A2'),
            ):
                try:
                    results_df[name] = [float(globals().get(val))] * len(results_df)
                except Exception:
                    results_df[name] = [np.nan] * len(results_df)
            # Time resolution and window
            try:
                results_df['meta_dt_hours'] = [float(dt_hours)] * len(results_df)
            except Exception:
                results_df['meta_dt_hours'] = [np.nan] * len(results_df)
            try:
                results_df['meta_time_start'] = [str(pd.to_datetime(time_index[0]))] * len(results_df)
                results_df['meta_time_end'] = [str(pd.to_datetime(time_index[-1]))] * len(results_df)
            except Exception:
                pass
            # BESS parameters used in DA plan
            try:
                results_df['meta_bess_eff'] = [float(bess_eff)] * len(results_df)
                results_df['meta_bess_initial_soc'] = [float(bess_initial_soc)] * len(results_df)
                results_df['meta_bess_capacity_mwh'] = [float(bess_capacity_mwh)] * len(results_df)
            except Exception:
                pass
            # Also copy price into a meta-prefixed column for robustness
            try:
                results_df['meta_electricity_price_eur_mwh'] = results_df['electricity_price_eur_mwh']
            except Exception:
                pass
        except Exception:
            pass
        # Append robust recourse policy summaries for transparency
        try:
            if ENABLE_RT_POLICIES:
                # Buy-back removed: exclude y_cap and gamma columns
                results_df['chi0_mw'] = [chi0_vars[t].x for t in time_steps]
                results_df['chi_minus'] = [chi_minus_vars[t].x for t in time_steps]
                results_df['lambda0_mw'] = [lambda0_vars[t].x for t in time_steps]
                results_df['lambda_plus'] = [lambda_plus_vars[t].x for t in time_steps]
                results_df['lambda_minus'] = [lambda_minus_vars[t].x for t in time_steps]
                results_df['rho_plus0_mw'] = [rho_plus0_vars[t].x for t in time_steps]
                results_df['rho_plus1'] = [rho_plus1_vars[t].x for t in time_steps]
                results_df['rho_minus0_mw'] = [rho_minus0_vars[t].x for t in time_steps]
                results_df['rho_minus1'] = [rho_minus1_vars[t].x for t in time_steps]
                # Baseline intercept decomposition & SoC envelope (conditional: may not exist if no BESS)
                try:
                    results_df['p0_dis_mw'] = [p0_dis_vars[t].x if t in p0_dis_vars else 0.0 for t in time_steps]
                    results_df['p0_ch_mw'] = [p0_ch_vars[t].x if t in p0_ch_vars else 0.0 for t in time_steps]
                except Exception:
                    pass
                try:
                    results_df['E_down_mwh'] = [E_down_vars[t].x if t in E_down_vars else np.nan for t in time_steps]
                    results_df['E_up_mwh'] = [E_up_vars[t].x if t in E_up_vars else np.nan for t in time_steps]
                except Exception:
                    pass
                results_df['D_plus_max_mw'] = [D_plus_max[t] for t in time_steps]
                results_df['D_minus_max_mw'] = [D_minus_max[t] for t in time_steps]
                # Also expose DA aggregated flexible curtailment to verify the tie to y_cap
                results_df['ycap_mw'] = [ycap_var.x if ("ycap_var" in locals() and ycap_var is not None) else 0.0 for _ in time_steps]
                # Baseline flexible (time-synchronized) loads per bus (MW & kW)
                try:
                    baseline_flex_dict = globals().get('flexible_time_synchronized_loads_P', {})
                    if baseline_flex_dict:
                        all_buses = set()
                        for tmap in baseline_flex_dict.values():
                            all_buses.update(tmap.keys())
                        buses_sorted = sorted(list(all_buses))
                        for bus in buses_sorted:
                            series_mw = []
                            for t in time_steps:
                                v = 0.0
                                if t in baseline_flex_dict and bus in baseline_flex_dict[t]:
                                    v = float(baseline_flex_dict[t][bus])
                                series_mw.append(v)
                            results_df[f'baseline_flex_bus_{bus}_mw'] = series_mw
                            results_df[f'baseline_flex_bus_{bus}_kw'] = [val * 1000.0 for val in series_mw]
                except Exception as _baseline_export_err:
                    print(f"WARNING: baseline flex export failed: {_baseline_export_err}")
                # Per-bus flexible load realized power (MW) — flat columns for plotting (unaggregated)
                try:
                    for bus in flexible_load_buses:
                        col_name = f'flex_load_bus_{bus}_mw'
                        results_df[col_name] = [flexible_load_P_vars[t][bus].x if (t in flexible_load_P_vars and bus in flexible_load_P_vars[t]) else 0.0 for t in time_steps]
                except Exception:
                    pass
        except Exception:
            pass
        
        # Build DRCC-aware suffix for CSV filename
        try:
            eps_val = float(DRCC_EPSILON)
        except Exception:
            eps_val = None
        # Simplified suffix logic per user request:
        # Only reflect network tightening (transformers/lines/voltages) in the filename.
        # If master tightening enabled AND at least one subcomponent flag is True => drcc_true_epsilon_<token>
        # Otherwise => drcc_false (independent of RT budgets usage).
        try:
            network_tightening_active = bool(
                ENABLE_DRCC_NETWORK_TIGHTENING and (DRCC_TIGHTEN_TRAFO or DRCC_TIGHTEN_LINES or DRCC_TIGHTEN_VOLTAGES)
            )
        except Exception:
            network_tightening_active = False

        if eps_val is not None and not (0.0 < eps_val < 1.0):
            eps_val = None  # sanitize invalid epsilon

        if network_tightening_active:
            if eps_val is not None:
                eps_token = f"{eps_val:.2f}".replace('.', '_')
                suffix = f"drcc_true_epsilon_{eps_token}"
            else:
                suffix = "drcc_true_epsilon_NA"
        else:
            suffix = "drcc_false"

        csv_filename = f"dso_model_v2_results_{suffix}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")
        
        # Expose pv_gen results at module level for plotting helpers that read globals()
        try:
            pv_gen_results = results.get('pv_gen', {})
            globals()['pv_gen_results'] = pv_gen_results
        except Exception:
            globals()['pv_gen_results'] = {}

        # Create comprehensive plots
        print("\nGenerating comprehensive plots...")
        try:    
            # Build a hp_power_values array (kW) for plotting. prefer hp_elec_mw column if present.
            if 'hp_elec_mw' in results_df.columns:
                hp_power_kW = results_df['hp_elec_mw'].to_numpy() * 1000.0
            else:
                hp_power_kW = np.zeros(len(results_df))

            create_comprehensive_plots(
                results_df,
                hp_power_values=hp_power_kW,
                ambient_temps_c=temp_profile_c,
                non_flexible_load_p=results.get('non_flexible_load_p', None),
                flexible_load_p=results.get('flexible_load_p', None),
                electricity_price=electricity_price,
                storage_soc_values=None,
                slack_power_values=None
            )
 
        except Exception:
            # Fallback: still call with correct signature using safe defaults
            if 'hp_elec_mw' in results_df.columns:
                hp_power_kW = results_df['hp_elec_mw'].to_numpy() * 1000.0
            else:
                hp_power_kW = np.zeros(len(results_df))
            create_comprehensive_plots(
                results_df,
               hp_power_values=hp_power_kW,
                ambient_temps_c=temp_profile_c,
                non_flexible_load_p=results.get('non_flexible_load_p', None),
                flexible_load_p=results.get('flexible_load_p', None)
            )
        print(f"="*80)

        # ------------------------------------------------------------------
        # DEBUG: Baseline vs Realized Flexible Load for a Selected Timestep
        # ------------------------------------------------------------------
        try:
            baseline_flex_dict = globals().get('flexible_time_synchronized_loads_P', {})
            realized_flex_dict = results.get('flexible_load_p', {})
            if baseline_flex_dict:
                # Choose first timestep with non-zero baseline total, else 0
                candidate_ts = sorted(baseline_flex_dict.keys())
                t_debug = None
                for tt in candidate_ts:
                    try:
                        if sum(float(v) for v in baseline_flex_dict[tt].values()) > 1e-9:
                            t_debug = tt
                            break
                    except Exception:
                        pass
                if t_debug is None:
                    t_debug = candidate_ts[0]
                base_map = baseline_flex_dict.get(t_debug, {})
                real_map = realized_flex_dict.get(t_debug, {})
                all_buses = sorted(set(base_map.keys()) | set(real_map.keys()))
                print(f"\n[DEBUG FLEX] Timestep t={t_debug}: baseline_vs_realized (MW) for first 25 buses:")
                print(f"{'Bus':>6} {'Baseline_MW':>14} {'Realized_MW':>14} {'Delta_MW':>12}")
                for bus in all_buses[:25]:
                    b = float(base_map.get(bus, 0.0))
                    r = float(real_map.get(bus, 0.0))
                    print(f"{bus:>6} {b:14.6f} {r:14.6f} {r-b:12.6f}")
                # Bar plot (kW) saved to file
                import matplotlib.pyplot as _plt
                fig_dbg, ax_dbg = _plt.subplots(figsize=(10,4))
                bw = 0.4
                x_idx = np.arange(len(all_buses))
                base_kw = [float(base_map.get(b, 0.0))*1000.0 for b in all_buses]
                real_kw = [float(real_map.get(b, 0.0))*1000.0 for b in all_buses]
                ax_dbg.bar(x_idx - bw/2, base_kw, width=bw, label='Baseline kW')
                ax_dbg.bar(x_idx + bw/2, real_kw, width=bw, label='Realized kW')
                ax_dbg.set_title(f'Baseline vs Realized Flexible Load (t={t_debug})')
                ax_dbg.set_xlabel('Bus')
                ax_dbg.set_ylabel('Power (kW)')
                if len(all_buses) <= 40:
                    ax_dbg.set_xticks(x_idx)
                    ax_dbg.set_xticklabels(all_buses, rotation=90, fontsize=8)
                else:
                    # sparse ticks
                    step = max(1, len(all_buses)//40)
                    sel = list(range(0, len(all_buses), step))
                    ax_dbg.set_xticks([x_idx[i] for i in sel])
                    ax_dbg.set_xticklabels([all_buses[i] for i in sel], rotation=90, fontsize=8)
                ax_dbg.grid(alpha=0.3)
                ax_dbg.legend()
                debug_fig_name = f'baseline_vs_realized_flex_t{t_debug}.png'
                fig_dbg.tight_layout()
                fig_dbg.savefig(debug_fig_name, dpi=150)
                _plt.close(fig_dbg)
                print(f"[DEBUG FLEX] Saved comparison figure: {debug_fig_name}")
            else:
                print("[DEBUG FLEX] No baseline flexible load dictionary present (flexible_time_synchronized_loads_P empty or undefined).")
        except Exception as _flex_dbg_err:
            print(f"[DEBUG FLEX] Failed generating baseline vs realized debug: {_flex_dbg_err}")
        
        return results, results_data
    
    elif model.status == GRB.INFEASIBLE:
        print("✗ Multi-period model is infeasible")
       
        # If the model is infeasible, write the model to an ILP file for debugging
        print("OPF Optimization failed - model is infeasible. Writing model to 'cmes_infeasible_model.ilp'")
        model.computeIIS()  # Compute IIS to identify the infeasible set

        model.write("cmes_infeasible_model.ilp")
        return None
    else:
        print(f"OPF Optimization failed with status: {model.status}")
        return None


def load_input_data_from_csv(time_index):
    """Load input data aligned to the DHN time_index.
       Prefer columns from vdi_profiles/all_house_profiles.csv (same CSV as DHN).
         Strict mode: read required columns from vdi_profiles/all_house_profiles.csv and align exactly.
    """
    import pandas as pd
    import numpy as np

    # We won't invent generic columns here. Provide safe defaults for price/pv
    electricity_price = np.full(len(time_index), 150.0)
    const_pv = np.zeros(len(time_index))
    const_load_household_P = np.zeros(len(time_index))
    const_load_heatpump = np.zeros(len(time_index))

    # Reactive power from PF=0.9 for household loads (unchanged)
    const_load_household_Q = const_load_household_P * np.tan(np.arccos(0.9))

    # Ambient temperature: use the same temperature dataset as DHN and the SAME INDEX
    temp_df = pd.read_csv('temperature_data_complete.csv')
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    temp_df.set_index('datetime', inplace=True)
    T_amb = temp_df.reindex(time_index)['temperature_K'].to_numpy()
    # Also align temperature std (K) for DRCC if available
    try:
        T_amb_std = temp_df.reindex(time_index)['temperature_std_K'].to_numpy()
    except Exception:
        T_amb_std = np.zeros(len(time_index))
    # Expose as global for solve_opf (to avoid changing function signature)
    globals()['T_amb_std'] = T_amb_std

    # Compute per-day std of daily average temperature using equicorrelation RHO_TEMP_AVG
    try:
        rho = float(globals().get('RHO_TEMP_AVG', 1.0))
    except Exception:
        rho = 1.0
    rho = max(0.0, min(1.0, rho))

    # Group temperature std by day aligned to time_index and compute std of the average
    ti = pd.to_datetime(time_index)
    sigma_Tavg_by_day = {}
    # For each day, obtain stds of the samples belonging to that day
    for day, idx in pd.Series(range(len(ti)), index=ti).groupby(ti.date):
        inds = idx.values
        sigmas = np.asarray(T_amb_std[inds], dtype=float)
        if len(sigmas) == 0:
            sigma_Tavg_by_day[day] = 0.0
            continue
        N = len(sigmas)
        sumsig = float(np.sum(sigmas))
        sumsqs = float(np.sum(sigmas**2))
        var_avg = (1.0/(N*N)) * ((1.0 - rho) * sumsqs + rho * (sumsig**2))
        sigma_Tavg_by_day[day] = float(np.sqrt(max(0.0, var_avg)))
    globals()['sigma_Tavg_by_day'] = sigma_Tavg_by_day

    # Compute Var(HDD_t) for each t using closed forms for positive part of normal
    # HDD_t = max(0, Tbase - T_C_t). We approximate μ_t = T_C_t (Celsius forecast)
    # and s_t = T_amb_std[t] (K/C). Then set m = a - μ_t, s = s_t, η = m/s.
    Tbase = 10.0
    Var_HDD_by_t = {}
    # Convert Kelvin to Celsius for the mean temperature series
    T_C = (T_amb - 273.15) if isinstance(T_amb, np.ndarray) else np.asarray(T_amb, dtype=float) - 273.15
    from math import sqrt
    import math
    # Standard normal pdf and cdf
    def _phi(x):
        return 1.0/math.sqrt(2.0*math.pi) * math.exp(-0.5*x*x)
    def _Phi(x):
        # Use error function
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    for t, dt in enumerate(ti):
        s = float(T_amb_std[t])
        m = float(Tbase - T_C[t])
        if s <= 1e-12:
            # Deterministic HDD variance 0 (unless m crosses 0, but s=0 ⇒ no randomness)
            Var_HDD_by_t[t] = 0.0
            continue
        eta = m / s
        phi = _phi(eta)
        Phi = _Phi(eta)
        EY = s*phi + m*Phi
        EY2 = (s*s + m*m)*Phi + m*s*phi
        varY = max(0.0, EY2 - EY*EY)
        Var_HDD_by_t[t] = float(varY)
    globals()['Var_HDD_by_t'] = Var_HDD_by_t

    # --- Load PV profile and align to the DHN/optimization time_index ---
    # const_pv is expected to be a normalized factor (0..1) that will later be multiplied
    # by installed peak capacity (from electrical_pv_systems.csv / net.sgen.p_mw) elsewhere.
    try:
        import os
        pv_path = os.path.join('.', 'pv_profiles_output.csv')
        if os.path.exists(pv_path):
            pv_df = pd.read_csv(pv_path, parse_dates=['datetime'])
            pv_df.set_index('datetime', inplace=True)
            if 'normalized_output' in pv_df.columns:
                base = pv_df['normalized_output']
            elif 'power_output_kw' in pv_df.columns:
                s = pv_df['power_output_kw']
                maxv = s.max() if s.max() != 0 else 1.0
                base = s / maxv
            else:
                base = None

            if base is not None:
                # Align exactly to the provided time_index (same as temperature alignment)
                aligned = base.reindex(time_index).fillna(0.0).to_numpy()
                const_pv = aligned
                # Also align PV normalized std if present
                if PV_STD_FROM_CSV and 'normalized_output_std' in pv_df.columns:
                    try:
                        const_pv_std = pv_df['normalized_output_std'].reindex(time_index).fillna(0.0).to_numpy()
                    except Exception:
                        const_pv_std = np.zeros(len(time_index))
                else:
                    # Fallback: use relative std fraction of normalized PV
                    try:
                        const_pv_std = (float(PV_RELATIVE_STD) * const_pv).astype(float)
                    except Exception:
                        const_pv_std = np.zeros(len(time_index))
                globals()['const_pv_std'] = const_pv_std
                try:
                    print(f"Loaded PV profile: range {const_pv.min():.3f} - {const_pv.max():.3f}")
                except Exception:
                    pass
            else:
                print("Warning: pv_profiles_output.csv missing 'normalized_output' or 'power_output_kw'; using zero PV profile")
        else:
            print(f"Note: pv_profiles_output.csv not found at {pv_path}; using zero PV profile")
    except Exception as e:
        print('Warning: failed to load/align pv_profiles_output.csv; using zero PV profile', e)

    # Attempt to load market electricity prices and align them to the DHN time_index
    try:
        price_df = pd.read_csv('market_prices_15min.csv')
        if 'datetime' in price_df.columns:
            price_df['datetime'] = pd.to_datetime(price_df['datetime'])
            price_df.set_index('datetime', inplace=True)
        else:
            # Try common alternative column names
            raise KeyError('datetime column not found in market_prices_15min.csv')

        # Use the price_EUR_MWh column if present
        if 'price_EUR_MWh' in price_df.columns:
            # Reindex to the DHN time_index; if market data is 15-min and DHN uses 15-min this will align directly
            aligned_prices = price_df.reindex(time_index)['price_EUR_MWh']

            # If there are missing values after reindexing, try forward/backfill
            if aligned_prices.isnull().any():
                aligned_prices = aligned_prices.ffill().bfill()

            # If still null (no overlap), keep the default and warn
            if aligned_prices.isnull().any():
                print("Warning: market_prices_15min.csv does not overlap requested time window. Using default electricity price.")
            else:
                electricity_price = aligned_prices.to_numpy()
                print(f"✓ Loaded electricity prices from market_prices_15min.csv and aligned to time_index ({len(electricity_price)} steps)")
        else:
            print("Warning: 'price_EUR_MWh' column not found in market_prices_15min.csv - using default electricity price")

    except FileNotFoundError:
        print("Warning: market_prices_15min.csv not found - using default electricity price")
    except Exception as e:
        print(f"Warning: failed to load/align market_prices_15min.csv: {e}. Using default electricity price")

    # Expose mapped per-component series if created earlier
    hex_ts = globals().get('hex_thermal_demands', {})
    el_ts = globals().get('electrical_time_series', {})

    return {
        'electricity_price': electricity_price,
        'const_pv': const_pv,
        'const_load_household_P': const_load_household_P,
        'const_load_household_Q': const_load_household_Q,
        'const_load_heatpump': const_load_heatpump,
        'T_amb': T_amb,
        'hex_thermal_demands': hex_ts,
        'electrical_time_series': el_ts
    }



if __name__ == "__main__":
    """Main execution block - runs the optimization with CSV data."""
    
    try:
        # Create network from CSV files
        print("\n🔧 Creating network structure from CSV files...")
        net = create_network_from_csv()
        print(f"✅ Network created: {len(net.bus)} buses, {len(net.line)} lines, {len(net.trafo)} transformers")
        

        # Use the DHN time index as canonical
        input_data = load_input_data_from_csv(time_index)

        # Backwards compatibility: some code expects input_data['time_steps'] to exist.
        # Ensure it's present and aligned to the DHN time_index (list of integer step indices).
        if 'time_steps' not in input_data:
            input_data['time_steps'] = list(range(len(time_index)))
        

        time_steps = list(range(len(time_index)))                  # <-- unified horizon
        print(f"\n🚀 Starting optimization with {len(time_steps)} time steps...")

        results = solve_opf(
            net,
            time_steps=time_steps,
            electricity_price=input_data['electricity_price'],
            const_pv=input_data['const_pv'],
            const_load_household_P=input_data['const_load_household_P'],
            const_load_household_Q=input_data['const_load_household_Q'],
            const_load_heatpump=input_data['const_load_heatpump'],
            T_amb=input_data['T_amb'],
        )
        
        if results is not None:
            print("\n🎉 Optimization completed successfully!")
            print("\n📊 OPTIMIZATION RESULTS SUMMARY:")
            print("-" * 50)
            
            # Display key results
            try:
                if 'pv_generation' in results:
                    pv_values = results['pv_generation']
                    if isinstance(pv_values, dict):
                        total_pv = sum(sum(v.values()) if isinstance(v, dict) else [v] for v in pv_values.values())
                    else:
                        total_pv = sum(pv_values) if hasattr(pv_values, '__iter__') else pv_values
                    print(f"Total PV Generation: {total_pv:.2f} MWh")
                
                if 'ext_grid_import_p' in results:
                    import_values = results['ext_grid_import_p']
                    if isinstance(import_values, dict):
                        total_import = sum(sum(v.values()) if isinstance(v, dict) else [v] for v in import_values.values())
                    else:
                        total_import = sum(import_values) if hasattr(import_values, '__iter__') else import_values
                    print(f"Total Grid Import: {total_import:.2f} MWh")
            except Exception as e:
                print(f"Note: Could not calculate totals - {e}")
                        
        else:
            print("\n❌ Optimization failed!")
            
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)