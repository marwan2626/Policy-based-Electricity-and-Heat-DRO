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

# Allow runtime override of START_DATE/DURATION_HOURS via CLI args or environment
try:
    import argparse
    import os
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--start-date', dest='start_date', help='Override START_DATE (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--duration-hours', dest='duration_hours', type=int, help='Override DURATION_HOURS (int)')
    # parse_known_args so other scripts importing this module won't fail
    args, _ = parser.parse_known_args()
    # Priority: CLI arg > environment variable > file default
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
except Exception:
    # If argparse/import fails for any reason, fall back to file defaults
    pass

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
el_norm_map = {normalize_name(n): n for n in el_names}

# Containers for assigned time series (kW)
el_time_series = {}


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
    
    # 8. Junction Temperatures Overview - Return Temperatures  
    plt.subplot(7, 2, 8)
    return_cols = [col for col in results_df.columns if 'return' in col.lower() and 'temp' in col.lower()]
    if return_cols:
        return_temps = results_df[return_cols].values
        for i, col in enumerate(return_cols[:10]):  # Show first 10 for clarity
            plt.plot(hours, results_df[col], alpha=0.7, label=f'Junction {i+1}')
        plt.title('Return Junction Temperatures (First 10)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 9. Electrical Bus Voltages
    plt.subplot(7, 2, 9)
    voltage_cols = [col for col in results_df.columns if 'voltage_pu' in col.lower()]
    if voltage_cols:
        # Plot voltage profiles for all buses
        for i, col in enumerate(voltage_cols):
            bus_number = col.split('_')[1]  # Extract bus number from column name
            plt.plot(hours, results_df[col], alpha=0.7, linewidth=1.5, label=f'Bus {bus_number}')
        
        # Add voltage limits
        #plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.8, label='Min Limit (0.95 p.u.)')
        #plt.axhline(y=1.05, color='red', linestyle='--', alpha=0.8, label='Max Limit (1.05 p.u.)')
        plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Nominal (1.0 p.u.)')
        
        plt.title('Electrical Bus Voltages Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Voltage (p.u.)')
        plt.grid(True, alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.ylim(0.85, 1.15)  # Set reasonable voltage range
    
    # 10. Thermal Storage State of Charge Analysis
    plt.subplot(7, 2, 10)
    if storage_soc_values is not None:
        storage_soc_percent = [soc * 100 for soc in storage_soc_values]  # Convert to percentage
        
        plt.plot(hours, storage_soc_percent, 'g-', linewidth=2, label='Storage SOC')
        plt.title('Thermal Storage State of Charge', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('State of Charge (%)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)  # SOC ranges from 0% to 100%
        
        # Add horizontal lines for reference
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% SOC')
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% SOC (Low)')
        plt.axhline(y=80, color='blue', linestyle='--', alpha=0.7, label='80% SOC (High)')
        
        # Add statistics text
        min_soc = min(storage_soc_percent)
        max_soc = max(storage_soc_percent)
        avg_soc = np.mean(storage_soc_percent)
        plt.text(0.05, 0.95, f'Min: {min_soc:.1f}%\nMax: {max_soc:.1f}%\nAvg: {avg_soc:.1f}%', 
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.legend(loc='upper right', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No Storage SOC Data Available', 
                 transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
        plt.title('Thermal Storage State of Charge', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('State of Charge (%)')
        plt.grid(True, alpha=0.3)
    
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
    
    # 13. PV Generation Over Time (replaces temperature summary)
    plt.subplot(7, 2, 13)
    # pv_gen results are expected in the saved results dict and results_df does not contain per-pv columns by default
    # Try to extract PV columns from results_df (if present) else use the passed-in results dict via globals()
    pv_cols = [col for col in results_df.columns if 'pv' in col.lower() or 'p_pv' in col.lower()]
    if pv_cols:
        # plot any explicit pv columns in results_df (assumed MW -> convert to kW)
        for col in pv_cols:
            plt.plot(hours, results_df[col].values * 1000.0, label=col)
        plt.title('PV Generation per Column (kW)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.grid(True, alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # Use pv_gen_results from the optimization results structure if available
        pv_ts = globals().get('pv_gen_results', {})
        if pv_ts and len(pv_ts) > 0:
            # build per-bus series (kW)
            per_pv = {}
            T = len(hours)
            for t in range(T):
                if t in pv_ts:
                    for bus, mw in pv_ts[t].items():
                        per_pv.setdefault(bus, [0.0]*T)
                        per_pv[bus][t] = mw * 1000.0

            colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(per_pv))))
            for i, (bus, series_kw) in enumerate(sorted(per_pv.items(), key=lambda x: x[0])):
                plt.plot(hours, series_kw, linewidth=1.2, alpha=0.9, color=colors[i % len(colors)], label=f'PV bus {bus}')

            plt.title('PV Generation per Bus (kW)', fontsize=14, fontweight='bold')
            plt.xlabel('Hour')
            plt.ylabel('Power (kW)')
            plt.grid(True, alpha=0.3)
            if len(per_pv) <= 20:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No PV generation data available', transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
            plt.title('PV Generation Over Time', fontsize=14, fontweight='bold')
    
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
    plt.savefig('dso_model_v1_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    bess_cost_per_mwh = 5 # Cost per MWh of BESS capacity

    ### Define the variables ###
    epsilon = 100e-9  # Small positive value to ensure some external grid usage

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


    # Temporary dictionary to store updated load values per time step
    flexible_time_synchronized_loads_P = {t: {} for t in time_steps}
    flexible_time_synchronized_loads_Q = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads_P = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads_Q = {t: {} for t in time_steps}

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
                    print(f"Added {len(bev_loads_df)} BEV loads to net.load from '{bev_csv_path}'")
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

        # plug in your chosen model’s coefficients (example: ridge dev_norm_Tavg_tod)
        Pmax = 0.30
        b0   = 0.065855
        bHDD = 0.013056
        bpi  = -0.000494
        bTav = -0.010871
        a1   = -0.017413
        a2   = -0.009293

        y_dev = b0 + bHDD*HDD_t + bpi*price_t + bTav*T_avg_d + a1*sin24 + a2*cos24
        P_t = P_base + Pmax * y_dev
        P_t = min(max(P_t, 0.0), Pmax)      # clip to [0, Pmax]

        p_hp_pred = P_t

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
                #print(f"Flexible load found: {load_name} at bus {bus}")
                # Flexible load: map its time series (same behavior as non-flexible loads)
                load_name = getattr(load, 'name', '')
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
                    load_p_mw_t = mapped_series[t]
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

        # Extract the bus indices where PV generators and BESS are connected (from net.sgen)
        # Distinguish by the 'type' column (e.g., 'PV' vs 'BESS') so we can treat them separately.
        if 'type' in net.sgen.columns:
            pv_mask = net.sgen['type'].astype(str).str.contains('PV', na=False)
            bess_mask = net.sgen['type'].astype(str).str.contains('BESS', na=False)
        else:
            # Fallback: assume all sgen entries without a type are PV
            pv_mask = pd.Series([True] * len(net.sgen), index=net.sgen.index)
            bess_mask = pd.Series([False] * len(net.sgen), index=net.sgen.index)

        # Ensure unique, sorted bus lists for deterministic ordering
        pv_buses = sorted(set(net.sgen.loc[pv_mask, 'bus'].astype(int).values))
        bess_buses = sorted(set(net.sgen.loc[bess_mask, 'bus'].astype(int).values))

        # Base installed PV per bus (sum if multiple sgen entries on same bus)
        base_pv_bus_limits = {bus: net.sgen.loc[(net.sgen['bus'] == bus) & pv_mask, 'p_mw'].sum() for bus in pv_buses}
        # Base BESS power per bus (sum if multiple BESS entries on same bus)
        base_bess_bus_limits = {bus: net.sgen.loc[(net.sgen['bus'] == bus) & bess_mask, 'p_mw'].sum() for bus in bess_buses}

        # Scale installed peak by the normalized PV profile for this timestep
        # const_pv is expected to be in 0..1; pv_bus_limits_t gives available potential (MW)
        pv_bus_limits_t = {bus: float(base_pv_bus_limits.get(bus, 0.0)) * float(const_pv[t]) for bus in pv_buses}

        # PV availability scaled per timestep (no verbose debug printing)

        # Create PV generation variables for this time step using the scaled availability
        if len(pv_buses) > 0:
            pv_gen_vars[t] = model.addVars(pv_buses, lb=0, ub=pv_bus_limits_t, name=f'pv_gen_{t}')
            curtailment_vars[t] = model.addVars(pv_buses, lb=0, ub=pv_bus_limits_t, name=f'curtailment_{t}')

            for bus in pv_buses:
                # available potential this timestep
                available_mw = float(base_pv_bus_limits.get(bus, 0.0)) * float(const_pv[t])
                model.addConstr(
                    curtailment_vars[t][bus] == available_mw - pv_gen_vars[t][bus], 
                    name=f'curtailment_constraint_{t}_{bus}'
                )

        if len(bess_buses) > 0:
            # Create per-bus negative lower bounds by negating the base limits dict
            bess_charge_vars[t] = model.addVars(bess_buses, lb=0, ub=base_bess_bus_limits, name=f'bess_charge_{t}')
            bess_discharge_vars[t] = model.addVars(bess_buses, lb=0, ub=base_bess_bus_limits, name=f'bess_discharge_{t}')
            bess_energy_vars[t] = model.addVars(bess_buses, lb=0, ub=bess_capacity_mwh, name=f'bess_energy_{t}')
            if t == time_steps[0]:
                # Set initial state-of-charge for each BESS (do not overwrite the vars dict)
                for bus in bess_buses:
                    model.addConstr(
                        bess_energy_vars[t][bus] == bess_initial_soc * bess_capacity_mwh,
                        name=f'bess_energy_initial_{t}_{bus}'
                    )
            else:
                # bess energy time t = bess energy t-1 + charge*efficiency
                for bus in bess_buses:
                    model.addConstr(
                        bess_energy_vars[t][bus] == bess_energy_vars[t-1][bus] 
                        + bess_charge_vars[t][bus] * bess_eff - bess_discharge_vars[t][bus] / bess_eff,
                        name=f'bess_energy_update_{t}_{bus}'
                    )
                # Optional: enforce cyclical SOC (end SOC = start SOC)
                if t == time_steps[-1]:
                    for bus in bess_buses:
                        model.addConstr(
                            bess_energy_vars[t][bus] == bess_energy_vars[time_steps[0]][bus],
                            name=f'bess_energy_cyclical_{t}_{bus}'
                        )
            
        # External grid power variables for import and export at the slack bus (bus 0)
        ext_grid_import_P_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_P_{t}')  # Import is non-negative
        ext_grid_import_Q_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_Q_{t}')  # Import is non-negative
        ext_grid_export_P_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_P_{t}')  # Export is non-negative
        ext_grid_export_Q_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_Q_{t}')  # Export is non-negative
        ext_grid_P_net = model.addVars(time_steps, lb=-GRB.INFINITY, name="P_net")
        ext_grid_Q_net = model.addVars(time_steps, lb=-GRB.INFINITY, name="Q_net")

        #model.addConstr(ext_grid_import_P_vars[t] + ext_grid_export_P_vars[t] >= epsilon, name=f'nonzero_ext_grid_P_usage_{t}')
        #model.addConstr(ext_grid_import_Q_vars[t] + ext_grid_export_Q_vars[t] >= epsilon, name=f'nonzero_ext_grid_Q_usage_{t}')


        flexible_load_P_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            name=f'flexible_load_P_{t}'
        )

        flexible_load_Q_vars[t] = model.addVars(
            flexible_load_buses,
            name=f'flexible_load_Q_{t}'
        )

        # Define heat pump load variables for buses with HP loads
        if len(hp_load_buses) > 0:
            p_hp_vars[t] = model.addVars(
                hp_load_buses,
                lb=0,
                ub=HP_PMAX_MW,
                name=f'p_hp_{t}'
            )
            # Debug: Print heat pump variable creation for first time step
            if t == 0:
                print(f"  Created p_hp_vars[{t}] for buses: {hp_load_buses}")
                            
       
    
    non_slack_buses = [bus for bus in net.bus.index if bus != slack_bus_index]

    
    V_vars = model.addVars(time_steps, net.bus.index, name="V")
    V_reduced_vars = model.addVars(time_steps, non_slack_buses, name="V_reduced")
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

    # Add power balance and load flow constraints for each time step
    for t in time_steps:
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
                    P_injected[bus] -= non_flexible_time_synchronized_loads_P[t][bus]
                    # non_flexible_time_synchronized_loads_Q already computed using household PF (MVar)
                    Q_injected[bus] -= non_flexible_time_synchronized_loads_Q[t][bus]

                if bus in hp_load_buses:
                    # For heat pump loads, use the heat pump variable
                    P_injected[bus] -= p_hp_vars[t][bus]
                    # Add reactive consumption for heat pump using its PF (inductive)
                    # p_hp_vars is MW (since other P are in MW), convert to MVar via qfactor_heatpump capacitive
                    Q_injected[bus] += p_hp_vars[t][bus] * qfactor_heatpump

            if len(pv_buses) > 0 and bus in pv_buses:
                if bus in pv_buses:
                    # Only add PV generation if the bus has PV (i.e., in net.sgen.bus)
                    P_injected[bus] += pv_gen_vars[t][bus]

            if len(bess_buses) > 0 and bus in bess_buses:
                if bus in bess_buses:
                    # BESS charging is negative injection, discharging is positive injection
                    P_injected[bus] -= bess_charge_vars[t][bus]
                    P_injected[bus] += bess_discharge_vars[t][bus]
                    # Assume BESS operates at unity power factor (no reactive power)

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

        # Define voltage magnitude constraints using correct indexing
        for i, bus in enumerate(non_slack_buses):
            model.addConstr(
                V_reduced_vars[t, bus] == 1 +
                 2* (gp.quicksum(R[i, j] * P_pu[non_slack_buses[j]] for j in range(len(non_slack_buses))) +
                gp.quicksum(X[i, j] * Q_pu[non_slack_buses[j]] for j in range(len(non_slack_buses)))),
                name=f"voltage_magnitude_{t}_{bus}"
        )

        # Map V_reduced_vars to V_vars for non-slack buses
        for bus in non_slack_buses:
                model.addConstr(V_vars[t, bus] == V_reduced_vars[t, bus], name=f"voltage_assignment_{t}_{bus}")

                #tight_v_min = 0.95 + par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("V_variance", bus)]
                #tight_v_max = 1.05 - par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("V_variance", bus)]

                tight_v_min = 0.90**2
                tight_v_max = 1.10**2

                #model.addConstr(V_vars[t, bus] >= tight_v_min, name=f"voltage_min_drcc_{t}_{bus}")
                #model.addConstr(V_vars[t, bus] <= tight_v_max, name=f"voltage_max_drcc_{t}_{bus}")
        

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
            #tight_line_limit = (par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("line_pl_mw", line_idx)])/par.hp_pf
            
            #S_branch_limit = 0.8 * S_rated_line - tight_line_limit
            S_branch_limit = 0.8 * S_rated_line

            #model.addGenConstrNorm(S_branch_vars[t, line_idx], [P_branch_vars[t, line_idx], Q_branch_vars[t, line_idx]], 2, name=f"S_branch_calc_{t}_{line_idx}")
            # model.addQConstr(
            #     P_branch_vars[t, line_idx]*P_branch_vars[t, line_idx] +
            #     Q_branch_vars[t, line_idx]*Q_branch_vars[t, line_idx]
            #     <= S_branch_limit**2,
            #     name=f"S_branch_limit_{t}_{line_idx}"
            # )

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
        
            #tight_trafo_limit = par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("S_trafo", trafo_idx)]
            #S_limit = 0.8*S_rated - tight_trafo_limit

            S_limit = 0.8*S_rated 

            # model.addQConstr(
            #     P_trafo_vars[t, trafo_idx]*P_trafo_vars[t, trafo_idx] +
            #     Q_trafo_vars[t, trafo_idx]*Q_trafo_vars[t, trafo_idx]
            #     <= S_limit**2,
            #     name=f"S_trafo_limit_{t}_{trafo_idx}"
            # )

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
    Pmax = 0.30  # MW
    Tbase = 10.0 # °C (from the fit)
    b0   = 0.065855
    bHDD = 0.013056
    bpi  = -0.000494
    bTav = -0.010871
    a1   = -0.017413
    a2   = -0.009293

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

    
        if len(flexible_load_buses) > 0:
            # Treat flexible electrical loads as an aggregated curtailable resource (capacity buyback).
            # Individual buses are allowed to be reduced, but curtailment is constrained only at the aggregate level.
            # Per-bus: 0 <= flexible_load_P_vars[t][bus] <= baseline_P_bus
            # Aggregate: sum_b flexible_load_P_vars[t][b] == sum_b baseline_P_b - flex_curtail_P_vars[t]
            # Create one aggregated curtailment variable for this timestep
            flex_curtail_P_vars[t] = model.addVar(
                lb=0,
                ub=sum(float(flexible_time_synchronized_loads_P[t][bus]) for bus in flexible_load_buses),
                name=f'flex_curtail_P_{t}'
            )

            # Per-bus upper bounds (allow each bus to be curtailed individually but only constrained by aggregate)
            for bus in flexible_load_buses:
                model.addConstr(
                    flexible_load_P_vars[t][bus] <= flexible_time_synchronized_loads_P[t][bus],
                    name=f"flexible_load_upper_P_t{t}_bus{bus}"
                )
                # Reactive load similarly bounded by its baseline (no individual reactive curtailment control)
                model.addConstr(
                    flexible_load_Q_vars[t][bus] <= flexible_time_synchronized_loads_Q[t][bus],
                    name=f"flexible_load_upper_Q_t{t}_bus{bus}"
                )

            # Enforce aggregated curtailment relationship: total flexible consumption = baseline_total - aggregated_curtail
            model.addConstr(
                gp.quicksum(flexible_load_P_vars[t][bus] for bus in flexible_load_buses)
                == gp.quicksum(float(flexible_time_synchronized_loads_P[t][bus]) for bus in flexible_load_buses) - flex_curtail_P_vars[t],
                name=f"flexible_agg_curtail_P_t{t}"
            )


    # Optimize
    print(f"\nStarting multi-period optimization...")
    
    
    print(f"COST-BASED Objective: Minimize total operational cost over {NUM_PERIODS} periods")
        
    electricity_cost =  gp.quicksum(electricity_price[t] * (ext_grid_import_P_vars[t] + ext_grid_export_P_vars[t]) for t in time_steps)
    bess_cost = gp.quicksum(bess_cost_per_mwh * (bess_charge_vars[t][bus] + bess_discharge_vars[t][bus]) for bus in bess_buses for t in time_steps) if len(bess_buses) > 0 else 0
    y_cap_cost = gp.quicksum(2 * electricity_price[t] * flex_curtail_P_vars[t] for t in time_steps)
    pv_curtail_cost = gp.quicksum(electricity_price[t] * curtailment_vars[t][bus] for bus in pv_buses for t in time_steps) if len(pv_buses) > 0 else 0


    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = electricity_cost + bess_cost + y_cap_cost + pv_curtail_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # After adding all constraints and variables
    #model.setParam('OutputFlag', 0)
    #model.setParam('Presolve', 0)
    #model.setParam('NonConvex', 2)

    model.update()

    # Optimize the model
    model.optimize()

    # Check if optimization was successful
    if model.status == GRB.OPTIMAL:
        print(f"OPF Optimal Objective Value: {model.ObjVal}")
        #print("\n--- Debugging P_abs and P_branch Values ---\n")
    
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
            flex_curtail_P_results[t] = flex_curtail_P_vars[t].x 
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
            electricity_cost_value = sum(electricity_price[t] * (ext_grid_import_P_results.get(t, 0.0) + ext_grid_export_P_results.get(t, 0.0)) for t in time_steps)
        except Exception:
            electricity_cost_value = None

        try:
            if len(bess_buses) > 0:
                bess_cost_value = sum(bess_cost_per_mwh * (sum(bess_charge_results[t].values()) + sum(bess_discharge_results[t].values())) for t in time_steps)
            else:
                bess_cost_value = 0.0
        except Exception:
            bess_cost_value = None

        try:
            y_cap_cost_value = sum(electricity_price[t] * flex_curtail_P_results.get(t, 0.0) for t in time_steps)
        except Exception:
            y_cap_cost_value = None

        try:
            pv_curtail_cost_value = 0.0
            if len(pv_buses) > 0:
                for t in time_steps:
                    pv_curtail_cost_value += electricity_price[t] * sum(curtailment_vars[t][bus].x for bus in pv_buses)
        except Exception:
            pv_curtail_cost_value = None

        try:
            total_cost_value = None
            # Prefer computing from components when available
            if None not in (electricity_cost_value, bess_cost_value, y_cap_cost_value, pv_curtail_cost_value):
                total_cost_value = electricity_cost_value + bess_cost_value + y_cap_cost_value + pv_curtail_cost_value
            else:
                # Fallback to model objective value if components couldn't be computed
                total_cost_value = model.ObjVal
        except Exception:
            total_cost_value = model.ObjVal if hasattr(model, 'ObjVal') else None

        print("\nCOST BREAKDOWN:")
        print(f"  electricity_cost = {electricity_cost_value}")
        print(f"  bess_cost = {bess_cost_value}")
        print(f"  y_cap_cost (flex curtail) = {y_cap_cost_value}")
        print(f"  pv_curtail_cost = {pv_curtail_cost_value}")
        print(f"  total_cost (components sum or model.ObjVal) = {total_cost_value}")
        try:
            print(f"  model.ObjVal = {model.ObjVal}")
        except Exception:
            pass
        print(f"✓ MULTI-PERIOD OPTIMIZATION SUCCESSFUL!")
        print(f"="*80)
        
        
        results_data = {
            'period': [t+1 for t in time_steps],
        }

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
        # Total HP electrical MW (sum across buses)
        results_data['hp_elec_mw'] = [sum(results['p_hp'][t].values()) if t in results['p_hp'] else 0.0 for t in time_steps]
        # Gross load MW (flexible + non-flexible + hp)
        results_data['gross_load_mw'] = [
            (sum(results['flexible_load_p'][t].values()) if t in results.get('flexible_load_p', {}) else 0.0) +
            (sum(results['non_flexible_load_p'][t].values()) if t in results.get('non_flexible_load_p', {}) else 0.0) +
            (sum(results['p_hp'][t].values()) if t in results.get('p_hp', {}) else 0.0)
            for t in time_steps
        ]
        
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
        
        results_df.to_csv('dso_model_v1_results.csv', index=False)
        print(f"\nResults saved to: dso_model_v1_results.csv")
        
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