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
START_DATE = "2023-06-10 00:00:00"  # Can be modified when running the optimization
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

    print(f"Using START_DATE={START_DATE}, DURATION_HOURS={DURATION_HOURS}")
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

#print("\nMapping profiles to network components (exact name matching, strict)")
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
hex_time_series = {}
el_time_series = {}

unmatched_bases = []
for base, parts in profile_parts.items():
    base_norm = normalize_name(base)
    # Thermal assignment requires both heating and hotwater
    if 'heating' in parts or 'hotwater' in parts:
        if 'heating' not in parts or 'hotwater' not in parts:
            raise ValueError(f"Missing heating or hotwater column for base '{base}'. Found parts: {list(parts.keys())}")
        total_kw = parts['heating'] + parts['hotwater']

        # Candidate hex key: prefer exact normalized matches (avoid unsafe substring matching)
        candidate_keys = [normalize_name(base), normalize_name('hex_' + base)]
        matched = None
        for k in candidate_keys:
            if k in hex_norm_map:
                matched = hex_norm_map[k]
                break
        if not matched:
            # No exact normalized match found — do NOT fallback to substring matching (avoids e.g. '_3' matching '_30')
            zero_series = np.zeros(NUM_PERIODS)
            guessed_hex = 'hex_' + base
            hex_time_series[guessed_hex] = zero_series
            print(f"Warning: thermal base '{base}' did not match any heat exchanger by exact name. Assigned zero-series to '{guessed_hex}'.")
        else:
            hex_time_series[matched] = total_kw
            #print(f"Assigned thermal profile -> {matched} (base='{base}')")

    # Electrical parts are mapped deterministically after we finish gathering profile_parts
    # (do not guess or create guessed_el entries here)

if unmatched_bases:
    # We already assign zero-series for unmatched components; log a summary
    for b, typ in unmatched_bases:
        print(f"Warning: Base '{b}' could not be matched to network {typ} component — zero-series assigned.")

if not hex_time_series and not el_time_series:
    raise ValueError("No valid house profiles (heating/hotwater/electric) were found in the CSV.")

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

# Create aggregated total thermal load (kW) for sizing and downstream use
total_load = np.zeros(NUM_PERIODS)
for hex_name, series in hex_time_series.items():
    if len(series) != NUM_PERIODS:
        raise ValueError(f"Time series length mismatch for {hex_name}: expected {NUM_PERIODS}, got {len(series)}")
    total_load += series
    #print(f"{hex_name}: Load range {series.min():.2f} - {series.max():.2f} kW")

if total_load.max() <= 0:
    raise ValueError("Aggregate thermal load is zero or negative after mapping - aborting")

base_load = total_load.max()
load_profile = total_load / base_load

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

# # Verify alignment
# print(f"\nDATA ALIGNMENT VERIFICATION (first 5 periods):")
# for i in range(min(5, NUM_PERIODS)):
#     print(f"Period {i+1}: Load factor = {load_profile[i]:.3f}, Ambient temp = {temp_profile_c[i]:.1f}°C")

####################################################################################################
# IMPORT EXACT FUNCTIONS FROM PANDAPIPES_SIMPLE.PY

# Create network data structures from CSV (same constants and logic as original)
rho = 983  # kg/m³
c_w = 4188  # J/kg.K (corrected value from conversation)
delta_t_s = 3600  # timestep in seconds (1 hour for hourly optimization)

# Build junctions dictionary from CSV
junctions = {}
junction_names = {}  # For name lookup
for _, node in nodes_df.iterrows():
    junctions[node['junction_id']] = {"name": node['name']}
    junction_names[int(node['junction_id'])] = node['name']

# Collect pipe data from CSV (same structure as pandapipes version)
pipes = []
for _, pipe in pipes_df.iterrows():
    length = pipe["length_km"] * 1000  # meters
    diameter = pipe["diameter_m"]
    area = np.pi * (diameter / 2) ** 2
    u = pipe["u_w_per_m2k"]
    from_junc = pipe["from_junction"]
    to_junc = pipe["to_junction"]
    pipes.append({
        "pipe_idx": pipe['pipe_id'],
        "from": junctions[from_junc]["name"],
        "to": junctions[to_junc]["name"],
        "length_m": length,
        "diameter_m": diameter,
        "area_m2": area,
        "u_w_per_m2k": u,
    })

# Load pump parameters from gas boiler CSV data
pump_data = boiler_df[boiler_df["name"] == "central_circulation_pump"].iloc[0]
mdot_kg_s = pump_data["mdot_flow_kg_per_s"]
T_in_C = pump_data["t_flow_k"]
pump_inlet_junction = int(pump_data["flow_junction"])
pump_return_junction = int(pump_data["return_junction"])

print(f"Loaded pump data: Mass flow = {mdot_kg_s:.1f} kg/s, Inlet temp = {T_in_C:.1f} K")
print(f"Pump junctions: inlet = {pump_inlet_junction} ({junction_names[pump_inlet_junction]}), return = {pump_return_junction} ({junction_names[pump_return_junction]})")

# Load heat exchangers from CSV
loads = {}
base_loads = {}  # Store base load values for scaling

for _, hx in hex_df.iterrows():
    from_junc = int(hx["from_junction"])
    to_junc = int(hx["to_junction"])
    qext_base = hx["qext_w"]  # Base load [W]
    
    # Get mass flow from flow control data
    matching_fc = flow_control_df[flow_control_df["to_junction"] == from_junc]
    
    if not matching_fc.empty:
        mdot = matching_fc.iloc[0]["controlled_mdot_kg_per_s"]
    else:
        print(f"Warning: No flow control found for HEX from {junction_names[from_junc]} to {junction_names[to_junc]}")
        mdot = mdot_kg_s  # fallback

    loads[to_junc] = {
        "type": "heat_exchanger",
        "from": from_junc,
        "to": to_junc,
        "qext_base": qext_base,  # Base load for scaling
        "mass_flow": mdot
    }
    
    base_loads[to_junc] = qext_base
    #print(f"Heat exchanger: {junction_names[to_junc]} - Base load: {qext_base/1000:.1f} kW")

print(f"Found {len(loads)} heat exchangers with total base load: {sum(base_loads.values())/1000:.1f} kW")

# Map junctions from CSV data
junctions = {}
for _, row in nodes_df.iterrows():
    junctions[int(row["junction_id"])] = {"name": row["name"]}

# Find flow control connections from CSV data
flow_control_connections = {}
for _, fc in flow_control_df.iterrows():
    from_junc = int(fc["from_junction"])
    to_junc = int(fc["to_junction"])
    mdot = fc["controlled_mdot_kg_per_s"]
    
    from_name = junction_names[from_junc]
    to_name = junction_names[to_junc]
    
    flow_control_connections[(from_junc, to_junc)] = {
        "from_name": from_name,
        "to_name": to_name,
        "mass_flow": mdot,
        "type": "flow_control"
    }

# Extract return processing order from CSV data
def extract_return_processing_order():
    """Extract the correct processing order for return junctions from network topology"""
    import networkx as nx
    
    # Create a directed graph of return pipe connections
    return_graph = nx.DiGraph()
    
    # Add return junctions to graph
    return_junctions = {}
    for _, junction in nodes_df.iterrows():
        if 'return' in junction['name'] and 'heat_pump' not in junction['name']:
            idx = int(junction['junction_id'])
            return_junctions[idx] = junction['name']
            return_graph.add_node(idx)
    
    # Add return pipe connections in normal flow direction
    for _, pipe in pipes_df.iterrows():
        from_junc = int(pipe['from_junction'])
        to_junc = int(pipe['to_junction'])
        
        # Only add if both junctions are return junctions
        if from_junc in return_junctions and to_junc in return_junctions:
            return_graph.add_edge(from_junc, to_junc)
    
    try:
        # Get topological order (from source to end)
        topo_order = list(nx.topological_sort(return_graph))
        processing_order_ids = list(topo_order)
        processing_order_names = [return_junctions[jid] for jid in processing_order_ids]
        return processing_order_names
        
    except nx.NetworkXError as e:
        # Fallback to manual order if automatic extraction fails
        manual_order = [
            'close_loop_return', 'bus_43_return', 'bus_42_return', 'bus_41_return', 
            'bus_40_return', 'bus_39_return', 'bus_38_return', 'bus_37_return',
            'bus_36_return', 'bus_35_return', 'bus_4_return', 'bus_13_return',
            'bus_30_return', 'bus_24_return', 'bus_6_return', 'bus_10_return',
            'bus_7_return', 'bus_28_return', 'bus_22_return', 'bus_5_return',
            'bus_17_return', 'bus_26_return', 'bus_20_return', 'bus_25_return',
            'bus_33_return', 'bus_27_return', 'bus_31_return', 'bus_18_return',
            'bus_16_return', 'bus_23_return', 'bus_11_return', 'bus_1_return',
            'bus_12_return', 'bus_3_return', 'bus_2_return', 'bus_32_return',
            'bus_8_return', 'bus_21_return', 'bus_19_return', 'bus_9_return',
            'bus_34_return', 'bus_14_return', 'bus_29_return', 'bus_15_return'
        ]
        
        # Filter to only include junctions that exist in the network
        existing_manual_order = [name for name in manual_order if name in return_junctions.values()]
        return existing_manual_order

# Extract the actual return processing order from the network
automatic_return_order = extract_return_processing_order()

# Assemble final network structure (same as pandapipes_simple.py)
network_dict = {
    "junctions": junctions,
    "pipes": pipes,
    "loads": loads,
    "flow_controls": flow_control_connections,
    "return_processing_order": automatic_return_order,
    "pump": {
        "inlet_junction": pump_inlet_junction,  # From gas boiler CSV data
        "return_junction": pump_return_junction,  # From gas boiler CSV data
        "mass_flow_kg_s": mdot_kg_s,
        "temperature_C": T_in_C - 273.15
    }
}

# --- Assign time series from VDI profiles to heat exchangers and electrical loads ---
print("Assigning mapped time series to network components...")
# hex_time_series and el_time_series are created earlier during strict CSV parsing
try:
    hex_time_series  # noqa: F821
    el_time_series   # noqa: F821
except NameError:
    # If they don't exist, the strict loader should have errored earlier
    raise RuntimeError("Internal error: expected mapped time series (hex_time_series / el_time_series) not found. Ensure profile mapping ran before network assembly.")

# Create hex_thermal_demands keyed by heat exchanger name used downstream (values in kW)
hex_thermal_demands = {}
for hex_name, series in hex_time_series.items():
    if len(series) != NUM_PERIODS:
        const_pv = np.zeros(len(time_index))
    # many downstream functions expect indexing by hx_id or hex name; store by name
    hex_thermal_demands[hex_name] = series

# For electrical loads, build a mapping from electrical load name -> p_mw time series (kW)
electrical_time_series = {}
for el_name, series in el_time_series.items():
    if len(series) != NUM_PERIODS:
        raise ValueError(f"Electrical series length mismatch for {el_name}: expected {NUM_PERIODS}, got {len(series)}")
    electrical_time_series[el_name] = series

print(f"Assigned {len(hex_thermal_demands)} thermal series and {len(electrical_time_series)} electrical series.")

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

                print(f"BEV profile assignment complete: {assigned} BEV loads assigned from {len(profile_cols)} profile columns.")
except Exception as _e:
    print(f"Warning: failed to assign BEV profiles: {_e}")

# Sanity check: ensure at least one thermal series assigned
if len(hex_thermal_demands) == 0:
    raise ValueError("No thermal profiles assigned to heat exchangers - aborting")


print(f"Heating Network structure built: {len(junctions)} junctions, {len(pipes)} pipes, {len(loads)} loads")

####################################################################################################
# PLOTTING FUNCTIONS (defined early to be available for use)

def create_comprehensive_plots(results_df, hp_power_values, load_factors, ambient_temps_c=None, storage_soc_values=None, slack_power_values=None, non_flexible_load_p=None, flexible_load_p=None, electricity_price=None):
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

    # DEBUG: print summary stats for each BESS energy column before plotting to diagnose
    try:
        for col in bess_cols:
            arr = results_df[col].to_numpy()
            # Count negatives and get a small sample of indices where values < 0
            neg_idx = np.where(arr < 0)[0]
            neg_count = len(neg_idx)
            neg_sample = neg_idx.tolist()[:10]
            print(f"DEBUG plot: {col}: dtype={arr.dtype}, min={arr.min():.12g}, max={arr.max():.12g}, negatives={neg_count}, neg_indices_sample={neg_sample}")
        # Save the raw BESS columns to a CSV for offline inspection if needed
        try:
            results_df[bess_cols].to_csv('debug_bess_energy_plot.csv', index=False)
            print("DEBUG: Saved BESS energy columns to debug_bess_energy_plot.csv")
        except Exception as _e:
            print("DEBUG: Failed to save debug_bess_energy_plot.csv:", _e)
    except Exception as _e:
        print("DEBUG: Error while inspecting bess columns:", _e)

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
    else:
        # Fallback if no grid data found
        plt.plot(hours, load_factors, 'r-', linewidth=2, label='Normalized Load Factor')
        plt.title('Load Factor Profile Over Time (No Grid Data)', fontsize=14, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Load Factor (0-1)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
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
        # legend removed for transformer loading per user request
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
        # Accept scalars, lists, pandas Series, or numpy arrays and coerce to numpy array
        try:
            arr = np.asarray(electricity_price)
            # If a 0-d array (scalar), broadcast to match hours length
            if arr.ndim == 0:
                price_series = np.full(len(hours), float(arr))
            else:
                price_series = arr
        except Exception:
            price_series = None
    else:
        # Try fallback to module-level global if available
        price_series = globals().get('electricity_price', None)
        if price_series is not None:
            try:
                arr = np.asarray(price_series)
                if arr.ndim == 0:
                    price_series = np.full(len(hours), float(arr))
                else:
                    price_series = arr
            except Exception:
                price_series = None

    # Check price_series safely and broadcast/truncate to match hp_power_values length
    try:
        if price_series is not None:
            # If price_series is not sized correctly, broadcast or truncate
            if getattr(price_series, 'ndim', 1) == 0:
                price_series = np.full(len(hours), float(price_series))
            if len(price_series) < len(hours):
                # pad with last value
                pad_val = float(price_series[-1]) if len(price_series) > 0 else 0.0
                price_series = np.concatenate([np.asarray(price_series), np.full(len(hours) - len(price_series), pad_val)])

        if price_series is not None and len(price_series) > 0:
            # Align/truncate to hours length
            price_to_plot = np.asarray(price_series)[:len(hours)]
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
    except Exception as e:
        # If anything goes wrong while handling price data, fallback to informative message
        print(f"Warning: failed to plot electricity price: {e}")
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
    # legend removed for comprehensive junction temperature overview per user request
    
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
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
    # legend removed for electrical bus voltages per user request
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
        
        # legends removed for line loading overview per user request
            
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
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
    plt.savefig('fully_coordinated_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All plots have been generated and saved!")

####################################################################################################
# IMPORT MASS FLOW CALCULATION (same as single-period)

def calculate_mass_flows_proper(network):
    """Mass flow calculation - same as single-period version"""
    mdot_global = network["pump"]["mass_flow_kg_s"]
    name_to_id = {v["name"]: k for k, v in network["junctions"].items()}
    id_to_name = {k: v["name"] for k, v in network["junctions"].items()}
    
    # Build NetworkX graph from pipe connections
    import networkx as nx
    G = nx.DiGraph()
    
    for pipe in network["pipes"]:
        f_name, t_name = pipe["from"], pipe["to"]
        f_id = name_to_id[f_name]
        t_id = name_to_id[t_name]
        G.add_edge(f_id, t_id, pipe_name=(f_name, t_name))
    
    # AUTOMATED PUMP DETECTION
    pump_outlet_id = network["pump"]["inlet_junction"]  # Pump outlet (where flow starts)
    pump_inlet_id = network["pump"]["return_junction"]   # Pump inlet (where flow returns)
    
    pump_outlet_name = id_to_name[pump_outlet_id]
    pump_inlet_name = id_to_name[pump_inlet_id]
    
    # AUTOMATED DISCOVERY OF COMPLETE CIRCULATION LOOP
    circulation_path = []
    
    def trace_complete_dhn_loop(start_junction_id, graph, id_to_name):
        """Automatically trace the complete DHN circulation loop"""
        loop_path = []
        visited = set()
        current = start_junction_id
        while current not in visited:
            visited.add(current)
            loop_path.append(current)
            current_name = id_to_name[current]
            
            # Find next junction in circulation
            successors = list(graph.successors(current))
            
            if not successors:
                break
                
            # Prioritize continuing on supply side, then close_loop connections
            next_junction = None
            
            # If we're on supply side, continue on supply side
            if 'supply' in current_name:
                supply_successors = [s for s in successors if 'supply' in id_to_name[s] and s not in visited]
                if supply_successors:
                    next_junction = supply_successors[0]
                else:
                    # Look for close_loop_return connection
                    loop_successors = [s for s in successors if 'close_loop_return' in id_to_name[s]]
                    if loop_successors:
                        next_junction = loop_successors[0]
            
            # If we're at close_loop_return, start return side
            elif 'close_loop_return' in current_name:
                # Find return side connections
                return_successors = [s for s in successors if 'return' in id_to_name[s] and s not in visited]
                if return_successors:
                    # Sort return junctions to find the right starting point
                    return_successors.sort(key=lambda x: id_to_name[x])
                    next_junction = return_successors[0]
            
            # If we're on return side, continue on return side or find heat pump
            elif 'return' in current_name:
                return_successors = [s for s in successors if 'return' in id_to_name[s] and s not in visited]
                if return_successors:
                    next_junction = return_successors[0]
                else:
                    # Look for heat pump or pump connection
                    other_successors = [s for s in successors if s not in visited]
                    if other_successors:
                        next_junction = other_successors[0]
            
            # Default: take first unvisited successor
            if next_junction is None and successors:
                unvisited = [s for s in successors if s not in visited]
                if unvisited:
                    next_junction = unvisited[0]
            
            if next_junction is None:
                break
                
            current = next_junction
            
            # Stop if we've completed the loop (returned to start or reached pump inlet)
            if current == start_junction_id or current == pump_inlet_id:
                visited.add(current)
                loop_path.append(current)
                break
        
        return loop_path
    
    # Trace the complete circulation loop automatically
    circulation_path = trace_complete_dhn_loop(pump_outlet_id, G, id_to_name)
    
    # Find junctions with flow controls (consumers) 
    consumer_flows = {}
    flow_control_junctions = set()
    
    if "flow_controls" in network:
        for (f_id, t_id), fc_data in network["flow_controls"].items():
            flow_control_junctions.add(f_id)  # Junction where flow control is located
            consumer_flows[f_id] = fc_data["mass_flow"]
    
    # AUTOMATED MASS FLOW CALCULATION USING DISCOVERED CIRCULATION PATH
    pipe_flows = {}
    
    # Split circulation path into supply and return sections
    supply_section = []
    return_section = []
    connection_section = []
    
    for jid in circulation_path:
        jname = id_to_name[jid]
        if 'supply' in jname and 'close_loop' not in jname:
            supply_section.append(jid)
        elif 'close_loop' in jname:
            connection_section.append(jid)
        elif 'return' in jname and 'heat_pump' not in jname:
            return_section.append(jid)
        elif 'heat_pump' in jname:
            connection_section.append(jid)
    
    # SUPPLY SIDE: Flow DECREASES as consumers take flow out
    current_flow = mdot_global
    
    for i in range(len(supply_section)):
        current_jid = supply_section[i]
        current_name = id_to_name[current_jid]
        
        # Check if current junction has consumers (flow controls)
        if current_jid in consumer_flows:
            consumed = consumer_flows[current_jid]
            current_flow -= consumed
        
        # Assign flow to pipe going to next junction
        if i < len(supply_section) - 1:
            next_jid = supply_section[i + 1]
            next_name = id_to_name[next_jid]
            pipe_key = (current_name, next_name)
            pipe_flows[pipe_key] = current_flow
    
    # CONNECTION: From last supply to first return via close_loop
    if supply_section and connection_section:
        last_supply_name = id_to_name[supply_section[-1]]
        first_connection_name = id_to_name[connection_section[0]]
        pipe_key = (last_supply_name, first_connection_name)
        pipe_flows[pipe_key] = current_flow
        
        # Through close_loop connection
        for i in range(len(connection_section) - 1):
            current_name = id_to_name[connection_section[i]]
            next_name = id_to_name[connection_section[i + 1]]
            pipe_key = (current_name, next_name)
            pipe_flows[pipe_key] = current_flow
    
    # RETURN SIDE: Flow INCREASES as consumers add flow back
    
    # Start with the minimal flow from close_loop connection
    return_flow = current_flow  # This is the reduced flow after all supply consumers
    
    for i in range(len(return_section)):
        current_jid = return_section[i]
        current_name = id_to_name[current_jid]
        
        # Check if current junction has consumers that add flow back
        # Find the corresponding supply junction for this return junction
        supply_junction_name = current_name.replace('_return', '_supply')
        supply_jid = name_to_id.get(supply_junction_name)
        
        if supply_jid and supply_jid in consumer_flows:
            added_back = consumer_flows[supply_jid]
            return_flow += added_back
        
        # Assign flow to pipe going to next junction
        if i < len(return_section) - 1:
            next_jid = return_section[i + 1]
            next_name = id_to_name[next_jid]
            pipe_key = (current_name, next_name)
            pipe_flows[pipe_key] = return_flow
    
    # Add the missing connection from close_loop_return to first return junction
    if connection_section and return_section:
        close_loop_return_name = 'close_loop_return'
        first_return_name = id_to_name[return_section[0]]
        pipe_key = (close_loop_return_name, first_return_name)
        pipe_flows[pipe_key] = current_flow
    
    # Final connection from last return to heat pump should have full flow
    if return_section and connection_section:
        last_return_name = id_to_name[return_section[-1]]
        heat_pump_name = 'heat_pump_return_intermediate'
        if heat_pump_name in name_to_id:
            pipe_key = (last_return_name, heat_pump_name)
            pipe_flows[pipe_key] = return_flow
            
            # From heat pump back to pump should also be full circulation flow
            pump_inlet_name = id_to_name[pump_inlet_id]
            if pump_inlet_name != heat_pump_name:
                heat_pump_pipe_key = (heat_pump_name, pump_inlet_name)
                pipe_flows[heat_pump_pipe_key] = mdot_global  # Full circulation flow
    
    # Add consumer pipes (to heat exchangers)
    consumer_pipe_count = 0
    for pipe in network["pipes"]:
        f_name, t_name = pipe["from"], pipe["to"]
        pipe_key = (f_name, t_name)
        
        if '_heatex_in' in t_name and pipe_key not in pipe_flows:
            # Consumer pipe - use flow control specification
            found_flow = False
            if "flow_controls" in network:
                for (f_id, t_id), fc_data in network["flow_controls"].items():
                    if fc_data["from_name"] == f_name and fc_data["to_name"] == t_name:
                        pipe_flows[pipe_key] = fc_data["mass_flow"]
                        consumer_pipe_count += 1
                        found_flow = True
                        break
            
            if not found_flow:
                pipe_flows[pipe_key] = 1.0  # Default
    
    return pipe_flows

# Calculate mass flows using proven method
pipe_flows = calculate_mass_flows_proper(network_dict)

# print(f"\n" + "="*80)
# print("PIPE TRANSPORT DELAY ANALYSIS WITH PRECALCULATED COEFFICIENTS")
# print("="*80)

# Analyze pipe characteristics to understand why enhanced model shows similar results
delta_t_s = 3600  # 1 hour timestep
rho = 1000  # kg/m³
c_w = 4188  # J/kg.K

# print(f"Timestep: {delta_t_s} seconds ({delta_t_s/3600:.1f} hours)")
# print(f"Water density: {rho} kg/m³")
# print(f"Specific heat: {c_w} J/kg.K")
# print()

total_pipe_volume = 0
total_water_mass = 0
significant_delay_pipes = 0
pipe_analysis = []

for pipe in network_dict["pipes"]:
    f_name, t_name = pipe["from"], pipe["to"]
    length_m = pipe["length_m"]
    diameter_m = pipe["diameter_m"]
    u_w_per_m2k = pipe.get("u_w_per_m2k", 10.0)  # Default if missing
    
    # Get mass flow for this pipe
    pipe_key = (f_name, t_name)
    mdot_pipe = pipe_flows.get(pipe_key, 1.0)  # Default if not found
    
    # Calculate pipe characteristics
    pipe_volume = np.pi * (diameter_m/2)**2 * length_m  # m³
    water_mass_in_pipe = pipe_volume * rho  # kg
    residence_time_s = water_mass_in_pipe / mdot_pipe if mdot_pipe > 0 else 3600  # seconds
    transport_delay_timesteps = residence_time_s / delta_t_s
    gamma_p = int(transport_delay_timesteps)
    
    # Thermal analysis
    thermal_mass = water_mass_in_pipe * c_w  # J/K
    pipe_surface_area = np.pi * diameter_m * length_m  # m²
    heat_loss_coeff = u_w_per_m2k * pipe_surface_area  # W/K
    thermal_time_constant = thermal_mass / heat_loss_coeff if heat_loss_coeff > 0 else float('inf')  # seconds
    
    # Heat retention factor
    lambda_val = u_w_per_m2k * np.pi * diameter_m
    W = np.exp(-length_m * lambda_val / (mdot_pipe * c_w)) if mdot_pipe > 0 else 0
    
    total_pipe_volume += pipe_volume
    total_water_mass += water_mass_in_pipe
    
    if transport_delay_timesteps >= 1:
        significant_delay_pipes += 1
    
    pipe_analysis.append({
        'name': f"{f_name}->{t_name}",
        'length_m': length_m,
        'diameter_m': diameter_m,
        'mdot_kg_s': mdot_pipe,
        'volume_m3': pipe_volume,
        'water_mass_kg': water_mass_in_pipe,
        'residence_time_s': residence_time_s,
        'transport_delay_timesteps': transport_delay_timesteps,
        'gamma_p': gamma_p,
        'thermal_time_constant_s': thermal_time_constant,
        'heat_retention_W': W,
        'heat_loss_coeff_W_K': heat_loss_coeff
    })

# Sort by transport delay for easier analysis
pipe_analysis.sort(key=lambda x: x['transport_delay_timesteps'], reverse=True)

# print(f"PIPE TRANSPORT DELAY SUMMARY:")
# print(f"Total network pipe volume: {total_pipe_volume:.1f} m³")
# print(f"Total water mass in pipes: {total_water_mass:.0f} kg")
# print(f"Pipes with significant delay (≥1 timestep): {significant_delay_pipes}/{len(network_dict['pipes'])}")
# print()

# print(f"TOP 10 PIPES BY TRANSPORT DELAY:")
# print(f"{'Pipe':<30} {'L(m)':<6} {'D(m)':<8} {'Flow(kg/s)':<10} {'τ(min)':<8} {'γ':<4} {'τ_th(min)':<10} {'W':<6}")
# print(f"-" * 85)

for i, pipe in enumerate(pipe_analysis[:10]):
    residence_min = pipe['residence_time_s'] / 60
    thermal_min = pipe['thermal_time_constant_s'] / 60 if pipe['thermal_time_constant_s'] != float('inf') else 999
    # print(f"{pipe['name']:<30} {pipe['length_m']:<6.0f} {pipe['diameter_m']:<8.3f} {pipe['mdot_kg_s']:<10.1f} "
    #       f"{residence_min:<8.1f} {pipe['gamma_p']:<4} {thermal_min:<10.1f} {pipe['heat_retention_W']:<6.3f}")

# print()
# print(f"ANALYSIS INSIGHTS:")
# print(f"1. Transport delay vs timestep:")
# max_delay = max(p['transport_delay_timesteps'] for p in pipe_analysis)
# avg_delay = np.mean([p['transport_delay_timesteps'] for p in pipe_analysis])
# print(f"   - Maximum transport delay: {max_delay:.2f} timesteps ({max_delay*60:.0f} minutes)")
# print(f"   - Average transport delay: {avg_delay:.2f} timesteps ({avg_delay*60:.0f} minutes)")
# print(f"   - Pipes with delay ≥ 1 timestep: {significant_delay_pipes}/{len(pipe_analysis)} ({significant_delay_pipes/len(pipe_analysis)*100:.1f}%)")

# print(f"2. Thermal time constants:")
# finite_thermal = [p['thermal_time_constant_s'] for p in pipe_analysis if p['thermal_time_constant_s'] != float('inf')]
# if finite_thermal:
#     avg_thermal = np.mean(finite_thermal) / 60  # minutes
#     max_thermal = max(finite_thermal) / 60  # minutes
#     print(f"   - Average thermal time constant: {avg_thermal:.1f} minutes")
#     print(f"   - Maximum thermal time constant: {max_thermal:.1f} minutes")

# print(f"3. Heat retention factors:")
# heat_retentions = [p['heat_retention_W'] for p in pipe_analysis]
# avg_retention = np.mean(heat_retentions)
# min_retention = min(heat_retentions)
# print(f"   - Average heat retention factor W: {avg_retention:.3f}")
# print(f"   - Minimum heat retention factor W: {min_retention:.3f} (most heat loss)")

# print(f"4. Why enhanced model may show similar results:")
# if max_delay < 1.0:
#     print(f"   - No pipes have significant transport delay (≥1 timestep)")
#     print(f"   - All pipes use original instantaneous model")
# elif significant_delay_pipes < 3:
#     print(f"   - Only {significant_delay_pipes} pipes have significant delays")
#     print(f"   - Most of the network uses instantaneous heat transfer")
# if avg_retention > 0.8:
#     print(f"   - High average heat retention ({avg_retention:.3f}) means low heat loss during transport")
#     print(f"   - Temperature changes are primarily due to transport delay, not heat loss")
# if avg_thermal < 60:
#     print(f"   - Short thermal time constants ({avg_thermal:.1f} min) relative to 1-hour timestep")
#     print(f"   - Pipes reach thermal equilibrium quickly within each timestep")

def calculate_precalculated_transfer_coefficients(network_dict, pipe_flows, NUM_PERIODS, delta_t_s, rho, c_w):
    """
    Calculate precalculated transfer delay coefficients K^S_ℓ,k,t for each pipe
    Based on the node method from Weiye Zheng's approach
    
    Returns:
    - K_coefficients: dict with pipe_key -> {(k,t): coefficient_value}
    - W_insulation: dict with pipe_key -> insulation_factor  
    - pipe_delay_info: dict with analysis information for each pipe
    """
    # print(f"\n" + "="*60)
    # print("CALCULATING PRECALCULATED TRANSFER DELAY COEFFICIENTS")
    # print("="*60)
    
    K_coefficients = {}  # K^S_ℓ,k,t coefficients
    W_insulation = {}    # W^ins_ℓ insulation factors
    pipe_delay_info = {}
    
    for pipe in network_dict["pipes"]:
        f_name, t_name = pipe["from"], pipe["to"]
        pipe_key = (f_name, t_name)
        
        # Get pipe properties
        length_m = pipe["length_m"]
        diameter_m = pipe["diameter_m"]
        u_w_per_m2k = pipe.get("u_w_per_m2k", 10.0)  # Default if missing
        
        # Get mass flow for this pipe
        mdot_pipe = pipe_flows.get(pipe_key, 1.0)  # Default if not found
        
        # Calculate transport delay parameters
        pipe_volume = np.pi * (diameter_m/2)**2 * length_m  # m³
        water_mass_in_pipe = pipe_volume * rho  # kg
        residence_time_s = water_mass_in_pipe / mdot_pipe if mdot_pipe > 0 else 3600  # seconds
        transport_delay_timesteps = residence_time_s / delta_t_s
        
        # Calculate integer and fractional delay parts
        gamma_p = int(transport_delay_timesteps)  # Integer delay timesteps
        alpha_p = transport_delay_timesteps - gamma_p  # Fractional part
        
        # Calculate insulation factor W^ins_ℓ (heat retention during transport)
        lambda_val = u_w_per_m2k * np.pi * diameter_m  # Heat loss parameter
        W_ins = np.exp(-length_m * lambda_val / (mdot_pipe * c_w)) if mdot_pipe > 0 else 0
        W_insulation[pipe_key] = W_ins
        
        # Store pipe delay information
        pipe_delay_info[pipe_key] = {
            'residence_time_s': residence_time_s,
            'transport_delay_timesteps': transport_delay_timesteps,
            'gamma_p': gamma_p,
            'alpha_p': alpha_p,
            'W_ins': W_ins,
            'volume_m3': pipe_volume,
            'mdot_kg_s': mdot_pipe
        }
        
        # Calculate K^S_ℓ,k,t coefficients for each time period
        K_pipe = {}
        
        for t in range(NUM_PERIODS):
            # For each time t, determine which previous inlet temperatures contribute
            # Based on the transport delay, we look back gamma_p timesteps with interpolation
            
            if gamma_p == 0:
                # No significant delay - current timestep only
                K_pipe[(t, t)] = 1.0
                
            elif t >= gamma_p:
                # Sufficient history available
                if alpha_p < 0.01:  # Negligible fractional part
                    K_pipe[(t-gamma_p, t)] = 1.0
                else:
                    # Linear interpolation between two timesteps
                    if t >= gamma_p + 1:
                        K_pipe[(t-gamma_p, t)] = 1.0 - alpha_p  # Weight for more recent
                        K_pipe[(t-gamma_p-1, t)] = alpha_p       # Weight for older
                    else:
                        # Not enough history for full interpolation
                        K_pipe[(0, t)] = 1.0  # Use earliest available
                        
            else:
                # Early timesteps - use current timestep (boundary condition)
                K_pipe[(t, t)] = 1.0
                
        K_coefficients[pipe_key] = K_pipe
    
    # print(f"Calculated coefficients for {len(K_coefficients)} pipes")
    # print(f"Max transport delay: {max(info['gamma_p'] for info in pipe_delay_info.values())} timesteps")
    # print(f"Pipes with significant delay (≥1 timestep): {sum(1 for info in pipe_delay_info.values() if info['gamma_p'] >= 1)}")
    
    return K_coefficients, W_insulation, pipe_delay_info

# Calculate precalculated transfer delay coefficients for enhanced pipe model
K_coefficients, W_insulation, pipe_delay_info = calculate_precalculated_transfer_coefficients(
    network_dict, pipe_flows, NUM_PERIODS, delta_t_s, rho, c_w
)

####################################################################################################
# MULTI-PERIOD GUROBI OPTIMIZATION
#####################################################################################################


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


def solve_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb):
    #variance_net= gd.setup_grid_IAS_variance(season)
    #var_results = calculate_variance_propagation(time_steps, variance_net)

    #k_epsilon = np.sqrt((1 - par.epsilon) / par.epsilon)


    pd.set_option('display.precision', 10)
    model = gp.Model("opf_with_ldf_lc")

    # Define the costs
    curtailment_cost = electricity_price
    #storage_cost = par.c_cost
    r = 0.05 #interest rate
    n= 20 #lifetime of the storage

    #storage_cost_levelized = storage_cost * ((r*(1+r)**n) / (((1+r)**n) - 1))
    #print(f"Levelized Storage Cost: {storage_cost_levelized}")

    bess_eff = 0.95  # Round-trip efficiency
    bess_initial_soc = 0.5  # Initial state of charge as a percentage of capacity
    bess_capacity_mwh = 0.25  # BESS capacity in MWh
    bess_cost_per_mwh = 5.1 # Cost per MWh of BESS capacity

    ### Define the variables ###
    epsilon = 100e-9  # Small positive value to ensure some external grid usage

    # Extract transformer capacity in MW (assuming sn_mva is in MVA)
    transformer_capacity_mw = net.trafo['sn_mva'].values[0]
    #print(f"Transformer Capacity: {transformer_capacity_mw}")


    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
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
    pf_household = 0.99  # households (non-flexible)
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

    # Validation: print a direct comparison for non-flexible electrical loads
    # print('\nVALIDATION: comparing non-flexible electrical loads (network) to CSV values (first 5 timesteps)')
    # sample_ts = list(range(min(5, NUM_PERIODS)))
    # for load in net.load.itertuples():
    #     # skip flexible and HP loads
    #     if getattr(load, 'controllable', False):
    #         continue
    #     lname = getattr(load, 'name', '')
    #     if not lname or str(lname).upper().startswith('HP'):
    #         continue
    #     # base static p_mw from network (MW)
    #     base_p_mw = getattr(load, 'p_mw', None)
    #     csv_series = None
    #     if 'electrical_time_series' in globals() and lname in electrical_time_series:
    #         csv_series = electrical_time_series[lname]

    #     if csv_series is None:
    #         print(f"{lname} (bus {load.bus}): network base p_mw={base_p_mw:.6f} MW | CSV: MISSING -> will use zero-series")
    #     else:
    #         # ensure numpy array
    #         arr = np.array(csv_series)
    #         cols = []
    #         for t in sample_ts:
    #             if t < len(arr):
    #                 cols.append(f"t{t}: csv={arr[t]:.2f} kW (used={arr[t]/1000.0:.6f} MW)")
    #             else:
    #                 cols.append(f"t{t}: csv=MISSING")
    #         print(f"{lname} (bus {load.bus}): network base p_mw={base_p_mw:.6f} MW | " + ", ".join(cols))

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
                name=f'p_hp_{t}'
            )
                            
       
    
    non_slack_buses = [bus for bus in net.bus.index if bus != slack_bus_index]

    
    V_vars = model.addVars(time_steps, net.bus.index, name="V")
    V_reduced_vars = model.addVars(time_steps, non_slack_buses, name="V_reduced")
    # Set slack bus voltage to 1.0 p.u. at all time steps
    for t in time_steps:
        model.addConstr(V_vars[t, slack_bus_index] == 1, name=f"slack_voltage_fixed_{t}")

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

                model.addConstr(V_vars[t, bus] >= tight_v_min, name=f"voltage_min_drcc_{t}_{bus}")
                model.addConstr(V_vars[t, bus] <= tight_v_max, name=f"voltage_max_drcc_{t}_{bus}")
        

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
            model.addQConstr(
                P_branch_vars[t, line_idx]*P_branch_vars[t, line_idx] +
                Q_branch_vars[t, line_idx]*Q_branch_vars[t, line_idx]
                <= S_branch_limit**2,
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
        
            #tight_trafo_limit = par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("S_trafo", trafo_idx)]
            #S_limit = 0.8*S_rated - tight_trafo_limit

            S_limit = 0.8*S_rated 

            model.addQConstr(
                P_trafo_vars[t, trafo_idx]*P_trafo_vars[t, trafo_idx] +
                Q_trafo_vars[t, trafo_idx]*Q_trafo_vars[t, trafo_idx]
                <= S_limit**2,
                name=f"S_trafo_limit_{t}_{trafo_idx}"
            )


            # model.addConstr(S_trafo_vars <= (0.8*S_rated)-tight_trafo_limit, name=f"S_trafo_limit_{t}_{trafo_idx}")

            #model.addConstr(transformer_loading_perc_vars[t, trafo_idx] == (S_trafo_vars / S_rated) * 100, name=f"trafo_loading_{t}_{trafo_idx}")


        #     model.addConstr(
        #     transformer_loading_perc_vars[t, trafo_idx] <= tight_trafo_limit,  # Enforce 80% limit
        #     name=f"trafo_loading_limit_{t}_{trafo_idx}"
        # )

        # External Grid Balance
        model.addConstr(ext_grid_P_net[t] == -gp.quicksum(P_trafo_vars[t, i] for i in net.trafo.index),
                        name=f"P_balance_slack_{t}")
        model.addConstr(ext_grid_Q_net[t] == -gp.quicksum(Q_trafo_vars[t, i] for i in net.trafo.index),
                        name=f"Q_balance_slack_{t}")
        # Create Gurobi model
    #model = gp.Model("DHN_Heat_Pump_Multi_Period_Optimization")
    
    # Map junction names to IDs
    name_to_id = {v["name"]: k for k, v in network_dict["junctions"].items()}
    id_to_name = {k: v["name"] for k, v in network_dict["junctions"].items()}
    
    # Create temperature variables for ALL junctions and ALL time periods
    all_junctions = list(network_dict["junctions"].keys())
    T_junction = model.addVars(all_junctions, time_steps, name="T_junction", 
                              lb=273.15, ub=400)  # Kelvin: 0°C to 127°C
    
    # Heat pump heat input as control variable for each time period
    Q_hp = model.addVars(time_steps, name="Q_hp", lb=0, ub=800000)  # Heat pump heat input [W]

    # SLACK HEAT SOURCE VARIABLES (at circulation pump outlet to meet 85°C constraint)
    # This is the heat source that the optimization should minimize
    Q_slack = model.addVars(time_steps, name="Q_slack", lb=0, ub=GRB.INFINITY)  # Slack heat input [W] at bus_15_supply
    
    # THERMAL STORAGE VARIABLES (Series Configuration with Fixed Temperature)
    # Storage is connected in series: Heat Pump → Storage → Circulation Pump
    # SIMPLIFICATION: Storage temperature is fixed at 85°C to avoid complexity
    Q_storage = model.addVars(time_steps, name="Q_storage", lb=-1e8, ub=10e8)  # Storage heat transfer [W]: +charge, -discharge, 0=pass-through
    E_storage = model.addVars(time_steps, name="E_storage", lb=0, ub=GRB.INFINITY)     # Storage energy content [J]
    T_storage_in = model.addVars(time_steps, name="T_storage_in", lb=273.15, ub=400)   # Storage inlet temperature [K] (from heat pump)
    T_storage_out = model.addVars(time_steps, name="T_storage_out", lb=273.15, ub=400) # Storage outlet temperature [K] (to circulation pump)
    
    # ENHANCED PIPE MODEL: Lossless outlet temperature variables
    # θ̃^out_ℓ,t - intermediate lossless outlet temperatures before heat loss correction
    # Only create variables for actual physical pipes, not heat exchangers/pumps/storage
    actual_pipe_keys = [(pipe["from"], pipe["to"]) for pipe in network_dict["pipes"]]
    T_pipe_lossless = model.addVars(actual_pipe_keys, time_steps, name="T_pipe_lossless", 
                                   lb=273.15, ub=400)  # Lossless pipe outlet temperatures [K]
    
    # CRITICAL: Update model after adding variables
    model.update()
    print(f"Added variables: {len(all_junctions) * NUM_PERIODS} temperature vars + {NUM_PERIODS} heat pump vars + {NUM_PERIODS} slack vars + {4 * NUM_PERIODS} thermal storage vars + {len(actual_pipe_keys) * NUM_PERIODS} pipe vars")
    print(f"Total variables: {model.NumVars}")
    
    
    ####################################################################################################
    # THERMAL STORAGE PARAMETERS (Series Configuration)
    
    # Storage parameters
    STORAGE_CAPACITY_J = 5000e6        # Storage capacity: 5000 MJ (≈1390 kWh thermal)
    STORAGE_EFFICIENCY = 0.98         # Round-trip efficiency (charging/discharging)
    STORAGE_HEAT_LOSS_COEFF = 2/1e6       # Heat loss coefficient
    STORAGE_CP_J_KG_K = 4188         # Specific heat capacity of storage medium [J/kg.K]
    INITIAL_SOC = 0.5                # Initial state of charge (50%)
    STORAGE_TEMP_K = 85 + 273.15     # Fixed storage temperature [K] = 85°C (simplified assumption)
    
    print(f"Thermal Storage parameters:")
    print(f"  Capacity: {STORAGE_CAPACITY_J/1e6:.0f} MJ ({STORAGE_CAPACITY_J/3.6e6:.0f} kWh)")
    print(f"  Efficiency: {STORAGE_EFFICIENCY*100:.1f}%")
    print(f"  Heat loss coeff: {STORAGE_HEAT_LOSS_COEFF:.0f} W/K")
    print(f"  Initial SOC: {INITIAL_SOC*100:.0f}%")
    print(f"  Fixed storage temp: {STORAGE_TEMP_K - 273.15:.1f}°C")
    
    ####################################################################################################
    # IMPLEMENT MULTI-PERIOD DHN PHYSICS WITH THERMAL STORAGE
    
    # Constants (same as single-period, TEMPERATURES IN KELVIN)
    # NOTE: T_ambient_K is now time-varying, defined above as temp_profile_k[t]
    target_supply_temp_K = 85.0 + 273.15  # 85°C target in Kelvin
    
    print(f"Temperature units: Target = {target_supply_temp_K:.1f}K")
    print(f"Ambient temperature range: {temp_profile_k.min():.1f}K to {temp_profile_k.max():.1f}K")
    print(f"Ambient temperature variability: {temp_profile_k.max() - temp_profile_k.min():.1f}K range")
    
    # KEY INSIGHT: Each time period is independent (no thermal storage modeled)
    # For each time period, create the same constraints as single-period version
    
    pump_inlet_id = network_dict["pump"]["inlet_junction"]
    pump_return_id = network_dict["pump"]["return_junction"]
    
    # Print pump junction information for debugging
    print(f"Pump junction IDs:")
    print(f"  Circulation pump outlet (inlet_junction): {pump_inlet_id} -> {id_to_name.get(pump_inlet_id, 'Name not found')}")
    print(f"  Circulation pump inlet (return_junction): {pump_return_id} -> {id_to_name.get(pump_return_id, 'Name not found')}")
    
    # Heat pump junction
    heat_pump_name = 'heat_pump_return_intermediate'
    heat_pump_id = name_to_id[heat_pump_name] if heat_pump_name in name_to_id else None
    
    print(f"Processing {NUM_PERIODS} time periods...")
    
    # Find heat pump inlet once (outside the loop)
    heat_pump_inlet_id = None
    if heat_pump_id:
        for pipe in network_dict["pipes"]:
            if pipe["to"] == heat_pump_name:
                heat_pump_inlet_id = name_to_id[pipe["from"]]
                print(f"Heat pump inlet found: {pipe['from']} -> {pipe['to']}")
                break
        
        if heat_pump_inlet_id is None:
            # Check heat exchangers
            for load in network_dict["loads"].values():
                if load["to"] == heat_pump_id:
                    heat_pump_inlet_id = load["from"]
                    print(f"Heat pump inlet via HEX: {id_to_name[heat_pump_inlet_id]} -> {heat_pump_name}")
                    break
        
        if heat_pump_inlet_id is None:
            print(f"⚠️  WARNING: Could not find heat pump inlet junction!")
    
    # Initialize counters for pipe model types
    pipe_model_stats = {
        'close_loop': 0,
        'enhanced_precalc': 0,      # New: Using precalculated coefficients with transport delay
        'enhanced_fallback': 0,     # New: No delay, instantaneous transfer with precalc system
        'enhanced_no_coeff': 0,     # New: No coefficients found, using fallback model
        'connection_instant': 0,    # New: Non-pipe connections (HEX, pumps, storage)
        'flow_control': 0,
        'skipped_mixing': 0
    }
    
    for t in time_steps:
        
        # Heat pump constraint for time period t
        if heat_pump_id:
            if heat_pump_inlet_id:
                # Heat pump adds heat: T_out = T_in + Q_hp[t] / (mdot * cp)
                model.addConstr(
                    T_junction[heat_pump_id, t] == T_junction[heat_pump_inlet_id, t] + Q_hp[t] / (mdot_kg_s * c_w),
                    name=f"heat_pump_boost_t{t}"
                )
            else:
                # Fallback: heat pump starts from ambient temperature
                model.addConstr(
                    T_junction[heat_pump_id, t] == temp_profile_k[t] + Q_hp[t] / (mdot_kg_s * c_w),
                    name=f"heat_pump_boost_from_ambient_t{t}"
                )
            
            # THERMAL STORAGE CONSTRAINTS (Series Configuration)
            # Series connection: Heat Pump → Storage → Circulation Pump
            
            # 1. Storage inlet temperature = Heat pump outlet temperature
            model.addConstr(
                T_storage_in[t] == T_junction[heat_pump_id, t],
                name=f"storage_inlet_connection_t{t}"
            )
            
            # 2. Storage energy balance implementing your specified equation
            if t == 0:
                # Initial condition: start with initial SOC and NO storage action
                initial_energy_J = INITIAL_SOC * STORAGE_CAPACITY_J
                model.addConstr(
                    E_storage[t] == initial_energy_J,
                    name=f"storage_energy_initial_t{t}"
                )
                
                # FIXED: Force Q_storage[0] = 0 to prevent "free" discharge at timestep 0
                model.addConstr(
                    Q_storage[t] == 0,
                    name=f"storage_no_action_t{t}"
                )
                
                # initial storage constraint info suppressed
            else:
                heat_loss_power = STORAGE_HEAT_LOSS_COEFF * E_storage[t-1] * (STORAGE_TEMP_K - temp_profile_k[t]) / STORAGE_CAPACITY_J
                
                model.addConstr(
                    E_storage[t] == E_storage[t-1] + 
                    Q_storage[t] * STORAGE_EFFICIENCY * 3600 - 
                    heat_loss_power * 3600,
                    name=f"storage_energy_balance_t{t}"
                )
                
                # energy balance debug suppressed for brevity
            
            # 3. Storage capacity limits
            model.addConstr(
                E_storage[t] <= STORAGE_CAPACITY_J,
                name=f"storage_capacity_upper_t{t}"
            )
            
            model.addConstr(
                E_storage[t] >= 0,
                name=f"storage_capacity_lower_t{t}"
            )
            
            # 4. Storage outlet temperature (Series flow with heat exchange)
            model.addConstr(
                T_storage_out[t] == T_storage_in[t] - Q_storage[t] / (mdot_kg_s * c_w),
                name=f"storage_outlet_temperature_t{t}"
            )
            
            # 5. Circulation pump inlet = Storage outlet + Slack heat (series connection)
            model.addConstr(
                T_junction[pump_inlet_id, t] == T_storage_out[t] + Q_slack[t] / (mdot_kg_s * c_w),
                name=f"circulation_pump_connection_with_slack_t{t}"
            )
        
        else:
            # No heat pump - direct connection (fallback case)
            model.addConstr(
                T_junction[pump_inlet_id, t] == temp_profile_k[t],
                name=f"direct_pump_connection_t{t}"
            )
        
        # Create pipe constraints for time period t (same logic as single-period)
        mixing_return_junctions = set()
        for load in network_dict["loads"].values():
            j_to = load["to"] 
            to_name = id_to_name[j_to]
            if '_return' in to_name and j_to != heat_pump_id:
                mixing_return_junctions.add(j_to)
        
        # Pipe heat transfer constraints
        for pipe_key, mdot_pipe in pipe_flows.items():
            f_name, t_name = pipe_key
            f_id = name_to_id[f_name]
            t_id = name_to_id[t_name]
            
            # Skip pipes ending at mixing return junctions
            if t_id in mixing_return_junctions:
                if t == 0:  # Count only once
                    pipe_model_stats['skipped_mixing'] += 1
                continue
            
            # Find the corresponding pipe data
            pipe_data = None
            for pipe in network_dict["pipes"]:
                if pipe["from"] == f_name and pipe["to"] == t_name:
                    pipe_data = pipe
                    break
            
            if pipe_data is None:
                if f_name == 'bus_15_return' and t_name == 'heat_pump_return_intermediate':
                    continue  # Handled by heat pump constraint
                else:
                    if t == 0:  # Count only once
                        pipe_model_stats['flow_control'] += 1
                    continue  # Flow control connection
            
            if mdot_pipe > 0:
                # Handle close_loop connections (no heat loss)
                if 'close_loop' in f_name or 'close_loop' in t_name:
                    model.addConstr(
                        T_junction[t_id, t] == T_junction[f_id, t],
                        name=f"connection_{f_name}_to_{t_name}_t{t}"
                    )
                    if t == 0:  # Count only once
                        pipe_model_stats['close_loop'] += 1
                else:
                    # ENHANCED PIPE MODEL WITH PRECALCULATED TRANSFER DELAY COEFFICIENTS
                    # Based on node method with θ̃^out_ℓ,t and θ^out_ℓ,t formulation
                    
                    pipe_key = (f_name, t_name)
                    
                    # Check if this is an actual pipe (has lossless temperature variable)
                    if pipe_key in T_pipe_lossless:
                        # Get precalculated coefficients and insulation factor for this pipe
                        K_pipe = K_coefficients.get(pipe_key, {})
                        W_ins = W_insulation.get(pipe_key, 1.0)  # Default no heat loss
                        delay_info = pipe_delay_info.get(pipe_key, {})
                        
                        # Step 1: Calculate lossless outlet temperature θ̃^out_ℓ,t
                        # θ̃^out_ℓ,t = Σ_{k=t-t^lb} K^S_ℓ,k,t · θ^in_ℓ,k
                        if K_pipe:
                            # Create expression for weighted sum of inlet temperatures
                            lossless_temp_expr = 0
                            has_valid_coeffs = False
                            for (k_time, t_time), coeff in K_pipe.items():
                                if t_time == t and coeff > 1e-6:  # Only include non-zero coefficients for current time t
                                    lossless_temp_expr += coeff * T_junction[f_id, k_time]
                                    has_valid_coeffs = True
                            
                            if has_valid_coeffs:  # Valid expression found
                                # Constraint: θ̃^out_ℓ,t = Σ K^S_ℓ,k,t · θ^in_ℓ,k
                                model.addConstr(
                                    T_pipe_lossless[pipe_key][t] == lossless_temp_expr,
                                    name=f"pipe_lossless_temp_{f_name}_to_{t_name}_t{t}"
                                )
                                
                                # Step 2: Calculate actual outlet temperature with heat loss
                                # θ^out_ℓ,t = θ_amb + W^ins_ℓ · (θ̃^out_ℓ,t - θ_amb)
                                model.addConstr(
                                    T_junction[t_id, t] == temp_profile_k[t] + W_ins * (T_pipe_lossless[pipe_key][t] - temp_profile_k[t]),
                                    name=f"pipe_outlet_temp_{f_name}_to_{t_name}_t{t}"
                                )
                                
                                if t == 0:  # Count only once
                                    pipe_model_stats['enhanced_precalc'] = pipe_model_stats.get('enhanced_precalc', 0) + 1
                                    # Debug info
                                    gamma_p = delay_info.get('gamma_p', 0)
                                    alpha_p = delay_info.get('alpha_p', 0)
                                    residence_time_s = delay_info.get('residence_time_s', 0)
                                    print(f"Pipe {f_name}->{t_name}: K coefficients={len([c for c in K_pipe.values() if abs(c) > 1e-6])}, "
                                          f"γ={gamma_p}, α={alpha_p:.3f}, W_ins={W_ins:.3f}, τ={residence_time_s:.0f}s")
                            else:
                                # Fallback: no delay, instantaneous transfer
                                model.addConstr(
                                    T_junction[t_id, t] == temp_profile_k[t] + W_ins * (T_junction[f_id, t] - temp_profile_k[t]),
                                    name=f"pipe_instant_{f_name}_to_{t_name}_t{t}"
                                )
                                if t == 0:
                                    pipe_model_stats['enhanced_fallback'] = pipe_model_stats.get('enhanced_fallback', 0) + 1
                        else:
                            # Fallback: no coefficients found, use instantaneous model
                            model.addConstr(
                                T_junction[t_id, t] == temp_profile_k[t] + W_ins * (T_junction[f_id, t] - temp_profile_k[t]),
                                name=f"pipe_no_coeff_{f_name}_to_{t_name}_t{t}"
                            )
                            if t == 0:
                                pipe_model_stats['enhanced_no_coeff'] = pipe_model_stats.get('enhanced_no_coeff', 0) + 1
                    else:
                        # This is not an actual pipe (probably heat exchanger, pump, or storage connection)
                        # Use simple instantaneous model without lossless temperature variable
                        W_ins = W_insulation.get(pipe_key, 1.0)  # Get insulation factor if available
                        model.addConstr(
                            T_junction[t_id, t] == temp_profile_k[t] + W_ins * (T_junction[f_id, t] - temp_profile_k[t]),
                            name=f"connection_instant_{f_name}_to_{t_name}_t{t}"
                        )
                        if t == 0:
                            pipe_model_stats['connection_instant'] = pipe_model_stats.get('connection_instant', 0) + 1
                        
        
        # Flow control constraints (no temperature drop)
        for (f_id, t_id_fc), fc_data in network_dict["flow_controls"].items():
            model.addConstr(
                T_junction[t_id_fc, t] == T_junction[f_id, t],
                name=f"flow_control_{fc_data['from_name']}_to_{fc_data['to_name']}_t{t}"
            )
        
        # Heat exchanger constraints with TIME-VARYING LOADS
        for load in network_dict["loads"].values():
            j_from = load["from"]
            j_to = load["to"] 
            from_name = id_to_name[j_from]
            to_name = id_to_name[j_to]
            
            # Skip the heat pump
            if j_to == heat_pump_id:
                continue
                
            # Apply time-varying temperature drop
            if '_return' in to_name and '_heatex_in' in from_name:
                # Scale the base load by the load profile
                qext_base = load["qext_base"]
                qext_t = qext_base * load_profile[t]  # Time-varying load
                
                mdot_hex = load["mass_flow"]
                delta_T_K = qext_t / (mdot_hex * c_w)
                
                # Check if this return junction needs mixing
                if j_to in mixing_return_junctions:
                    # Find pipe input to this return junction
                    pipe_from_id = None
                    pipe_mdot = 0
                    
                    for pipe_key, flow in pipe_flows.items():
                        pf_name, pt_name = pipe_key
                        if pt_name == to_name:
                            pipe_from_id = name_to_id[pf_name]
                            pipe_mdot = flow
                            break
                    
                    if pipe_from_id is not None:
                        # Mixing constraint with time-varying loads
                        total_mdot = mdot_hex + pipe_mdot
                        
                        model.addConstr(
                            total_mdot * T_junction[j_to, t] + mdot_hex * delta_T_K == 
                            mdot_hex * T_junction[j_from, t] + 
                            pipe_mdot * T_junction[pipe_from_id, t],
                            name=f"mixing_{to_name}_t{t}"
                        )
                    else:
                        # Fallback to simple heat exchanger drop
                        model.addConstr(
                            T_junction[j_to, t] == T_junction[j_from, t] - delta_T_K,
                            name=f"heat_exchanger_drop_{from_name}_to_{to_name}_t{t}"
                        )
                else:
                    # Simple heat exchanger drop
                    model.addConstr(
                        T_junction[j_to, t] == T_junction[j_from, t] - delta_T_K,
                        name=f"heat_exchanger_drop_{from_name}_to_{to_name}_t{t}"
                    )
        
        # Target temperature constraint for each time period
        # CORRECTED: Target temperature should be at circulation pump outlet (first supply node) with slack heat source
        # The slack heat source Q_slack adds heat directly at pump outlet to meet 85°C constraint
        # Temperature rise from slack: ΔT = Q_slack / (mdot_circ_pump * cp)
        
        # Use pump_inlet_id directly - this is the circulation pump outlet (first supply node)
        target_junction_id = pump_inlet_id
        
        # Target constraint creation (verbose debug suppressed)
        
        # Find the input temperature to pump outlet (from storage outlet via circulation pump)
        # pump_outlet = T_circulation_pump_inlet + Q_slack / (mdot * cp)
        # T_circulation_pump_inlet = T_storage_out (from series storage configuration)
        pump_inlet_temp = T_storage_out[t]  # Temperature from storage outlet to circulation pump
        
        try:
            # internal debug details suppressed
            
            # FIXED: Only constraint needed - temperature must meet target
            # (Junction temperature is already set by circulation pump connection constraint above)
            constr1 = model.addConstr(
                T_junction[target_junction_id, t] >= target_supply_temp_K,
                name=f"target_temperature_constraint_t{t}"
            )
            
            # target constraint added for period t
                
        except Exception as e:
            print(f"  ✗ ERROR adding target temperature constraints for period {t}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # THERMAL STORAGE CYCLIC CONSTRAINT
    # Ensure storage ends at the same SOC it started (50%) for energy balance
    final_period = NUM_PERIODS - 1
    model.addConstr(
        E_storage[final_period] == INITIAL_SOC * STORAGE_CAPACITY_J,
        name="storage_cyclic_energy_balance"
    )
    print(f"Added cyclic constraint: Storage must end at {INITIAL_SOC*100:.0f}% SOC")
    
    # Print pipe model statistics
    # print(f"\nPIPE MODEL STATISTICS:")
    # print(f"=" * 40)
    # total_pipes = sum(pipe_model_stats.values())
    # for model_type, count in pipe_model_stats.items():
    #     percentage = (count / total_pipes * 100) if total_pipes > 0 else 0
    #     print(f"{model_type.replace('_', ' ').title()}: {count} pipes ({percentage:.1f}%)")
    # print(f"Total pipes processed: {total_pipes}")
    
    # # Analysis of enhanced precalculated coefficient model impact
    # enhanced_precalc_pipes = pipe_model_stats.get('enhanced_precalc', 0)
    # enhanced_fallback_pipes = pipe_model_stats.get('enhanced_fallback', 0)
    # enhanced_no_coeff_pipes = pipe_model_stats.get('enhanced_no_coeff', 0)
    # print(f"\nENHANCED PRECALCULATED COEFFICIENT MODEL IMPACT ASSESSMENT:")
    # if enhanced_precalc_pipes > 0:
    #     print(f"✓ {enhanced_precalc_pipes} pipes use precalculated transfer delay coefficients")
    #     print(f"  These pipes have K^S_ℓ,k,t coefficients for weighted inlet temperature history")
    # if enhanced_fallback_pipes > 0:
    #     print(f"⚠️  {enhanced_fallback_pipes} pipes use instantaneous fallback (no significant delay)")
    # if enhanced_no_coeff_pipes > 0:
    #     print(f"⚠️  {enhanced_no_coeff_pipes} pipes had no coefficients and use basic fallback model")
    
    # total_enhanced = enhanced_precalc_pipes + enhanced_fallback_pipes + enhanced_no_coeff_pipes
    # if total_enhanced > 0:
    #     delay_fraction = enhanced_precalc_pipes / total_enhanced
    #     print(f"Transport delay significance: {delay_fraction*100:.1f}% of pipes have meaningful delays")
    # else:
    #     print(f"⚠️  No pipes processed with enhanced model - check coefficient calculation")
    
    # STORAGE RAMPING CONSTRAINTS to prevent aggressive cycling
    # Limit maximum storage power change between periods
    MAX_STORAGE_RAMP_RATE = 100000  # W/hour (100 kW per hour maximum change)
    #print(f"Adding storage ramping constraints: max {MAX_STORAGE_RAMP_RATE/1000:.1f} kW per hour")
    
    for t in range(1, NUM_PERIODS):
        model.addConstr(
            Q_storage[t] - Q_storage[t-1] <= MAX_STORAGE_RAMP_RATE,
            name=f"storage_ramp_up_limit_t{t}"
        )
        model.addConstr(
            Q_storage[t-1] - Q_storage[t] <= MAX_STORAGE_RAMP_RATE,
            name=f"storage_ramp_down_limit_t{t}"
        )
    
    # OPTIONAL: Add ramping constraints for heat pump (DISABLED for testing)
    # MAX_RAMP_RATE = 20000  # W/hour (20 kW per hour - more conservative for hourly intervals)
    # print(f"Adding ramping constraints: max {MAX_RAMP_RATE/1000:.1f} kW per hour")
    
    # DISABLED: Ramping constraints to keep model identical to single-period
    # for t in range(1, NUM_PERIODS):
    #     model.addConstr(
    #         Q_hp[t] - Q_hp[t-1] <= MAX_RAMP_RATE,
    #         name=f"ramp_up_limit_t{t}"
    #     )
    #     model.addConstr(
    #         Q_hp[t-1] - Q_hp[t] <= MAX_RAMP_RATE,
    #         name=f"ramp_down_limit_t{t}"
    #     )
    
    #print("Ramping constraints disabled - using pure multi-period replication of single-period model")
    
    # DEBUG: Check if basic model is feasible
    #print(f"Checking model feasibility...")
    
    # Update model before optimization
    model.update()
    
    #print(f"\nFINAL MULTI-PERIOD MODEL SUMMARY:")
    #print(f"Variables: {model.NumVars}")
    #print(f"Constraints: {model.NumConstrs}")
    #print(f"Time periods: {NUM_PERIODS}")
    #print(f"Expected constraints per period: ~127 (from single-period)")
    #print(f"Actual average constraints per period: {model.NumConstrs / NUM_PERIODS:.1f}")
    
    # Count constraint types to compare with single-period
    constraint_names = [constr.ConstrName for constr in model.getConstrs()]
    constraint_types = {}
    
    for name in constraint_names:
        # Extract constraint type (before _t{number})
        if '_t' in name:
            constraint_type = name.split('_t')[0]
        else:
            constraint_type = name
        
        constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1
    
    #print(f"\nConstraint type breakdown (total across all periods):")
    #for ctype, count in sorted(constraint_types.items()):
    #    avg_per_period = count / NUM_PERIODS
    #    print(f"  {ctype}: {count} total ({avg_per_period:.1f} per period)")
    
    # Check for critical constraint types
    # critical_types = ['heat_pump_boost', 'circulation_pump_connection', 'target_temperature_constraint', 
    #                  'storage_inlet_connection', 'storage_energy_balance', 'storage_temperature', 'storage_outlet_temperature']
    # for ctype in critical_types:
    #     if ctype in constraint_types:
    #         expected_count = NUM_PERIODS
    #         actual_count = constraint_types[ctype]
    #         status = "✓" if actual_count == expected_count else "✗"
    #         print(f"  {status} {ctype}: {actual_count}/{expected_count}")
    #     else:
    #         print(f"  ✗ {ctype}: MISSING!")
    
    # Basic constraint checks completed (detailed debug suppressed)
    
    # Check if heat pump junction exists
    # if heat_pump_id:
    #     print(f"✓ Heat pump junction found: {heat_pump_name}")
    # else:
    #     print(f"✗ Heat pump junction missing: {heat_pump_name}")
    
    # Check target junction
    # target_junction_name = 'bus_15_supply'
    # if target_junction_name in name_to_id:
    #     target_junction_id = name_to_id[target_junction_name]
    #     print(f"✓ Target junction found: {target_junction_name}")
    # else:
    #     print(f"✗ Target junction missing: {target_junction_name}")
    
    # Check if we have reasonable temperature bounds
    # print(f"Temperature bounds: [{T_junction[list(all_junctions)[0], 0].LB:.1f}, {T_junction[list(all_junctions)[0], 0].UB:.1f}] K")
    # print(f"Target temperature: {target_supply_temp_K:.1f} K")
    
    if target_supply_temp_K > T_junction[list(all_junctions)[0], 0].UB:
        print(f"⚠️  WARNING: Target temperature exceeds variable upper bound!")
        # Increase upper bound
        for j_id in all_junctions:
            for t in time_steps:
                T_junction[j_id, t].UB = 450  # Increase to 177°C

    print(f"adding coupling constraint between dhn and electrical network...")
    # Coupling constraint: Electrical power consumption of heat pump
    # Parameters for COP formula (fixed per user)
    DELTA_THETA = 2.1966   # ΔΘ in K (updated per user)
    ETA_C0 = 0.6           # η^{C0} (Carnot efficiency fraction) - fixed

    # Precompute COP per timestep using the supply temperature (target_supply_temp_K)
    # COP_t = ETA_C0 * (theta_s + DELTA_THETA) / (theta_s - theta_amb_t + 2*DELTA_THETA)
    cop_profile = []
    for t in time_steps:
        # temp_profile_k[t] is ambient temperature in K; target_supply_temp_K is supply temp in K
        denom = (target_supply_temp_K - temp_profile_k[t] + 2.0 * DELTA_THETA)
        if denom == 0:
            cop_t = 1.0
        else:
            cop_t = ETA_C0 * (target_supply_temp_K + DELTA_THETA) / denom
        # enforce reasonable bounds
        cop_t = max(1.0, min(cop_t, 6.0))
        cop_profile.append(cop_t)

    print(f"Heat pump buses found: {hp_load_buses}")
    print(f"Number of heat pump buses: {len(hp_load_buses)}")

    for t in time_steps:
        if len(hp_load_buses) > 0:
            # Single heat pump - assign all thermal power to the first (and only) bus
            bus = hp_load_buses[0]

            # Use precomputed COP scalar for this timestep to keep constraint linear
            cop_t = cop_profile[t]
            model.addConstr(
                p_hp_vars[t][bus] == Q_hp[t] / (cop_t * 1e6),
                name=f"heat_pump_electrical_coupling_t{t}_bus{bus}"
            )

            if t < 3:
                print(f"  t={t}: Q_hp (thermal) -> p_hp_vars[bus_{bus}] using COP={cop_t:.3f}")

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

    # Determine timestep duration in hours for energy-based costs
    try:
        if 'time_index' in globals() and len(time_index) >= 2:
            dt_hours = max(1/60.0, (time_index[1] - time_index[0]).total_seconds() / 3600.0)
        else:
            dt_hours = 1.0
    except Exception:
        dt_hours = 1.0
    
    COST_SLACK = 100         # Cost per Wh of slack heat (expensive backup)
    
    total_slack_cost = gp.quicksum(Q_slack[t] * COST_SLACK / 1e6 for t in time_steps)
    
    print(f"COST-BASED Objective: Minimize total operational cost over {NUM_PERIODS} periods")

    electricity_cost =  gp.quicksum(electricity_price[t] * (ext_grid_import_P_vars[t] + ext_grid_export_P_vars[t]) * dt_hours for t in time_steps)
    bess_cost = gp.quicksum(bess_cost_per_mwh * (bess_charge_vars[t][bus] + bess_discharge_vars[t][bus]) * dt_hours for bus in bess_buses for t in time_steps) if len(bess_buses) > 0 else 0
    # Aggregate flexible curtailment cost (if any flexible curtail vars were created)
    y_cap_cost = gp.quicksum(2 * electricity_price[t] * flex_curtail_P_vars[t] * dt_hours for t in flex_curtail_P_vars.keys()) if len(flex_curtail_P_vars) > 0 else 0
    pv_curtail_cost = gp.quicksum(electricity_price[t] * curtailment_vars[t][bus] * dt_hours for bus in pv_buses for t in time_steps) if len(pv_buses) > 0 else 0

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = electricity_cost + bess_cost + y_cap_cost + pv_curtail_cost + total_slack_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # After adding all constraints and variables
    #model.setParam('OutputFlag', 0)
    #model.setParam('Presolve', 0)
    #model.setParam('NonConvex', 2)
    #model.setParam("NumericFocus", 3)
    model.setParam("BarHomogeneous", 1)
    model.setParam("Crossover", 0)
    model.setParam("BarQCPConvTol", 1e-8)

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

            

        # Return results in a structured format. Use the extracted *_results dicts (populated above)
        # so the stored values exactly match the optimization solution.
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
        
        # Save the results to a file
        # if results is not None:
        #     filename = f"drcc_results_drcc_{par.DRCC_FLG}_{par.epsilon}.pkl"
        #     save_optim_results(results, filename)

        print(f"\n" + "="*80)

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
            total_slack_cost_value = sum(Q_slack[t].x * COST_SLACK / 1e6 for t in time_steps)
        except Exception:
            total_slack_cost_value = None

        try:
            total_cost_value = None
            # Prefer computing from components when available
            if None not in (electricity_cost_value, bess_cost_value, y_cap_cost_value, pv_curtail_cost_value, total_slack_cost_value):
                total_cost_value = electricity_cost_value + bess_cost_value + y_cap_cost_value + pv_curtail_cost_value + total_slack_cost_value
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
        print(f"  total_slack_cost = {total_slack_cost_value}")
        print(f"  total_cost (components sum or model.ObjVal) = {total_cost_value}")
        try:
            print(f"  model.ObjVal = {model.ObjVal}")
        except Exception:
            pass
        print(f"✓ MULTI-PERIOD OPTIMIZATION SUCCESSFUL!")
        print(f"="*80)
        
        # Calculate total energy consumption
        total_hp_energy_J = sum(Q_hp[t].x * delta_t_s for t in time_steps)
        total_hp_energy_kWh = total_hp_energy_J / 3.6e6  # Convert J to kWh
        
        total_slack_energy_J = sum(Q_slack[t].x * delta_t_s for t in time_steps)
        total_slack_energy_kWh = total_slack_energy_J / 3.6e6  # Convert J to kWh
        
        print(f"Total heat pump energy: {total_hp_energy_kWh:.2f} kWh")
        print(f"Total slack energy: {total_slack_energy_kWh:.2f} kWh")
        print(f"Total combined energy: {total_hp_energy_kWh + total_slack_energy_kWh:.2f} kWh")
        print(f"Slack percentage of total: {(total_slack_energy_kWh/(total_hp_energy_kWh + total_slack_energy_kWh))*100:.1f}%")
        print(f"Average heat pump power: {total_hp_energy_J / (NUM_PERIODS * delta_t_s) / 1000:.1f} kW")
        print(f"Average slack power: {total_slack_energy_J / (NUM_PERIODS * delta_t_s) / 1000:.1f} kW")
        
        # # Show results for first few and last few periods
        # print(f"\n" + "="*80)
        # print(f"HEAT SOURCES POWER SCHEDULE WITH STORAGE DETAILS")
        # print(f"="*80)
        # print(f"{'Period':<8} {'Load Factor':<12} {'Q_hp (kW)':<12} {'Q_storage (kW)':<15} {'SOC (%)':<10} {'Supply Temp (°C)':<15}")
        # print(f"-" * 90)
        
        target_junction_id = name_to_id['bus_15_supply']
        
        # Show first 10 periods
        for t in range(min(10, NUM_PERIODS)):
            load_factor = load_profile[t]
            q_hp_kw = Q_hp[t].x / 1000
            q_slack_kw = Q_slack[t].x / 1000
            q_storage_kw = Q_storage[t].x / 1000
            soc_percent = E_storage[t].x / STORAGE_CAPACITY_J * 100
            supply_temp_c = T_junction[target_junction_id, t].x - 273.15
            #print(f"{t+1:<8} {load_factor:<12.3f} {q_hp_kw:<12.1f} {q_storage_kw:<15.1f} {soc_percent:<10.1f} {supply_temp_c:<15.2f}")
        
        if NUM_PERIODS > 10:
            print(f"...")
            # Show last 5 periods
            for t in range(max(10, NUM_PERIODS-5), NUM_PERIODS):
                load_factor = load_profile[t]
                q_hp_kw = Q_hp[t].x / 1000
                q_slack_kw = Q_slack[t].x / 1000
                q_storage_kw = Q_storage[t].x / 1000
                soc_percent = E_storage[t].x / STORAGE_CAPACITY_J * 100
                supply_temp_c = T_junction[target_junction_id, t].x - 273.15
                #print(f"{t+1:<8} {load_factor:<12.3f} {q_hp_kw:<12.1f} {q_storage_kw:<15.1f} {soc_percent:<10.1f} {supply_temp_c:<15.2f}")
        
        # Calculate statistics with storage and slack
        q_hp_values = [Q_hp[t].x / 1000 for t in time_steps]
        q_storage_values = [Q_storage[t].x / 1000 for t in time_steps]  # kW
        q_slack_values = [Q_slack[t].x / 1000 for t in time_steps]  # kW
        e_storage_values = [E_storage[t].x / 3.6e6 for t in time_steps]  # kWh
        storage_soc_values = [E_storage[t].x / STORAGE_CAPACITY_J for t in time_steps]  # State of charge
        
        print(f"\n" + "="*80)
        print(f"OPTIMIZATION STATISTICS WITH THERMAL STORAGE AND SLACK")
        print(f"="*80)
        print(f"Heat pump power range: {min(q_hp_values):.1f} - {max(q_hp_values):.1f} kW")
        print(f"Heat pump power average: {np.mean(q_hp_values):.1f} kW")
        print(f"Slack power range: {min(q_slack_values):.1f} - {max(q_slack_values):.1f} kW")
        print(f"Slack power average: {np.mean(q_slack_values):.1f} kW")
        print(f"Storage power range: {min(q_storage_values):.1f} - {max(q_storage_values):.1f} kW")
        print(f"Storage energy range: {min(e_storage_values):.1f} - {max(e_storage_values):.1f} kWh")
        print(f"Storage SOC range: {min(storage_soc_values)*100:.1f}% - {max(storage_soc_values)*100:.1f}%")
        
        # Calculate total energies
        total_hp_energy_kWh = sum(q_hp_values)  # Total heat pump energy
        total_slack_energy_kWh = sum(q_slack_values)  # Total slack energy (objective)
        total_combined_energy_kWh = total_hp_energy_kWh + total_slack_energy_kWh
                
        print(f"Peak HP power period: {np.argmax(q_hp_values) + 1}")
        print(f"Peak slack power period: {np.argmax(q_slack_values) + 1}")
        print(f"Min power period: {np.argmin(q_hp_values) + 1}")
        print(f"Max charging period: {np.argmax(q_storage_values) + 1} ({max(q_storage_values):.1f} kW)")
        print(f"Max discharging period: {np.argmin(q_storage_values) + 1} ({min(q_storage_values):.1f} kW)")
        
        # Storage statistics
        charging_periods = [q for q in q_storage_values if q > 0.1]
        discharging_periods = [q for q in q_storage_values if q < -0.1]
        passthrough_periods = [q for q in q_storage_values if abs(q) <= 0.1]
        
        # print(f"\nStorage Operation Analysis:")
        # print(f"Charging periods: {len(charging_periods)} ({len(charging_periods)/NUM_PERIODS*100:.1f}%)")
        # print(f"Discharging periods: {len(discharging_periods)} ({len(discharging_periods)/NUM_PERIODS*100:.1f}%)")
        # print(f"Pass-through periods: {len(passthrough_periods)} ({len(passthrough_periods)/NUM_PERIODS*100:.1f}%)")
        
        if charging_periods:
            print(f"Average charging power: {np.mean(charging_periods):.1f} kW")
        if discharging_periods:
            print(f"Average discharging power: {np.mean([abs(q) for q in discharging_periods]):.1f} kW")
        
        # DETAILED STORAGE ENERGY BALANCE ANALYSIS
        # print(f"\nDETAILED STORAGE ANALYSIS:")
        # print(f"=" * 50)
        
        # # Calculate energy balance components for first few periods
        # print(f"Storage energy balance for first 5 periods:")
        # print(f"{'Period':<8} {'E_start (kWh)':<15} {'Q_storage (kW)':<15} {'Heat_loss (kW)':<15} {'E_end (kWh)':<15}")
        # print(f"-" * 75)
        
        # for t in range(min(5, NUM_PERIODS)):
        #     e_start = E_storage[t-1].x / 3.6e6 if t > 0 else INITIAL_SOC * STORAGE_CAPACITY_J / 3.6e6
        #     q_storage_kw = Q_storage[t].x / 1000
            
        #     # FIXED: Heat loss calculation matching constraint formula
        #     if t > 0:
        #         e_storage_prev_j = E_storage[t-1].x  # Previous energy in Joules
        #         heat_loss_kw = STORAGE_HEAT_LOSS_COEFF * e_storage_prev_j * (STORAGE_TEMP_K - temp_profile_k[t]) / STORAGE_CAPACITY_J / 1000
        #     else:
        #         # For t=0, no previous storage state, so use initial SOC energy
        #         e_storage_prev_j = INITIAL_SOC * STORAGE_CAPACITY_J
        #         heat_loss_kw = STORAGE_HEAT_LOSS_COEFF * e_storage_prev_j * (STORAGE_TEMP_K - temp_profile_k[t]) / STORAGE_CAPACITY_J / 1000
            
        #     e_end = E_storage[t].x / 3.6e6
            
        #     #print(f"{t+1:<8} {e_start:<15.1f} {q_storage_kw:<15.1f} {heat_loss_kw:<15.1f} {e_end:<15.1f}")
            
        
        
        # Save results to CSV with ALL junction temperatures
        results_data = {
            'period': [t+1 for t in time_steps],
            'load_factor': load_profile,
            'q_hp_kw': q_hp_values,
            'q_slack_kw': q_slack_values,
            'q_storage_kw': q_storage_values,
            'e_storage_kwh': e_storage_values,
            'storage_soc': storage_soc_values
        }

        # Add COP profile (per-timestep) to results if available
        try:
            # cop_profile was computed earlier as a list indexed by time_steps
            results_data['cop_t'] = [cop_profile[t] for t in time_steps]
        except Exception:
            # Fallback to ones if cop_profile not available
            results_data['cop_t'] = [1.0 for _ in time_steps]
        
        # Add all junction temperatures
        for j_id in all_junctions:
            junction_name = id_to_name[j_id]
            temp_col_name = f"{junction_name}_temp_c"
            results_data[temp_col_name] = [T_junction[j_id, t].x - 273.15 for t in time_steps]
        
        # Add all bus voltages from electrical network results
        for bus in net.bus.index:
            voltage_col_name = f"bus_{bus}_voltage_pu"
            results_data[voltage_col_name] = [results['voltage'][t][bus] for t in time_steps]
        
        # Add heat pump electrical power results
        for bus in net.bus.index:
            hp_power_col_name = f"bus_{bus}_p_hp_mw"
            if bus in hp_load_buses and len(hp_load_buses) > 0 and time_steps[0] in results['p_hp']:
                if bus in results['p_hp'][time_steps[0]]:  # Check if this bus has heat pump
                    # Store as MW (no conversion needed - already in MW from optimization)
                    results_data[hp_power_col_name] = [results['p_hp'][t][bus] for t in time_steps]
                else:
                    results_data[hp_power_col_name] = [0.0 for t in time_steps]
            else:
                results_data[hp_power_col_name] = [0.0 for t in time_steps]  # No heat pump on this bus
        
        # Add line loading results (percentage)
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
        
        results_df.to_csv('fully_coordinated_model_results.csv', index=False)
        print(f"\nResults saved to: fully_coordinated_model_results.csv")
        
        # Expose pv_gen results at module level for plotting helpers that read globals()
        try:
            pv_gen_results = results.get('pv_gen', {})
            globals()['pv_gen_results'] = pv_gen_results
        except Exception:
            globals()['pv_gen_results'] = {}

        # Create comprehensive plots
        print("\nGenerating comprehensive plots...")
        # Pass electricity_price from the outer scope (solve_opf argument) if available
        try:
            create_comprehensive_plots(results_df, q_hp_values, load_profile, temp_profile_c, storage_soc_values, q_slack_values, results.get('non_flexible_load_p', None), results.get('flexible_load_p', None), electricity_price=electricity_price)
        except Exception:
            # Fallback without price if variable not available in this scope
            create_comprehensive_plots(results_df, q_hp_values, load_profile, temp_profile_c, storage_soc_values, q_slack_values, results.get('non_flexible_load_p', None), results.get('flexible_load_p', None))
        
        print(f"="*80)
        
        return results, results_data
    
    elif model.status == GRB.INFEASIBLE:
        print("✗ Multi-period model is infeasible")
        # Analyze load scaling
        max_load_factor = max(load_profile)
        min_load_factor = min(load_profile)
        total_scaled_load = sum(base_loads.values()) * max_load_factor
        
        print(f"\nLoad Analysis:")
        print(f"Base total load: {sum(base_loads.values())/1000:.1f} kW")
        print(f"Peak total load: {total_scaled_load/1000:.1f} kW")
        print(f"Heat pump upper bound: {Q_hp[0].UB/1000:.1f} kW")
        
        if total_scaled_load > Q_hp[0].UB:
            print(f"⚠️  Peak load exceeds heat pump capacity!")
        else:
            print(f"✓ Heat pump capacity is sufficient for peak load")
        
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
                print(f"Loaded electricity prices from market_prices_15min.csv and aligned to time_index ({len(electricity_price)} steps)")
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
        print("\nCreating electrical network structure from CSV files...")
        net = create_network_from_csv()
        print(f"Network created: {len(net.bus)} buses, {len(net.line)} lines, {len(net.trafo)} transformers")
        

        # Use the DHN time index as canonical
        input_data = load_input_data_from_csv(time_index)

        # Backwards compatibility: some code expects input_data['time_steps'] to exist.
        # Ensure it's present and aligned to the DHN time_index (list of integer step indices).
        if 'time_steps' not in input_data:
            input_data['time_steps'] = list(range(len(time_index)))
        
        # Run the optimization       

        time_steps = list(range(len(time_index)))
        NUM_PERIODS = len(load_profile)
        print(f"\nOptimization setup:")
        print(f"Number of time steps: {NUM_PERIODS} (15-min intervals)")
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
            print("\nOptimization completed successfully!")
            
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
            

            #print(f"\nResults saved to: drcc_results_drcc_{par.DRCC_FLG}_{par.epsilon}.pkl")
            
        else:
            print("\n❌ Optimization failed!")
            
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)