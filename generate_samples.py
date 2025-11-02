"""
Generate reproducible Monte Carlo samples for out-of-sample analysis.

How to use (no command-line needed):
- Edit the USER CONFIG block below (RESULTS_CSV, N_SAMPLES, SEED, OUTDIR, CLEAN_OLD)
- Save this file; then run it (F5 or right-click Run Python File in VS Code)

Inputs:
- A v2 results CSV (from dso_model_v2.py) containing timestamps and metadata
	needed to parameterize uncertainty. If RESULTS_CSV is None, the latest
	dso_model_v2_results_*.csv in the current folder is used.

Outputs (CSV files under --outdir):
- samples_temperature_c.csv                  # long format: [timestamp, sample_id, temperature_c]
- samples_pv.csv                             # wide per bus: [timestamp, sample_id, pv_bus_{b}_mw, ...]
- samples_hp_residual.csv                    # wide per bus: [timestamp, sample_id, hp_residual_bus_{b}_mw, ...]
- samples_meta.json                          # metadata of generation settings and parameters

This ensures all studies use the exact same sample set.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ===========================
# USER CONFIG (edit these)
# ===========================
# Path to the v2 results CSV. If None, we auto-pick the most recent dso_model_v2_results_*.csv
RESULTS_CSV: str | None = None

# Number of Monte Carlo samples to generate per timestamp
N_SAMPLES: int = 1000

# Distribution type for random draws (affects all stochastic components).
# Options: "gaussian" (standard normal), "uniform" (variance-matched U(-√3, +√3)),
#          "contaminated" (95% N(0,1) + 5% N(0,(k)^2) normalized to unit variance; k≈5 for outliers),
#          "studentt" (Student-t with df=3, scaled to unit variance)
DISTRIBUTION: str = "studentt"

# Contaminated normal parameters
CONTAM_RATE: float = 0.05  # 5% of draws from the inflated-variance component
CONTAM_SCALE_K: float = 5.0  # outlier std multiplier (kσ)

# Random seed for reproducibility. Use any integer to change the random stream.
SEED: int = 42

# Output directory where CSVs will be written
OUTDIR: str = "samples"

# If True, remove any previously generated per-bus CSVs in OUTDIR to avoid clutter
CLEAN_OLD: bool = True


def _infer_latest_results_csv(workdir: str) -> str | None:
	"""Pick the most recent dso_model_v2_results_*.csv in the working directory."""
	candidates = [
		f for f in os.listdir(workdir)
		if f.startswith("dso_model_v2_results_") and f.endswith(".csv")
	]
	if not candidates:
		return None
	# sort by mtime descending
	candidates_full = [os.path.join(workdir, f) for f in candidates]
	candidates_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return candidates_full[0]


def _load_results(results_csv: str) -> pd.DataFrame:
	df = pd.read_csv(results_csv)
	# Prefer explicit timestamp column we export in v2; if absent, fall back to period index
	if 'timestamp' in df.columns:
		ts = pd.to_datetime(df['timestamp'])
	else:
		# Try to reconstruct timestamps from meta_time_start and dt_hours
		start = pd.to_datetime(df.get('meta_time_start', [None])[0])
		dt_hours = float(df.get('meta_dt_hours', [1.0])[0])
		if pd.isna(start):
			# fallback to integer periods starting at 0
			ts = pd.date_range(start=pd.Timestamp("1970-01-01"), periods=len(df), freq=f"{int(dt_hours*60)}min")
		else:
			ts = pd.date_range(start=start, periods=len(df), freq=f"{int(dt_hours*60)}min")
	df.index = ts
	return df


def _align_series_to_index(path: str, col: str, index: pd.DatetimeIndex) -> np.ndarray:
	"""Read a CSV with 'datetime' column and align the requested column to index."""
	raw = pd.read_csv(path, parse_dates=['datetime'])
	raw.set_index('datetime', inplace=True)
	return raw.reindex(index)[col].to_numpy()


def _get_bus_ids_from_columns(df: pd.DataFrame, prefix: str, suffix: str) -> List[int]:
	ids: List[int] = []
	pre = prefix
	suf = suffix
	for c in df.columns:
		if c.startswith(pre) and c.endswith(suf):
			try:
				mid = c[len(pre):-len(suf)]
				ids.append(int(mid))
			except Exception:
				continue
	return sorted(list(set(ids)))


def _compute_pv_installed_from_avail(df: pd.DataFrame, pv_bus_ids: List[int], const_pv: np.ndarray) -> Dict[int, float]:
	"""installed_mw[bus] ~= median over t with const_pv>0 of pv_avail_bus_{bus}_mw / const_pv[t]."""
	installed: Dict[int, float] = {}
	eps = 1e-6
	mask = const_pv > eps
	for b in pv_bus_ids:
		col = f"pv_avail_bus_{b}_mw"
		if col not in df.columns:
			installed[b] = 0.0
			continue
		ratio = np.where(mask, df[col].to_numpy() / np.maximum(const_pv, eps), np.nan)
		# robust estimate ignoring NaNs and zeros
		r = pd.Series(ratio[np.isfinite(ratio)]).quantile(0.5)
		installed[b] = float(max(0.0, r if pd.notna(r) else 0.0))
	return installed


def _var_of_daily_mean_with_equicorr(sigmas: np.ndarray, rho: float) -> float:
	"""Compute Var(mean) for unequal stds with equicorrelation rho over a day.
	Var(sum) = (1-rho)*sum(σ_i^2) + rho*(sum σ_i)^2, so Var(mean) = Var(sum)/n^2.
	Returns std (sqrt of Var(mean)).
	"""
	s = np.asarray(sigmas, dtype=float)
	n = max(1, len(s))
	sum_sq = float(np.sum(s**2))
	sum_s = float(np.sum(s))
	var_sum = (1.0 - rho) * sum_sq + rho * (sum_s ** 2)
	var_mean = var_sum / (n ** 2)
	return float(np.sqrt(max(0.0, var_mean)))


def generate_samples(
	results_csv: str,
	n_samples: int = 1000,
	seed: int = 42,
    outdir: str = "samples",
    distribution: str = "gaussian",
) -> None:
	os.makedirs(outdir, exist_ok=True)

	df = _load_results(results_csv)
	index = df.index

	# Metadata (with robust fallbacks)
	meta = {
		'pv_rho': float(df['meta_pv_std_correlation'].iloc[0]) if 'meta_pv_std_correlation' in df.columns else 1.0,
		'rho_temp_avg': float(df['meta_rho_temp_avg'].iloc[0]) if 'meta_rho_temp_avg' in df.columns else 0.0,
		'hp_residual_sigma_norm': float(df['meta_hp_residual_sigma_norm'].iloc[0]) if 'meta_hp_residual_sigma_norm' in df.columns else 0.0,
		'hp_pred_pmax_mw': float(df['meta_hp_pred_pmax_mw'].iloc[0]) if 'meta_hp_pred_pmax_mw' in df.columns else 0.0,
		'dt_hours': float(df['meta_dt_hours'].iloc[0]) if 'meta_dt_hours' in df.columns else 1.0,
	}

	# Load normalized PV profile and std aligned to index
	pv_profile_path = os.path.join(os.getcwd(), 'pv_profiles_output.csv')
	if not os.path.exists(pv_profile_path):
		raise FileNotFoundError(f"pv_profiles_output.csv not found at {pv_profile_path}")
	const_pv = _align_series_to_index(pv_profile_path, 'normalized_output', index)
	try:
		const_pv_std = _align_series_to_index(pv_profile_path, 'normalized_output_std', index)
	except Exception:
		# fallback: relative std not available, assume zero
		const_pv_std = np.zeros(len(index))

	# Determine PV and HP bus ids from v2 CSV
	pv_bus_ids = _get_bus_ids_from_columns(df, 'pv_avail_bus_', '_mw')
	hp_bus_ids = _get_bus_ids_from_columns(df, 'hp_elec_bus_', '_mw')

	# Reconstruct installed PV capacity per bus
	installed_pv_mw = _compute_pv_installed_from_avail(df, pv_bus_ids, const_pv)

	# Determine PV mean availability per bus/time from v2 CSV
	pv_avail_means = {b: df[f'pv_avail_bus_{b}_mw'].to_numpy() if f'pv_avail_bus_{b}_mw' in df.columns else np.zeros(len(index)) for b in pv_bus_ids}

	# Ambient temperature mean from v2 CSV (Celsius)
	if 'ambient_temp_c' in df.columns:
		T_mean_c = df['ambient_temp_c'].to_numpy(dtype=float)
	else:
		# if absent, try temperature_data_complete.csv directly (Kelvin -> C)
		temp_path = os.path.join(os.getcwd(), 'temperature_data_complete.csv')
		if not os.path.exists(temp_path):
			raise FileNotFoundError("ambient_temp_c not in results, and temperature_data_complete.csv not found")
		T_mean_k = _align_series_to_index(temp_path, 'temperature_K', index)
		T_mean_c = T_mean_k - 273.15

	# Temperature per-interval std (Kelvin) from input file, aligned
	temp_path = os.path.join(os.getcwd(), 'temperature_data_complete.csv')
	if not os.path.exists(temp_path):
		raise FileNotFoundError(f"temperature_data_complete.csv not found at {temp_path}")
	try:
		T_std_k = _align_series_to_index(temp_path, 'temperature_std_K', index)
	except Exception:
		T_std_k = np.zeros(len(index))

	# Group per day to compute std of daily average with equicorrelation rho_temp_avg
	ti = pd.to_datetime(index)
	day_index = pd.Series(range(len(ti)), index=ti).groupby(ti.date)
	sigma_Tavg_by_day: Dict[datetime.date, float] = {}
	for day, idx in day_index:
		# idx is a Series whose values are positional indices for this day
		pos = np.asarray(idx.values, dtype=int)
		sigmas = T_std_k[pos]
		sigma_Tavg_by_day[day] = _var_of_daily_mean_with_equicorr(sigmas, meta['rho_temp_avg'])

	# Prep RNG & validate distribution
	rng = np.random.default_rng(seed)
	distribution = distribution.lower().strip()
	if distribution not in {"gaussian", "uniform", "contaminated", "studentt"}:
		raise ValueError(f"Unsupported distribution '{distribution}'. Use 'gaussian', 'uniform', 'contaminated', or 'studentt'.")

	# For contaminated distribution, implement sample-wise contamination:
	# choose a fixed subset of sample_ids that get higher sigma across all draws.
	# We normalize by sqrt((1-p) + p*k^2) so overall variance across samples remains ~1.
	contam_scale_per_sample: np.ndarray | None = None
	contam_meta: Dict[str, object] | None = None
	if distribution == "contaminated":
		p = float(CONTAM_RATE)
		k = float(CONTAM_SCALE_K)
		# Fixed mask over samples (trajectories): True => high-sigma sample
		high_sigma_mask = rng.random(n_samples) < p
		var_mix = (1.0 - p) * 1.0 + p * (k ** 2)
		norm = float(np.sqrt(var_mix)) if var_mix > 0 else 1.0
		contam_scale_per_sample = np.where(high_sigma_mask, k, 1.0).astype(float) / norm
		# stash meta for later
		contaminated_sample_ids = [f"s{str(i+1).zfill(4)}" for i, m in enumerate(high_sigma_mask) if m]
		contam_meta = {
			'contamination_mode': 'sample-wise',
			'contamination_rate': p,
			'contamination_scale_k': k,
			'contaminated_sample_ids': contaminated_sample_ids,
			'contamination_variance_normalizer': var_mix,
		}

	def _draw_standard(size):
		"""Draw standard (mean 0, var 1) variates according to selected distribution.
		In 'contaminated' mode, applies a per-sample scaling on the last axis (length n_samples).
		"""
		if distribution == "gaussian":
			return rng.standard_normal(size=size)
		if distribution == "uniform":
			# Uniform with same variance as standard normal: U(-sqrt(3), +sqrt(3)) has Var=1
			bound = np.sqrt(3.0)
			return rng.uniform(-bound, bound, size=size)
		if distribution == "studentt":
			# Student-t with df=3 has Var = df/(df-2) = 3; scale by 1/sqrt(3) to get Var=1
			df_t = 3.0
			scale = 2.0 / np.sqrt(df_t / (df_t - 2.0))  # = 1/sqrt(3)
			return rng.standard_t(df_t, size=size) * scale
		# Contaminated normal (sample-wise): base standard normal scaled per sample id
		z = rng.standard_normal(size=size)
		# Ensure we can broadcast per-sample scaling along the last axis
		assert contam_scale_per_sample is not None
		# Build a view with ones on all leading dims and n_samples on the last
		try:
			last_dim = z.shape[-1]  # type: ignore[attr-defined]
		except Exception:
			last_dim = None
		if last_dim is None:
			# scalar draw not supported in contaminated mode
			raise ValueError("Contaminated mode expects draws with a samples axis; got scalar.")
		if last_dim != n_samples:
			raise ValueError(
				f"Contaminated mode expects the last dimension to equal n_samples={n_samples}, got {last_dim}."
			)
		scale = contam_scale_per_sample.reshape((1,) * (z.ndim - 1) + (n_samples,))
		return z * scale

	# Contract for outputs
	nT = len(index)
	sample_cols = [f"s{str(i+1).zfill(4)}" for i in range(n_samples)]

	# 1) Temperature samples: per-interval equicorrelated structure within each day
	#    For times t in a day, with per-interval std s_t and equicorrelation rho among times of that day:
	#      T_t = mu_t + s_t * (sqrt(rho) * Z_c + sqrt(1-rho) * Z_t), Z_c ~ N(0,1), Z_t ~ N(0,1) iid
	#    This matches per-interval variance s_t^2 and yields Var(daily_mean) per v2 DRCC formula.
	temp_samples = np.zeros((nT, n_samples), dtype=float)
	# Group indices by day
	idx_to_day = [ts.date() for ts in index]
	unique_days = sorted(set(idx_to_day))
	rho = float(np.clip(meta['rho_temp_avg'], 0.0, 1.0))
	sqrt_rho = np.sqrt(rho)
	sqrt_1mrho = np.sqrt(max(0.0, 1.0 - rho))
	# Iterate by day and fill blocks
	for day in unique_days:
		pos = np.array([i for i, d in enumerate(idx_to_day) if d == day], dtype=int)
		if pos.size == 0:
			continue
		s_t = T_std_k[pos].astype(float)
		mu_t = T_mean_c[pos].astype(float)
		# Common shock per sample for this day
		zc = _draw_standard(size=n_samples)
		# Idiosyncratic shocks per time per sample
		zi = _draw_standard(size=(pos.size, n_samples))
		# Compose per interval/samples
		# Broadcast: s_t[:,None] * (sqrt_rho*zc[None,:] + sqrt_1mrho*zi)
		block = mu_t[:, None] + (s_t[:, None] * (sqrt_rho * zc[None, :] + sqrt_1mrho * zi))
		temp_samples[pos, :] = block

	# 2) PV samples per PV bus with equicorrelation across buses within each timestep
	pv_samples_by_bus: Dict[int, np.ndarray] = {b: np.zeros((nT, n_samples), dtype=float) for b in pv_bus_ids}
	rho_pv = float(np.clip(meta['pv_rho'], 0.0, 1.0))
	sqrt_rho = np.sqrt(rho_pv)
	sqrt_1mrho = np.sqrt(max(0.0, 1.0 - rho_pv))
	# For each timestep, draw common and idiosyncratic components
	for t in range(nT):
		zc = _draw_standard(size=n_samples)  # common shock per t
		for b in pv_bus_ids:
			zi = _draw_standard(size=n_samples)
			sigma_bus_t = float(const_pv_std[t]) * float(installed_pv_mw.get(b, 0.0))
			mu_bus_t = float(pv_avail_means[b][t])
			noise = sigma_bus_t * (sqrt_rho * zc + sqrt_1mrho * zi)
			samples = mu_bus_t + noise
			# physical limits [0, installed]
			cap = float(installed_pv_mw.get(b, 0.0))
			samples = np.clip(samples, 0.0, cap)
			pv_samples_by_bus[b][t, :] = samples

	# 3) HP residual samples per HP bus (MW), zero-mean Normal
	hp_resid_sigma_mw = meta['hp_pred_pmax_mw'] * meta['hp_residual_sigma_norm']
	hp_samples_by_bus: Dict[int, np.ndarray] = {}
	for b in hp_bus_ids:
		if hp_resid_sigma_mw <= 0:
			hp_samples_by_bus[b] = np.zeros((nT, n_samples), dtype=float)
		else:
			standard_draws = _draw_standard(size=(nT, n_samples))
			hp_samples_by_bus[b] = hp_resid_sigma_mw * standard_draws

	# Write outputs (consolidated to minimize file count)
	# Build long-form vectors: t repeats each sample; sample_id cycles fastest per time
	ts_vec = np.repeat(index.astype(str).values, n_samples)
	sid_vec = np.tile(sample_cols, nT)

	# Choose suffix for distribution in filenames
	file_suffix = f"_{distribution}"

	# Temperature CSV (long format)
	temp_flat = temp_samples.reshape(nT * n_samples, order='C')
	temp_df = pd.DataFrame({
		'timestamp': ts_vec,
		'sample_id': sid_vec,
		'temperature_c': temp_flat,
	})
	temp_out = os.path.join(outdir, f'samples_temperature_c{file_suffix}.csv')
	temp_df.to_csv(temp_out, index=False)

	# PV samples consolidated across buses (wide per bus, long over samples)
	pv_cols: Dict[str, np.ndarray] = {}
	for b in pv_bus_ids:
		pv_cols[f'pv_bus_{b}_mw'] = pv_samples_by_bus[b].reshape(nT * n_samples, order='C')
	pv_df = pd.DataFrame({'timestamp': ts_vec, 'sample_id': sid_vec, **pv_cols})
	pv_out = os.path.join(outdir, f'samples_pv{file_suffix}.csv')
	pv_df.to_csv(pv_out, index=False)

	# HP residual consolidated across buses (wide per bus, long over samples)
	hp_cols: Dict[str, np.ndarray] = {}
	for b in hp_bus_ids:
		hp_cols[f'hp_residual_bus_{b}_mw'] = hp_samples_by_bus[b].reshape(nT * n_samples, order='C')
	hp_df = pd.DataFrame({'timestamp': ts_vec, 'sample_id': sid_vec, **hp_cols})
	hp_out = os.path.join(outdir, f'samples_hp_residual{file_suffix}.csv')
	hp_df.to_csv(hp_out, index=False)

	# Metadata JSON
	meta_out = {
		'generated_at': datetime.utcnow().isoformat() + 'Z',
		'results_csv': os.path.abspath(results_csv),
		'n_samples': n_samples,
		'seed': seed,
		'index_start': str(index[0]) if len(index) else None,
		'index_end': str(index[-1]) if len(index) else None,
		'dt_hours': meta['dt_hours'],
		'pv_rho': rho_pv,
		'rho_temp_avg': meta['rho_temp_avg'],
		'temperature_model': 'per_interval_equicorr',
		'hp_residual_sigma_norm': meta['hp_residual_sigma_norm'],
		'hp_pred_pmax_mw': meta['hp_pred_pmax_mw'],
		'pv_installed_mw': installed_pv_mw,
		'pv_bus_ids': pv_bus_ids,
		'hp_bus_ids': hp_bus_ids,
		'notes': "PV samples represent available PV MW (pre-curtailment); temperature samples add daily-average uncertainty; HP residual is additive to HP predictor."
	}
	meta_out['distribution'] = distribution
	if distribution == 'contaminated' and contam_meta is not None:
		# enrich metadata with contamination details
		meta_out.update(contam_meta)
	if distribution == 'studentt':
		meta_out['student_t_df'] = 3
	with open(os.path.join(outdir, 'samples_meta.json'), 'w', encoding='utf-8') as f:
		json.dump(meta_out, f, indent=2)

	print(f"✓ Wrote temperature samples: {temp_out}")
	print(f"✓ Wrote PV samples ({len(pv_bus_ids)} buses) to {pv_out}")
	print(f"✓ Wrote HP residual samples ({len(hp_bus_ids)} buses) to {hp_out}")
	print(f"✓ Wrote metadata: {os.path.join(outdir, 'samples_meta.json')}")


def main():
	parser = argparse.ArgumentParser(description="Generate reproducible Monte Carlo samples aligned to a v2 results CSV", add_help=True)
	parser.add_argument('--results-csv', type=str, default=None, help='Optional: path to dso_model_v2 results CSV; if omitted, uses USER CONFIG or latest in CWD')
	parser.add_argument('--n-samples', type=int, default=None, help='Optional: override USER CONFIG N_SAMPLES (or ENV GEN_N_SAMPLES)')
	parser.add_argument('--seed', type=int, default=None, help='Optional: override USER CONFIG SEED')
	parser.add_argument('--distribution', '--dist', type=str, choices=['gaussian', 'uniform', 'contaminated', 'studentt'], default=None, help='Override distribution (gaussian|uniform|contaminated|studentt). If omitted, uses USER CONFIG DISTRIBUTION.')
	parser.add_argument('--outdir', type=str, default=None, help='Optional: override USER CONFIG OUTDIR')
	parser.add_argument('--clean-old', action='store_true', help='Optional: override USER CONFIG CLEAN_OLD to True')
	args = parser.parse_args()

	# Resolve settings: CLI override > USER CONFIG > sensible defaults
	results_csv = args.results_csv if args.results_csv is not None else (RESULTS_CSV or _infer_latest_results_csv(os.getcwd()))
	if not results_csv or not os.path.exists(results_csv):
		raise FileNotFoundError("No results CSV provided and none found matching dso_model_v2_results_*.csv in current directory")

	# Determine number of samples: CLI > USER CONFIG > ENV > default
	try:
		n_samples = int(args.n_samples) if args.n_samples is not None else int(N_SAMPLES)
	except Exception:
		n_samples = int(os.environ.get('GEN_N_SAMPLES', '1000'))

	# Seed: CLI > USER CONFIG > default 42
	seed = int(args.seed) if args.seed is not None else int(SEED)

	# Outdir: CLI > USER CONFIG > default "samples"
	outdir = args.outdir if args.outdir is not None else OUTDIR

	# Clean old: CLI flag or USER CONFIG
	clean_old = bool(args.clean_old or CLEAN_OLD)

	# Optionally clean old per-bus sample files to avoid thousands of files
	if clean_old:
		try:
			for fname in os.listdir(outdir):
				if (fname.startswith('samples_pv_bus_') or fname.startswith('samples_hp_residual_bus_')) and fname.endswith('.csv'):
					os.remove(os.path.join(outdir, fname))
		except FileNotFoundError:
			pass

	# Distribution: CLI override or USER CONFIG
	distribution = args.distribution if args.distribution is not None else DISTRIBUTION

	generate_samples(
		results_csv=results_csv,
		n_samples=n_samples,
		seed=seed,
		outdir=outdir,
		distribution=distribution,
	)


if __name__ == '__main__':
	main()
