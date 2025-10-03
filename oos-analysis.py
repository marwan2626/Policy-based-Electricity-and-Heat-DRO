# Out-of-sample analysis for v3 results
# Builds an overview figure with costs vs epsilon and violation counts vs epsilon.

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === User config ===
RESULTS_DIR = "v3_oos"
EPSILONS: List[float] = [0.25, 0.20, 0.15, 0.10, 0.05]
OUT_FIG = "oos_overview.png"
OUT_CSV = "oos_overview_summary.csv"
SHOW: bool = False  # set True to display interactively

# Cost model parameters for OOS components
PV_CURT_PRICE_FACTOR = 1.0  # EUR per MWh of curtailed PV is factor * price
BESS_THROUGHPUT_COST_EUR_PER_MWH = 0.0  # cost per MWh of RT BESS throughput (set >0 if you price cycling)


def epsilon_token(eps: float) -> str:
    return f"{eps:.2f}".replace(".", "_")


def load_summary_for_epsilon(eps: float) -> pd.DataFrame:
    token = epsilon_token(eps)
    path = os.path.join(RESULTS_DIR, f"v3_summary_epsilon_{token}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary for epsilon={eps}: {path}")
    return pd.read_csv(path)


def load_meta_for_epsilon(eps: float) -> Dict:
    token = epsilon_token(eps)
    meta_path = os.path.join(RESULTS_DIR, f"v3_meta_epsilon_{token}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
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
    # Core import cost (already computed in v3)
    import_cost_mean = float(df.get("energy_cost_eur", pd.Series([0.0] * len(df))).mean())

    # RT PV curtailment cost (approx): total curtailed MWh * avg price * factor
    pv_curt_mwh_mean = float(df.get("pv_curtail_mwh", pd.Series([0.0] * len(df))).mean())
    pv_rt_curt_cost_mean = pv_curt_mwh_mean * max(avg_price_eur_mwh, 0.0) * float(PV_CURT_PRICE_FACTOR)

    # BESS throughput cost (optional): throughput * unit cost
    bess_throughput_mwh_mean = float(df.get("bess_rt_energy_throughput_mwh", pd.Series([0.0] * len(df))).mean())
    bess_rt_cycle_cost_mean = bess_throughput_mwh_mean * float(BESS_THROUGHPUT_COST_EUR_PER_MWH)

    # Violation counts (sum across trajectories)
    v_steps = int(df.get("steps_voltage_violation", pd.Series([0] * len(df))).sum())
    l_steps = int(df.get("steps_line_over_100pct", pd.Series([0] * len(df))).sum())
    t_steps = int(df.get("steps_trafo_over_100pct", pd.Series([0] * len(df))).sum())

    return {
        # individual components
        "import_cost_eur_mean": import_cost_mean,
        "pv_rt_curt_cost_eur_mean": pv_rt_curt_cost_mean,
        "bess_rt_cycle_cost_eur_mean": bess_rt_cycle_cost_mean,
        # totals and counts
        "total_cost_eur_mean": import_cost_mean + pv_rt_curt_cost_mean + bess_rt_cycle_cost_mean,
        "voltage_steps": v_steps,
        "line_steps": l_steps,
        "trafo_steps": t_steps,
    }


def main() -> None:
    rows: List[Dict[str, float]] = []
    for eps in EPSILONS:
        df = load_summary_for_epsilon(eps)
        meta = load_meta_for_epsilon(eps)
        avg_price = compute_avg_price_from_v2(meta)
        agg = aggregate_metrics(df, avg_price)
        agg["epsilon"] = eps
        rows.append(agg)

    summary = pd.DataFrame(rows).sort_values("epsilon")
    summary.to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)

    # Recompute core metrics directly from v3 summaries for RT components and DA import cost
    rt_rows = []
    for eps in EPSILONS:
        token = epsilon_token(eps)
        path = os.path.join(RESULTS_DIR, f"v3_summary_epsilon_{token}.csv")
        if not os.path.exists(path):
            continue
        df_eps = pd.read_csv(path)
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
        rt_rows.append({
            'epsilon': eps,
            'da_import_cost_mean': da_import_cost_mean,
            'rt_imbalance_cost_mean': rt_imb_cost_mean,
            'rt_pv_cost_mean': rt_pv_cost_mean,
            'rt_bess_cost_mean': rt_bess_cost_mean,
            'trafo_steps': trafo_steps
        })
    rt_summary = pd.DataFrame(rt_rows).sort_values('epsilon')

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
    x = np.arange(len(rt_summary))
    width = 0.6

    c_imb = rt_summary['rt_imbalance_cost_mean'].to_numpy()
    c_pv = rt_summary['rt_pv_cost_mean'].to_numpy()
    c_bess = rt_summary['rt_bess_cost_mean'].to_numpy()
    axes[0].bar(x, c_imb, width=width, label='RT Imbalance')
    axes[0].bar(x, c_pv, width=width, bottom=c_imb, label='RT PV Curtail')
    axes[0].bar(x, c_bess, width=width, bottom=c_imb + c_pv, label='RT BESS Cycle')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{e:.2f}" for e in rt_summary['epsilon']])
    axes[0].set_xlabel('epsilon')
    axes[0].set_ylabel('EUR (mean across samples)')
    axes[0].set_title('RT cost components (stacked)')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    c_da = rt_summary['da_import_cost_mean'].to_numpy()
    axes[1].bar(x, c_da, width=width, color='#4c72b0', label='DA Import Cost')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{e:.2f}" for e in rt_summary['epsilon']])
    axes[1].set_xlabel('epsilon')
    axes[1].set_ylabel('EUR (mean across samples)')
    axes[1].set_title('DA import cost')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    t_steps = rt_summary['trafo_steps'].to_numpy()
    axes[2].bar(x, t_steps, width=width, color='#dd8452', label='Transformer steps > threshold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{e:.2f}" for e in rt_summary['epsilon']])
    axes[2].set_xlabel('epsilon')
    axes[2].set_ylabel('Steps (sum across trajectories)')
    axes[2].set_title('Transformer loading violations')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].legend()

    out_path = os.path.join(RESULTS_DIR, OUT_FIG)
    fig.savefig(out_path, dpi=150)
    print(f"✓ Overview saved: {out_path}")
    print(f"✓ Summary CSV: {os.path.join(RESULTS_DIR, OUT_CSV)}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()

