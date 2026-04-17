#!/usr/bin/env python3
"""
Unified benchmark runner.

Usage:
    python scripts/run_benchmark.py --dataset rees46 --data-dir data/raw/rees46 \
        --output-dir results/rees46

    python scripts/run_benchmark.py --dataset retailrocket \
        --data-dir data/raw/retailrocket --output-dir results/retailrocket

    python scripts/run_benchmark.py --dataset movielens \
        --data-dir data/raw/ml-1m --output-dir results/movielens

    python scripts/run_benchmark.py --dataset simulation --output-dir results/simulation

The script saves per-user (or per-seed for simulation) raw data, bootstrap
summary, paired p-values, Friedman omnibus, Nemenyi post-hoc matrices, and
bar-chart PNGs for conversion/trust/volume.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lamf.datasets import load_movielens_1m, load_retailrocket, load_rees46_cosmetics
from lamf.evaluation import (
    evaluate_policies_on_event_log,
    bootstrap_summary,
    friedman_nemenyi,
)
from lamf.simulation import run_simulation, SimulationConfig
from scipy import stats


def _save_sim_outputs(sim_df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sim_df.to_csv(os.path.join(out_dir, "sim_per_seed.csv"), index=False)

    policies = ["Static", "Triggers", "Liquid+Rules", "Liquid+Bandits"]
    conv_pivot = sim_df.pivot(index="seed", columns="policy", values="conversion_rate")[
        policies
    ]
    trust_pivot = sim_df.pivot(index="seed", columns="policy", values="trust_event_rate")[
        policies
    ]

    try:
        import scikit_posthocs as sp
    except ImportError:
        print("WARNING: scikit-posthocs not installed; skipping Nemenyi", file=sys.stderr)
        sp = None

    omnibus = []
    for metric_name, data in [
        ("conversion_rate", conv_pivot),
        ("trust_event_rate", trust_pivot),
    ]:
        stat, p = stats.friedmanchisquare(*[data[pol].values for pol in policies])
        n = len(data)
        W = stat / (n * 3)
        omnibus.append(
            {
                "metric": metric_name,
                "n": n,
                "k": 4,
                "friedman_chi2": round(stat, 4),
                "df": 3,
                "p_value": p,
                "kendall_W": round(W, 4),
            }
        )
        if sp is not None:
            nem = sp.posthoc_nemenyi_friedman(data.values)
            nem.index = policies
            nem.columns = policies
            suffix = "conv" if "conv" in metric_name else "trust"
            nem.to_csv(os.path.join(out_dir, f"sim_posthoc_{suffix}.csv"))

    pd.DataFrame(omnibus).to_csv(os.path.join(out_dir, "sim_friedman.csv"), index=False)
    print(f"  -> {out_dir}/sim_per_seed.csv ({len(sim_df)} rows)")
    print(f"  -> {out_dir}/sim_friedman.csv")


def _plot_bars(summary: pd.DataFrame, out_prefix: str, dataset_name: str) -> None:
    colors = ["#d9534f", "#f0ad4e", "#5cb85c", "#0275d8"]
    policies = summary["policy"].tolist()
    for metric, ylabel, suffix in [
        ("conversion", "Conversion proxy", "conversion"),
        ("trust", "Trust proxy (off-category rate)", "trust"),
        ("volume", "Volume proxy (action rate)", "volume"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = np.arange(len(summary))
        y = summary[f"{metric}_mean"].values
        yerr = np.vstack(
            [
                y - summary[f"{metric}_ci95_low"].values,
                summary[f"{metric}_ci95_high"].values - y,
            ]
        )
        ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=5, color="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset_name} benchmark: {suffix} with 95% CI", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{metric}.png", dpi=200)
        plt.close(fig)


def _run_replay(loader_fn, loader_kwargs, out_dir, dataset_label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {dataset_label}...")
    events, item_to_cat_idx, n_categories = loader_fn(**loader_kwargs)
    print(
        f"  {len(events)} events, {events['user_id'].nunique()} users, {n_categories} categories"
    )

    print("Evaluating policies...")
    per_user = evaluate_policies_on_event_log(
        events=events,
        item_to_cat_idx=item_to_cat_idx,
        n_categories=n_categories,
    )
    print(f"  {len(per_user)} users produced test decisions")
    per_user.to_csv(os.path.join(out_dir, "per_user.csv"), index=False)

    print("Bootstrap (500 resamples)...")
    summary, pvals = bootstrap_summary(per_user, n_bootstrap=500, seed=42)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pvals.to_csv(os.path.join(out_dir, "pvals.csv"), index=False)

    print("Friedman + Nemenyi...")
    omnibus, posthocs = friedman_nemenyi(per_user)
    omnibus.to_csv(os.path.join(out_dir, "friedman.csv"), index=False)
    for metric_prefix, nem in posthocs.items():
        nem.to_csv(os.path.join(out_dir, f"posthoc_{metric_prefix}.csv"))

    print("Plotting...")
    _plot_bars(summary, os.path.join(out_dir, f"fig_{dataset_label.lower()}"), dataset_label)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== FRIEDMAN ===")
    print(omnibus.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run LAMF benchmark.")
    parser.add_argument(
        "--dataset",
        choices=["simulation", "movielens", "retailrocket", "rees46"],
        required=True,
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-seeds", type=int, default=10, help="Simulation only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "simulation":
        print(f"Running simulation with {args.n_seeds} seeds...")
        cfg = SimulationConfig()
        sim_df = run_simulation(n_seeds=args.n_seeds, cfg=cfg)
        _save_sim_outputs(sim_df, args.output_dir)
        return

    if args.data_dir is None:
        parser.error(f"--data-dir is required for dataset={args.dataset}")

    if args.dataset == "movielens":
        _run_replay(load_movielens_1m, {"data_dir": args.data_dir}, args.output_dir, "MovieLens")
    elif args.dataset == "retailrocket":
        _run_replay(
            load_retailrocket, {"data_dir": args.data_dir}, args.output_dir, "RetailRocket"
        )
    elif args.dataset == "rees46":
        _run_replay(
            load_rees46_cosmetics,
            {"data_dir": args.data_dir},
            args.output_dir,
            "REES46",
        )


if __name__ == "__main__":
    main()
