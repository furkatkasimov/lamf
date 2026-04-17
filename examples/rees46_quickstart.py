#!/usr/bin/env python3
"""
Minimal working example: load REES46, run the full LAMF evaluation,
and print the key results.

Expects REES46 data in data/raw/rees46/ (see docs/DATASETS.md).

Run from the repo root:
    python examples/rees46_quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lamf.datasets import load_rees46_cosmetics
from lamf.evaluation import (
    evaluate_policies_on_event_log,
    bootstrap_summary,
    friedman_nemenyi,
)


def main():
    data_dir = "data/raw/rees46"

    print("1. Loading REES46 Cosmetics (Oct-Dec 2019, subsampled)...")
    events, item_to_cat_idx, n_categories = load_rees46_cosmetics(data_dir)
    print(f"   {len(events):,} events, {events['user_id'].nunique():,} users, "
          f"{n_categories} categories\n")

    print("2. Running policy evaluation (takes ~30 seconds)...")
    per_user = evaluate_policies_on_event_log(
        events=events,
        item_to_cat_idx=item_to_cat_idx,
        n_categories=n_categories,
        half_life_fast_days=1.0,
        half_life_slow_days=7.0,
        theta=0.55,
        theta_safe=0.65,
        top_k=10,
        bandit_epsilon=0.05,
        verbose=True,
    )
    print(f"   {len(per_user):,} users produced test decisions\n")

    print("3. Bootstrap summary (500 resamples)...")
    summary, pvals = bootstrap_summary(per_user, n_bootstrap=500, seed=42)
    print(summary.to_string(index=False))
    print()

    print("4. Friedman omnibus + Nemenyi post-hoc...")
    omnibus, posthocs = friedman_nemenyi(per_user)
    print(omnibus.to_string(index=False))
    print()

    print("5. Nemenyi adjusted p-values for trust proxy:")
    print(posthocs["trust"].round(6).to_string())
    print()

    print("Expected pattern (all three benchmarks in the paper):")
    print("   Static trust >> Triggers trust > Liquid+Rules = Liquid+Bandits = 0")
    print("   Friedman on trust: p ≈ 0, Kendall's W > 0.5")
    print("   Nemenyi (Liquid vs Triggers) p_adj < 1e-7")
    print("   Nemenyi (Liquid+Rules vs Liquid+Bandits) p_adj = 1.0")


if __name__ == "__main__":
    main()
