# LAMF: Liquid Audience Measurement Framework

Reference implementation and reproducibility package for the paper
**"Liquid Audiences: Measuring Intent-Based Segmentation in Real Time"**
(Kasimov, 2025).

This repository provides the core framework described in the manuscript, the
policies evaluated in Sections 5.1–5.5, the evaluation machinery
(bootstrap CIs, Friedman omnibus, Nemenyi post-hoc), and scripts that
reproduce every numerical result and figure in the paper.

## What this is

LAMF is a **measurement and governance layer** for real-time intent-based
segmentation. The paper makes three concrete claims:

1. Intent can be represented as a decaying segment-of-one state
   (half-life, TTL).
2. Governed policies (consent, caps, do-nothing, capped exploration)
   reduce trust-damaging contacts relative to always-on baselines.
3. The same governance pattern holds across three public benchmarks
   (MovieLens 1M, RetailRocket, REES46 Cosmetics) and a controlled
   simulation.

This repository is the executable version of those claims. A reviewer
can clone it, install the dependencies, run the tests, run the
benchmarks, and compare the numerical output against the tables and
figures in the paper.

## Relationship to the paper

| Paper section | Code |
|---|---|
| Section 3.2, Eq. 4 (decayed accumulator) | `lamf/intent.py` |
| Section 3.5 (four policies) | `lamf/policies.py` |
| Section 3.6 (evaluation protocol) | `lamf/evaluation.py` |
| Section 5.1–5.2 (simulation) | `lamf/simulation.py` |
| Section 5.3 (MovieLens 1M replay) | `scripts/run_benchmark.py --dataset movielens` |
| Section 5.4 (RetailRocket replay) | `scripts/run_benchmark.py --dataset retailrocket` |
| Section 5.5 (REES46 Cosmetics replay) | `scripts/run_benchmark.py --dataset rees46` |
| Friedman + Nemenyi (Table S2) | `lamf.evaluation.friedman_nemenyi` |

## Repository layout

```
lamf/                       # Core package
├── intent.py               # Decayed accumulator (Eq. 4), IntentState class
├── policies.py             # Static, Triggers, Liquid+Rules, Liquid+Bandits
├── evaluation.py           # Event-log replay, bootstrap, Friedman, Nemenyi
├── simulation.py           # Section 5.1 illustrative simulation
└── datasets.py             # Loaders for MovieLens, RetailRocket, REES46

scripts/
├── run_benchmark.py        # Unified CLI runner for all four experiments
└── reproduce_all.sh        # One-shot reproduction of all paper results

tests/
└── test_lamf.py            # 30 verification tests (pytest)

configs/
└── default.yaml            # Default hyperparameters (theta, half-lives, etc.)

docs/
├── DATASETS.md             # How to obtain each public dataset
└── EXTENDING.md            # How to add a new dataset or policy

examples/
└── rees46_quickstart.py    # Minimal working example end-to-end

data/raw/                   # Place downloaded datasets here (see DATASETS.md)
results/                    # Benchmark outputs land here
```

## Quick start

### Prerequisites

- Python 3.9 or later
- ~4 GB RAM (REES46 is the largest workload)
- Optional: ~8 GB disk space if all three public datasets are downloaded

### Installation

```bash
git clone https://github.com/furkatkasimov/lamf.git
cd lamf
pip install -r requirements.txt
pip install -e .
```

### Verify installation (recommended before running benchmarks)

```bash
pytest tests/ -v
```

All 30 tests should pass in under 60 seconds.

### Run the illustrative simulation

The simulation needs no external data and takes ~1 minute:

```bash
python scripts/run_benchmark.py --dataset simulation --output-dir results/simulation
```

This reproduces the results in Section 5.1–5.2 (Table S1 simulation rows,
Table S2 simulation rows, Figure 2).

### Run a public-data replay (REES46 Cosmetics example)

Download the REES46 Cosmetics Shop dataset from Kaggle
(`kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop`)
and unzip the CSV files into `data/raw/rees46/`. Then run:

```bash
python scripts/run_benchmark.py \
    --dataset rees46 \
    --data-dir data/raw/rees46 \
    --output-dir results/rees46
```

Runtime: ~1 minute on a modern laptop. See `docs/DATASETS.md` for the
other datasets.

### Reproduce every result in the paper

Once all datasets are in place:

```bash
bash scripts/reproduce_all.sh
```

This runs the simulation, MovieLens 1M, RetailRocket, and REES46
benchmarks in sequence and writes results to `results/`.

## Output format

Every benchmark (simulation or replay) writes the following files into its
output directory:

| File | Description |
|---|---|
| `per_user.csv` (or `sim_per_seed.csv`) | Raw per-user / per-seed metric values |
| `summary.csv` | Policy means with 95% bootstrap CIs |
| `pvals.csv` | Paired-bootstrap P-values vs. Triggers |
| `friedman.csv` | Friedman omnibus ($\chi^2_F$, df, p, Kendall's W) |
| `posthoc_conv.csv` | Nemenyi pairwise adj. p-values — conversion |
| `posthoc_trust.csv` | Nemenyi pairwise adj. p-values — trust |
| `posthoc_vol.csv` | Nemenyi pairwise adj. p-values — volume |
| `fig_<name>_conversion.png` | Conversion bar chart with 95% CI |
| `fig_<name>_trust.png` | Trust bar chart with 95% CI |
| `fig_<name>_volume.png` | Volume bar chart with 95% CI |

## How to add a new dataset

The expected input schema is a pandas DataFrame with columns:

```
user_id (int), item_id (int), ts (float, seconds), cat_idx (int), weight (float)
```

Plus a dictionary `item_to_cat_idx: Dict[int, int]` and the number of
categories. To add a new dataset:

1. Write a loader function in `lamf/datasets.py` that returns
   `(events_df, item_to_cat_idx, n_categories)`.
2. Add a CLI option in `scripts/run_benchmark.py`.
3. Run the same evaluation pipeline:

```python
from lamf.evaluation import evaluate_policies_on_event_log, bootstrap_summary, friedman_nemenyi
from lamf.datasets import load_your_new_dataset

events, item_to_cat, n_cats = load_your_new_dataset("data/raw/your_dataset")

per_user = evaluate_policies_on_event_log(
    events=events,
    item_to_cat_idx=item_to_cat,
    n_categories=n_cats,
    half_life_fast_days=1.0,   # tune for your domain
    half_life_slow_days=7.0,   # tune for your domain
    theta=0.55,
    theta_safe=0.65,
    top_k=10,
)

summary, pvals = bootstrap_summary(per_user, n_bootstrap=500)
omnibus, posthocs = friedman_nemenyi(per_user)
```

See `docs/EXTENDING.md` for a worked example adding the Taobao UserBehavior
dataset.

## How to tune for a different domain

Hyperparameters controlling the policy behaviour, listed in
`configs/default.yaml`:

| Parameter | Default | Meaning |
|---|---|---|
| `half_life_fast_days` | 1.0 | Decay half-life for the fast/exploration accumulator |
| `half_life_slow_days` | 7.0 | Decay half-life for the slow/stable accumulator |
| `theta` | 0.55 | Activation threshold (min confidence to act) |
| `theta_safe` | 0.65 | Safety threshold (single vs top-k behaviour) |
| `top_k` | 10 | Size of the recommendation list above theta_safe |
| `bandit_epsilon` | 0.05 | Bandit exploration cap |

Guidance from Section 5:

- Fast e-commerce browsing: `half_life_fast ≈ 1 day`, `half_life_slow ≈ 7 days`
- Movie ratings / slower content consumption: `half_life_fast ≈ 3 days`,
  `half_life_slow ≈ 14 days`
- Urgency-driven intents (price-checks): use shorter half-lives
- Replenishment intents: use longer half-lives (weeks)

## Interpreting the metrics

The three proxies reported are deliberately simple so they transfer across
domains. They are **not** a substitute for field measurement; they diagnose
the governance pattern of a policy, not its real-world profit.

- **Conversion proxy**: fraction of decision points where the user's
  next-interacted item appears in the recommended set.
- **Trust proxy**: fraction of decisions where the top-recommended item
  lies outside the user's two strongest slow-decay categories
  (an "off-category contact"). Near-zero trust for governed policies
  is partly a design consequence of confidence gating and category-aligned
  recommendation; interpret accordingly.
- **Volume proxy**: fraction of decision points where the policy emits
  any recommendation.
- **Net reward** (for framing only): `conversion − 0.4 × trust`. The 0.4
  weight is a transparent compromise, not tuned; reweighting can change
  which policy has the highest net reward without new data.

## Reproducibility notes

- All random seeds are fixed (simulation seeds 0–9; bootstrap seed 42;
  evaluation seed 0). Running the same benchmark twice should produce
  byte-identical output.
- The REES46 loader subsamples 15,000 users by default (`random_seed=42`);
  this matches the paper. Different subsamples produce slightly different
  numbers but the qualitative findings hold (see paper limitation 8).
- Results in `results/` match the tables and figures in the paper to
  within floating-point tolerance.

## Citation

If you use this code, please cite:

```bibtex
@article{kasimov2025liquid,
  title   = {Liquid Audiences: Measuring Intent-Based Segmentation in Real Time},
  author  = {Kasimov, Furkat},
  year    = {2025},
  note    = {Preprint},
}
```

## License

MIT. See `LICENSE`.

## Contact

For questions about the framework or the paper, open an issue on this
repository or contact the author via the ORCID on the manuscript.
