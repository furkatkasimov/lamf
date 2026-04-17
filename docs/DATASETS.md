# Datasets

All three public datasets are free to download. None are distributed
with this repository; you must obtain them from the original sources
and place the files in `data/raw/<dataset_name>/`.

## 1. MovieLens 1M (Section 5.3)

**Source:** https://grouplens.org/datasets/movielens/1m/
**Size:** ~24 MB compressed
**License:** Non-commercial research use (GroupLens terms)

```bash
mkdir -p data/raw/ml-1m
cd data/raw/ml-1m
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
mv ml-1m/* .
rmdir ml-1m ml-1m.zip
# Files expected: ratings.dat, movies.dat, users.dat
```

Run:
```bash
python scripts/run_benchmark.py \
    --dataset movielens \
    --data-dir data/raw/ml-1m \
    --output-dir results/movielens
```

## 2. RetailRocket (Section 5.4)

**Source:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
**Size:** ~350 MB uncompressed
**License:** CC BY-NC-ND 4.0 (research use)
**Requires:** Free Kaggle account

Download the dataset zip from Kaggle and extract into `data/raw/retailrocket/`.
Expected files:
- `events.csv` (~90 MB)
- `category_tree.csv`
- `item_properties_part1.csv`, `item_properties_part2.csv`

Pre-extract the item→category mapping once (saves ~20 seconds on repeated runs):

```bash
python - <<'EOF'
import csv, pandas as pd
item_cat = {}
for p in ("data/raw/retailrocket/item_properties_part1.csv",
          "data/raw/retailrocket/item_properties_part2.csv"):
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 4 and row[2] == "categoryid":
                try: item_cat[int(row[1])] = int(row[3])
                except: pass
pd.DataFrame(list(item_cat.items()), columns=["itemid","categoryid"]) \
  .to_csv("data/raw/retailrocket/item_categories.csv", index=False)
print(f"Saved {len(item_cat)} items")
EOF
```

Run:
```bash
python scripts/run_benchmark.py \
    --dataset retailrocket \
    --data-dir data/raw/retailrocket \
    --output-dir results/retailrocket
```

## 3. REES46 Cosmetics (Section 5.5)

**Source:** https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop
**Size:** ~2.5 GB uncompressed
**License:** CC BY-NC-SA 4.0 (research use)
**Requires:** Free Kaggle account

Download and extract into `data/raw/rees46/`. Expected files:
- `2019-Oct.csv`, `2019-Nov.csv`, `2019-Dec.csv` (used for the paper)
- `2020-Jan.csv`, `2020-Feb.csv` (not used by default)

By default the loader uses Oct+Nov+Dec 2019. You can change which months
to use by editing the `months` argument in `lamf/datasets.py` or by passing
additional parameters.

Run:
```bash
python scripts/run_benchmark.py \
    --dataset rees46 \
    --data-dir data/raw/rees46 \
    --output-dir results/rees46
```

## Dataset size considerations

The REES46 monthly CSVs are large. If memory is a constraint, you can:

1. Use only a single month: edit `months=("2019-Oct.csv",)` in the
   `load_rees46_cosmetics` call.
2. Reduce the user subsample: pass `subsample_users=5000` to the loader.
3. Increase the minimum events per user: `min_events_per_user=25`.

All three reduce data volume without changing the methodology; the
qualitative finding (trust proxy = 0 for governed policies) is robust
to these choices.

## What results should look like

The paper reports the following values for the trust proxy — any run of
the scripts on the same data should reproduce these to within a few
tenths of a percentage point (exact match for deterministic code paths;
tiny variation only from bootstrap resampling).

| Policy | MovieLens 1M | RetailRocket | REES46 Cosmetics |
|---|---|---|---|
| Static | 31.9% | 83.7% | 97.5% |
| Triggers | 24.4% | 3.1% | 20.0% |
| Liquid+Rules | 0.0% | 0.0% | 0.0% |
| Liquid+Bandits | 0.0% | 0.0% | 0.0% |

If you see substantially different numbers, check:
1. Dataset file integrity (compare row counts with those in the paper).
2. Subsample seed (REES46 defaults to `random_seed=42`).
3. Python and library versions (see `requirements.txt`).
