"""
Dataset loaders: normalize different public datasets into the common schema
expected by lamf.evaluation.

All loaders return a tuple:
    (events_df, item_to_cat_idx, n_categories)

events_df columns:
    user_id (int), item_id (int), ts (float seconds),
    cat_idx (int), weight (float)

To add a new dataset, implement a loader with the same signature and call
it from scripts/run_benchmark.py.
"""

import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# MovieLens 1M
# ─────────────────────────────────────────────────────────────────────
def load_movielens_1m(data_dir: str) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """Load MovieLens 1M. Expects ratings.dat and movies.dat in data_dir."""
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path = os.path.join(data_dir, "movies.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "ts"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )

    # Intent ontology: primary genre of each movie
    all_genres = set()
    for g in movies["genres"].dropna():
        all_genres.update(g.split("|"))
    genre_list = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    movies["primary_genre"] = movies["genres"].fillna("").apply(
        lambda s: s.split("|")[0] if s else None
    )
    movies = movies.dropna(subset=["primary_genre"])
    item_to_cat = {
        int(r["item_id"]): genre_to_idx[r["primary_genre"]] for _, r in movies.iterrows()
    }

    ratings = ratings[ratings["item_id"].isin(item_to_cat)].copy()
    ratings["cat_idx"] = ratings["item_id"].map(item_to_cat).astype(int)
    # Weight: only positive ratings contribute to intent
    ratings["weight"] = (ratings["rating"] > 2.5).astype(float) * ratings["rating"] / 5.0

    events = ratings[["user_id", "item_id", "ts", "cat_idx", "weight"]].copy()
    events = events[events["weight"] > 0]

    return events, item_to_cat, len(genre_list)


# ─────────────────────────────────────────────────────────────────────
# RetailRocket
# ─────────────────────────────────────────────────────────────────────
def load_retailrocket(data_dir: str) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """Load RetailRocket. Expects events.csv and category_tree.csv;
    optionally item_properties_part1.csv / _part2.csv for category mapping."""
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    events = events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    events["ts"] = events["timestamp"] / 1000.0
    event_weights = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}
    events["weight"] = events["event"].map(event_weights).fillna(1.0)

    # Build item -> category mapping
    cat_path = os.path.join(data_dir, "item_categories.csv")
    if os.path.exists(cat_path):
        icat = pd.read_csv(cat_path)
        item_cat = dict(zip(icat["itemid"].astype(int), icat["categoryid"].astype(int)))
    else:
        # Extract from item_properties files
        import csv
        item_cat = {}
        for fname in ("item_properties_part1.csv", "item_properties_part2.csv"):
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 4 and row[2] == "categoryid":
                        try:
                            item_cat[int(row[1])] = int(row[3])
                        except ValueError:
                            pass

    # Collapse via category tree to top-level categories
    tree_path = os.path.join(data_dir, "category_tree.csv")
    if os.path.exists(tree_path):
        ct = pd.read_csv(tree_path)
        tree = {
            int(r["categoryid"]): (int(r["parentid"]) if pd.notna(r["parentid"]) else None)
            for _, r in ct.iterrows()
        }

        def _top(c, depth=0):
            if depth > 10:
                return c
            parent = tree.get(c)
            if parent is None or parent == c:
                return c
            return _top(parent, depth + 1)

        top_map = {c: _top(c) for c in set(item_cat.values())}
        item_cat = {it: top_map.get(cat, cat) for it, cat in item_cat.items()}

    events["category"] = events["item_id"].map(item_cat)
    events = events.dropna(subset=["category"])
    events["category"] = events["category"].astype(int)

    unique_cats = sorted(events["category"].unique())
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    events["cat_idx"] = events["category"].map(cat_to_idx).astype(int)
    item_to_cat_idx = {it: cat_to_idx[cat] for it, cat in item_cat.items() if cat in cat_to_idx}

    # Filter to users with >=10 events
    uc = events.groupby("user_id").size()
    keep = uc[uc >= 10].index
    events = events[events["user_id"].isin(keep)].copy()

    return (
        events[["user_id", "item_id", "ts", "cat_idx", "weight"]].copy(),
        item_to_cat_idx,
        len(unique_cats),
    )


# ─────────────────────────────────────────────────────────────────────
# REES46 Cosmetics
# ─────────────────────────────────────────────────────────────────────
def load_rees46_cosmetics(
    data_dir: str,
    months: Tuple[str, ...] = ("2019-Oct.csv", "2019-Nov.csv", "2019-Dec.csv"),
    n_top_categories: int = 30,
    min_events_per_user: int = 15,
    subsample_users: int = 15_000,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """Load REES46 Cosmetics (Kaggle mkechinov dataset).
    Expects monthly CSVs like 2019-Oct.csv in data_dir.
    """
    cols = ["event_time", "event_type", "product_id", "category_id", "user_id"]
    parts = []
    for fname in months:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing REES46 file: {fpath}")
        parts.append(pd.read_csv(fpath, usecols=cols))
    events = pd.concat(parts, ignore_index=True)
    del parts

    events["ts"] = pd.to_datetime(events["event_time"]).astype(np.int64) // 10**9

    event_weights = {"view": 1.0, "cart": 2.0, "remove_from_cart": 0.0, "purchase": 3.0}
    events["weight"] = events["event_type"].map(event_weights).fillna(1.0)

    # Collapse categories: keep top-N by volume, bucket the rest into -1
    cat_counts = events["category_id"].value_counts()
    top_cats = set(cat_counts.head(n_top_categories).index)
    events["top_cat"] = events["category_id"].apply(lambda c: c if c in top_cats else -1)

    # Filter users with >=min_events_per_user
    uc = events.groupby("user_id").size()
    keep = uc[uc >= min_events_per_user].index
    events = events[events["user_id"].isin(keep)].copy()

    # Subsample users for tractability
    rng = np.random.default_rng(random_seed)
    uids = events["user_id"].unique()
    if len(uids) > subsample_users:
        uids = rng.choice(uids, size=subsample_users, replace=False)
        events = events[events["user_id"].isin(uids)].copy()

    unique_cats = sorted(events["top_cat"].unique())
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    events["cat_idx"] = events["top_cat"].map(cat_to_idx).astype(int)

    # Build item -> category index mapping
    item_cat = dict(
        zip(
            events["product_id"].astype(int),
            events["top_cat"].astype(int),
        )
    )
    item_to_cat_idx = {it: cat_to_idx[c] for it, c in item_cat.items() if c in cat_to_idx}

    events = events.sort_values(["user_id", "ts"])
    return (
        events.rename(columns={"product_id": "item_id"})[
            ["user_id", "item_id", "ts", "cat_idx", "weight"]
        ].copy(),
        item_to_cat_idx,
        len(unique_cats),
    )
