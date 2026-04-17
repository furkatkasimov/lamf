# Extending LAMF

## Adding a new dataset

LAMF's evaluation pipeline consumes a canonical event-log schema, so
supporting a new dataset is mostly a matter of writing a loader.

### Step 1: Write a loader in `lamf/datasets.py`

The loader must return three things:

```python
def load_mydataset(data_dir: str) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """
    Returns:
        events: DataFrame with columns user_id, item_id, ts, cat_idx, weight
        item_to_cat_idx: mapping from item_id -> category index
        n_categories: size of the intent ontology K
    """
    ...
```

The `weight` column is the event-to-intent weight (paper Section 3.2):
typical choices are 1.0 for views, 2.0 for add-to-cart, 3.0 for
purchases. Events with weight 0 are ignored by the intent state but can
still appear in the log.

### Step 2: Wire into the CLI

In `scripts/run_benchmark.py`, add:

```python
parser.add_argument("--dataset", choices=[..., "mydataset"], required=True)
...
elif args.dataset == "mydataset":
    _run_replay(load_mydataset, {"data_dir": args.data_dir}, args.output_dir, "MyDataset")
```

### Step 3: Tune hyperparameters for the domain

Different domains have different natural decay timescales. If your
dataset is grocery/replenishment, you probably want longer half-lives;
for urgent intents (price-check), shorter. Override via
`evaluate_policies_on_event_log(...)`:

```python
per_user = evaluate_policies_on_event_log(
    events=events,
    item_to_cat_idx=item_to_cat,
    n_categories=n_cats,
    half_life_fast_days=2.0,    # <-- tune
    half_life_slow_days=14.0,   # <-- tune
)
```

### Worked example: Taobao UserBehavior

Suppose you have the Taobao UserBehavior.csv file (columns:
`user_id, item_id, category_id, behavior_type, timestamp` with
behavior_type in `{pv, cart, fav, buy}`).

```python
def load_taobao(data_dir: str):
    df = pd.read_csv(
        os.path.join(data_dir, "UserBehavior.csv"),
        names=["user_id", "item_id", "category_id", "behavior_type", "ts"],
    )
    weights = {"pv": 1.0, "cart": 2.0, "fav": 1.5, "buy": 3.0}
    df["weight"] = df["behavior_type"].map(weights).fillna(1.0)

    # Filter short histories
    uc = df.groupby("user_id").size()
    df = df[df["user_id"].isin(uc[uc >= 10].index)].copy()

    # Build category index
    unique_cats = sorted(df["category_id"].unique())
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    df["cat_idx"] = df["category_id"].map(cat_to_idx).astype(int)

    item_to_cat_idx = dict(zip(df["item_id"].astype(int), df["cat_idx"].astype(int)))
    return (
        df[["user_id", "item_id", "ts", "cat_idx", "weight"]].copy(),
        item_to_cat_idx,
        len(unique_cats),
    )
```

That's it. The same evaluation pipeline, bootstrap, Friedman/Nemenyi,
and plotting work without changes.

## Adding a new policy

Policies live in `lamf/policies.py`. Each policy is a dataclass with a
`select(...)` method returning a list of recommended item IDs (empty for
do-nothing):

```python
@dataclass
class MyPolicy:
    name: str = "MyPolicy"

    def select(
        self,
        pop_rank: List[int],
        seen: Set[int],
        intent: Optional[IntentState] = None,
        t_now: Optional[float] = None,
        item_to_cat_idx: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> List[int]:
        # your logic here
        return [some_item_id]
```

To include your policy in the benchmark, extend the evaluator in
`lamf/evaluation.py` (the loop over four policies is explicit, not
dynamic, by design — the paper reports a fixed comparison set).

## Changing the intent weight scheme

Event-to-intent weights live in the dataset loader (paper Section 3.2,
Eq. 4). For instance, to treat `remove_from_cart` as a strong negative
signal:

```python
weights = {
    "view": 1.0,
    "cart": 2.0,
    "remove_from_cart": -1.0,   # negative signal
    "purchase": 3.0,
}
```

Note: the current `decayed_category_prefs` guards against negative
preferences with a `if effective[i] > 0` check. If you want negative
signals to propagate, remove that guard and ensure downstream code
(e.g. `top_k_categories`) handles sign correctly.

## Alternative trust proxies

The default trust proxy is off-category mismatch against top-2
slow-decay preferences. If your domain has a better signal (e.g.
unsubscribes, complaints, `remove_from_cart` events), substitute your
own function in `lamf/evaluation.py::trust_mismatch`. The function
signature:

```python
def trust_mismatch(
    recommended_item: Optional[int],
    item_to_cat_idx: Dict[int, int],
    top2_slow_cats: List[int],
) -> float: ...
```
