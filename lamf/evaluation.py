"""
Evaluation: runs a policy comparison on an event log and computes inference.

Pipeline:
  1. evaluate_policies_on_event_log(events_df) -> per-user DataFrame of
     conversion/trust/volume proxies for each policy.
  2. bootstrap_summary(per_user_df)           -> 95% CIs and paired P-values.
  3. friedman_nemenyi(per_user_df)            -> omnibus + post-hoc tables.

Conventions
-----------
events_df must contain columns:
  - user_id (int)
  - item_id (int)
  - ts (float, seconds)
  - cat_idx (int, category index into the ontology)
  - weight (float, event-to-intent weight)

Conversion proxy = 1 if the user's next item appears in the recommendation set.
Trust proxy      = 1 if the top recommended item is outside the user's two
                   strongest slow-decay categories.
Volume proxy     = 1 if the policy emitted at least one recommendation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from lamf.intent import IntentState, top_k_categories
from lamf.policies import (
    StaticPolicy,
    TriggerPolicy,
    LiquidRulesPolicy,
    LiquidBanditsPolicy,
    BetaBandit,
)


POLICIES_DEFAULT = ["Static", "Triggers", "Liquid+Rules", "Liquid+Bandits"]


def trust_mismatch(
    recommended_item: Optional[int],
    item_to_cat_idx: Dict[int, int],
    top2_slow_cats: List[int],
) -> float:
    """1.0 if the recommendation's category is not in the user's top-2 slow
    preferences; 0.0 if no recommendation or no preferences yet."""
    if recommended_item is None:
        return 0.0
    cat = item_to_cat_idx.get(int(recommended_item))
    if cat is None or not top2_slow_cats:
        return 0.0
    return 0.0 if cat in top2_slow_cats else 1.0


def _split_temporal(events: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user 80/20 temporal split; keeps users with >=10 events."""
    events = events.sort_values(["user_id", "ts"])
    train_parts, test_parts = [], []
    for uid, grp in events.groupby("user_id"):
        n = len(grp)
        if n < 10:
            continue
        split = max(n - 5, int(np.floor(n * (1 - test_frac))))
        split = min(max(split, 5), n - 1)
        train_parts.append(grp.iloc[:split])
        test_parts.append(grp.iloc[split:])
    if not train_parts:
        return (
            events.iloc[:0].copy(),
            events.iloc[:0].copy(),
        )
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def evaluate_policies_on_event_log(
    events: pd.DataFrame,
    item_to_cat_idx: Dict[int, int],
    n_categories: int,
    half_life_fast_days: float = 1.0,
    half_life_slow_days: float = 7.0,
    theta: float = 0.55,
    theta_safe: float = 0.65,
    top_k: int = 10,
    bandit_epsilon: float = 0.05,
    test_frac: float = 0.2,
    min_test_decisions: int = 3,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all four policies on a per-user temporal-split replay.

    Returns per-user DataFrame with columns conv_{policy}, trust_{policy},
    vol_{policy} for each of the four policies.
    """
    rng = np.random.default_rng(seed)
    train, test = _split_temporal(events, test_frac=test_frac)

    if len(train) == 0 or len(test) == 0:
        return pd.DataFrame()

    pop_rank = (
        train.groupby("item_id").size().sort_values(ascending=False).index.astype(int).tolist()
    )

    train_by_user = {u: g.sort_values("ts") for u, g in train.groupby("user_id")}
    test_by_user = {u: g.sort_values("ts") for u, g in test.groupby("user_id")}

    static = StaticPolicy()
    trigger = TriggerPolicy()
    rules = LiquidRulesPolicy(theta=theta, theta_safe=theta_safe, top_k=top_k)
    bandit_policy = LiquidBanditsPolicy(
        theta=theta, theta_safe=theta_safe, top_k=top_k, epsilon=bandit_epsilon,
        bandit=BetaBandit(n_arms=3),
    )

    per_user_rows = []
    n_total = len(test_by_user)
    for i, (uid, tg) in enumerate(test_by_user.items()):
        hist = train_by_user.get(uid)
        if hist is None or len(hist) < 5 or len(tg) < 2:
            continue

        seen = set(hist["item_id"].astype(int).tolist())
        last_item = int(hist.iloc[-1]["item_id"])

        # Build intent state with history
        intent = IntentState(
            n_categories=n_categories,
            half_life_fast_days=half_life_fast_days,
            half_life_slow_days=half_life_slow_days,
        )
        for _, row in hist.iterrows():
            intent.update(int(row["cat_idx"]), float(row["weight"]), float(row["ts"]))

        t_items = tg["item_id"].values.astype(int)
        t_ts = tg["ts"].values.astype(float)
        t_ci = tg["cat_idx"].values.astype(int)
        t_w = tg["weight"].values.astype(float)

        conv = {p: [] for p in POLICIES_DEFAULT}
        trust = {p: [] for p in POLICIES_DEFAULT}
        vol = {p: [] for p in POLICIES_DEFAULT}

        for j in range(len(tg) - 1):
            t_now = float(t_ts[j])
            next_item = int(t_items[j + 1])

            top2_slow = top_k_categories(intent.slow_prefs(t_now), 2)

            def _record(name, recs):
                first = recs[0] if recs else None
                conv[name].append(1.0 if next_item in recs else 0.0)
                trust[name].append(trust_mismatch(first, item_to_cat_idx, top2_slow))
                vol[name].append(1.0 if first is not None else 0.0)

            # Static
            s_recs = static.select(pop_rank=pop_rank, seen=seen)
            _record("Static", s_recs)

            # Triggers (based on last-item category)
            last_cat = item_to_cat_idx.get(int(last_item))
            t_recs = trigger.select(
                pop_rank=pop_rank,
                seen=seen,
                last_item_cat_idx=last_cat,
                item_to_cat_idx=item_to_cat_idx,
            )
            _record("Triggers", t_recs)

            # Liquid + Rules
            r_recs = rules.select(
                pop_rank=pop_rank,
                seen=seen,
                intent=intent,
                t_now=t_now,
                item_to_cat_idx=item_to_cat_idx,
            )
            _record("Liquid+Rules", r_recs)

            # Liquid + Bandits (returns (recs, arm))
            b_recs, arm = bandit_policy.select(
                pop_rank=pop_rank,
                seen=seen,
                intent=intent,
                t_now=t_now,
                item_to_cat_idx=item_to_cat_idx,
                rng=rng,
            )
            _record("Liquid+Bandits", b_recs)

            # Update bandit with observed reward (conversion - 0.4*trust)
            cb = 1.0 if b_recs and next_item in b_recs else 0.0
            tb = trust_mismatch(b_recs[0] if b_recs else None, item_to_cat_idx, top2_slow)
            reward_obs = float(cb - 0.4 * tb)
            bandit_policy.bandit.update(arm, max(0.0, min(1.0, (reward_obs + 1.0) / 2.0)))

            # Advance history: fold the current test event into intent/seen
            intent.update(int(t_ci[j]), float(t_w[j]), float(t_ts[j]))
            last_item = int(t_items[j])
            seen.add(int(t_items[j]))

        if len(conv["Triggers"]) < min_test_decisions:
            continue

        row = {"user_id": int(uid), "n_decisions": len(conv["Triggers"])}
        for p in POLICIES_DEFAULT:
            row[f"conv_{p}"] = float(np.mean(conv[p]))
            row[f"trust_{p}"] = float(np.mean(trust[p]))
            row[f"vol_{p}"] = float(np.mean(vol[p]))
        per_user_rows.append(row)

        if verbose and (i + 1) % 3000 == 0:
            print(f"  {i + 1}/{n_total} users...", flush=True)

    return pd.DataFrame(per_user_rows)


def bootstrap_summary(
    per_user_df: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int = 42,
    policies: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary with 95% CIs, pairwise-vs-Triggers P-values)."""
    if policies is None:
        policies = POLICIES_DEFAULT
    rng = np.random.default_rng(seed)
    n = len(per_user_df)
    boot: Dict[Tuple[str, str], list] = {
        (m, p): [] for m in ("conv", "trust", "vol") for p in policies
    }
    diff: Dict[Tuple[str, str], list] = {}
    non_triggers = [p for p in policies if p != "Triggers"]
    for m in ("conv", "trust"):
        for p in non_triggers:
            diff[(m, p)] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s = per_user_df.iloc[idx]
        for p in policies:
            for m in ("conv", "trust", "vol"):
                boot[(m, p)].append(float(s[f"{m}_{p}"].mean()))
        if "Triggers" in policies:
            tc = s["conv_Triggers"].values
            tt = s["trust_Triggers"].values
            for p in non_triggers:
                diff[("conv", p)].append(float(np.mean(s[f"conv_{p}"].values - tc)))
                diff[("trust", p)].append(float(np.mean(s[f"trust_{p}"].values - tt)))

    def _ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(np.mean(vals)), float(lo), float(hi)

    summary_rows = []
    for p in policies:
        cm, clo, chi = _ci(boot[("conv", p)])
        tm, tlo, thi = _ci(boot[("trust", p)])
        vm, vlo, vhi = _ci(boot[("vol", p)])
        summary_rows.append(
            {
                "policy": p,
                "conversion_mean": cm,
                "conversion_ci95_low": clo,
                "conversion_ci95_high": chi,
                "trust_mean": tm,
                "trust_ci95_low": tlo,
                "trust_ci95_high": thi,
                "volume_mean": vm,
                "volume_ci95_low": vlo,
                "volume_ci95_high": vhi,
            }
        )
    summary = pd.DataFrame(summary_rows)

    pval_rows = []
    for p in non_triggers:
        for m, label in (("conv", "conversion"), ("trust", "trust")):
            diffs = np.array(diff[(m, p)])
            p_two = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
            pval_rows.append(
                {
                    "comparison": f"{p} vs Triggers",
                    "metric": label,
                    "diff_mean": float(diffs.mean()),
                    "p_value_bootstrap": float(p_two),
                }
            )
    pvals = pd.DataFrame(pval_rows)
    return summary, pvals


def friedman_nemenyi(
    per_user_df: pd.DataFrame,
    policies: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Friedman omnibus (blocked by user) + Nemenyi post-hoc per metric."""
    try:
        import scikit_posthocs as sp
    except ImportError as e:
        raise ImportError(
            "scikit-posthocs is required for Nemenyi post-hoc. "
            "Install with: pip install scikit-posthocs"
        ) from e

    if policies is None:
        policies = POLICIES_DEFAULT
    n = len(per_user_df)
    omnibus_rows = []
    posthocs = {}

    for metric_prefix, label in (
        ("conv", "conversion_proxy"),
        ("trust", "trust_proxy"),
        ("vol", "volume_proxy"),
    ):
        cols = [f"{metric_prefix}_{p}" for p in policies]
        data = per_user_df[cols].values
        stat, p = stats.friedmanchisquare(*[data[:, i] for i in range(len(policies))])
        W = stat / (n * (len(policies) - 1))
        omnibus_rows.append(
            {
                "metric": label,
                "n": n,
                "k": len(policies),
                "friedman_chi2": round(stat, 4),
                "df": len(policies) - 1,
                "p_value": p,
                "kendall_W": round(W, 4),
            }
        )
        nem = sp.posthoc_nemenyi_friedman(data)
        nem.index = policies
        nem.columns = policies
        posthocs[metric_prefix] = nem

    return pd.DataFrame(omnibus_rows), posthocs
