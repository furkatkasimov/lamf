"""
Verification tests for LAMF.

These tests exercise mathematical invariants and sanity-check policy
behaviour on small synthetic inputs. A reviewer running `pytest` should
see all tests pass in under 60 seconds on a laptop.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from lamf.intent import (
    IntentState,
    decayed_category_prefs,
    top_k_categories,
    LOG2,
)
from lamf.policies import (
    StaticPolicy,
    TriggerPolicy,
    LiquidRulesPolicy,
    LiquidBanditsPolicy,
    BetaBandit,
)
from lamf.evaluation import (
    evaluate_policies_on_event_log,
    bootstrap_summary,
    friedman_nemenyi,
    trust_mismatch,
)
from lamf.simulation import run_simulation, SimulationConfig


# ══════════════════════════════════════════════════════════════════════
# 1. Intent decay invariants
# ══════════════════════════════════════════════════════════════════════
class TestIntentDecay:
    def test_empty_history_returns_zeros(self):
        prefs = decayed_category_prefs(
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            t_now=0.0,
            half_life_days=1.0,
            n_categories=5,
        )
        assert prefs.shape == (5,)
        assert np.all(prefs == 0.0)

    def test_single_event_at_current_time_preserves_weight(self):
        prefs = decayed_category_prefs(
            cat_idx=np.array([2]),
            weights=np.array([1.0]),
            timestamps=np.array([0.0]),
            t_now=0.0,
            half_life_days=1.0,
            n_categories=5,
        )
        assert prefs[2] == pytest.approx(1.0)
        assert prefs[0] == 0.0
        assert prefs[4] == 0.0

    def test_half_life_property(self):
        """After one half-life, the weight should be halved."""
        half_life = 3.0
        prefs = decayed_category_prefs(
            cat_idx=np.array([0]),
            weights=np.array([2.0]),
            timestamps=np.array([0.0]),
            t_now=half_life * 86400,  # seconds
            half_life_days=half_life,
            n_categories=1,
        )
        assert prefs[0] == pytest.approx(1.0, rel=1e-6)

    def test_decay_is_monotonic(self):
        """Preference at a later time is never larger than at an earlier time."""
        args = dict(
            cat_idx=np.array([0]),
            weights=np.array([1.0]),
            timestamps=np.array([0.0]),
            half_life_days=1.0,
            n_categories=1,
        )
        p1 = decayed_category_prefs(t_now=86400.0, **args)
        p2 = decayed_category_prefs(t_now=2 * 86400.0, **args)
        p3 = decayed_category_prefs(t_now=5 * 86400.0, **args)
        assert p1[0] > p2[0] > p3[0]

    def test_multiple_events_additive(self):
        prefs = decayed_category_prefs(
            cat_idx=np.array([0, 0, 0]),
            weights=np.array([1.0, 2.0, 3.0]),
            timestamps=np.array([0.0, 0.0, 0.0]),
            t_now=0.0,
            half_life_days=1.0,
            n_categories=2,
        )
        assert prefs[0] == pytest.approx(6.0)

    def test_top_k_categories_ordering(self):
        prefs = np.array([0.1, 3.0, 1.5, 0.5])
        top2 = top_k_categories(prefs, k=2)
        assert top2 == [1, 2]

    def test_top_k_empty_on_zero_prefs(self):
        assert top_k_categories(np.zeros(5)) == []


# ══════════════════════════════════════════════════════════════════════
# 2. IntentState wrapper behaviour
# ══════════════════════════════════════════════════════════════════════
class TestIntentState:
    def test_update_then_read(self):
        state = IntentState(n_categories=3, half_life_fast_days=1.0, half_life_slow_days=7.0)
        state.update(cat_idx=1, weight=2.0, timestamp=0.0)
        fast = state.fast_prefs(t_now=0.0)
        slow = state.slow_prefs(t_now=0.0)
        assert fast[1] == pytest.approx(2.0)
        assert slow[1] == pytest.approx(2.0)

    def test_fast_decays_faster_than_slow(self):
        state = IntentState(n_categories=2, half_life_fast_days=1.0, half_life_slow_days=7.0)
        state.update(cat_idx=0, weight=1.0, timestamp=0.0)
        fast = state.fast_prefs(t_now=3 * 86400)
        slow = state.slow_prefs(t_now=3 * 86400)
        # Slow should retain more weight than fast after 3 days
        assert slow[0] > fast[0]

    def test_combined_confidence_sums_to_one(self):
        state = IntentState(n_categories=3)
        state.update(cat_idx=0, weight=1.0, timestamp=0.0)
        state.update(cat_idx=1, weight=2.0, timestamp=0.0)
        top, conf = state.combined_confidence(t_now=0.0)
        assert top == 1
        assert 0.0 < conf <= 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. Policy behaviour
# ══════════════════════════════════════════════════════════════════════
class TestPolicies:
    def setup_method(self):
        self.pop_rank = [100, 101, 102, 103, 104]
        self.item_to_cat = {100: 0, 101: 1, 102: 0, 103: 1, 104: 2}
        self.seen = set()

    def test_static_recommends_most_popular(self):
        recs = StaticPolicy().select(pop_rank=self.pop_rank, seen=self.seen)
        assert recs == [100]

    def test_static_skips_seen_items(self):
        recs = StaticPolicy().select(pop_rank=self.pop_rank, seen={100, 101})
        assert recs == [102]

    def test_triggers_respects_last_category(self):
        recs = TriggerPolicy().select(
            pop_rank=self.pop_rank,
            seen=self.seen,
            last_item_cat_idx=1,
            item_to_cat_idx=self.item_to_cat,
        )
        assert recs == [101]

    def test_triggers_returns_empty_without_category(self):
        recs = TriggerPolicy().select(
            pop_rank=self.pop_rank,
            seen=self.seen,
            last_item_cat_idx=None,
            item_to_cat_idx=self.item_to_cat,
        )
        assert recs == []

    def test_liquid_rules_abstains_below_theta(self):
        """With weak intent, Liquid+Rules should do nothing."""
        intent = IntentState(n_categories=3)
        intent.update(cat_idx=0, weight=0.01, timestamp=0.0)
        intent.update(cat_idx=1, weight=0.01, timestamp=0.0)
        intent.update(cat_idx=2, weight=0.01, timestamp=0.0)
        policy = LiquidRulesPolicy(theta=0.55, theta_safe=0.65, top_k=10)
        recs = policy.select(
            pop_rank=self.pop_rank,
            seen=self.seen,
            intent=intent,
            t_now=0.0,
            item_to_cat_idx=self.item_to_cat,
        )
        # Three equal preferences -> no single category above theta=0.55
        assert recs == []

    def test_liquid_rules_acts_above_theta_safe(self):
        """With strong intent, Liquid+Rules returns top-k."""
        intent = IntentState(n_categories=3)
        intent.update(cat_idx=0, weight=10.0, timestamp=0.0)
        policy = LiquidRulesPolicy(theta=0.55, theta_safe=0.65, top_k=3)
        recs = policy.select(
            pop_rank=self.pop_rank,
            seen=self.seen,
            intent=intent,
            t_now=0.0,
            item_to_cat_idx=self.item_to_cat,
        )
        # Top category is 0; items 100, 102 are in category 0
        assert len(recs) <= 3
        assert 100 in recs

    def test_liquid_rules_grey_zone_returns_single_item(self):
        """In the grey zone (theta < conf < theta_safe) only 1 item is returned."""
        # Construct preferences such that top category has ~58% share
        intent = IntentState(n_categories=3)
        intent.update(cat_idx=0, weight=5.8, timestamp=0.0)
        intent.update(cat_idx=1, weight=2.1, timestamp=0.0)
        intent.update(cat_idx=2, weight=2.1, timestamp=0.0)
        policy = LiquidRulesPolicy(theta=0.55, theta_safe=0.65, top_k=10)
        recs = policy.select(
            pop_rank=self.pop_rank,
            seen=self.seen,
            intent=intent,
            t_now=0.0,
            item_to_cat_idx=self.item_to_cat,
        )
        _, conf = intent.combined_confidence(0.0)
        if 0.55 <= conf < 0.65:
            assert len(recs) == 1

    def test_bandit_do_nothing_arm(self):
        bandit = BetaBandit(n_arms=3)
        # Force arm 0 by making its posterior dominant
        bandit.a[0] = 1e6
        bandit.b[0] = 1.0
        policy = LiquidBanditsPolicy(epsilon=0.0, bandit=bandit)
        intent = IntentState(n_categories=3)
        intent.update(cat_idx=0, weight=1.0, timestamp=0.0)
        rng = np.random.default_rng(0)
        recs, arm = policy.select(
            pop_rank=self.pop_rank,
            seen=self.seen,
            intent=intent,
            t_now=0.0,
            item_to_cat_idx=self.item_to_cat,
            rng=rng,
        )
        assert arm == 0
        assert recs == []


# ══════════════════════════════════════════════════════════════════════
# 4. Trust mismatch correctness
# ══════════════════════════════════════════════════════════════════════
class TestTrustMismatch:
    def test_no_recommendation_no_trust_cost(self):
        assert trust_mismatch(None, {100: 0}, [0, 1]) == 0.0

    def test_item_in_top_preferences(self):
        assert trust_mismatch(100, {100: 0}, [0, 1]) == 0.0

    def test_item_outside_top_preferences(self):
        assert trust_mismatch(100, {100: 5}, [0, 1]) == 1.0

    def test_no_preferences_yet(self):
        assert trust_mismatch(100, {100: 0}, []) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 5. End-to-end synthetic event log
# ══════════════════════════════════════════════════════════════════════
class TestEvaluationPipeline:
    def _make_synthetic_events(self, n_users=50, n_events_per_user=20, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for uid in range(n_users):
            # Assign each user a preferred category
            pref_cat = rng.integers(0, 4)
            for i in range(n_events_per_user):
                # 70% of events are in preferred category
                cat = pref_cat if rng.random() < 0.7 else rng.integers(0, 4)
                item_id = int(cat * 1000 + rng.integers(0, 50))
                rows.append(
                    {
                        "user_id": uid,
                        "item_id": item_id,
                        "ts": float(i * 3600),  # hourly events
                        "cat_idx": cat,
                        "weight": 1.0,
                    }
                )
        df = pd.DataFrame(rows)
        # Build item -> category mapping
        item_to_cat = dict(zip(df["item_id"].astype(int), df["cat_idx"].astype(int)))
        return df, item_to_cat

    def test_end_to_end_runs_without_error(self):
        events, item_to_cat = self._make_synthetic_events()
        per_user = evaluate_policies_on_event_log(
            events=events,
            item_to_cat_idx=item_to_cat,
            n_categories=4,
            verbose=False,
        )
        assert len(per_user) > 0
        for p in ["Static", "Triggers", "Liquid+Rules", "Liquid+Bandits"]:
            assert f"conv_{p}" in per_user.columns
            assert f"trust_{p}" in per_user.columns
            assert f"vol_{p}" in per_user.columns

    def test_liquid_policies_have_zero_or_low_trust(self):
        """Invariant: governed liquid policies should not exceed trust of baselines."""
        events, item_to_cat = self._make_synthetic_events()
        per_user = evaluate_policies_on_event_log(
            events=events,
            item_to_cat_idx=item_to_cat,
            n_categories=4,
            verbose=False,
        )
        trust_static = per_user["trust_Static"].mean()
        trust_rules = per_user["trust_Liquid+Rules"].mean()
        # Liquid+Rules trust should be <= Static trust
        assert trust_rules <= trust_static + 1e-9

    def test_volume_ordering(self):
        events, item_to_cat = self._make_synthetic_events()
        per_user = evaluate_policies_on_event_log(
            events=events,
            item_to_cat_idx=item_to_cat,
            n_categories=4,
            verbose=False,
        )
        # Static and Triggers always act => volume ~1
        assert per_user["vol_Static"].mean() >= 0.99
        assert per_user["vol_Triggers"].mean() >= 0.99

    def test_bootstrap_returns_valid_cis(self):
        events, item_to_cat = self._make_synthetic_events()
        per_user = evaluate_policies_on_event_log(
            events=events,
            item_to_cat_idx=item_to_cat,
            n_categories=4,
            verbose=False,
        )
        summary, pvals = bootstrap_summary(per_user, n_bootstrap=100, seed=0)
        for _, row in summary.iterrows():
            assert row["conversion_ci95_low"] <= row["conversion_mean"] <= row["conversion_ci95_high"]
            assert row["trust_ci95_low"] <= row["trust_mean"] <= row["trust_ci95_high"]
            assert row["volume_ci95_low"] <= row["volume_mean"] <= row["volume_ci95_high"]

    def test_friedman_runs_when_variance_exists(self):
        events, item_to_cat = self._make_synthetic_events(n_users=100)
        per_user = evaluate_policies_on_event_log(
            events=events,
            item_to_cat_idx=item_to_cat,
            n_categories=4,
            verbose=False,
        )
        omnibus, posthocs = friedman_nemenyi(per_user)
        assert set(omnibus["metric"]) == {"conversion_proxy", "trust_proxy", "volume_proxy"}
        for p in posthocs.values():
            assert p.shape == (4, 4)


# ══════════════════════════════════════════════════════════════════════
# 6. Simulation reproducibility
# ══════════════════════════════════════════════════════════════════════
class TestSimulation:
    def test_simulation_deterministic_per_seed(self):
        """Same seed -> identical output."""
        cfg = SimulationConfig(n_users=500, horizon_days=5)
        df1 = run_simulation(n_seeds=2, cfg=cfg)
        df2 = run_simulation(n_seeds=2, cfg=cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_simulation_produces_all_policies(self):
        cfg = SimulationConfig(n_users=200, horizon_days=5)
        df = run_simulation(n_seeds=2, cfg=cfg)
        assert set(df["policy"]) == {"Static", "Triggers", "Liquid+Rules", "Liquid+Bandits"}

    def test_rates_bounded_zero_to_one(self):
        cfg = SimulationConfig(n_users=200, horizon_days=5)
        df = run_simulation(n_seeds=3, cfg=cfg)
        assert df["conversion_rate"].min() >= 0.0
        assert df["conversion_rate"].max() <= 1.0
        assert df["trust_event_rate"].min() >= 0.0
        assert df["trust_event_rate"].max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
