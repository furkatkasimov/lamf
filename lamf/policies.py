"""
Policies: rule-based and bandit-based next-best-action selection.

Each policy implements select(...) -> list of recommended item IDs (possibly
empty for do-nothing). Signatures are kept uniform so the evaluator can call
any policy identically.

The four policies used in the paper:
  - StaticPolicy       : top-popularity, ignores intent state
  - TriggerPolicy      : popularity within last-event category
  - LiquidRulesPolicy  : confidence-gated, grey-zone aware
  - LiquidBanditsPolicy: Beta-Thompson over {do-nothing, single, top-K}
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np

from lamf.intent import IntentState, top_k_categories


def _popularity_rec(
    pop_rank: List[int],
    item_to_cat_idx: Dict[int, int],
    allowed_cat_idxs: List[int],
    seen: Set[int],
    k: int,
) -> List[int]:
    """Pick top-k unseen items, preferring those in allowed categories."""
    recs: List[int] = []
    # First pass: in-category items
    for item_id in pop_rank:
        if item_id in seen:
            continue
        cat = item_to_cat_idx.get(item_id)
        if cat is not None and cat in allowed_cat_idxs:
            recs.append(item_id)
        if len(recs) >= k:
            break
    # Second pass: fall back to any popular item
    if len(recs) < k:
        for item_id in pop_rank:
            if item_id in seen or item_id in recs:
                continue
            recs.append(item_id)
            if len(recs) >= k:
                break
    return recs[:k]


# ─────────────────────────────────────────────────────────────────────
# Policy 1: Static (always acts, no intent)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class StaticPolicy:
    """Recommend the single most-popular unseen item. Always acts."""

    name: str = "Static"

    def select(
        self,
        pop_rank: List[int],
        seen: Set[int],
        **kwargs,
    ) -> List[int]:
        for item_id in pop_rank:
            if item_id not in seen:
                return [item_id]
        return []


# ─────────────────────────────────────────────────────────────────────
# Policy 2: Triggers (always acts, uses last-event category)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class TriggerPolicy:
    """Recommend popularity-ranked unseen item in the last-interacted category."""

    name: str = "Triggers"

    def select(
        self,
        pop_rank: List[int],
        seen: Set[int],
        last_item_cat_idx: Optional[int] = None,
        item_to_cat_idx: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> List[int]:
        if last_item_cat_idx is None or item_to_cat_idx is None:
            return []
        return _popularity_rec(
            pop_rank, item_to_cat_idx, [last_item_cat_idx], seen, k=1
        )


# ─────────────────────────────────────────────────────────────────────
# Policy 3: Liquid + Rules
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LiquidRulesPolicy:
    """Confidence-gated rule policy with grey-zone handling."""

    theta: float = 0.55
    theta_safe: float = 0.65
    top_k: int = 10
    name: str = "Liquid+Rules"

    def select(
        self,
        pop_rank: List[int],
        seen: Set[int],
        intent: Optional[IntentState] = None,
        t_now: Optional[float] = None,
        item_to_cat_idx: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> List[int]:
        if intent is None or t_now is None or item_to_cat_idx is None:
            return []
        top_cat, conf = intent.combined_confidence(t_now)
        if conf < self.theta:
            return []
        k = 1 if conf < self.theta_safe else self.top_k
        return _popularity_rec(pop_rank, item_to_cat_idx, [top_cat], seen, k=k)


# ─────────────────────────────────────────────────────────────────────
# Policy 4: Liquid + Bandits (Beta-Thompson)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class BetaBandit:
    """Three-arm Beta-Bernoulli Thompson sampling bandit."""

    n_arms: int = 3

    def __post_init__(self):
        self.a = np.ones(self.n_arms)
        self.b = np.ones(self.n_arms)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.a, self.b)

    def update(self, arm: int, reward: float) -> None:
        """Binarise reward at 0.5 and update posterior."""
        y = 1 if reward >= 0.5 else 0
        self.a[arm] += y
        self.b[arm] += 1 - y


@dataclass
class LiquidBanditsPolicy:
    """
    Beta-Thompson bandit over action indices {0: do-nothing, 1: single, 2: top-K}.
    Exploration rate epsilon picks an arm uniformly within the allowed set.
    In the grey zone, arm 2 (top-K) is disallowed.
    """

    theta: float = 0.55
    theta_safe: float = 0.65
    top_k: int = 10
    epsilon: float = 0.05
    name: str = "Liquid+Bandits"
    bandit: Optional[BetaBandit] = None

    def __post_init__(self):
        if self.bandit is None:
            self.bandit = BetaBandit(n_arms=3)

    def select(
        self,
        pop_rank: List[int],
        seen: Set[int],
        intent: Optional[IntentState] = None,
        t_now: Optional[float] = None,
        item_to_cat_idx: Optional[Dict[int, int]] = None,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> tuple:
        """
        Returns (recommended_items, arm_chosen) so the evaluator can record
        which arm was pulled and update the bandit after observing reward.
        """
        if rng is None:
            rng = np.random.default_rng()
        if intent is None or t_now is None or item_to_cat_idx is None:
            return [], 0
        top_cat, conf = intent.combined_confidence(t_now)
        allowed_arms = [0, 1] if conf < self.theta_safe else [0, 1, 2]

        if rng.random() < self.epsilon:
            arm = int(rng.choice(allowed_arms))
        else:
            samples = self.bandit.sample(rng)
            arm = int(max(allowed_arms, key=lambda a: samples[a]))

        if arm == 0:
            return [], arm
        k = 1 if arm == 1 else self.top_k
        slow_top = top_k_categories(intent.slow_prefs(t_now), 1) or top_k_categories(
            intent.fast_prefs(t_now), 1
        )
        if not slow_top:
            return [], arm
        recs = _popularity_rec(pop_rank, item_to_cat_idx, slow_top, seen, k=k)
        return recs, arm
