"""
Intent state: decayed segment-of-one accumulator (Eq. 4 in the paper).

s_{u,k}(t) = sum_{(e, t_e) in E_u, t_e <= t} w(e, k) * exp(-lambda_k * (t - t_e))

where:
  - w(e, k) is the event-to-intent weight (e.g. view=1, cart=2, purchase=3)
  - lambda_k is the decay rate for intent k
  - t - t_e is the elapsed time since the event

Half-life relates to lambda by t_{1/2} = ln(2) / lambda.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

LOG2 = np.log(2.0)


def decayed_category_prefs(
    cat_idx: np.ndarray,
    weights: np.ndarray,
    timestamps: np.ndarray,
    t_now: float,
    half_life_days: float,
    n_categories: int,
) -> np.ndarray:
    """
    Compute the decayed-accumulator intent state over categories at time t_now.

    Parameters
    ----------
    cat_idx : array of int, shape (n_events,)
        Category index for each event in the user's history.
    weights : array of float, shape (n_events,)
        Event-to-intent weights (e.g. view=1, cart=2, purchase=3).
    timestamps : array of float, shape (n_events,)
        Event timestamps in seconds (Unix epoch or arbitrary origin).
    t_now : float
        Current time in seconds. Events after t_now are ignored.
    half_life_days : float
        Exponential decay half-life in days.
    n_categories : int
        Size of the intent ontology K.

    Returns
    -------
    prefs : array of float, shape (n_categories,)
        Decayed-accumulator preference vector. Non-negative.
    """
    if len(cat_idx) == 0:
        return np.zeros(n_categories)
    dt_days = (t_now - timestamps) / 86400.0
    time_weights = np.exp(-LOG2 * dt_days / half_life_days)
    effective = weights * time_weights
    prefs = np.zeros(n_categories)
    for i in range(len(cat_idx)):
        if effective[i] > 0:
            prefs[cat_idx[i]] += effective[i]
    return prefs


def top_k_categories(prefs: np.ndarray, k: int = 2) -> List[int]:
    """Return indices of the top-k categories by preference score."""
    if prefs.sum() == 0:
        return []
    return list(np.argsort(prefs)[-k:][::-1])


@dataclass
class IntentState:
    """
    Segment-of-one intent state for a user.

    Maintains two decayed accumulators (fast and slow half-lives) over a
    fixed category ontology. Use `update(...)` to fold in new events and
    `fast_prefs(...)` / `slow_prefs(...)` to read state at a given time.

    Parameters
    ----------
    n_categories : int
        Number of intent categories in the ontology K.
    half_life_fast_days : float, default=1.0
        Half-life of the fast/exploration accumulator.
    half_life_slow_days : float, default=7.0
        Half-life of the slow/stable-preference accumulator.
    """

    n_categories: int
    half_life_fast_days: float = 1.0
    half_life_slow_days: float = 7.0
    cat_idx: List[int] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def update(self, cat_idx: int, weight: float, timestamp: float) -> None:
        """Record an event. Does not mutate derived state until queried."""
        self.cat_idx.append(int(cat_idx))
        self.weights.append(float(weight))
        self.timestamps.append(float(timestamp))

    def fast_prefs(self, t_now: float) -> np.ndarray:
        return decayed_category_prefs(
            np.array(self.cat_idx, dtype=int),
            np.array(self.weights, dtype=float),
            np.array(self.timestamps, dtype=float),
            t_now,
            self.half_life_fast_days,
            self.n_categories,
        )

    def slow_prefs(self, t_now: float) -> np.ndarray:
        return decayed_category_prefs(
            np.array(self.cat_idx, dtype=int),
            np.array(self.weights, dtype=float),
            np.array(self.timestamps, dtype=float),
            t_now,
            self.half_life_slow_days,
            self.n_categories,
        )

    def combined_confidence(self, t_now: float) -> tuple:
        """Return (top_category_idx, normalized_confidence) for the combined state."""
        combined = self.fast_prefs(t_now) + self.slow_prefs(t_now)
        total = combined.sum() + 1e-9
        top = int(np.argmax(combined))
        conf = float(combined[top] / total)
        return top, conf
