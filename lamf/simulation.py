"""
Illustrative simulation (Section 5.1 of the paper).

Parameters from Table 2 of the manuscript are the defaults. The simulation
produces one row per (seed, policy) with conversion_rate, trust_event_rate,
messages_sent, eligible_activations.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

LOG2 = np.log(2.0)


@dataclass
class SimulationConfig:
    n_users: int = 25_000
    horizon_days: int = 45
    step_days: float = 0.25  # 6-hour resolution
    half_life_days: float = 3.0
    hard_ttl_hours: float = 48.0
    theta: float = 1.0
    lag_static_days: float = 2.0
    freq_cap_per_week: int = 3
    explore_cap: float = 0.05
    alpha: float = 1.2  # intent weight in conversion logit
    gamma: float = 2.0  # late-delivery penalty
    delta_action: Dict[str, float] = None
    eta: tuple = (-4.2, 0.35, 1.2, 1.0)  # trust logit coefficients

    def __post_init__(self):
        if self.delta_action is None:
            self.delta_action = {
                "do_nothing": 0.00,
                "trigger": 0.20,
                "rules": 0.35,
                "bandit": 0.45,
            }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _run_one_seed(seed: int, cfg: SimulationConfig) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_steps = int(cfg.horizon_days / cfg.step_days)
    decay_rate = LOG2 / cfg.half_life_days

    onset = rng.uniform(0, cfg.horizon_days * 0.7, size=cfg.n_users)
    intensity = rng.exponential(1.5, size=cfg.n_users)

    policies = ["Static", "Triggers", "Liquid+Rules", "Liquid+Bandits"]
    conversions = {p: 0 for p in policies}
    trust_events = {p: 0 for p in policies}
    messages_sent = {p: 0 for p in policies}
    eligible = {p: 0 for p in policies}

    msg_count_week = {p: np.zeros(cfg.n_users, dtype=int) for p in policies}

    bandit_a = np.ones(3)
    bandit_b = np.ones(3)

    for step in range(n_steps):
        t = step * cfg.step_days
        if step > 0 and (step * cfg.step_days) % 7 < cfg.step_days:
            for p in policies:
                msg_count_week[p][:] = 0

        time_since = t - onset
        active = time_since > 0
        raw = np.where(active, intensity * np.exp(-decay_rate * time_since), 0.0)
        event_prob = 1.0 - np.exp(-raw * cfg.step_days * 0.3)
        events = rng.random(cfg.n_users) < event_prob

        score = raw.copy()
        score[events] += intensity[events] * 0.5
        activated = score >= cfg.theta
        ttl_expired = (time_since > (cfg.hard_ttl_hours / 24.0)) & (~events)

        for p in policies:
            if p == "Static":
                lagged_t = t - cfg.lag_static_days
                lagged_active = (lagged_t - onset) > 0
                lagged_score = np.where(
                    lagged_active,
                    intensity * np.exp(-decay_rate * (lagged_t - onset)),
                    0.0,
                )
                elig_mask = lagged_score >= cfg.theta
            elif p == "Triggers":
                elig_mask = events
            else:
                elig_mask = activated & (~ttl_expired)

            can_send = msg_count_week[p] < cfg.freq_cap_per_week
            elig_mask = elig_mask & can_send
            n_elig = int(elig_mask.sum())
            if n_elig == 0:
                continue

            eligible[p] += n_elig

            if p == "Static":
                action_effect = cfg.delta_action["trigger"]
                late = np.ones(n_elig)
                mismatch = rng.random(n_elig) < 0.3
            elif p == "Triggers":
                action_effect = cfg.delta_action["trigger"]
                late = np.zeros(n_elig)
                mismatch = rng.random(n_elig) < 0.1
            elif p == "Liquid+Rules":
                action_effect = cfg.delta_action["rules"]
                late = np.zeros(n_elig)
                mismatch = rng.random(n_elig) < 0.05
            else:  # Liquid+Bandits
                explore = rng.random() < cfg.explore_cap
                if explore:
                    arm = int(rng.integers(0, 3))
                else:
                    samples = rng.beta(bandit_a, bandit_b)
                    arm = int(np.argmax(samples))
                if arm == 0:
                    continue
                action_effect = cfg.delta_action["bandit"]
                late = np.zeros(n_elig)
                mismatch = rng.random(n_elig) < 0.04

            elig_scores = score[elig_mask]
            p_conv = _sigmoid(cfg.alpha * elig_scores + action_effect - cfg.gamma * late)
            converted = rng.random(n_elig) < p_conv

            freq_proxy = (
                msg_count_week[p][elig_mask].astype(float) / max(cfg.freq_cap_per_week, 1)
            )
            p_trust = _sigmoid(
                cfg.eta[0]
                + cfg.eta[1] * freq_proxy
                + cfg.eta[2] * late
                + cfg.eta[3] * mismatch
            )
            trust_fired = rng.random(n_elig) < p_trust

            conversions[p] += int(converted.sum())
            trust_events[p] += int(trust_fired.sum())
            messages_sent[p] += n_elig
            msg_count_week[p][elig_mask] += 1

            if p == "Liquid+Bandits" and n_elig > 0:
                reward = float(converted.mean() - 0.4 * trust_fired.mean())
                y = 1 if reward > 0 else 0
                bandit_a[arm] += y
                bandit_b[arm] += 1 - y

    results = {}
    for p in ["Static", "Triggers", "Liquid+Rules", "Liquid+Bandits"]:
        ea = max(eligible[p], 1)
        ms = max(messages_sent[p], 1)
        results[p] = {
            "eligible_activations": eligible[p],
            "conversions": conversions[p],
            "conversion_rate": conversions[p] / ea,
            "messages_sent": messages_sent[p],
            "trust_events": trust_events[p],
            "trust_event_rate": trust_events[p] / ms,
        }
    return results


def run_simulation(n_seeds: int = 10, cfg: SimulationConfig = None) -> pd.DataFrame:
    """Run the Section 5.1 simulation across n_seeds and return a long-form DataFrame."""
    if cfg is None:
        cfg = SimulationConfig()
    rows = []
    for seed in range(n_seeds):
        res = _run_one_seed(seed, cfg)
        for policy, metrics in res.items():
            rows.append({"seed": seed, "policy": policy, **metrics})
    return pd.DataFrame(rows)
