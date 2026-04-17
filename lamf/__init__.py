"""
LAMF: Liquid Audience Measurement Framework — reference implementation.

This package provides the core primitives used in the paper:
  - Decayed segment-of-one intent state (Eq. 4)
  - Governed policies: Static, Triggers, Liquid+Rules, Liquid+Bandits
  - Evaluation: bootstrap CIs, Friedman omnibus, Nemenyi post-hoc

Public API:
    from lamf import IntentState, policies, evaluation, simulation
"""

__version__ = "0.1.0"

from lamf.intent import IntentState, decayed_category_prefs, top_k_categories
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
from lamf.simulation import run_simulation

__all__ = [
    "IntentState",
    "decayed_category_prefs",
    "top_k_categories",
    "StaticPolicy",
    "TriggerPolicy",
    "LiquidRulesPolicy",
    "LiquidBanditsPolicy",
    "BetaBandit",
    "evaluate_policies_on_event_log",
    "bootstrap_summary",
    "friedman_nemenyi",
    "trust_mismatch",
    "run_simulation",
]
