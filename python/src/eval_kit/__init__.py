"""eval-kit -- Statistical confidence for LLM evaluation."""

from eval_kit.stats import (
    Stats,
    WelchResult,
    approx_p_value,
    descriptive_stats,
    required_n,
    t_critical,
    welch_t_test,
)

__version__ = "0.0.1"
__all__ = [
    "Stats",
    "WelchResult",
    "approx_p_value",
    "descriptive_stats",
    "required_n",
    "t_critical",
    "welch_t_test",
]
