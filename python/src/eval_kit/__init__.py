"""eval-kit -- Statistical confidence for LLM evaluation."""

from eval_kit.stats import (
    PairedResult,
    Stats,
    VerdictResult,
    WelchResult,
    approx_p_value,
    descriptive_stats,
    p_value,
    paired_t_test,
    required_n,
    required_n_paired,
    t_critical,
    verdict,
    welch_t_test,
)

__version__ = "0.1.0"
__all__ = [
    "PairedResult",
    "Stats",
    "VerdictResult",
    "WelchResult",
    "approx_p_value",
    "descriptive_stats",
    "p_value",
    "paired_t_test",
    "required_n",
    "required_n_paired",
    "t_critical",
    "verdict",
    "welch_t_test",
]
