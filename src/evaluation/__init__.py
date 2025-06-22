"""Evaluation metrics and statistical analysis utilities."""

from .metrics import calculate_correctness_indicators
from .statistical_analysis import (
    test_normality,
    compare_two_groups,
    compare_multiple_groups,
    run_gender_analysis,
    run_age_analysis
)

__all__ = [
    "calculate_correctness_indicators", 
    "test_normality",
    "compare_two_groups",
    "compare_multiple_groups",
    "run_gender_analysis",
    "run_age_analysis"
] 