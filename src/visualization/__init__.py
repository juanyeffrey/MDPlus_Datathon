"""Visualization utilities for generating plots and charts."""

from .plots import (
    plot_accuracy_by_gender,
    plot_accuracy_by_age, 
    plot_age_distribution,
    save_all_plots
)

__all__ = [
    "plot_accuracy_by_gender",
    "plot_accuracy_by_age",
    "plot_age_distribution", 
    "save_all_plots"
] 