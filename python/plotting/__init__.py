"""Central plotting API for GDM: publication style, diagnostics, and figures."""
from .style import PALETTE, save_figure, set_publication_style
from .benchmark import (
    plot_benchmark_results,
    plot_benchmark_summary,
    plot_benchmark_summary_file,
    plot_benchmark_raw_file,
)
from .training import plot_training_history, plot_selfsupervised_history
from .dataset import audit_dataset
from .stage_d_pde import (
    plot_stage_d_pde_benchmark,
    plot_l2_error_over_time,
    plot_integrated_and_final_errors,
    plot_computational_cost_comparison,
    plot_geometric_quality_evolution,
    plot_safety_filter_intervention,
    plot_interp_vs_extrap_comparison,
)

__all__ = [
    "PALETTE",
    "save_figure",
    "set_publication_style",
    "plot_benchmark_results",
    "plot_benchmark_summary",
    "plot_benchmark_summary_file",
    "plot_benchmark_raw_file",
    "plot_training_history",
    "plot_selfsupervised_history",
    "audit_dataset",
    "plot_stage_d_pde_benchmark",
    "plot_l2_error_over_time",
    "plot_integrated_and_final_errors",
    "plot_computational_cost_comparison",
    "plot_geometric_quality_evolution",
    "plot_safety_filter_intervention",
    "plot_interp_vs_extrap_comparison",
]