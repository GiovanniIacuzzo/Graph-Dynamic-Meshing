"""Central plotting API for GDM: publication style, diagnostics, and figures."""
from .style import PALETTE, save_figure, set_publication_style
from .benchmark import plot_benchmark_results, plot_benchmark_summary, plot_benchmark_summary_file, plot_benchmark_raw_file
from .training import plot_training_history, plot_selfsupervised_history
from .dataset import audit_dataset

__all__ = [
    "PALETTE", "save_figure", "set_publication_style",
    "plot_benchmark_results", "plot_benchmark_summary", "plot_benchmark_summary_file", "plot_benchmark_raw_file",
    "plot_training_history", "plot_selfsupervised_history", "audit_dataset",
]
