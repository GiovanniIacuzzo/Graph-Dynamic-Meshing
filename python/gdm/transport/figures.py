"""Three publication panels from recorded measurements, exported as vector PDF."""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .design import ACTIONS
from .statistics import average_seeds


STYLE = {
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.65, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "pdf.fonttype": 42,
}
GRAY, BLUE, GREEN, OCHRE = "#292929", "#315c82", "#27765a", "#97652b"
PROFILE = {"transport": "Free transport", "turning": "Forced turning"}


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Graph Dynamic Meshing"})
    plt.close(fig)


def _panel(ax, label, title):
    ax.set_title(f"({label})  {title}", loc="left", pad=8)


def _effects(ax, data, labels, colors=None):
    ax.axvline(0, ls="--", lw=0.7, color="0.5", zorder=0)
    for i, row in enumerate(data):
        color = colors[i] if colors is not None else GREEN
        interval = row["bootstrap_low"], row["bootstrap_high"]
        if None not in interval:
            ax.hlines(i, 100 * interval[0], 100 * interval[1], color=color, lw=1.15)
        ax.plot(100 * row["mean_improvement"], i, "o", color=color, ms=4.3)
    ax.set_yticks(np.arange(len(data)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Integrated $L^2$ improvement (pp)")
    ax.grid(axis="x", color="0.90", lw=0.5)


def _mechanism(directory, contrasts):
    targets = [r for r in contrasts if r["metric"] == "relative_l2_average"
               and (r["baseline"], r["candidate"]) in {("uniform", "early"), ("remap", "uniform")}]
    if not targets:
        return
    fig, (ax, forest) = plt.subplots(1, 2, figsize=(7.1, max(2.5, 1.1 + .14 * len(targets))),
                                      gridspec_kw={"width_ratios": [1, 1.8]}, layout="constrained")
    t = np.linspace(0, 1, 100)
    _panel(ax, "a", "Identical endpoints")
    ax.plot(t, t, color=BLUE, label="Uniform ALE")
    ax.plot(t, t + .7 * t * (1 - t), color=GREEN, label="Early ALE")
    if any(r["baseline"] == "remap" for r in targets):
        ax.plot([0, 0, 1], [0, 1, 1], color=OCHRE, ls="--", label="Instantaneous remap")
    ax.set(xlabel=r"Time within window, $t/H$", ylabel="Path fraction", xlim=(0, 1), ylim=(0, 1.03))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .96))
    _panel(forest, "b", "Paired effects")
    labels = [f'{PROFILE.get(r["profile"], r["profile"])} — '
              f'{"early vs. uniform" if r["candidate"] == "early" else "uniform vs. remap"}'
              for r in targets]
    colors = [GREEN if r["candidate"] == "early" else OCHRE for r in targets]
    _effects(forest, targets, labels, colors)
    _save(fig, directory / "01_trajectory_mechanism.pdf")


def _learning(directory, contrasts, windows):
    targets = [r for r in contrasts if r["metric"] == "relative_l2_average"
               and r["candidate"] == "gnn_window" and r["baseline"] in {"gnn_instantaneous", "transport"}]
    chosen = [row for row in windows if row["method"] in {"gnn_instantaneous", "gnn_window"}
              and row.get("action", "") != ""]
    if not targets or not chosen:
        return
    fig, (forest, ax) = plt.subplots(1, 2, figsize=(7.1, 2.65),
                                     gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained")
    _panel(forest, "a", "Held-out error")
    _effects(forest, targets, [
        f'{PROFILE.get(r["profile"], r["profile"])}\n'
        f'{"Instantaneous GNN" if r["baseline"] == "gnn_instantaneous" else "Transport control"}'
        for r in targets])
    _panel(ax, "b", "Learned decision schedule")
    profiles = sorted({row["profile"] for row in chosen})
    styles = [(-.7, BLUE, "Late"), (0.0, "#c5cbd2", "Uniform"), (.7, GREEN, "Early")]
    for i, profile in enumerate(profiles):
        for offset, method in ((-.19, "gnn_instantaneous"), (.19, "gnn_window")):
            actions = [ACTIONS[int(row["action"])].timing for row in chosen
                       if row["profile"] == profile and row["method"] == method]
            if not actions:
                continue
            bottom = 0.0
            for schedule, color, _ in styles:
                proportion = 100 * actions.count(schedule) / len(actions)
                ax.bar(i + offset, proportion, width=.32, bottom=bottom, color=color, edgecolor="white",
                       lw=.45, hatch="//" if method == "gnn_instantaneous" else None)
                bottom += proportion
    labels = [f"{PROFILE.get(profile, profile)}\nInstant.  Window" for profile in profiles]
    ax.set_xticks(np.arange(len(profiles)), labels)
    ax.set(ylabel="Decision windows (%)", ylim=(0, 100))
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=color, label=label) for _, color, label in styles],
              loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    _save(fig, directory / "02_learning_ablation.pdf")


def _work_precision(directory, summaries):
    styles = {"fixed_fem": (GRAY, "s", "Fixed FEM"),
              "transport": (BLUE, "^", "Transport control"),
              "gnn_window": (GREEN, "o", "Window GNN")}
    groups = defaultdict(list)
    for row in average_seeds(summaries):
        if row["method"] in styles:
            key = (row["profile"], row["method"], int(row["resolution"]), float(row["dt"]))
            groups[key].append(row)
    if not groups:
        return
    profiles = sorted({key[0] for key in groups})
    fig, axes = plt.subplots(1, len(profiles), figsize=(7.1, 2.7), squeeze=False,
                             layout="constrained")
    used = set()
    for i, (ax, profile) in enumerate(zip(axes[0], profiles)):
        _panel(ax, chr(97+i), PROFILE.get(profile, profile))
        for (kind, method, n, dt), rows in sorted(groups.items()):
            if kind != profile:
                continue
            seconds = np.median([float(row["solver_and_controller_seconds"]) for row in rows])
            error = 100 * np.mean([float(row["relative_l2_average"]) for row in rows])
            if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(error):
                raise ValueError("Work–precision requires positive timing and finite errors.")
            color, marker, _ = styles[method]
            # A hollow marker identifies the coarser temporal discretization.
            finest_dt = min(key[3] for key in groups if key[0] == profile)
            ax.scatter(seconds, error, marker=marker, color=color, s=27, linewidths=.8,
                       facecolors=color if dt == finest_dt else "none")
            used.add(method)
        ax.set(xscale="log", xlabel="Median online time (s)", ylabel=r"Mean integrated $L^2$ error (%)")
        ax.grid(color="0.91", ls=":", lw=.6)
    handles = [Line2D([], [], marker=marker, ls="", color=color, label=label)
               for method, (color, marker, label) in styles.items() if method in used]
    axes[0, -1].legend(handles=handles, frameon=False)
    _save(fig, directory / "03_work_precision.pdf")


def write_figures(directory, summaries, contrasts, windows, cfg):
    options = cfg["campaigns"]
    setting = int(options["figure_resolution"]), float(options["figure_dt"])

    def selected(row):
        return (int(row["resolution"]), float(row["dt"])) == setting

    with plt.rc_context(STYLE):
        _mechanism(directory, [row for row in contrasts if selected(row)])
        _learning(directory, [row for row in contrasts if selected(row)],
                  [row for row in windows if selected(row)])
        _work_precision(directory, summaries)
