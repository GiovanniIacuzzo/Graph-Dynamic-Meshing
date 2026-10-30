# Graph Dynamic Meshing (GDM)

**A native C++/deal.II numerical core with a graph-neural-network mesh adapter for transient PDE simulations.**

Graph Dynamic Meshing (GDM) investigates whether a graph neural network (GNN) can predict admissible nodal mesh displacements that improve the accuracy–cost trade-off of finite-element simulations, while preserving a strict separation between the numerical engine and the learning/orchestration layer.

The repository is organized around reproducible experimental campaigns (Stages A–D). It does not claim that a learned mesh is universally more accurate or faster: every proposed benefit must be demonstrated against fixed-mesh and classical/native numerical baselines using the same PDE, time step, mesh resolution, and measurement protocol.

## Research objective

The workflow has four stages:

- **Stage A — Static adaptation:** learn a mapping from a nodal field and its derivatives to mesh displacements for analytical Gaussian-bump fields. Offline references are finite-budget, quality-screened optimizer outputs; they are not described as certified global or stationary optima unless the corresponding diagnostics support that claim.
- **Stage B — Native transient solver check:** compare fixed and adapted meshes and transfer operators using the C++/deal.II runtime.
- **Stage C — Repeated end-to-end evaluation:** measure solution error, geometric admissibility, conservation/GCL diagnostics where applicable, and total/component runtime over repeated transient simulations.
- **Stage D — Trajectory-aware extensions:** investigate supervision over adaptation windows and a self-supervised/pathwise-safety alternative.

## Architecture

```text
python/gdm/                  Typed wrappers and orchestration of _gdm_core
cpp_core/                    C++20 numerical core and pybind11 bindings
python/oracle/               Analytical fields and offline objective/optimizer
python/dataset/              Graph construction and dataset loading
python/model/                GNN architecture
python/plotting/             Shared publication-figure and dataset-audit code
python/evaluation/           Aggregated learning/benchmark metrics
training/                    Dataset generation, training, inference entry point
experiments/stage_a/         Stage-A benchmark campaign and evaluation
experiments/stage_b/         Stage-B transient solver/ablation campaign
experiments/stage_c/         Stage-C repeated PDE and transfer-operator studies
experiments/stage_d/         Stage-D data, training variants, and evaluation
config/                      Four canonical campaign configurations (A–D)
tools/diagnostics/           Explicitly non-production diagnostic utilities
tests/                       Python integration/unit tests for the numerical workflow
```

### Numerical boundary

The triangulation state, mesh metrics, admissibility/safety filter, FEM solver, ALE/remapping routines, and relevant PDE diagnostics are provided by the compiled `gdm._gdm_core` C++ extension. deal.II supplies the mesh/DoF and finite-element/linear-algebra infrastructure where used by the corresponding kernels. Some mesh-generation, ALE, remapping, geometry, and diagnostic formulas are implemented directly in C++ around deal.II data structures; the project does not claim that every arithmetic operation is implemented by a deal.II API call.

Python is used for experiment orchestration, data serialization, the analytical Gaussian field and offline SciPy oracle, GNN training/inference, plotting, and report generation. The differentiable self-supervised objective remains in PyTorch so that gradients can flow through the predicted mesh displacement. Oracle provenance records this division explicitly.

## Requirements

- Python 3.10 or newer.
- C++20 compiler, CMake, and pybind11 (Ninja is recommended but optional).
- deal.II 9.7.1 or a compatible newer release.
- Python packages listed in `environment.yml` and `pyproject.toml` (NumPy, SciPy, PyTorch, PyTorch Geometric, Matplotlib, PyYAML, and pytest for development).

The deal.II installation must be compatible with the local compiler and Python toolchain. If CMake does not locate it automatically, set `DEAL_II_DIR` to the directory containing `deal.IIConfig.cmake`.

## Installation and build

From the repository root:

```bash
bash prepare.sh
```

For a manual build in an already prepared environment. This reuses an existing build directory generator; if you need to switch generators, create a separate build directory instead of changing it in place:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python -m pip install --no-build-isolation --no-deps -e .
```

The `_gdm_core` shared library is placed alongside `python/gdm/`. A Python-only installation is not sufficient for tests or experiments that call mesh/FEM/ALE/remapping operations; those operations intentionally do not fall back to duplicate Python numerical backends.

## Canonical configurations

There are four top-level YAML configurations, one per campaign:

| File | Purpose |
|---|---|
| `config/stage_a.yaml` | Gaussian reference dataset, supervised training, and static benchmark defaults |
| `config/stage_b.yaml` | Transient native FEM/ALE comparison |
| `config/stage_c.yaml` | Repeated end-to-end PDE benchmark and transfer-operator study |
| `config/stage_d.yaml` | Trajectory-aware experiment plus a nested `self_supervised` profile |

Diagnostic modes are command-line options, not separate top-level config files. Generated run configurations are saved as `reproduced_config.yaml` alongside results. The Stage-A v4 raw data and manifest paths are kept stable (`data/raw_oracle_v4/` and `data/manifest_oracle_v4.json`) so the dataset can be consumed by the training, evaluation and PDE stages.

## Recommended execution order

Run commands from the repository root with the intended Conda environment active. Keep generated `data/` and `results/` directories outside source-only repository transfers; the canonical configurations preserve the existing Stage-A v4 paths, but source archives do not include generated datasets or checkpoints.

### 1. Tests

```bash
PYTHONPATH=python python -m pytest -q -rs
ctest --test-dir build --output-on-failure
```

A skipped native-integration test is not a successful verification of C++ behavior. Confirm that `_gdm_core` imports and the C++ tests are actually executed before generating publication results.

### 2. Stage A: dataset and training

Optional isolated diagnostic pilot (10 samples, 4/2/2/2 across the four splits):

```bash
PYTHONPATH=python python training/generate_dataset.py --config config/stage_a.yaml --pilot
```

Full production dataset (210 samples by default):

```bash
PYTHONPATH=python python training/generate_dataset.py --config config/stage_a.yaml
```

Training is intentionally blocked unless the production dataset audit passes:

```bash
PYTHONPATH=python python training/train.py --config config/stage_a.yaml
```

Evaluate the trained checkpoint on the independent Stage-A splits:

```bash
PYTHONPATH=python python -m experiments.stage_a.evaluate \
  --config config/stage_a.yaml \
  --checkpoint results/stage_a_oracle_v4/checkpoints/best_model.pt \
  --output_dir results/stage_a_oracle_v4/benchmark
```

Run the wider Stage-A paper campaign (safety stress, runtime scaling, OOD sweep, ablations and qualitative mesh views):

```bash
PYTHONPATH=python python -m experiments.stage_a.campaign
```

### 3. Stage B: transient solver baseline

```bash
PYTHONPATH=python python -m experiments.stage_b.campaign \
  --config config/stage_b.yaml \
  --checkpoint results/stage_a_oracle_v4/checkpoints/best_model.pt \
  --output_dir results/stage_b_v4
```

Optional diagnostics and ablations live in the same Stage-B directory:

```bash
PYTHONPATH=python python -m experiments.stage_b.diagnostics --config config/stage_b.yaml
PYTHONPATH=python python -m experiments.stage_b.ablations
```

### 4. Stage C: repeated PDE benchmark

```bash
PYTHONPATH=python python -m experiments.stage_c.campaign \
  --config config/stage_c.yaml \
  --checkpoint results/stage_a_oracle_v4/checkpoints/best_model.pt \
  --output_dir results/stage_c_v4 \
  --repeats 5 --nx 32 --ny 32 --T 0.5 --dt 0.005 --adapt-interval 10
```

The transfer-operator benchmark and replay diagnostic use stored Stage-C trajectories:

```bash
PYTHONPATH=python python -m experiments.stage_c.transfer_benchmark --config config/stage_c.yaml
PYTHONPATH=python python -m experiments.stage_c.diagnose_transfer --config config/stage_c.yaml
```

### 5. Stage D: trajectory-aware and self-supervised variants

Generate trajectory-aware labels and train with the shared trainer:

```bash
PYTHONPATH=python python -m experiments.stage_d.generate_trajectory_dataset --config config/stage_d.yaml
PYTHONPATH=python python training/train.py --config config/stage_d.yaml
PYTHONPATH=python python -m experiments.stage_d.evaluate_trajectory --config config/stage_d.yaml
```

The self-supervised/pathwise-safety variant is nested in the same Stage-D configuration:

```bash
PYTHONPATH=python python -m experiments.stage_d.generate_selfsupervised_dataset --config config/stage_d.yaml
PYTHONPATH=python python -m experiments.stage_d.train_selfsupervised --config config/stage_d.yaml
PYTHONPATH=python python -m experiments.stage_d.evaluate_selfsupervised --config config/stage_d.yaml
```

### 6. Inference demo

```bash
PYTHONPATH=python python training/inference.py \
  --config config/stage_a.yaml \
  --checkpoint results/stage_a_oracle_v4/checkpoints/best_model.pt \
  --sample data/raw_oracle_v4/sample_00000_train.pkl
```

## Figure policy

All experiment figures are written **only as vector PDF** through the shared utilities in `python/plotting/`. PDF fonts are configured for publication embedding. The code intentionally does not emit PNG or SVG copies. Figure source data, JSON/CSV summaries, and LaTeX tables should be retained alongside the PDFs so every figure can be regenerated without rerunning the numerical experiment whenever raw results are available.

## Performance and reproducibility

For a 32×32 rectangular grid the initial triangulation has approximately 1,089 nodes and 2,048 triangles. The Stage-A oracle uses a composite degree-five Dunavant quadrature with 112 points per triangle, i.e. about 229,376 field-evaluation points per objective/gradient evaluation. L-BFGS-B then repeats that evaluation up to the configured iteration budget, and each sample is optimized independently. The observed generation rate of roughly 38–41 seconds per sample is therefore computationally plausible for the current Python/SciPy objective, although it is expensive. At that measured rate, the 120-sample training split takes about 80 minutes and the complete 210-sample data set about 2 hours 20 minutes, excluding retries and extra audit work. The configured `device: mps` accelerates eligible PyTorch training operations; it does not accelerate this NumPy/SciPy L-BFGS-B objective, which runs on the CPU. The optimized field hot path skips computing the unused analytic Laplacian inside the objective; it does not change the functional or quadrature rule. Runtime records now include iterations, function evaluations and oracle wall time to make future measurements auditable.

For every run retain the configuration, random seed, dataset manifest/provenance, per-sample oracle diagnostics, model checkpoint, training history, per-sample benchmark JSON and aggregated report. Report all test-set outcomes—including failures—and do not select hyperparameters based on the final test sets. Runtime comparisons should include end-to-end wall time and component timings and should be repeated on the same hardware under the same protocol.

## Scientific status

GDM is a research implementation. The contribution of learned mesh adaptation must be supported by independent in-domain and extrapolation tests and by the transient PDE experiments. A lower displacement-regression loss or a smaller analytical interpolation error alone does not establish improved FEM accuracy, conservation, stability, or computational efficiency. The repository keeps these claims separate by design.
