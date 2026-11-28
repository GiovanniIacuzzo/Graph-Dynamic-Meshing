# Graph Dynamic Meshing

Controlled experiments on how the **path between two admissible meshes** affects a scalar transport solution. The repository implements a native FEM–ALE solver, deterministic moving-mesh controls and a graph policy that ranks feasible trajectory candidates. Start from the deterministic controls, then compare training on two objectives under the same data and architecture.

## Install and verify

Use Python 3.10+, a C++20 compiler and deal.II 9.7.1 or newer. Set `DEAL_II_DIR` to the compatible local installation if CMake does not find it. The first line assumes that a virtual environment is active.

```bash
bash prepare.sh
python -m pip install -e '.[learning,figures]'
gdm-transport design
gdm-campaign all --smoke --root results/smoke
```

The smoke workflow uses one case per profile and split, one paired training seed, two epochs, and an 8×8 mesh. It checks the workflow interfaces. Full evidence comes from the production run. Individual C++ and Python tests can also be run with `ctest --test-dir build --output-on-failure` and `python -m pytest -q -rs`.

## Production campaigns

Run each command from the repository root. Keep one run root and the same configuration across the six steps. The default is `config/transport.yaml`. Use an empty run root for a repeat with different settings. `--build-dir` selects a CMake build other than `build/`. `--device` applies to training only. Computational comparisons use CPU inference.

| Campaign | Command | Result |
|---|---|---|
| C0 · Verify | `gdm-campaign verify --root results/main` | Native CTest suite, static FEM–ALE comparison and diagnostic replay |
| C1 · Mechanism | `gdm-campaign mechanism --root results/main` | Advection–diffusion on development trajectories. Uniform and early ALE on identical endpoints, conservative remap, fixed and causal controls |
| C2 · Data | `gdm-campaign dataset --root results/main` | Nine candidate scores per state under each of two objectives, with whole-trajectory train/validation splits and per-sample hashes |
| C3 · Train | `gdm-campaign train --root results/main` | Two training arms for each of three paired seeds. Best checkpoints selected by validation decision regret |
| C4 · Evaluate | `gdm-campaign evaluate --root results/main` | Held-out rollouts on two resolutions and two time steps, numerical and geometric diagnostics, matched CPU timing repeats |
| C5 · Report | `gdm-campaign report --root results/main` | Case-paired contrasts, sensitivity by seed and three PDF figures |

Inspect C1 before C3. The case generator and numerical settings remain fixed across stages. Reserve C4 for the frozen models and final held-out comparison. `gdm-campaign all --root results/main` runs the steps in order. `gdm-transport --help` exposes the underlying commands. Completed training arms can be reused with the same run root. For an interrupted numerical run, retain its outputs for inspection and use a fresh run root or remove only the incomplete stage directory after inspecting it.

## Software boundaries

| Component | Responsibility |
|---|---|
| `cpp_core/` and `gdm.native_api` | FEM–ALE, geometry and physical state on the native solver side |
| `gdm.transport` | Study cases, admissible proposed paths, offline objectives, campaigns, metrics and plots |
| `gdm.learning` | Graph features, batching, shared message passing, candidate score head, checkpoint validation and inference. Usable with saved mesh observations or another compatible solver |
| `tests/` | Numerical, graph invariance, checkpoint and campaign reporting tests |

The learner consumes arrays for the current mesh, numerical field, nodal area weights and physical context, together with a candidate descriptor matrix. The action head scores candidates through a shared function. Field and graph encoders are used once per decision. Inference returns an action index and optional scores. The numerical backend applies that action under its own geometric constraints. Training and inference never need to import the native extension. See [learning protocol](docs/learning.md) for the public interface and the paired objectives.

## Publication figures

The report generates only the following result PDFs. The analysis also writes CSV files for any further audit and sensitivity tables.

| PDF | Scientific question |
|---|---|
| `01_trajectory_mechanism.pdf` | How much does the time schedule matter when the endpoint mesh is identical? How does the result change under instantaneous transfer? |
| `02_learning_ablation.pdf` | Does training on a time window improve held-out accuracy, and how does it change the timing decisions selected online? |
| `03_work_precision.pdf` | Is any accuracy benefit retained when comparing the total online time of Fixed FEM, the causal transport control and the learned controller? |

Figures 1 and 2 use the configured primary resolution and step. C5 uses the held-out evaluation for those figures. The optional C1 report provides development-only checks. Figure 3 uses all configurations. See [numerical protocol](docs/protocol.md) and [figure definitions](docs/figures.md).

## Provenance

The source snapshot derives from public commit [`63c2b075`](https://github.com/GiovanniIacuzzo/Graph-Dynamic-Meshing/tree/63c2b0754a11da85383ad8dd5cfe81c2ed2232b1) and the preceding research ZIP. Use a separate directory to reconcile any private local edits. The archive is source-only.
