# Stage F1

See `docs/stage_f1_methodology.md` for the mathematical formulation, fairness
protocol, monitor definitions, expected outputs and acceptance criteria.

## Run sequence

Run from the repository root, after the native extension has been built and
the Stage-D checkpoint/data are present.

```bash
# 1) End-to-end wiring smoke test (one validation case + one case per test split).
#    It uses separate smoke_tuning/ and smoke_evaluation/ output directories.
python -m experiments.stage_f.run all --smoke-test

# 2) Tune only the classical monitor strengths on the full validation split.
python -m experiments.stage_f.run tune

# 3) Full held-out comparison (one warm-up + five timed repetitions).
python -m experiments.stage_f.run evaluate
```

Validation tuning checkpoints every completed candidate/case pair and can resume
only with identical configuration, implementation hashes and validation case files.
The full evaluation automatically resumes from its checkpointed CSV outputs
when re-run with an identical configuration, implementation hashes, checkpoint
hash and held-out case file checksums.
To run a changed protocol, choose a new `experiment.output_dir` in
`config/stage_f.yaml`; do not overwrite previous results.

Expected prerequisites:
- compiled `gdm._gdm_core` importable by the active Python interpreter;
- `data/stage_d_trajectory_aware/raw/manifest.json` and raw case files;
- `results/stage_d_trajectory_aware/checkpoints/best_model.pt`;
- the Stage-E1 numerical configuration at `config/stage_e.yaml`.

The classical variational controller evaluates its frozen-monitor functional and its
analytical coordinate gradient with vectorized NumPy algebra; it does not invoke
PyTorch/autodifferentiation in the online baseline. The gradient is checked by an
independent Torch-autodiff reference and centred finite differences in the unit tests.
The predeclared pseudo-time iteration cap is 200; every event records whether it
converged by relative energy decrease or stopped at the cap.

The Stage-D benchmark has one backward-compatible optional configuration hook
for Stage F: the `stage_f_numerics` key is absent in all historical Stage-D
calls, so previous defaults remain unchanged. Stage F fills this key from the
Stage-E1 Galerkin and linear-solver settings, making its fixed-FEM and ALE
trajectories use the same verified native solver tolerances.

The code does not fabricate experimental results. A completed run writes its
actual metrics to `results/stage_f1/`.
